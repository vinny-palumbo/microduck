"""Arrival review is independent, strict, bounded, and fails closed on provider errors."""

import asyncio
import base64
import copy
import json

import aiohttp
import pytest

from duck_nav.arrival import GeminiArrivalReviewer


def report(**changes):
    return {
        "destination_visible": True,
        "inside_destination": False,
        "evidence": "A sink and counter are visible beyond an open doorway.",
        "uncertainty": "The camera appears to be in the adjoining room.",
        **changes,
    }


def response(args=None):
    return {
        "candidates": [
            {
                "finishReason": "STOP",
                "content": {
                    "parts": [
                        {"functionCall": {"name": "report_arrival", "args": args or report()}}
                    ]
                },
            }
        ]
    }


def views():
    return [{"label": "forward", "jpeg": b"first image"}, {"label": "left", "jpeg": b"second"}]


def install_http(monkeypatch, *, data=None, status=200, enter_error=None, json_error=None):
    captured = {}

    class Reply:
        async def __aenter__(self):
            if enter_error:
                raise enter_error
            return self

        async def __aexit__(self, *args):
            return False

        async def json(self):
            captured["read_body"] = True
            if json_error:
                raise json_error
            return copy.deepcopy(data if data is not None else response())

    class Session:
        def __init__(self, *, timeout):
            captured["timeout"] = timeout.total

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def post(self, url, **kwargs):
            captured.update(url=url, **kwargs)
            reply = Reply()
            reply.status = status
            return reply

    monkeypatch.setattr("duck_nav.arrival.aiohttp.ClientSession", Session)
    return captured


def test_payload_contains_only_goal_and_ordered_views():
    payload = GeminiArrivalReviewer("unused").payload("kitchen", views())
    assert len(payload["contents"]) == 1
    parts = payload["contents"][0]["parts"]
    assert json.loads(parts[0]["text"]) == {"goal": "kitchen"}
    assert json.loads(parts[1]["text"]) == {"label": "forward"}
    assert base64.b64decode(parts[2]["inlineData"]["data"]) == b"first image"
    assert json.loads(parts[3]["text"]) == {"label": "left"}
    assert base64.b64decode(parts[4]["inlineData"]["data"]) == b"second"
    assert len(parts) == 5
    config = payload["toolConfig"]["functionCallingConfig"]
    assert config == {"mode": "ANY", "allowedFunctionNames": ["report_arrival"]}
    declaration = payload["tools"][0]["functionDeclarations"][0]
    assert declaration["parametersJsonSchema"]["additionalProperties"] is False


@pytest.mark.parametrize("extra", ["simulator_truth", "odom", "history", "model_claim"])
def test_navigation_metadata_cannot_enter_reviewer(extra):
    contaminated = views()
    contaminated[0][extra] = "must not be sent"
    with pytest.raises(ValueError, match="only label and jpeg"):
        GeminiArrivalReviewer("unused").payload("kitchen", contaminated)


@pytest.mark.parametrize(
    "goal,images",
    [(None, views()), (" ", views()), ("x" * 2001, views()), ("kitchen", []), ("kitchen", None)],
)
def test_invalid_review_inputs(goal, images):
    with pytest.raises(ValueError):
        GeminiArrivalReviewer("unused").payload(goal, images)


@pytest.mark.parametrize("change", [{"label": ""}, {"label": 3}, {"jpeg": "bytes"}, {"jpeg": b""}])
def test_invalid_views(change):
    image = {"label": "forward", "jpeg": b"image", **change}
    with pytest.raises(ValueError):
        GeminiArrivalReviewer("unused").payload("kitchen", [image])


@pytest.mark.parametrize("inside", [False, True])
def test_valid_assessment_is_copied(inside):
    data = response(report(inside_destination=inside))
    result = GeminiArrivalReviewer.parse(data)
    assert result == report(inside_destination=inside)
    result["evidence"] = "changed"
    assert (
        data["candidates"][0]["content"]["parts"][0]["functionCall"]["args"]["evidence"]
        != "changed"
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"destination_visible": 1},
        {"inside_destination": "true"},
        {"inside_destination": True, "destination_visible": False},
        {"evidence": ""},
        {"evidence": "x" * 1501},
        {"uncertainty": "x" * 1001},
        {"uncertainty": None},
        {"position": [1, 2]},
    ],
)
def test_invalid_report_fails_closed(changes):
    with pytest.raises(ValueError, match="invalid Gemini arrival assessment"):
        GeminiArrivalReviewer.parse(response(report(**changes)))


@pytest.mark.parametrize("bad", [None, [], {}, {"candidates": []}, {"candidates": [None]}])
def test_malformed_response(bad):
    with pytest.raises(ValueError, match="invalid Gemini arrival assessment"):
        GeminiArrivalReviewer.parse(bad)


@pytest.mark.parametrize("failure", ["blocked", "multiple", "missing", "wrong_tool", "bad_args"])
def test_invalid_function_response(failure):
    data = response()
    candidate = data["candidates"][0]
    parts = candidate["content"]["parts"]
    if failure == "blocked":
        candidate["finishReason"] = "MAX_TOKENS"
    elif failure == "multiple":
        parts.append(copy.deepcopy(parts[0]))
    elif failure == "missing":
        parts.clear()
    elif failure == "wrong_tool":
        parts[0]["functionCall"]["name"] = "move"
    else:
        parts[0]["functionCall"]["args"] = "{}"
    with pytest.raises(ValueError, match="invalid Gemini arrival assessment"):
        GeminiArrivalReviewer.parse(data)


@pytest.mark.asyncio
async def test_http_request_has_header_key_timeout_and_no_redirects(monkeypatch):
    sent = install_http(monkeypatch)
    reviewer = GeminiArrivalReviewer("secret")
    assert await reviewer.review("kitchen", views()) == report()
    assert sent["url"].endswith("/gemini-robotics-er-2-preview:generateContent")
    assert "secret" not in sent["url"]
    assert sent["headers"] == {"x-goog-api-key": "secret"}
    assert sent["timeout"] == 30
    assert sent["allow_redirects"] is False


@pytest.mark.asyncio
async def test_visible_destination_with_uncertain_threshold_remains_rejected(monkeypatch):
    # The independent reviewer must receive every side view, preserve their order,
    # and return its geometry rejection unchanged even if the destination is visible.
    ordered = [
        {"label": label, "jpeg": f"actual-{label}-bytes".encode()}
        for label in ("front", "left45", "right45", "front_final")
    ]
    assessment = report(
        destination_visible=True,
        inside_destination=False,
        evidence="Appliances identify the kitchen; right45 places an entry jamb beside the camera.",
        uncertainty="The robot may straddle the threshold; full body entry is not established.",
    )
    sent = install_http(monkeypatch, data=response(assessment))
    result = await GeminiArrivalReviewer("secret").review("kitchen", ordered)
    assert result == assessment
    parts = sent["json"]["contents"][0]["parts"]
    assert [json.loads(parts[index]["text"])["label"] for index in (1, 3, 5, 7)] == [
        view["label"] for view in ordered
    ]
    assert [base64.b64decode(parts[index]["inlineData"]["data"]) for index in (2, 4, 6, 8)] == [
        view["jpeg"] for view in ordered
    ]
    assert set(result) == {"destination_visible", "inside_destination", "evidence", "uncertainty"}


@pytest.mark.asyncio
async def test_http_error_does_not_read_provider_body(monkeypatch):
    sent = install_http(monkeypatch, status=403, data={"error": "secret provider body"})
    with pytest.raises(RuntimeError, match="^Gemini arrival review HTTP 403$"):
        await GeminiArrivalReviewer("secret").review("kitchen", views())
    assert "read_body" not in sent


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [aiohttp.ClientError("secret"), TimeoutError("secret")])
async def test_transport_errors_are_redacted(monkeypatch, error):
    install_http(monkeypatch, enter_error=error)
    with pytest.raises((RuntimeError, TimeoutError)) as caught:
        await GeminiArrivalReviewer("secret").review("kitchen", views())
    assert "secret" not in str(caught.value)


@pytest.mark.asyncio
async def test_invalid_json_is_redacted(monkeypatch):
    install_http(monkeypatch, json_error=ValueError("secret response"))
    with pytest.raises(ValueError, match="^invalid Gemini arrival response$"):
        await GeminiArrivalReviewer("secret").review("kitchen", views())


@pytest.mark.asyncio
async def test_cancellation_is_not_an_arrival_result(monkeypatch):
    install_http(monkeypatch, enter_error=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await GeminiArrivalReviewer("secret").review("kitchen", views())
