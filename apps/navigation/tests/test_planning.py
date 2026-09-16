"""Visual planning cannot receive raw truth, change the goal, or bypass tool validation."""

import asyncio
import base64
import copy
import io
import json

import aiohttp
import pytest
from PIL import Image

from duck_nav.live import declarations
from duck_nav.planning import ALLOWED_TOOLS, GeminiVisualPlanner


def planner():
    return GeminiVisualPlanner("secret", declarations())


def jpeg(color="black", *, format="JPEG"):
    output = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(output, format=format)
    return output.getvalue()


JPEG = jpeg()


def camera(**changes):
    return {
        "label": "front",
        "yaw_deg": 0.0,
        "pitch_deg": 0.0,
        "received_at": 100.0,
        "state_received_at": 99.99,
        "age_s": 0.0,
        "view_id": "frame-0001",
        "body_position_delta_m": 0.0,
        "body_heading_delta_deg": 0.0,
        **changes,
    }


def context(**changes):
    return {
        "goal": "kitchen",
        "ready": True,
        "guard_reason": None,
        "depth": {},
        "odometry": {"position": [0.1, 0.2, 0.12], "yaw": 0.1},
        "motion": {"requested": [0, 0, 0]},
        "recent_actions": [{"source": "robot_odometry_not_simulator_ground_truth"}],
        "remembered_places": {},
        "recovery": None,
        "step": 4,
        "arrival_claims": 1,
        "arrival_review": {"destination_visible": False, "inside_destination": False},
        **changes,
    }


def response(name="advance", args=None):
    if args is None:
        args = {"distance_m": 0.1, "heading_deg": -20, "reason": "Align with the visible doorway."}
    return {
        "candidates": [
            {
                "finishReason": "STOP",
                "content": {"parts": [{"functionCall": {"name": name, "args": args}}]},
            }
        ]
    }


def install_http(monkeypatch, *, status=200, error=None, json_error=None):
    sent = {}

    class Reply:
        async def __aenter__(self):
            if error is not None:
                raise error
            return self

        async def __aexit__(self, *args):
            return False

        async def json(self):
            sent["read_body"] = True
            if json_error:
                raise json_error
            return response()

    class Session:
        def __init__(self, *, timeout):
            sent["timeout"] = timeout.total

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def post(self, url, **kwargs):
            sent.update(url=url, **kwargs)
            reply = Reply()
            reply.status = status
            return reply

    monkeypatch.setattr("duck_nav.planning.aiohttp.ClientSession", Session)
    return sent


def test_live_declarations_convert_to_filtered_http_tools():
    original = declarations()
    saved = copy.deepcopy(original)
    model = GeminiVisualPlanner("secret", original)
    payload = model.payload(context(), JPEG)
    tools = payload["tools"][0]["functionDeclarations"]
    assert {tool["name"] for tool in tools} == ALLOWED_TOOLS
    assert all(set(tool) == {"name", "description", "parametersJsonSchema"} for tool in tools)
    assert original == saved
    config = payload["toolConfig"]["functionCallingConfig"]
    assert config["mode"] == "ANY"
    assert set(config["allowedFunctionNames"]) == ALLOWED_TOOLS


def test_stateless_payload_has_one_current_image_and_explicit_context():
    model = planner()
    old, new = jpeg("red"), jpeg("blue")
    first = model.payload(context(goal="kitchen"), old)
    second = model.payload(context(goal="bedroom"), new)
    assert len(second["contents"]) == 1
    parts = second["contents"][0]["parts"]
    assert len(parts) == 3
    assert json.loads(parts[0]["text"]) == context(goal="bedroom")
    assert json.loads(parts[1]["text"]) == {"view": "current", "camera": None}
    assert base64.b64decode(parts[2]["inlineData"]["data"]) == new
    assert "kitchen" not in parts[0]["text"]
    first["tools"][0]["functionDeclarations"].clear()
    assert len(model.declarations) == 5


@pytest.mark.parametrize("reverse_prior_order", [False, True])
def test_prior_views_are_chronological_and_current_view_is_last(reverse_prior_order):
    current = context(camera=camera(), ready=False, guard_reason="obstacle")
    views = [
        {
            "camera": camera(
                label="left",
                yaw_deg=44.0,
                received_at=80.0,
                state_received_at=79.99,
                age_s=20.0,
                view_id="frame-left",
                body_position_delta_m=0.01,
                body_heading_delta_deg=2.0,
            ),
            "jpeg": jpeg("red"),
        },
        {
            "camera": camera(
                label="right",
                yaw_deg=-45.0,
                pitch_deg=2.0,
                received_at=90.0,
                state_received_at=89.99,
                age_s=10.0,
                view_id="frame-right",
            ),
            "jpeg": jpeg("blue"),
        },
    ]
    if reverse_prior_order:
        views.reverse()
    original = copy.deepcopy((current, views))
    payload = planner().payload(current, JPEG, views=views)
    parts = payload["contents"][0]["parts"]
    assert len(parts) == 7
    assert json.loads(parts[0]["text"]) == current
    assert json.loads(parts[-2]["text"]) == {"view": "current", "camera": current["camera"]}
    assert base64.b64decode(parts[-1]["inlineData"]["data"]) == JPEG
    for index, view in enumerate(sorted(views, key=lambda view: view["camera"]["received_at"])):
        assert json.loads(parts[1 + index * 2]["text"]) == {
            "view": "prior_stationary_scan",
            "camera": view["camera"],
        }
        assert base64.b64decode(parts[2 + index * 2]["inlineData"]["data"]) == view["jpeg"]
    assert (current, views) == original
    instruction = payload["systemInstruction"]["parts"][0]["text"]
    assert "relative to the BODY, not the camera" in instruction
    assert "Prior scans never override the current ready state or current depth" in instruction
    assert "does not clear a physical obstacle" in instruction


@pytest.mark.parametrize("camera_value", [None, camera()])
def test_current_camera_can_be_known_or_explicitly_unknown(camera_value):
    parts = planner().payload(context(camera=camera_value), JPEG)["contents"][0]["parts"]
    assert json.loads(parts[1]["text"])["camera"] == camera_value


@pytest.mark.parametrize("placement", ["current", "prior"])
@pytest.mark.parametrize(
    "changes",
    [
        {"label": "back"},
        {"label": None},
        {"view_id": ""},
        {"view_id": " "},
        {"view_id": 1},
        {"view_id": "x" * 201},
        {"yaw_deg": float("nan")},
        {"pitch_deg": float("inf")},
        {"received_at": float("-inf")},
        {"state_received_at": 10**1000},
        {"received_at": -1.0},
        {"state_received_at": -0.01},
        {"age_s": -0.01},
        {"age_s": 30.01},
        {"body_position_delta_m": -0.001},
        {"body_position_delta_m": 0.0251},
        {"body_heading_delta_deg": -5.01},
        {"body_heading_delta_deg": 5.01},
        {"yaw_deg": "0.0"},
        {"age_s": False},
        {"image_path": "/tmp/local.jpg"},
        {"ground_truth": {"room": "kitchen"}},
    ],
)
def test_invalid_camera_metadata_is_rejected_in_every_image(placement, changes):
    metadata = camera(**changes)
    with pytest.raises(ValueError):
        if placement == "current":
            planner().payload(context(camera=metadata), JPEG)
        else:
            planner().payload(context(), JPEG, views=[{"camera": metadata, "jpeg": JPEG}])


@pytest.mark.parametrize("missing", list(camera()))
@pytest.mark.parametrize("placement", ["current", "prior"])
def test_camera_metadata_requires_all_fields(missing, placement):
    metadata = camera()
    del metadata[missing]
    with pytest.raises(ValueError, match="camera metadata requires exactly"):
        if placement == "current":
            planner().payload(context(camera=metadata), JPEG)
        else:
            planner().payload(context(), JPEG, views=[{"camera": metadata, "jpeg": JPEG}])


@pytest.mark.parametrize(
    "views",
    [
        "frame.jpg",
        {},
        (),
        [None],
        [{}],
        [{"jpeg": JPEG}],
        [{"camera": None, "jpeg": JPEG}],
        [{"camera": camera()}],
        [{"camera": camera(), "jpeg": JPEG, "path": "/tmp/private.jpg"}],
        [{"camera": camera(), "jpeg": JPEG}] * 3,
    ],
)
def test_prior_views_fail_closed_on_missing_unknown_or_excess_data(views):
    with pytest.raises(ValueError):
        planner().payload(context(), JPEG, views=views)


@pytest.mark.parametrize("value", [None, [], "front"])
def test_prior_camera_cannot_have_unknown_orientation(value):
    with pytest.raises(ValueError):
        planner().payload(context(), JPEG, views=[{"camera": value, "jpeg": JPEG}])


@pytest.mark.parametrize("heading_delta", [-5.0, 5.0])
def test_stationary_scan_bounds_are_inclusive(heading_delta):
    view = {
        "camera": camera(
            age_s=30.0, body_position_delta_m=0.025, body_heading_delta_deg=heading_delta
        ),
        "jpeg": JPEG,
    }
    assert len(planner().payload(context(), JPEG, views=[view])["contents"][0]["parts"]) == 5


@pytest.mark.parametrize("extra", ["state", "simulator_truth", "map", "target_coordinates"])
def test_unknown_context_is_rejected(extra):
    with pytest.raises(ValueError, match="unexpected visual planning context"):
        planner().payload(context(**{extra: "do not send"}), JPEG)


@pytest.mark.parametrize("key", ["simulator_truth", "ground_truth", "qpos", "qvel"])
def test_nested_raw_truth_is_rejected(key):
    with pytest.raises(ValueError, match="simulator truth is forbidden"):
        planner().payload(context(recent_actions=[{"result": {key: [1, 2, 3]}}]), JPEG)


def test_tuple_cannot_hide_serializable_truth():
    with pytest.raises(ValueError, match="simulator truth is forbidden"):
        planner().payload(context(recent_actions=({"simulator_truth": [1, 2, 3]},)), JPEG)


@pytest.mark.parametrize(
    "changes",
    [{"ready": 1}, {"goal": " "}, {"step": True}, {"arrival_claims": -1}, {"depth": float("nan")}],
)
def test_malformed_context_is_rejected(changes):
    with pytest.raises(ValueError):
        planner().payload(context(**changes), JPEG)


@pytest.mark.parametrize("image", [None, "not bytes", b"", b"image", jpeg(format="PNG"), JPEG[:50]])
@pytest.mark.parametrize("placement", ["current", "prior"])
def test_invalid_image(image, placement):
    with pytest.raises(ValueError):
        if placement == "current":
            planner().payload(context(), image)
        else:
            planner().payload(context(), JPEG, views=[{"camera": camera(), "jpeg": image}])


@pytest.mark.parametrize(
    "name,args",
    [
        ("observe", {"reason": "Inspect before moving."}),
        ("look_at", {"x": 1, "y": 0, "z": 0, "reason": "Recenter before moving."}),
        ("advance", {"distance_m": 0.1, "reason": "Clear floor ahead."}),
        (
            "remember_place",
            {"name": "doorway", "observation": "Counter beyond.", "explored": False},
        ),
        ("finish", {"status": "blocked", "reason": "No clear route."}),
    ],
)
def test_known_tools_parse_without_mutating_response(name, args):
    data = response(name, args)
    result = planner().parse(data)
    assert result == {"name": name, "args": args}
    result["args"].clear()
    assert data["candidates"][0]["content"]["parts"][0]["functionCall"]["args"] == args


@pytest.mark.parametrize("name", ["start_navigation", "say", "stop", "move_for", "unknown"])
def test_out_of_scope_tools_are_rejected(name):
    with pytest.raises(ValueError, match="invalid Gemini visual planning decision"):
        planner().parse(response(name, {"reason": "not allowed"}))


@pytest.mark.parametrize(
    "args",
    [
        {},
        {"distance_m": 0.1},
        {"distance_m": True, "reason": "visible"},
        {"distance_m": "0.1", "reason": "visible"},
        {"distance_m": float("nan"), "reason": "visible"},
        {"distance_m": float("inf"), "reason": "visible"},
        {"distance_m": 10**1000, "reason": "visible"},
        {"distance_m": 0.1, "reason": " "},
        {"distance_m": 0.1, "reason": "x" * 1001},
        {"distance_m": 0.1, "reason": "visible", "override_guard": True},
    ],
)
def test_invalid_action_arguments(args):
    with pytest.raises(ValueError, match="invalid Gemini visual planning decision"):
        planner().parse(response(args=args))


def test_runtime_retains_numeric_bound_enforcement():
    # Parsing validates the wire contract; the guarded executor owns action limits.
    args = {"distance_m": 0.3, "reason": "Runtime must reject this distance."}
    assert planner().parse(response(args=args))["args"] == args


@pytest.mark.parametrize(
    "name,args",
    [
        ("remember_place", {"name": "a", "observation": "b", "explored": 1}),
        ("finish", {"status": "success", "reason": "invented status"}),
    ],
)
def test_schema_booleans_and_enums_are_strict(name, args):
    with pytest.raises(ValueError):
        planner().parse(response(name, args))


@pytest.mark.parametrize("bad", [None, [], {}, {"candidates": []}, {"candidates": [None]}])
def test_malformed_provider_response(bad):
    with pytest.raises(ValueError):
        planner().parse(bad)


@pytest.mark.parametrize("failure", ["blocked", "multiple", "missing", "bad_args"])
def test_provider_must_return_one_complete_tool(failure):
    data = response()
    candidate = data["candidates"][0]
    parts = candidate["content"]["parts"]
    if failure == "blocked":
        candidate["finishReason"] = "MAX_TOKENS"
    elif failure == "multiple":
        parts.append(copy.deepcopy(parts[0]))
    elif failure == "missing":
        parts.clear()
    else:
        parts[0]["functionCall"]["args"] = "{}"
    with pytest.raises(ValueError):
        planner().parse(data)


@pytest.mark.asyncio
async def test_http_contract(monkeypatch):
    sent = install_http(monkeypatch)
    assert (await planner().decide(context(), JPEG))["name"] == "advance"
    assert sent["url"].endswith("/gemini-robotics-er-2-preview:generateContent")
    assert "secret" not in sent["url"]
    assert sent["headers"] == {"x-goog-api-key": "secret"}
    assert sent["timeout"] == 30
    assert sent["allow_redirects"] is False


@pytest.mark.asyncio
async def test_http_passes_validated_supplemental_images(monkeypatch):
    sent = install_http(monkeypatch)
    views = [{"camera": camera(label="left", yaw_deg=45.0), "jpeg": jpeg("red")}]
    assert (await planner().decide(context(camera=camera()), JPEG, views=views))[
        "name"
    ] == "advance"
    parts = sent["json"]["contents"][0]["parts"]
    assert json.loads(parts[1]["text"])["camera"] == views[0]["camera"]
    assert base64.b64decode(parts[2]["inlineData"]["data"]) == views[0]["jpeg"]
    assert json.loads(parts[-2]["text"]) == {"view": "current", "camera": camera()}
    assert base64.b64decode(parts[-1]["inlineData"]["data"]) == JPEG


@pytest.mark.asyncio
async def test_invalid_prior_image_fails_before_network(monkeypatch):
    sent = install_http(monkeypatch)
    with pytest.raises(ValueError, match="valid JPEG"):
        await planner().decide(context(), JPEG, views=[{"camera": camera(), "jpeg": b"invalid"}])
    assert sent == {}


@pytest.mark.asyncio
async def test_http_error_never_reads_body(monkeypatch):
    sent = install_http(monkeypatch, status=403)
    with pytest.raises(RuntimeError, match="^Gemini visual planning HTTP 403$"):
        await planner().decide(context(), JPEG)
    assert "read_body" not in sent


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [aiohttp.ClientError("secret"), TimeoutError("secret")])
async def test_transport_errors_are_redacted(monkeypatch, error):
    install_http(monkeypatch, error=error)
    with pytest.raises((RuntimeError, TimeoutError)) as caught:
        await planner().decide(context(), JPEG)
    assert "secret" not in str(caught.value)


@pytest.mark.asyncio
async def test_invalid_json_is_redacted(monkeypatch):
    install_http(monkeypatch, json_error=ValueError("secret body"))
    with pytest.raises(ValueError, match="^invalid Gemini visual planning response$"):
        await planner().decide(context(), JPEG)


@pytest.mark.asyncio
async def test_cancellation_propagates(monkeypatch):
    install_http(monkeypatch, error=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await planner().decide(context(), JPEG)
