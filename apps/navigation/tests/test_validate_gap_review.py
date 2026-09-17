"""Replay fixtures preserve exact annotations without leaking scorer expectations."""

import base64
import hashlib
import io
import json
import runpy
import shutil
import sys
from pathlib import Path

import pytest
from PIL import Image

from duck_nav.gap_review import GeminiGapReviewer

SCRIPT = Path(__file__).parents[1] / "scripts" / "validate_gap_review.py"
EXPECTED_SOURCES = {
    "018-view00059": {
        "run": "20260917T014258Z-f579a09f",
        "sha256": "9308ee869a585f3fc9c1d4ad78452d836f62d72c1dfaba9e1cb482343bce4f22",
        "camera": {
            "view_id": "view-00059",
            "label": "right",
            "yaw_deg": -38.03752770014445,
            "pitch_deg": -7.761009541354485,
        },
        "pair": [[560, 220], [495, 750]],
        "flags": [False, False, False],
        "at": 1789609620.7304387,
    },
    "018-view00053": {
        "run": "20260917T014258Z-f579a09f",
        "sha256": "fbf360194c8c21e52bf93030b6ad7c7216148710f62fc57ebc34e1856ddccad2",
        "camera": {
            "view_id": "view-00053",
            "label": "left",
            "yaw_deg": 26.497518612568047,
            "pitch_deg": -10.48912769752878,
        },
        "pair": [[540, 215], [465, 765]],
        "flags": [True, False, False],
        "at": 1789609595.614664,
    },
    "013-view00032": {
        "run": "20260917T000748Z-50e6f8a8",
        "sha256": "3b8d3d409151344d3faf5a418c3e31ac08890f9f4b87d6b4c65281735c13d596",
        "camera": {
            "view_id": "view-00032",
            "label": "left",
            "yaw_deg": 44.060892383614735,
            "pitch_deg": -8.422090289128398,
        },
        "pair": [[598, 325], [480, 960]],
        "flags": [True, True, True],
        "at": 1789603774.8265262,
    },
}


@pytest.fixture
def validation():
    return runpy.run_path(str(SCRIPT))


def copied_manifest(validation, tmp_path):
    shutil.copytree(validation["FIXTURES"], tmp_path, dirs_exist_ok=True)
    return json.loads((tmp_path / "manifest.json").read_text())


def save_manifest(tmp_path, manifest):
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))


def test_import_and_fixture_loading_do_not_read_credentials_or_open_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Offline import/fixture validation attempted external access")

    monkeypatch.setattr("duck_nav.credentials.load_gemini_key", forbidden)
    monkeypatch.setattr("aiohttp.ClientSession", forbidden)
    validation = runpy.run_path(str(SCRIPT))
    assert len(validation["load_cases"]()) == 3


def test_exact_jpegs_points_camera_and_recording_provenance_are_bundled(validation):
    cases = validation["load_cases"]()
    assert tuple(case["name"] for case in cases) == tuple(EXPECTED_SOURCES)
    assert sum(len(case["jpeg"]) for case in cases) == 50600
    for case in cases:
        expected = EXPECTED_SOURCES[case["name"]]
        image = case["image"]
        provenance = case["provenance"]
        assert hashlib.sha256(case["jpeg"]).hexdigest() == expected["sha256"]
        assert image["jpeg_sha256"] == provenance["source_sha256"] == expected["sha256"]
        assert case["inputs"]["camera"] == expected["camera"]
        assert case["inputs"]["proposed_endpoints"] == expected["pair"]
        assert case["inputs_sha256"] == validation["json_sha256"](case["inputs"])
        assert [case["expected"][key] for key in validation["VERDICT_KEYS"]] == expected["flags"]
        assert provenance["source_file"] == (
            f"runs/{expected['run']}/{expected['camera']['view_id']}.jpg"
        )
        assert provenance["source_observation_at"] == expected["at"]
        assert provenance["conversion"] == "unchanged JPEG bytes"
        assert provenance["probe_inputs_file"].endswith(f"/{case['name']}-inputs.json")
        assert len(provenance["probe_inputs_sha256"]) == 64
        with Image.open(io.BytesIO(case["jpeg"])) as picture:
            assert picture.format == "JPEG"
            assert picture.size == (image["width"], image["height"]) == (360, 640)
            picture.load()
    manifest = json.loads((validation["FIXTURES"] / "manifest.json").read_text())
    assert manifest["original_probe"]["call_count"] == 3
    assert "10px" in cases[-1]["annotation_note"]
    assert "not ground truth" in manifest["description"]
    assert "does not establish" in manifest["limits"]


async def test_provider_gets_only_exact_image_camera_and_points_once_per_case(validation):
    cases = validation["load_cases"]()
    calls, messages = [], []

    class Reviewer:
        model = "fixture"

        async def review(self, jpeg, *, camera, point, opposite_point):
            case = cases[len(calls)]
            assert jpeg == case["jpeg"]
            assert camera == case["inputs"]["camera"]
            assert [point, opposite_point] == case["inputs"]["proposed_endpoints"]
            payload = GeminiGapReviewer.payload(
                jpeg, camera=camera, point=point, opposite_point=opposite_point
            )
            parts = payload["contents"][0]["parts"]
            assert len(parts) == 2
            assert json.loads(parts[0]["text"]) == case["inputs"]
            assert base64.b64decode(parts[1]["inlineData"]["data"]) == jpeg
            assert not {"expected", "provenance", "annotation_note", "goal"} & set(
                json.loads(parts[0]["text"])
            )
            calls.append(payload)
            return {**case["expected"], "evidence": "Fixture response"}

    report = await validation["evaluate"](Reviewer(), cases, messages.append)
    assert report["all_passed"] is True
    assert report["actions_executed"] is False
    assert report["call_count"] == len(calls) == len(messages) == 3
    assert all("PASS" in message for message in messages)
    assert len(report["prompt_sha256"]) == len(report["schema_sha256"]) == 64
    assert report["results"][0]["provenance"] == cases[0]["provenance"]
    assert "jpeg" not in report["results"][0]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("case_index", range(3))
@pytest.mark.parametrize(
    "field", ["doorway_visible", "both_contacts_visible", "points_match_contacts"]
)
async def test_any_changed_verdict_fails_local_regression(validation, case_index, field):
    case = validation["load_cases"]()[case_index]

    class Reviewer:
        model = "fixture"

        async def review(self, jpeg, **kwargs):
            return {**case["expected"], field: not case["expected"][field], "evidence": "Changed"}

    report = await validation["evaluate"](Reviewer(), [case], lambda line: None)
    assert report["all_passed"] is False
    assert report["results"][0]["passed"] is False


async def test_failure_is_redacted_and_remaining_cases_run_once(validation):
    calls, messages = [], []

    class Reviewer:
        model = "fixture"

        async def review(self, jpeg, **kwargs):
            calls.append(jpeg)
            raise RuntimeError("private credential=DO_NOT_PRINT and provider URL")

    report = await validation["evaluate"](Reviewer(), validation["load_cases"](), messages.append)
    assert len(calls) == report["call_count"] == 3
    assert report["all_passed"] is False
    assert all(result["error_type"] == "RuntimeError" for result in report["results"])
    assert "DO_NOT_PRINT" not in json.dumps(report) + repr(messages)


def test_wrong_image_checksum_fails_before_review(validation, tmp_path):
    manifest = copied_manifest(validation, tmp_path)
    image = tmp_path / manifest["cases"][0]["image"]["file"]
    image.write_bytes(image.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="image checksum mismatch"):
        validation["load_cases"](tmp_path)


@pytest.mark.parametrize("change", ["empty", "missing", "duplicate", "extra"])
def test_missing_or_duplicate_cases_cannot_silently_reduce_coverage(validation, tmp_path, change):
    manifest = copied_manifest(validation, tmp_path)
    cases = manifest["cases"]
    if change == "empty":
        cases.clear()
    elif change == "missing":
        cases.pop()
    elif change == "duplicate":
        cases[2] = cases[1]
    else:
        cases.append(cases[0])
    save_manifest(tmp_path, manifest)
    with pytest.raises(ValueError, match="all three named cases exactly once"):
        validation["load_cases"](tmp_path)


@pytest.mark.parametrize("change", ["point", "camera"])
def test_changed_source_annotations_fail_input_checksum(validation, tmp_path, change):
    manifest = copied_manifest(validation, tmp_path)
    inputs = manifest["cases"][0]["inputs"]
    if change == "point":
        inputs["proposed_endpoints"][0][0] += 1
    else:
        inputs["camera"]["yaw_deg"] += 1
    save_manifest(tmp_path, manifest)
    with pytest.raises(ValueError, match="input checksum mismatch"):
        validation["load_cases"](tmp_path)


def test_missing_image_cannot_be_skipped(validation, tmp_path):
    manifest = copied_manifest(validation, tmp_path)
    (tmp_path / manifest["cases"][0]["image"]["file"]).unlink()
    with pytest.raises(FileNotFoundError):
        validation["load_cases"](tmp_path)


def test_fixture_cannot_read_image_outside_bundle(validation, tmp_path):
    manifest = copied_manifest(validation, tmp_path)
    manifest["cases"][0]["image"]["file"] = "../outside.jpg"
    save_manifest(tmp_path, manifest)
    with pytest.raises(ValueError, match="inside the fixture directory"):
        validation["load_cases"](tmp_path)


def test_source_checksum_disagreement_is_rejected(validation, tmp_path):
    manifest = copied_manifest(validation, tmp_path)
    manifest["cases"][0]["provenance"]["source_sha256"] = "0" * 64
    save_manifest(tmp_path, manifest)
    with pytest.raises(ValueError, match="provenance mismatch"):
        validation["load_cases"](tmp_path)


def test_explicit_cli_loads_fresh_key_and_makes_only_three_calls_per_invocation(
    validation, monkeypatch, tmp_path, capsys
):
    cases = validation["load_cases"]()
    keys, calls = [], []

    def fresh_key():
        key = f"PRIVATE_KEY_{len(keys)}"
        keys.append(key)
        return key

    class Reviewer:
        model = "fixture"

        def __init__(self, key):
            assert key == keys[-1]

        async def review(self, jpeg, **kwargs):
            case = cases[len(calls) % 3]
            calls.append(jpeg)
            return {**case["expected"], "evidence": "Fixture response"}

    namespace = validation["main"].__globals__
    monkeypatch.setitem(namespace, "load_cases", lambda: cases)
    monkeypatch.setitem(namespace, "load_gemini_key", fresh_key)
    monkeypatch.setitem(namespace, "GeminiGapReviewer", Reviewer)
    for invocation in range(2):
        output = tmp_path / f"report-{invocation}.json"
        monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--output", str(output)])
        assert validation["main"]() == 0
        report = json.loads(output.read_text())
        assert report["call_count"] == 3
        assert report["all_passed"] is True
        assert "PRIVATE_KEY" not in output.read_text()
    assert len(keys) == 2
    assert len(calls) == 6
    assert "PRIVATE_KEY" not in capsys.readouterr().out


def test_cli_credential_error_is_redacted(validation, monkeypatch, capsys):
    def failed_key():
        raise RuntimeError("PRIVATE_KEY must never be printed")

    namespace = validation["main"].__globals__
    monkeypatch.setitem(namespace, "load_gemini_key", failed_key)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT)])
    assert validation["main"]() == 2
    assert capsys.readouterr().out == "Gap review validation failed: RuntimeError\n"
