"""Recorded perception checks keep scorer labels out of model input and fail honestly."""

import copy
import hashlib
import io
import json
import runpy
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def validation():
    return runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "validate_arrival.py"))


def test_bundled_images_are_small_valid_jpegs_with_checked_provenance(validation):
    cases = validation["load_cases"]()
    assert [case["name"] for case in cases] == [
        "doorway_008",
        "interior",
        "wall_004",
        "non_kitchen_011",
    ]
    assert sum(case["repetitions"] for case in cases) == 8
    assert sum(len(view["jpeg"]) for case in cases for view in case["model_views"]) < 1_000_000
    for case in cases:
        for view, source in zip(case["model_views"], case["views"], strict=True):
            assert hashlib.sha256(view["jpeg"]).hexdigest() == source["jpeg_sha256"]
            assert source["provenance"]["source_file"]
            with Image.open(io.BytesIO(view["jpeg"])) as image:
                assert image.format == "JPEG"
                image.load()


async def test_provider_receives_only_goal_labels_and_exact_images(validation):
    cases = validation["load_cases"]()
    calls, emitted = [], []
    expected = [case for case in cases for _ in range(case["repetitions"])]

    class Reviewer:
        model = "fixture"

        async def review(self, goal, views):
            case = expected[len(calls)]
            assert goal == case["goal"]
            assert views == case["model_views"]
            assert all(set(view) == {"label", "jpeg"} for view in views)
            calls.append((goal, copy.deepcopy(views)))
            return {
                **case["expected"],
                "evidence": "Observed geometry",
                "uncertainty": "Model assessment",
            }

    report = await validation["evaluate"](Reviewer(), cases, emitted.append)
    assert report["all_passed"] is True
    assert len(calls) == len(report["results"]) == len(emitted) == 8
    assert all("PASS" in line for line in emitted)
    # The local report retains labels/provenance for auditing, without image bytes.
    assert "jpeg_sha256" in json.dumps(report)
    assert "expected" in report["results"][0]


@pytest.mark.parametrize("case_index", [0, 1, 2, 3])
@pytest.mark.parametrize("field", ["destination_visible", "inside_destination"])
async def test_wrong_perception_fails_the_local_scorer(validation, case_index, field):
    case = validation["load_cases"]()[case_index]
    case["repetitions"] = 1

    class Reviewer:
        model = "fixture"

        async def review(self, goal, views):
            return {
                **case["expected"],
                field: not case["expected"][field],
                "evidence": "Contradictory assessment",
                "uncertainty": "Unknown",
            }

    report = await validation["evaluate"](Reviewer(), [case], lambda line: None)
    assert report["all_passed"] is False
    assert report["results"][0]["passed"] is False


async def test_provider_failure_is_redacted_and_does_not_skip_other_cases(validation):
    calls, messages = [], []

    class Reviewer:
        model = "fixture"

        async def review(self, goal, views):
            calls.append(goal)
            raise RuntimeError("credential=DO_NOT_PRINT provider details")

    report = await validation["evaluate"](Reviewer(), validation["load_cases"](), messages.append)
    assert report["all_passed"] is False
    assert len(calls) == 8
    assert all(result["error_type"] == "RuntimeError" for result in report["results"])
    assert "DO_NOT_PRINT" not in json.dumps(report) + repr(messages)


def test_changed_fixture_bytes_are_rejected_before_review(validation, tmp_path):
    (tmp_path / "changed.jpg").write_bytes(b"changed")
    (tmp_path / "manifest.json").write_text(
        json.dumps({"cases": [{"views": [{"file": "changed.jpg", "jpeg_sha256": "old hash"}]}]})
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        validation["load_cases"](tmp_path)
