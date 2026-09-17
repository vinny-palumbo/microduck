"""Manually replay three recorded semantic gap-review cases without a robot.

Run `uv run python scripts/validate_gap_review.py` from apps/navigation. Explicit
invocation makes exactly three provider requests, one per checked fixture, using
a freshly loaded saved Gemini key. Importing this file or running pytest makes
no requests. The reviewer receives only the exact JPEG, measured camera fields,
and proposed [y,x] points. Expected flags and provenance stay in local scoring.

These labels reproduce a small initial probe, including ambiguous localization
in its positive case. Passing does not certify endpoints, clearance, or entry.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image

from duck_nav.credentials import load_gemini_key
from duck_nav.gap_review import REPORT_SCHEMA, SYSTEM, VERDICT_KEYS, GeminiGapReviewer

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "gap_review"
CASE_NAMES = ("018-view00059", "018-view00053", "013-view00032")


def json_sha256(value):
    """Hash structured inputs independently of manifest whitespace."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def load_cases(directory=FIXTURES):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    cases = manifest.get("cases")
    if (
        manifest.get("version") != 1
        or not isinstance(cases, list)
        or any(not isinstance(case, dict) for case in cases)
        or tuple(case.get("name") for case in cases) != CASE_NAMES
    ):
        raise ValueError("gap review manifest must contain all three named cases exactly once")
    loaded = []
    for case in cases:
        expected = case.get("expected")
        if (
            not isinstance(expected, dict)
            or set(expected) != set(VERDICT_KEYS)
            or any(type(value) is not bool for value in expected.values())
        ):
            raise ValueError("fixture expected flags must be three booleans")
        source = case["image"]
        path = (directory / source["file"]).resolve()
        if not path.is_relative_to(directory):
            raise ValueError("fixture image must be inside the fixture directory")
        jpeg = path.read_bytes()
        if hashlib.sha256(jpeg).hexdigest() != source["jpeg_sha256"]:
            raise ValueError("fixture image checksum mismatch")
        provenance = case["provenance"]
        if (
            provenance.get("source_sha256") != source["jpeg_sha256"]
            or not provenance.get("source_file")
            or provenance.get("conversion") != "unchanged JPEG bytes"
        ):
            raise ValueError("fixture image provenance mismatch")
        inputs = case["inputs"]
        if json_sha256(inputs) != case["inputs_sha256"]:
            raise ValueError("fixture input checksum mismatch")
        if (
            set(inputs)
            != {"view", "camera", "proposed_endpoints", "point_order", "coordinate_range"}
            or inputs["view"] != "CURRENT"
            or inputs["point_order"] != "y,x"
            or inputs["coordinate_range"] != [0, 1000]
            or not isinstance(inputs["proposed_endpoints"], list)
            or len(inputs["proposed_endpoints"]) != 2
        ):
            raise ValueError("fixture inputs must describe one exact upright view and point pair")
        point, opposite_point = inputs["proposed_endpoints"]
        # The pure public payload builder validates the same camera/point/JPEG
        # contract as a real request without constructing a session or loading a key.
        GeminiGapReviewer.payload(
            jpeg, camera=inputs["camera"], point=point, opposite_point=opposite_point
        )
        with Image.open(io.BytesIO(jpeg)) as picture:
            if picture.size != (source["width"], source["height"]):
                raise ValueError("fixture image dimensions mismatch")
        loaded.append({**case, "jpeg": jpeg})
    return loaded


async def evaluate(reviewer, cases, emit=print):
    results = []
    for case in cases:
        inputs = case["inputs"]
        point, opposite_point = inputs["proposed_endpoints"]
        result = {
            "case": case["name"],
            "expected": case["expected"],
            "image": case["image"],
            "inputs": inputs,
            "inputs_sha256": case["inputs_sha256"],
            "provenance": case["provenance"],
            "annotation_note": case["annotation_note"],
        }
        try:
            review = await reviewer.review(
                case["jpeg"],
                camera=inputs["camera"],
                point=point,
                opposite_point=opposite_point,
            )
            passed = all(review.get(key) is value for key, value in case["expected"].items())
            result.update(review=review, passed=passed)
            flags = " ".join(f"{key}={review.get(key)}" for key in VERDICT_KEYS)
            emit(f"{case['name']}: {'PASS' if passed else 'FAIL'} {flags}")
        except Exception as error:  # noqa: BLE001 - never expose credentials or raw provider errors
            result.update(error_type=type(error).__name__, passed=False)
            emit(f"{case['name']}: ERROR {type(error).__name__}")
        results.append(result)
    return {
        "at": datetime.now(UTC).isoformat(),
        "model": reviewer.model,
        "prompt_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest(),
        "schema_sha256": json_sha256(REPORT_SCHEMA),
        "structured_hash_encoding": "UTF-8 JSON, sorted keys, compact separators",
        "call_count": len(results),
        "actions_executed": False,
        "limits": "Three perception regression samples; no clearance or navigation certification.",
        "results": results,
        "all_passed": bool(results) and all(result["passed"] for result in results),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, help="JSON report path (default: timestamped runs file)"
    )
    args = parser.parse_args()
    try:
        cases = load_cases()
        reviewer = GeminiGapReviewer(load_gemini_key())
        report = asyncio.run(evaluate(reviewer, cases))
    except Exception as error:  # noqa: BLE001 - credentials and provider details remain private
        print(f"Gap review validation failed: {type(error).__name__}")
        return 2
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or FIXTURES.parents[2] / "runs" / f"gap-review-validation-{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved {output}")
    return 0 if report["all_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
