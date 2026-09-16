"""Manually check arrival perception against recorded simulator images.

Run `uv run python scripts/validate_arrival.py` from apps/navigation. This makes
eight real provider requests using the saved Gemini key and records the results
under runs/. It never connects to a robot. Normal pytest runs use no network.
Expected outcomes and source provenance remain local to this scorer; the reviewer
receives only each case's goal, view labels, and the exact checked JPEG bytes.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from duck_nav.arrival import SYSTEM, GeminiArrivalReviewer
from duck_nav.credentials import load_gemini_key

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "arrival"


def load_cases(directory=FIXTURES):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    cases = []
    for case in manifest["cases"]:
        views = []
        for source in case["views"]:
            path = (directory / source["file"]).resolve()
            if not path.is_relative_to(directory.resolve()):
                raise ValueError("fixture image must be inside the fixture directory")
            jpeg = path.read_bytes()
            if hashlib.sha256(jpeg).hexdigest() != source["jpeg_sha256"]:
                raise ValueError("fixture image checksum mismatch")
            views.append({"label": source["label"], "jpeg": jpeg})
        cases.append({**case, "model_views": views})
    return cases


async def evaluate(reviewer, cases, emit=print):
    results = []
    for case in cases:
        for repetition in range(1, case["repetitions"] + 1):
            result = {
                "case": case["name"],
                "repetition": repetition,
                "expected": case["expected"],
                "images": case["views"],
            }
            try:
                review = await reviewer.review(case["goal"], case["model_views"])
                passed = all(review.get(key) is value for key, value in case["expected"].items())
                result.update(review=review, passed=passed)
                emit(
                    f"{case['name']} {repetition}/{case['repetitions']}: "
                    f"{'PASS' if passed else 'FAIL'} "
                    f"visible={review['destination_visible']} inside={review['inside_destination']}"
                )
            except Exception as error:  # noqa: BLE001 - preserve failed checks without leaking provider errors
                result.update(error_type=type(error).__name__, passed=False)
                emit(
                    f"{case['name']} {repetition}/{case['repetitions']}: ERROR {type(error).__name__}"
                )
            results.append(result)
    return {
        "at": datetime.now(UTC).isoformat(),
        "model": reviewer.model,
        "prompt_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest(),
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
        reviewer = GeminiArrivalReviewer(load_gemini_key())
        report = asyncio.run(evaluate(reviewer, cases))
    except Exception as error:  # noqa: BLE001 - credentials and provider URLs must remain private
        print(f"Arrival validation failed: {type(error).__name__}")
        return 2
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or FIXTURES.parents[2] / "runs" / f"arrival-validation-{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved {output}")
    return 0 if report["all_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
