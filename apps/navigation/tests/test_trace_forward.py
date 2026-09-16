"""Heading phase attribution must survive the +/- pi wrap boundary."""

import math
import runpy
from pathlib import Path

import pytest


def load_summary(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    return runpy.run_path(str(scripts / "trace_forward.py"))["summarize"]


def test_phases_sum_to_settled_heading_across_wrap(monkeypatch):
    samples = []
    for tick in range(81):
        t = tick * 0.05
        delta = 10 * min(1, max(0, t - 1)) + 8 * min(0.5, max(0, t - 2))
        angle = math.radians(179 + delta)
        samples.append(
            {
                "t": t,
                "truth_yaw": math.atan2(math.sin(angle), math.cos(angle)),
                "trunk": [0, 0, 0.12],
                "requested": [0, 0, 0],
                "applied": [0, 0, 0],
            }
        )
    result = load_summary(monkeypatch)(samples, 1, 2)
    assert result["startup_0_5s_deg"] == pytest.approx(5)
    assert result["walking_after_0_5s_deg"] == pytest.approx(5)
    assert result["post_action_deg"] == pytest.approx(4)
    assert result["total_deg"] == pytest.approx(14)
    assert result["settled_span_deg"] == pytest.approx(0)


def test_missing_baseline_is_not_a_valid_trial(monkeypatch):
    sample = {"t": 1, "truth_yaw": 0}
    with pytest.raises(ValueError, match="insufficient"):
        load_summary(monkeypatch)([sample], 1, 2)
