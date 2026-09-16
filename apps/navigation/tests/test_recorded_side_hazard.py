"""The real trial-011 side scan must survive the following clear front view."""

import json
from pathlib import Path

from duck_nav.core import GuardedRobot, _unit


def test_recorded_close_side_obstacle_cannot_disappear_after_recentering():
    path = Path(__file__).parent / "fixtures/depth/side_hazard_011.json"
    fixture = json.loads(path.read_text())
    side, front = [sample["snapshot"] for sample in fixture["samples"]]
    robot = GuardedRobot(None)
    robot._beams = [_unit(beam, 3) for beam in fixture["tof_beams"]]

    # This isolates recorded depth geometry, not expired recording timestamps:
    # the live freshness/health gate is covered separately by core tests.
    front_alone, _ = robot._depth_guard(front)
    assert front_alone is None

    side_reason, side_depth = robot._depth_guard(side)
    assert side_reason == "head_not_forward"
    assert 0.12 < side_depth["nearest_obstacle_m"] < 0.13

    front_reason, front_depth = robot._depth_guard(front)
    assert front_depth["nearest_obstacle_m"] > 1.0
    assert front_reason == "obstacle"
    assert front_depth["retained_side_hazards"][0]["direction"] == "right"
