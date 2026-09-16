"""A comparison must not accidentally reverse the sign-preserving candidate."""

import math
import runpy
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def steering():
    script = Path(__file__).parents[1] / "scripts/probe_navigation_gait.py"
    return runpy.run_path(str(script))["steering"]


@pytest.mark.parametrize("desired", [-20, 20])
@pytest.mark.parametrize("turned", [-60, -20, 0, 20, 60, 359])
def test_clipped_candidate_never_countersteers(steering, desired, turned):
    command = steering("clipped", math.radians(desired), math.radians(turned))
    assert command * desired >= 0
    assert -0.3 <= command <= 0.5


def test_proportional_baseline_does_reverse_after_overshoot(steering):
    assert steering("proportional", math.radians(20), math.radians(30)) < 0
    assert steering("clipped", math.radians(20), math.radians(30)) == 0


@pytest.mark.parametrize("mode", ["fixed", "clipped"])
def test_straight_candidate_does_not_steer_against_gait_sway(steering, mode):
    for angle in [-0.5, 0.0, 0.5]:
        assert steering(mode, 0.0, angle) == 0


def test_fixed_candidate_uses_measured_asymmetric_rates(steering):
    assert steering("fixed", math.radians(20), 0) == 0.5
    assert steering("fixed", math.radians(-20), 0) == -0.3
