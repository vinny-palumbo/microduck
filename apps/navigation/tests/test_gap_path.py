"""Pure reference-geometry tests; no robot, simulator, map or model is used."""

import copy
import json
import math
from itertools import pairwise

import pytest

from duck_nav.gap_path import _at, _csc, _wall_distance, build_gap_path, path_step


def angle_difference(a, b):
    return math.atan2(math.sin(a - b), math.cos(a - b))


def integrate(start, yaw, segments):
    # Independent local-frame arc integration, rather than the implementation's
    # world-coordinate sine/cosine differences.
    x, y = start
    for length, curvature in segments:
        angle = length * curvature
        forward = math.sin(angle) / curvature if curvature else length
        left = (1 - math.cos(angle)) / curvature if curvature else 0
        x, y = (
            x + forward * math.cos(yaw) - left * math.sin(yaw),
            y + forward * math.sin(yaw) + left * math.cos(yaw),
        )
        yaw += angle
    return [x, y], yaw


@pytest.mark.parametrize("first,last", [(1, 1), (1, -1), (-1, 1), (-1, -1)])
def test_all_csc_tangent_signs_recover_known_forward_curves(first, last):
    start, yaw, radius = [0.2, -0.3], 0.7, 0.45
    expected = [(0.3 * radius, first / radius), (0.6, 0), (0.8 * radius, last / radius)]
    goal, goal_yaw = integrate(start, yaw, expected)
    actual = _csc(start, yaw, goal, goal_yaw, radius, first, last)
    assert actual is not None
    for found, known in zip(actual, expected):
        assert found == pytest.approx(known)
    end, heading = integrate(start, yaw, actual)
    assert end == pytest.approx(goal)
    assert angle_difference(heading, goal_yaw) == pytest.approx(0)


@pytest.mark.parametrize(
    "goal,goal_yaw,first,last,lengths",
    [
        ([2, 0], 0, 1, 1, [0, 2, 0]),
        ([1, 1], math.pi / 2, 1, 1, [0, 0, math.pi / 2]),
        ([1, -1], -math.pi / 2, -1, -1, [0, 0, math.pi / 2]),
        ([2, 2], math.pi / 2, 1, 1, [math.pi / 4, math.sqrt(2), math.pi / 4]),
        ([math.sqrt(2), 2 - math.sqrt(2)], 0, 1, -1, [math.pi / 4, 0, math.pi / 4]),
    ],
)
def test_csc_straight_pure_circle_and_zero_straight_tangency(goal, goal_yaw, first, last, lengths):
    segments = _csc([0, 0], 0, goal, goal_yaw, 1, first, last)
    assert segments is not None
    assert [length for length, _ in segments] == pytest.approx(lengths, abs=5e-8)
    final, yaw = integrate([0, 0], 0, segments)
    assert final == pytest.approx(goal, abs=5e-8)
    assert angle_difference(yaw, goal_yaw) == pytest.approx(0, abs=5e-8)


def test_csc_rejects_impossible_internal_tangent_and_arc_loops():
    assert _csc([0, 0], 0, [0, 1], 0, 1, 1, -1) is None
    assert _csc([0, 0], 0, [-1, 1], -math.pi / 2, 1, 1, 1) is None


def straight_plan(width=0.8):
    return build_gap_path([0, -1.5], math.pi / 2, [[-width / 2, 0], [width / 2, 0]])


def turning_plan():
    return build_gap_path([0, 0.6], math.pi / 2, [[-1, 1.6], [-1, 2.4]])


def measured_projection_plan():
    # Recorded trial013 view32 odometry and projected image endpoints only.
    # No simulator truth or scene geometry is used in this deterministic fixture.
    return build_gap_path(
        [-0.17224528180933163, 1.2315496698791024],
        1.798532883773177,
        [[-0.9897875566128226, 1.6171937539254257], [-1.0260855482045912, 2.3925312553453213]],
    )


def close_parallel_plan():
    return build_gap_path([-0.5, -1.5], math.pi / 2, [[0, -0.4], [0, 0.4]])


def test_reference_has_exact_staging_and_final_pose_with_bounded_sampling():
    plan = turning_plan()
    center, normal = plan["gap_center"], plan["gap_normal"]
    stage = _at(plan["samples"], plan["stage_s"])
    assert stage["position"] == pytest.approx(plan["stage_pose"]["position"])
    assert angle_difference(stage["yaw"], plan["stage_pose"]["yaw"]) == pytest.approx(0)
    final = plan["samples"][-1]
    assert final["position"] == pytest.approx([center[i] - 0.35 * normal[i] for i in range(2)])
    assert angle_difference(final["yaw"], math.atan2(-normal[1], -normal[0])) == pytest.approx(0)
    assert final["s"] == pytest.approx(plan["total_length"])
    assert plan["total_length"] <= 4
    for before, after in pairwise(plan["samples"]):
        assert 0 < after["s"] - before["s"] <= 0.02 + 1e-9
    for segment in plan["segments"]:
        assert abs(segment["length"] * segment["curvature"]) <= math.pi + 1e-9
    json.dumps(plan, allow_nan=False)


def transformed(point, angle, offset):
    c, s = math.cos(angle), math.sin(angle)
    return [c * point[0] - s * point[1] + offset[0], s * point[0] + c * point[1] + offset[1]]


def test_translation_rotation_equivariance():
    original = turning_plan()
    angle, offset = 1.3, [1.1, -0.7]
    changed = build_gap_path(
        transformed([0, 0.6], angle, offset),
        math.pi / 2 + angle,
        [transformed(p, angle, offset) for p in [[-1, 1.6], [-1, 2.4]]],
    )
    assert changed["total_length"] == pytest.approx(original["total_length"])
    assert changed["reference_radius"] == original["reference_radius"]
    assert len(changed["samples"]) == len(original["samples"])
    for before, after in zip(original["samples"], changed["samples"]):
        assert after["position"] == pytest.approx(transformed(before["position"], angle, offset))
        assert angle_difference(after["yaw"], before["yaw"] + angle) == pytest.approx(0)


def test_mirrored_geometry_and_reversed_endpoint_order_preserve_solution():
    original = turning_plan()
    mirrored = build_gap_path([0, 0.6], math.pi / 2, [[1, 1.6], [1, 2.4]])
    reversed_plan = build_gap_path([0, 0.6], math.pi / 2, [[-1, 2.4], [-1, 1.6]])
    assert mirrored["total_length"] == pytest.approx(original["total_length"])
    assert reversed_plan["total_length"] == pytest.approx(original["total_length"])
    for before, mirror, reverse in zip(
        original["samples"], mirrored["samples"], reversed_plan["samples"]
    ):
        assert mirror["position"] == pytest.approx([-before["position"][0], before["position"][1]])
        assert angle_difference(mirror["yaw"], math.pi - before["yaw"]) == pytest.approx(0)
        assert reverse["position"] == pytest.approx(before["position"])


def test_straight_geometry_selects_shortest_length_without_unnecessary_turns():
    plan = straight_plan()
    assert plan["total_length"] == pytest.approx(1.85)
    assert all(abs(sample["position"][0]) < 1e-9 for sample in plan["samples"])
    # A narrow but admissible straight gap must not be refused due to arbitrary
    # sample spacing; exact chord distance is used, not endpoint-only sampling.
    assert straight_plan(0.75)["total_length"] == pytest.approx(1.85)


@pytest.mark.parametrize("width", [0, 0.5, 0.74, 3.01])
def test_narrow_or_unsupported_gaps_are_refused(width):
    with pytest.raises(ValueError, match="width"):
        straight_plan(width)


def test_ambiguous_side_near_wall_and_unavailable_bounded_path_are_refused():
    endpoints = [[-0.4, 0], [0.4, 0]]
    with pytest.raises(ValueError, match="initial side"):
        build_gap_path([0, -0.09], math.pi / 2, endpoints)
    with pytest.raises(ValueError, match="initial pose"):
        build_gap_path([0.7, -0.39], math.pi / 2, endpoints)
    with pytest.raises(ValueError, match="no bounded"):
        build_gap_path([0, -5], math.pi / 2, endpoints)


def test_extended_wall_and_plane_band_detect_segment_interior_collision():
    assert _wall_distance([-1, 0.5], [-1, -0.5], 0.8) == pytest.approx(0)
    assert _wall_distance([0, 0.5], [0, -0.5], 0.8) == pytest.approx(0.4)
    assert _wall_distance([1, 0.5], [1, 0.4], 0.8) == pytest.approx(0.37)


@pytest.mark.parametrize(
    "factory", [straight_plan, turning_plan, measured_projection_plan, close_parallel_plan]
)
def test_every_ideal_reference_sample_is_followable_without_self_refusal(factory):
    plan = factory()
    progress = 0.0
    seen_complete = False
    for sample in plan["samples"]:
        step = path_step(plan, sample["position"], sample["yaw"], progress)
        assert step["status"] in ("advance", "complete"), (sample, step)
        assert progress <= step["progress_s"] <= progress + 0.3 + 1e-9
        progress = step["progress_s"]
        if step["status"] == "advance":
            assert 0.05 <= step["distance_m"] <= 0.1
            assert abs(step["heading_deg"]) <= 30
        else:
            seen_complete = True
    assert seen_complete


def test_measured_projection_has_a_short_reference_using_the_nearer_stage():
    plan = measured_projection_plan()
    assert plan["reference_radius"] == 0.35
    assert plan["stage_distance_m"] == 0.4
    assert plan["total_length"] == pytest.approx(1.7638466387672023)
    assert plan["limits"]["reference_clearance_m"] == 0.37
    assert plan["limits"]["wall_half_thickness_m"] == 0.03


def test_progress_window_cannot_jump_to_a_distant_later_path_section():
    plan = straight_plan()
    step = path_step(plan, [0, -0.4], math.pi / 2, 0)
    assert step["status"] == "refused"
    assert step["reason"] == "cross_track_limit"
    assert step["progress_s"] <= 0.3
    # Small apparent backwards motion does not decrease stored progress.
    step = path_step(plan, [0, -1.22], math.pi / 2, 0.3)
    assert step["progress_s"] == pytest.approx(0.3)


def test_cross_track_and_heading_limits_refuse_motion():
    plan = straight_plan()
    assert path_step(plan, [0.101, -1.5], math.pi / 2, 0)["reason"] == "cross_track_limit"
    assert (
        path_step(plan, [0, -1.5], math.pi / 2 + math.radians(61), 0)["reason"]
        == "reference_heading_limit"
    )


def test_near_plane_requires_current_heading_clearance_not_just_path_clearance():
    plan = straight_plan()
    step = path_step(plan, [0, -0.3], math.pi / 2 + math.radians(20), 1.2)
    assert step["status"] == "refused"
    assert step["reason"] == "crossing_alignment_clearance"
    assert step["crossing_clearance_m"] < 0.35
    assert path_step(plan, [0, -0.3], math.pi / 2, 1.2)["status"] == "advance"


@pytest.mark.parametrize("lateral,expected", [(0.03, "advance"), (-0.03, "refused")])
def test_off_center_crossing_heading_is_independent_of_endpoint_order(lateral, expected):
    endpoints = [[-0.4, 0], [0.4, 0]]
    results = []
    for points in [endpoints, list(reversed(endpoints))]:
        plan = build_gap_path([0, -1.5], math.pi / 2, points)
        result = path_step(plan, [lateral, -0.3], math.pi / 2 + math.radians(5), 1.2)
        assert result["status"] == expected
        results.append(result)
    assert results[0]["crossing_clearance_m"] == pytest.approx(results[1]["crossing_clearance_m"])


def test_reversed_endpoints_cannot_allow_a_heading_that_cuts_toward_a_jamb():
    endpoints = [[1, -0.4], [1, 0.4]]
    for points in [endpoints, list(reversed(endpoints))]:
        plan = build_gap_path([0, 0], 0, points)
        step = path_step(plan, [0.6, 0.03], 0.1, 0.5)
        assert step["status"] == "refused"
        assert step["reason"] == "crossing_alignment_clearance"
        assert step["crossing_clearance_m"] == pytest.approx(0.328218174494)


def test_completion_requires_passed_plane_progress_position_and_heading():
    plan = straight_plan()
    total, end = plan["total_length"], plan["samples"][-1]
    assert path_step(plan, end["position"], end["yaw"], total - 0.1)["status"] == "complete"
    assert path_step(plan, end["position"], end["yaw"], 0)["status"] == "refused"
    assert (
        path_step(plan, end["position"], end["yaw"] + math.radians(11), total - 0.1)["status"]
        != "complete"
    )
    assert path_step(plan, [0, -0.01], end["yaw"], 1.4)["status"] == "advance"


def test_wrapped_yaw_and_step_output_are_finite_and_inputs_are_unchanged():
    plan = turning_plan()
    before = copy.deepcopy(plan)
    original = path_step(plan, [0, 0.6], math.pi / 2, 0)
    wrapped = path_step(plan, [0, 0.6], math.pi / 2 + math.tau, 0)
    assert wrapped["heading_deg"] == pytest.approx(original["heading_deg"])
    assert plan == before
    json.dumps(wrapped, allow_nan=False)


@pytest.mark.parametrize(
    "start,yaw,endpoints",
    [
        ([float("nan"), 0], 0, [[0, 0], [1, 0]]),
        ([0, 0], float("inf"), [[0, 0], [1, 0]]),
        ([True, 0], 0, [[0, 0], [1, 0]]),
        ([0, 0, 0], 0, [[0, 0], [1, 0]]),
        ([0, 0], 0, [[0, 0], [float("inf"), 1]]),
        ([0, 0], 0, [[0, 0]]),
        ([0, 0], 0, None),
    ],
)
def test_invalid_build_inputs_are_refused(start, yaw, endpoints):
    with pytest.raises(ValueError):
        build_gap_path(start, yaw, endpoints)


@pytest.mark.parametrize(
    "position,yaw,progress",
    [
        ([float("nan"), 0], 0, 0),
        ([0, 0], float("inf"), 0),
        ([0, 0], 0, float("nan")),
        ([0, 0], 0, -1),
        ([0, 0], 0, 100),
        ([True, 0], 0, 0),
        ([0], 0, 0),
    ],
)
def test_invalid_follow_inputs_are_refused(position, yaw, progress):
    with pytest.raises(ValueError):
        path_step(straight_plan(), position, yaw, progress)


def test_corrupted_plan_sample_is_refused():
    plan = straight_plan()
    plan["samples"][3]["yaw"] = float("nan")
    with pytest.raises(ValueError):
        path_step(plan, [0, -1.5], math.pi / 2, 0)
