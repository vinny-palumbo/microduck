"""Entry evidence is body-relative, finite, fresh and independent of room claims."""

import copy
import json
import math

import pytest

from duck_nav.entry import DoorwayEntry
from duck_nav.gap_path import build_gap_path


def snapshot(x=0, y=0, z=0.12, yaw=0, *, at=10, connected=True):
    return {
        "connected": connected,
        "state": {
            "received_at": at,
            "data": {
                "odom": {"position": [x, y, z], "yaw": yaw},
                "simulator_truth": {"inside_kitchen": True},
                "frames": {"camera": {"pos": [20, 20, 20]}},
            },
        },
    }


def reference():
    return {
        "gap_center": [1, 0],
        "gap_normal": [-1, 0],
        "gap_tangent": [0, 1],
        "gap_width_m": 0.8,
        "samples": ["must not be retained"],
        "simulator_truth": {"inside_kitchen": True},
    }


def result(**changes):
    return {"completed": True, "stop": {"physical_settling_verified": True}, **changes}


def entry(start=None):
    start = start or snapshot()
    return DoorwayEntry(reference(), "view-00051", start, now=start["state"]["received_at"])


def walk(gate, before, *, x=None, y=None, z=None, yaw=None, dt=1, outcome=None):
    p = before["state"]["data"]["odom"]["position"]
    after = snapshot(
        x=p[0] if x is None else x,
        y=p[1] if y is None else y,
        z=p[2] if z is None else z,
        yaw=before["state"]["data"]["odom"]["yaw"] if yaw is None else yaw,
        at=before["state"]["received_at"] + dt,
    )
    summary = gate.note_motion(
        before, after, result() if outcome is None else outcome, now=after["state"]["received_at"]
    )
    return after, summary


@pytest.mark.parametrize("inside", [0.249, 0.25, 0.251])
def test_body_threshold_ignores_camera_projection_and_room_claim(inside):
    before = snapshot(x=0.98)
    gate = entry(before)
    after, summary = walk(gate, before, x=1 + inside)
    assert summary["status"] == ("outside" if inside < 0.25 else "inside")
    assert summary["signed_outside_m"] == pytest.approx(-inside)
    assert summary["required_inside_m"] == 0.25
    assert summary["arrival_verified"] is False
    assert summary["source"] == "observed_doorway_and_robot_odometry"
    assert gate.observation(after, now=after["state"]["received_at"]) == summary


@pytest.mark.parametrize(
    "lateral,expected",
    [(0.249, "inside"), (0.25, "inside"), (0.251, "outside"), (-0.251, "outside")],
)
def test_first_entry_needs_margin_from_both_jambs(lateral, expected):
    before = snapshot(x=0.98, y=lateral)
    after, summary = walk(entry(before), before, x=1.25)
    assert summary["status"] == expected
    assert after["state"]["data"]["odom"]["position"][1] == lateral


def test_proven_entry_allows_room_travel_but_exit_requires_new_margin_proof():
    before = snapshot(x=0.98)
    gate = entry(before)
    current, summary = walk(gate, before, x=1.25)
    assert summary["status"] == "inside"
    for lateral in (0.2, 0.4, 0.6):
        current, summary = walk(gate, current, y=lateral)
        assert summary["status"] == "inside"
    current, summary = walk(gate, current, x=0.99)
    assert summary["status"] == "outside"
    current, summary = walk(gate, current, x=1.25)
    assert summary["status"] == "outside"
    assert summary["reason"] == "doorway_crossing_not_established"
    for lateral in (0.4, 0.2):
        current, summary = walk(gate, current, y=lateral)
        assert summary["status"] == "outside"
    # Moving sideways after crossing outside the jamb margins cannot repair
    # missing crossing evidence; a new valid crossing is required.
    current, summary = walk(gate, current, x=0.99)
    assert summary["status"] == "outside"
    current, summary = walk(gate, current, x=1.25)
    assert summary["status"] == "inside"


def test_crossing_outside_jambs_then_moving_sideways_cannot_establish_entry():
    current = snapshot(x=0.9, y=0.4)
    gate = entry(current)
    for x, y in ((1.1, 0.4), (1.3, 0.4), (1.3, 0.2)):
        current, summary = walk(gate, current, x=x, y=y)
        assert summary["status"] == "outside"
        assert summary["crossing_evidence"] is None
    assert summary["signed_outside_m"] == pytest.approx(-0.3)
    assert summary["reason"] == "doorway_crossing_not_established"


@pytest.mark.parametrize("side", [-1, 1])
@pytest.mark.parametrize("before_lateral,expected", [(0.27, "inside"), (0.28, "outside")])
def test_interpolated_crossing_margin_not_later_endpoint_controls_evidence(
    side, before_lateral, expected
):
    before = snapshot(x=0.98, y=side * before_lateral)
    gate = entry(before)
    # Crossing is 10% along this measured segment: lateral 0.25 is admissible,
    # lateral 0.26 is not, although both later settled positions are within margins.
    current, summary = walk(gate, before, x=1.18, y=side * (before_lateral - 0.2))
    assert summary["status"] == "outside"  # Insufficient inside distance yet.
    current, summary = walk(gate, current, x=1.26)
    assert summary["status"] == expected
    if expected == "inside":
        assert summary["crossing_evidence"] == (
            "settled_odometry_segment_estimate_not_trajectory_certification"
        )
    else:
        assert summary["crossing_evidence"] is None


def test_gradual_crossing_at_plane_retains_evidence_when_backing_up_on_inside_half():
    current = snapshot(x=0.98)
    gate = entry(current)
    current, summary = walk(gate, current, x=1.0)
    assert summary["crossing_evidence"] is not None
    for x in (1.2, 1.01, 1.249):
        current, summary = walk(gate, current, x=x)
        assert summary["status"] == "outside"  # Depth threshold still unmet.
        assert summary["crossing_evidence"] is not None
    current, summary = walk(gate, current, x=1.25)
    assert summary["status"] == "inside"


def test_stationary_head_scan_drift_cannot_create_missing_plane_crossing_evidence():
    before = snapshot(x=0.99)
    gate = entry(before)
    drifted = snapshot(x=1.01, at=11)
    summary = gate.observation(drifted, now=11)
    assert summary["status"] == "outside"
    assert summary["crossing_evidence"] is None
    _, summary = walk(gate, drifted, x=1.26)
    assert summary["status"] == "outside"
    assert summary["reason"] == "doorway_crossing_not_established"


def test_stationary_return_to_outside_discards_previous_crossing_evidence():
    before = snapshot(x=0.99)
    gate = entry(before)
    current, summary = walk(gate, before, x=1.01)
    assert summary["crossing_evidence"] is not None
    summary = gate.observation(snapshot(x=0.99, at=12), now=12)
    assert summary["crossing_evidence"] is None
    # Even if it then drifts back across without a reported body action, the
    # previous valid crossing must not be reused to prove entry on later motion.
    current = snapshot(x=1.01, at=13)
    assert gate.observation(current, now=13)["crossing_evidence"] is None
    _, summary = walk(gate, current, x=1.26)
    assert summary["status"] == "outside"


@pytest.mark.parametrize("angle", [0, math.pi / 2, -2.1, math.pi])
@pytest.mark.parametrize("reverse", [False, True])
def test_rotated_and_unordered_observed_endpoints_give_same_entry(angle, reverse):
    def transform(x, y):
        return [
            2 + x * math.cos(angle) - y * math.sin(angle),
            -3 + x * math.sin(angle) + y * math.cos(angle),
        ]

    endpoints = [transform(1, -0.4), transform(1, 0.4)]
    if reverse:
        endpoints.reverse()
    start_xy = transform(0, 0)
    plan = build_gap_path(start_xy, angle, endpoints)
    current = snapshot(*start_xy, yaw=angle)
    gate = DoorwayEntry(plan, "view-00051", current, now=10)
    for distance in (0.25, 0.5, 0.75, 1.0, 1.25):
        xy = transform(distance, 0)
        current, summary = walk(gate, current, x=xy[0], y=xy[1])
    assert summary["status"] == "inside"
    assert summary["signed_outside_m"] == pytest.approx(-0.25)


@pytest.mark.parametrize("change", ["x", "height", "yaw"])
def test_unreported_pose_jump_permanently_invalidates(change):
    gate = entry()
    changed = snapshot(
        x=0.026 if change == "x" else 0,
        z=0.146 if change == "height" else 0.12,
        yaw=math.radians(5.1) if change == "yaw" else 0,
        at=11,
    )
    first = gate.observation(changed, now=11)
    assert first["status"] == "unknown"
    assert gate.observation(snapshot(at=12), now=12) == first
    assert gate.note_motion(snapshot(at=12), snapshot(x=0.1, at=13), result(), now=13) == first


def test_head_scans_and_small_pose_drift_never_reset_body_anchor():
    gate = entry()
    first = snapshot(x=0.02, at=11)
    first["state"]["data"]["frames"]["camera"]["pos"] = [-30, 20, 40]
    assert gate.observation(first, now=11)["status"] == "outside"
    assert gate.observation(snapshot(x=0.03, at=12), now=12)["status"] == "unknown"


def test_yaw_wrap_and_historical_before_sample_support_ordinary_settled_motion():
    before = snapshot(yaw=math.radians(179))
    gate = entry(before)
    after, summary = walk(gate, before, x=0.2, yaw=math.radians(-179), dt=3)
    assert summary["status"] == "outside"
    assert gate.observation(after, now=13)["status"] == "outside"
    # Ordinary movement has moved the anchor; this is not an unexpected 20 cm jump.
    after, summary = walk(gate, after, x=0.4, yaw=math.radians(-170))
    assert summary["status"] == "outside"


@pytest.mark.parametrize(
    "outcome",
    [
        {"completed": False, "stop": {"physical_settling_verified": True}},
        {"completed": True},
        {"completed": 1, "stop": {"physical_settling_verified": True}},
        {"completed": True, "stop": {"physical_settling_verified": 1}},
        {"completed": True, "stop": {"physical_settling_verified": False}},
        None,
    ],
)
def test_failed_partial_or_unsettled_motion_latches_unknown(outcome):
    gate = entry()
    summary = gate.note_motion(snapshot(), snapshot(x=0.1, at=11), outcome, now=11)
    assert summary["status"] == "unknown"
    assert gate.observation(snapshot(x=0.1, at=12), now=12) == summary


@pytest.mark.parametrize(
    "distance,yaw,expected",
    [(0.3, 0, "outside"), (0.3001, 0, "unknown"), (0.1, 50, "outside"), (0.1, 50.1, "unknown")],
)
def test_measured_motion_limits(distance, yaw, expected):
    _, summary = walk(entry(), snapshot(), x=distance, yaw=math.radians(yaw))
    assert summary["status"] == expected


def test_motion_cannot_hide_an_unreported_jump_before_dispatch():
    gate = entry()
    summary = gate.note_motion(snapshot(x=0.026, at=11), snapshot(x=0.126, at=12), result(), now=12)
    assert summary["status"] == "unknown"
    assert summary["reason"] == "entry_unexpected_pose_change"


@pytest.mark.parametrize(
    "issue",
    [
        "disconnected",
        "missing",
        "nonfinite",
        "stale",
        "future",
        "replayed",
        "expired",
        "clock_backwards",
    ],
)
def test_unknown_or_stale_frame_is_latched_without_numeric_evidence(issue):
    gate = entry()
    now, sample = 11, snapshot(at=11)
    if issue == "disconnected":
        sample["connected"] = False
    elif issue == "missing":
        del sample["state"]["data"]["odom"]
    elif issue == "nonfinite":
        sample["state"]["data"]["odom"]["position"][0] = math.nan
    elif issue == "stale":
        sample["state"]["received_at"] = 10.64
    elif issue == "future":
        sample["state"]["received_at"] = 11.01
    elif issue == "replayed":
        now, sample = 10.1, snapshot(at=9.99)
    elif issue == "expired":
        now, sample = 250.01, snapshot(at=250.01)
    else:
        now, sample = 9.9, snapshot(at=9.9)
    summary = gate.observation(sample, now=now)
    assert summary["status"] == "unknown"
    assert summary["signed_outside_m"] is None
    assert gate.observation(snapshot(at=251), now=251) == summary
    json.dumps(summary, allow_nan=False)


def test_reference_lifetime_includes_stationary_time_and_exact_boundary():
    gate = entry()
    assert gate.observation(snapshot(at=250), now=250)["status"] == "outside"
    assert gate.observation(snapshot(at=250.001), now=250.001)["status"] == "unknown"


@pytest.mark.parametrize("after_at", [10, 9.9])
def test_motion_requires_newer_after_telemetry(after_at):
    gate = entry()
    summary = gate.note_motion(snapshot(), snapshot(x=0.1, at=after_at), result(), now=10.1)
    assert summary["status"] == "unknown"


def test_motion_after_state_must_be_fresh():
    gate = entry()
    summary = gate.note_motion(snapshot(), snapshot(x=0.1, at=10.5), result(), now=11)
    assert summary["status"] == "unknown"
    assert summary["reason"] == "entry_state_stale"


def test_retains_only_copied_observed_geometry_and_body_pose():
    geometry, initial = reference(), snapshot()
    gate = DoorwayEntry(geometry, "view-00051", initial, now=10)
    geometry["gap_center"][0] = 99
    initial["state"]["data"]["odom"]["position"][0] = 99
    summary = gate.observation(snapshot(at=11), now=11)
    assert summary["signed_outside_m"] == 1
    summary["source_view_id"] = "modified"
    assert gate.observation(snapshot(at=11), now=11)["source_view_id"] == "view-00051"
    retained = json.dumps(vars(gate), allow_nan=False)
    assert "simulator_truth" not in retained
    assert "samples" not in retained
    assert "camera" not in retained


@pytest.mark.parametrize(
    "field,value",
    [
        ("gap_center", [0]),
        ("gap_center", [math.inf, 0]),
        ("gap_normal", [2, 0]),
        ("gap_tangent", [-1, 0]),
        ("gap_width_m", 0.3),
        ("gap_width_m", True),
        ("gap_width_m", math.nan),
    ],
)
def test_malformed_geometry_is_rejected(field, value):
    geometry = reference()
    geometry[field] = value
    with pytest.raises(ValueError):
        DoorwayEntry(geometry, "view-00051", snapshot(), now=10)


@pytest.mark.parametrize("bad", [None, "", " ", 5, "x" * 201])
def test_source_id_is_required(bad):
    with pytest.raises(ValueError):
        DoorwayEntry(reference(), bad, snapshot(), now=10)


@pytest.mark.parametrize(
    "initial,now",
    [
        (snapshot(x=1), 10),
        (snapshot(x=1.1), 10),
        (snapshot(connected=False), 10),
        (snapshot(), 11),
        (snapshot(), math.nan),
    ],
)
def test_invalid_initial_pose_or_time_is_rejected(initial, now):
    with pytest.raises(ValueError):
        DoorwayEntry(reference(), "view-00051", copy.deepcopy(initial), now=now)
