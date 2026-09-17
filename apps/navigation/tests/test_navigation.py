"""Action lifecycle regressions; actual gait performance is measured in MuJoCo."""

import asyncio
import json
import math
from dataclasses import replace

import pytest
from test_core import FakeRobot

from duck_nav.core import GuardConfig
from duck_nav.navigation import GaitConfig, GaitNavigator


class MovingRobot(FakeRobot):
    def __init__(self):
        super().__init__()
        self.state["move"] = {"requested": [0, 0, 0], "applied": [0, 0, 0]}
        self.walks = True

    def notify(self, method, params):
        super().notify(method, params)
        if method != "robot.move":
            return
        commands = [params["vx"], params["vy"], params["vyaw"]]
        self.state["move"] = {"requested": commands, "applied": commands}
        if self.walks:
            pose = self.state["odom"]
            pose["yaw"] += params["vyaw"] * 0.05
            pose["position"][0] += params["vx"] * math.cos(pose["yaw"]) * 0.05
            pose["position"][1] += params["vx"] * math.sin(pose["yaw"]) * 0.05


@pytest.fixture
async def pair():
    transport = MovingRobot()
    robot = GaitNavigator(
        transport,
        GuardConfig(pulse_period_s=0.01),
        GaitConfig(
            drive_timeout_s=0.3, settle_timeout_s=0.2, settle_window_s=0.03, settle_min_s=0.04
        ),
    )
    await robot.initialize()
    try:
        yield robot, transport
    finally:
        await robot.close()


async def test_arc_reports_settled_motion_and_holds_lock_during_settling(pair):
    robot, transport = pair
    action = asyncio.create_task(robot.advance(0.1, 20))
    while not transport.moving_pulses():
        await asyncio.sleep(0.001)
    while transport.pulses[-1][1]["vx"]:
        await asyncio.sleep(0.001)
    with pytest.raises(RuntimeError, match="another action"):
        await robot.advance(0.1)
    outcome = await action
    assert outcome["completed"]
    assert outcome["stop"]["physical_settling_verified"]
    assert outcome["distance_m"] >= 0.08
    assert outcome["heading_deg"] > 0
    assert transport.pulses[-1][1] == {"vx": 0, "vy": 0, "vyaw": 0}
    assert max(abs(p[1]["vyaw"]) for p in transport.pulses) <= 0.5


async def test_cancel_stops_arc_without_claiming_success(pair):
    robot, transport = pair
    action = asyncio.create_task(robot.advance(0.2))
    while not transport.moving_pulses():
        await asyncio.sleep(0.001)
    await robot.stop()
    result = await action
    assert not result["completed"]
    assert result["reason"] == "cancelled"
    assert transport.pulses[-1][1]["vx"] == 0
    assert not robot.course()["active"]


async def test_obstacle_aborts_arc_and_never_restarts(pair):
    robot, transport = pair
    action = asyncio.create_task(robot.advance(0.2))
    while not transport.moving_pulses():
        await asyncio.sleep(0.001)
    transport.depth["distance_mm"] = [200] * 64
    result = await action
    assert result["reason"] == "obstacle"
    assert not result["completed"]
    assert transport.pulses[-1][1]["vx"] == 0
    assert robot.course()["reset_reason"] == "obstacle"
    assert not robot.course()["active"]


async def test_frozen_pose_does_not_count_command_duration_as_progress(pair):
    robot, transport = pair
    transport.walks = False
    result = await robot.advance(0.1)
    assert not result["completed"]
    assert result["reason"] == "no_progress"


async def test_stop_ack_without_physical_settling_is_failure(pair):
    robot, transport = pair
    original = transport.snapshot

    def oscillating():
        result = original()
        if transport.pulses and transport.pulses[-1][1]["vx"] == 0:
            transport.state["odom"]["yaw"] += 0.1
        return result

    transport.snapshot = oscillating
    result = await robot.advance(0.1)
    assert not result["completed"]
    assert result["reason"] == "settle_timeout"


@pytest.mark.parametrize("distance,heading", [(0.01, 0), (0.3, 0), (0.1, 31), (True, 0)])
async def test_invalid_arc_never_moves(pair, distance, heading):
    robot, transport = pair
    with pytest.raises((ValueError, TypeError)):
        await robot.advance(distance, heading)
    assert not transport.moving_pulses()


def test_gait_envelope_cannot_be_silently_expanded():
    with pytest.raises(ValueError):
        replace(GaitConfig(), command_speed_m_s=0.5)


async def test_initial_snapshot_failure_always_stops_and_releases_action_lock(pair):
    robot, transport = pair
    original = transport.snapshot

    def unavailable():
        raise ValueError("bad camera rotation metadata")

    transport.snapshot = unavailable
    result = await robot.advance(0.1)
    assert not result["completed"]
    assert not robot._active
    assert transport.calls[-1][0] == "robot.stop"
    transport.snapshot = original
    assert (await robot.advance(0.1))["completed"]


async def test_lost_depth_during_deceleration_cannot_report_success(pair):
    robot, transport = pair
    original = transport.notify

    def lose_depth_at_stop(method, params):
        original(method, params)
        if method == "robot.move" and params["vx"] == 0:
            transport.depth["status"] = [0] * 64

    transport.notify = lose_depth_at_stop
    result = await robot.advance(0.1)
    assert not result["completed"]
    assert result["reason"] == "depth_quality"


async def test_zero_heading_recovers_previous_settled_error(pair):
    robot, transport = pair
    original = transport.notify
    disturbed = False

    def settle_with_right_bias(method, params):
        nonlocal disturbed
        original(method, params)
        if method == "robot.move" and params["vx"] == 0 and not disturbed:
            transport.state["odom"]["yaw"] -= math.radians(8)
            disturbed = True

    transport.notify = settle_with_right_bias
    first = await robot.advance(0.1)
    assert first["completed"]
    assert first["course"]["error_deg"] == pytest.approx(8)
    start = len(transport.moving_pulses())
    second = await robot.advance(0.1)
    assert second["completed"]
    assert second["requested_heading_deg"] == 0
    assert second["effective_heading_deg"] == pytest.approx(8)
    assert second["target_yaw_deg"] == pytest.approx(first["target_yaw_deg"])
    assert transport.moving_pulses()[start][1]["vyaw"] > 0
    assert abs(second["course"]["error_deg"]) < abs(first["course"]["error_deg"])


async def test_nonzero_heading_replaces_target_relative_to_current_pose(pair):
    robot, transport = pair
    first = await robot.advance(0.1, 20)
    current = math.degrees(transport.state["odom"]["yaw"])
    second = await robot.advance(0.1, -10)
    assert second["completed"]
    assert second["target_yaw_deg"] == pytest.approx(current - 10)
    assert second["target_yaw_deg"] != pytest.approx(first["target_yaw_deg"] - 10)
    assert second["effective_heading_deg"] == pytest.approx(-10)


async def test_course_wraps_across_pi_without_reversing_correction(pair):
    robot, transport = pair
    transport.state["odom"]["yaw"] = math.radians(179)
    first = await robot.advance(0.1, 20)
    assert first["target_yaw_deg"] == pytest.approx(-161)
    second = await robot.advance(0.1)
    assert second["completed"]
    assert 0 < second["effective_heading_deg"] < 20
    assert second["target_yaw_deg"] == pytest.approx(-161)
    assert abs(second["course"]["error_deg"]) < 20


@pytest.mark.parametrize("external_change", ["position", "yaw", "height"])
@pytest.mark.parametrize("observe_before_advance", [True, False])
async def test_external_motion_invalidates_course_and_reanchors_next_advance(
    pair, external_change, observe_before_advance
):
    robot, transport = pair
    await robot.advance(0.1, 20)
    if external_change == "yaw":
        transport.state["odom"]["yaw"] += math.radians(5.1)
    else:
        axis = 0 if external_change == "position" else 2
        transport.state["odom"]["position"][axis] += 0.026
    if observe_before_advance:
        observation = await robot.observe()
        assert observation["course"]["active"] is False
        assert observation["course"]["reset_reason"] == "external_pose_change"
    current_yaw = math.degrees(transport.state["odom"]["yaw"])
    result = await robot.advance(0.1)
    assert result["completed"]
    assert result["target_yaw_deg"] == pytest.approx(current_yaw)
    assert result["effective_heading_deg"] == pytest.approx(0)
    assert result["course_reset_reason"] == "external_pose_change"
    assert result["course"]["reset_reason"] is None


async def test_course_survives_small_post_settle_motion_and_successful_gaze(pair):
    robot, transport = pair
    first = await robot.advance(0.1, 20)
    transport.state["odom"]["position"][0] += 0.01
    transport.state["odom"]["yaw"] += math.radians(1)
    assert (await robot.look_at(1, 0, 0))["completed"]
    course = (await robot.observe())["course"]
    assert course["active"]
    assert course["target_yaw_deg"] == pytest.approx(first["target_yaw_deg"])


async def test_retained_correction_beyond_limit_refuses_without_motion(pair):
    robot, transport = pair
    original = transport.notify

    def cannot_turn(method, params):
        yaw = transport.state["odom"]["yaw"]
        original(method, params)
        transport.state["odom"]["yaw"] = yaw

    transport.notify = cannot_turn
    first = await robot.advance(0.1, 30)
    assert first["completed"] and not first["target_reached"]
    # This small uncommanded shift is inside the external-motion reset threshold,
    # but makes the retained target require more than the allowed 30-degree arc.
    transport.state["odom"]["yaw"] -= math.radians(2)
    count = len(transport.moving_pulses())
    refused = await robot.advance(0.1)
    assert refused["reason"] == "course_correction_limit"
    assert not refused["completed"] and not refused["target_reached"]
    assert refused["effective_heading_deg"] == pytest.approx(32)
    assert len(transport.moving_pulses()) == count
    assert not refused["course"]["active"]
    assert refused["course"]["reset_reason"] == "course_correction_limit"
    resumed = await robot.advance(0.1)
    assert resumed["completed"]
    assert resumed["effective_heading_deg"] == pytest.approx(0)


async def test_task_cancellation_clears_course_and_stops(pair):
    robot, transport = pair
    await robot.advance(0.1, 20)
    count = len(transport.moving_pulses())
    task = asyncio.create_task(robot.advance(0.2))
    while len(transport.moving_pulses()) == count:
        await asyncio.sleep(0.001)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert transport.pulses[-1][1]["vx"] == 0
    assert not robot.course()["active"]
    assert robot.course()["reset_reason"] == "cancelled"


@pytest.mark.parametrize(
    "operation,reason", [("stop", "public_stop"), ("initialize", "initialized")]
)
async def test_explicit_lifecycle_reset_discards_previous_course(pair, operation, reason):
    robot, transport = pair
    await robot.advance(0.1, 20)
    await getattr(robot, operation)()
    assert robot.course() == {
        "active": False,
        "target_yaw_deg": None,
        "error_deg": None,
        "reset_reason": reason,
    }
    current_yaw = math.degrees(transport.state["odom"]["yaw"])
    result = await robot.advance(0.1)
    assert result["target_yaw_deg"] == pytest.approx(current_yaw)
    assert result["course_reset_reason"] == reason
    assert result["course"]["reset_reason"] is None


async def test_failed_advance_discards_old_target_before_recovery(pair):
    robot, transport = pair
    first = await robot.advance(0.1, 20)
    transport.depth["distance_mm"] = [200] * 64
    failed = await robot.advance(0.1)
    assert failed["reason"] == "obstacle"
    assert not failed["course"]["active"]
    transport.depth["distance_mm"] = [2000] * 64
    current = math.degrees(transport.state["odom"]["yaw"])
    recovered = await robot.advance(0.1)
    assert recovered["target_yaw_deg"] == pytest.approx(current)
    assert recovered["target_yaw_deg"] != pytest.approx(first["target_yaw_deg"])
    assert recovered["course_reset_reason"] == "obstacle"
    assert recovered["effective_heading_deg"] == pytest.approx(0)


async def test_course_observation_never_emits_nonfinite_telemetry(pair):
    robot, transport = pair
    await robot.advance(0.1)
    transport.state["odom"]["yaw"] = math.nan
    observation = await robot.observe()
    assert observation["course"]["active"] is False
    assert observation["course"]["reset_reason"] == "telemetry_unavailable"
    json.dumps(observation["course"], allow_nan=False)


@pytest.mark.parametrize("failure", ["invalid_yaw", "disconnected"])
async def test_transient_observer_failure_cannot_clear_an_active_actions_course(pair, failure):
    robot, transport = pair
    task = asyncio.create_task(robot.advance(0.2, 20))
    while not transport.moving_pulses():
        await asyncio.sleep(0.001)
    original = transport.snapshot

    def transient_failure():
        snapshot = original()
        if failure == "invalid_yaw":
            snapshot["state"]["data"]["odom"]["yaw"] = math.nan
        else:
            snapshot["connected"] = False
        return snapshot

    transport.snapshot = transient_failure
    course = robot.course()
    assert course["active"] and course["error_deg"] is None
    transport.snapshot = original
    result = await task
    assert result["completed"]
    assert result["course"]["active"]
    assert result["course"]["target_yaw_deg"] == pytest.approx(20)


async def test_idle_disconnection_discards_retained_course(pair):
    robot, transport = pair
    await robot.advance(0.1, 20)
    transport.connected = False
    assert robot.course() == {
        "active": False,
        "target_yaw_deg": None,
        "error_deg": None,
        "reset_reason": "disconnected",
    }
