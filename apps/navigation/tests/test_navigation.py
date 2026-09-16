"""Action lifecycle regressions; actual gait performance is measured in MuJoCo."""

import asyncio
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
