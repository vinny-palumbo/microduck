"""Safety regressions use a fake transport; no robot or network is contacted."""

import asyncio
import copy
import itertools
import json
import math
import time
import unittest
from dataclasses import replace

from duck_nav.core import GuardConfig, GuardedRobot


class FakeRobot:
    def __init__(self):
        self.connected = True
        self.frozen = set()
        self.created = time.monotonic()
        self.calls = []
        self.pulses = []
        self.head_pulses = []
        self.health_delay = 0
        self.stop_delay = 0
        self.fail_notify = False
        self.rotate = False
        self.model = {
            "trunk_height_m": 0.12,
            "tof_beams": [],
            "joint_names": ["neck_pitch", "head_pitch", "head_yaw", "head_roll"],
            "joint_home": [0.3491, 0.3491, 0, 0],
        }
        self.look_result = {
            "head": dict.fromkeys(self.model["joint_names"], 0),
            "joint_targets": {
                "neck_pitch": 0.3491,
                "head_pitch": 0.3491,
                "head_yaw": 0,
                "head_roll": 0,
            },
            "clamped": False,
        }
        for row in range(8):
            elevation = math.radians(19.6875 - row * 5.625)
            for col in range(8):
                azimuth = math.radians(19.6875 - col * 5.625)
                self.model["tof_beams"].append(
                    [
                        math.cos(elevation) * math.cos(azimuth),
                        math.cos(elevation) * math.sin(azimuth),
                        math.sin(elevation),
                    ]
                )
        self.health = {
            "healthy": True,
            "control_loop": {"last_tick_age_ms": 1, "achieved_hz": 50},
            "bus": {"consecutive_errors": 0},
            "imu": {"ready": True, "consecutive_stale_blocks": 0},
        }
        self.state = {
            "t_ns": 1,
            "safety": {"fallen": False, "limp": False, "gravity": [0, 0, -1]},
            "policy": "walk",
            "loop": {"hz": 50},
            "joints": [0.3491, 0.3491, 0, 0],
            "head": [0, 0, 0, 0],
            "odom": {"position": [0, 0, 0.12], "yaw": 0},
            "frames": {
                "tof": {"pos": [0.03, 0, 0.05], "quat": [1, 0, 0, 0]},
                "camera": {"pos": [0, 0, 0], "quat": [math.sqrt(0.5), 0, math.sqrt(0.5), 0]},
            },
        }
        self.depth = {
            "t_ns": 1,
            "rows": 8,
            "cols": 8,
            "distance_mm": [2000] * 64,
            "status": [5] * 64,
        }

    async def request(self, method, params):
        self.calls.append((method, params))
        if method == "robot.model":
            return self.model
        if method == "robot.health":
            await asyncio.sleep(self.health_delay)
            return copy.deepcopy(self.health)
        if method == "robot.stop":
            await asyncio.sleep(self.stop_delay)
            if not self.connected:
                raise ConnectionError("dropped")
            return {"accepted": True}
        if method == "robot.look":
            return copy.deepcopy(self.look_result)
        raise AssertionError(method)

    def notify(self, method, params):
        if self.fail_notify or not self.connected:
            raise ConnectionError("dropped")
        if method == "robot.head":
            self.head_pulses.append((time.monotonic(), params.copy()))
            return
        self.pulses.append((time.monotonic(), params.copy()))
        if self.rotate:
            yaw = self.state["odom"]["yaw"] + params["vyaw"] * 0.1
            self.state["odom"]["yaw"] = math.atan2(math.sin(yaw), math.cos(yaw))

    def snapshot(self):
        now = time.monotonic()
        self.state["t_ns"] = self.depth["t_ns"] = int(now * 1e9)
        return {
            "connected": self.connected,
            "camera": {
                "sequence": int(now * 1000),
                "received_at": self.created if "camera" in self.frozen else now,
                "image": object(),
                "metadata": {"width": 640},
            },
            "state": {
                "sequence": 1,
                "received_at": self.created if "state" in self.frozen else now,
                "data": copy.deepcopy(self.state),
            },
            "depth": {
                "sequence": 1,
                "received_at": self.created if "depth" in self.frozen else now,
                "data": copy.deepcopy(self.depth),
            },
        }

    def moving_pulses(self):
        return [pulse for pulse in self.pulses if pulse[1]["vx"] or pulse[1]["vyaw"]]


class GuardTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.transport = FakeRobot()
        self.robot = GuardedRobot(
            self.transport,
            GuardConfig(
                pulse_period_s=0.01,
                state_max_age_s=0.07,
                depth_max_age_s=0.07,
                camera_max_age_s=0.07,
                health_poll_s=0.02,
                health_max_age_s=0.10,
                rpc_timeout_s=0.06,
                initial_sensor_timeout_s=0.1,
                turn_timeout_s=0.3,
                look_min_hold_s=0.04,
                look_settle_s=0.02,
                look_timeout_s=0.12,
            ),
        )
        await self.robot.initialize()

    async def asyncTearDown(self):
        await self.robot.close()

    async def wait_for_movement(self):
        for _ in range(100):
            if self.transport.moving_pulses():
                return
            await asyncio.sleep(0.002)
        self.fail("movement did not start")

    async def test_bounded_movement_stops_and_never_enables(self):
        result = await self.robot.move_for(0.08, 0.05)
        self.assertTrue(result["completed"])
        self.assertGreater(len(self.transport.moving_pulses()), 2)
        self.assertEqual(self.transport.pulses[-1][1], {"vx": 0, "vy": 0, "vyaw": 0})
        self.assertTrue(result["stop"]["acknowledged"])
        self.assertNotIn("robot.enable", [call[0] for call in self.transport.calls])
        json.dumps(await self.robot.observe(), allow_nan=False)

    async def test_depth_sectors_locate_obstacles_without_weakening_guard(self):
        self.transport.depth["status"] = [255] * 64
        for row in range(3):
            for col in range(3):
                index = row * 8 + col
                self.transport.depth["status"][index] = 5
                self.transport.depth["distance_mm"][index] = 250
        observation = await self.robot.observe()
        self.assertEqual(observation["guard_reason"], "obstacle")
        sectors = observation["depth_summary"]["sectors"]
        self.assertLess(sectors["left"]["nearest_obstacle_m"], 0.3)
        self.assertIsNone(sectors["right"]["nearest_obstacle_m"])
        self.assertEqual(sectors["right"]["known_zones"], 24)

    async def test_invalid_inputs_never_issue_commands(self):
        for speed, duration in [
            (-0.1, 1),
            (0, 1),
            (0.11, 1),
            (0.1, 2.1),
            (0.1, 0),
            (float("nan"), 1),
            (0.1, float("inf")),
            (True, 1),
            (".1", 1),
        ]:
            with (
                self.subTest(speed=speed, duration=duration),
                self.assertRaises((TypeError, ValueError)),
            ):
                await self.robot.move_for(speed, duration)
        for angle in [0, 31, -31, float("nan"), False]:
            with self.assertRaises((TypeError, ValueError)):
                await self.robot.turn_by(angle)
        for point in [(0, 0, 0), (float("nan"), 0, 0), (4, 0, 0)]:
            with self.assertRaises(ValueError):
                await self.robot.look_at(*point)
        self.assertEqual(self.transport.pulses, [])

    async def test_side_scan_reports_depth_but_cannot_authorize_motion(self):
        angle = math.radians(45)
        self.transport.state["frames"]["tof"]["quat"] = [
            math.cos(angle / 2),
            0,
            0,
            math.sin(angle / 2),
        ]
        observation = await self.robot.observe()
        self.assertFalse(observation["ready"])
        self.assertEqual(observation["guard_reason"], "head_not_forward")
        depth = observation["depth_summary"]
        self.assertAlmostEqual(depth["sensor_yaw_deg"], 45)
        self.assertEqual(depth["sectors"]["left"]["known_zones"], 64)
        self.assertEqual(depth["sectors"]["right"]["known_zones"], 0)
        self.assertEqual((await self.robot.move_for(0.1, 0.1))["reason"], "head_not_forward")
        self.assertEqual(self.transport.moving_pulses(), [])

    async def test_all_frozen_streams_refuse_motion(self):
        # A frozen source can still be delivered by the network: its trusted
        # received_at is only advanced by the transport for new source data.
        self.transport.created = time.monotonic() - 1
        for stream in ("camera", "state", "depth"):
            self.transport.frozen = {stream}
            result = await self.robot.move_for(0.1, 0.1)
            self.assertEqual(result["reason"], f"stale_{stream}")
        self.assertEqual(self.transport.moving_pulses(), [])

    async def test_depth_freezes_mid_move(self):
        move = asyncio.create_task(self.robot.move_for(0.1, 0.3))
        await self.wait_for_movement()
        self.transport.frozen = {"depth"}
        result = await move
        self.assertEqual(result["reason"], "stale_depth")
        self.assertLess(result["elapsed_s"], 0.2)

    async def test_obstacle_appears_mid_move(self):
        move = asyncio.create_task(self.robot.move_for(0.1, 0.3))
        await self.wait_for_movement()
        self.transport.depth["distance_mm"][27] = 200
        result = await move
        self.assertEqual(result["reason"], "obstacle")
        self.assertFalse(result["completed"])
        self.assertLess(result["elapsed_s"], 0.2)

    async def test_floor_projection_and_too_close_are_distinct(self):
        # A beam interrupted halfway to the floor is an obstacle; reaching the
        # floor is expected. Raw min(depth) cannot make this distinction.
        self.transport.state["odom"]["position"][2] = 0.06
        self.transport.state["frames"]["tof"]["pos"][2] = 0.02
        self.transport.depth["distance_mm"][59] = 250
        observation = await self.robot.observe()
        self.assertTrue(observation["ready"], observation)
        self.assertGreater(observation["depth_summary"]["floor_zones"], 0)
        self.transport.depth["distance_mm"][59] = 125
        self.assertEqual((await self.robot.observe())["guard_reason"], "obstacle")
        self.transport.depth["distance_mm"][27] = 20
        self.assertEqual((await self.robot.observe())["guard_reason"], "depth_too_close")

    async def test_floor_uses_gravity_when_trunk_leans(self):
        self.transport.state["safety"]["gravity"] = [0.5, 0, -math.sqrt(0.75)]
        self.transport.depth["distance_mm"] = [350] * 64
        observation = await self.robot.observe()
        self.assertGreater(observation["depth_summary"]["floor_zones"], 0)

    async def test_head_away_blocks_move_but_allows_look(self):
        self.transport.state["frames"]["tof"]["quat"] = [math.sqrt(0.5), 0, 0, math.sqrt(0.5)]
        self.assertEqual((await self.robot.move_for(0.1, 0.1))["reason"], "head_not_forward")
        self.assertTrue((await self.robot.look_at(1, 0, 0))["completed"])
        self.assertGreater(len(self.transport.head_pulses), 1)
        self.assertEqual(self.transport.moving_pulses(), [])

    async def test_look_waits_for_measured_aim_and_fresh_camera(self):
        self.transport.state["frames"]["camera"]["quat"] = [1, 0, 0, 0]
        self.assertEqual((await self.robot.look_at(1, 0, 0))["reason"], "look_timeout")
        self.transport.state["frames"]["camera"]["quat"] = [math.sqrt(0.5), 0, math.sqrt(0.5), 0]
        self.transport.frozen.add("camera")
        self.assertEqual((await self.robot.look_at(1, 0, 0))["reason"], "stale_camera")

    async def test_look_resends_offsets_but_settles_against_measured_camera(self):
        # Joint tracking bias is acceptable only when measured camera aim is correct.
        self.transport.state["joints"][1] = 0
        self.assertEqual((await self.robot.look_at(1, 0, 0))["reason"], "gaze_settled")
        self.assertTrue(self.transport.head_pulses)
        self.assertTrue(all(params["head_pitch"] == 0 for _, params in self.transport.head_pulses))

    async def test_look_requires_current_joint_target_contract(self):
        del self.transport.look_result["joint_targets"]
        self.assertEqual((await self.robot.look_at(1, 0, 0))["reason"], "invalid_telemetry")
        self.assertEqual(self.transport.head_pulses, [])

    async def test_repeated_looks_preserve_command_despite_neck_tracking_bias(self):
        # A lower measured angle must not become a progressively lower command.
        for measured in (0.27, 0.23, 0.19):
            self.transport.state["joints"][0] = measured
            self.assertTrue((await self.robot.look_at(1, 0, 0))["completed"])
        requests = [p for method, p in self.transport.calls if method == "robot.look"]
        self.assertEqual([p["neck_pitch"] for p in requests], [0.3491] * 3)

    async def test_look_preserves_nonzero_neck_command(self):
        self.transport.state["head"][0] = -0.05
        self.assertTrue((await self.robot.look_at(1, 0, 0))["completed"])
        params = next(p for method, p in self.transport.calls if method == "robot.look")
        self.assertAlmostEqual(params["neck_pitch"], 0.2991)

    async def test_joint_alignment_alone_does_not_prove_camera_aim(self):
        self.transport.state["frames"]["camera"]["quat"] = [1, 0, 0, 0]
        self.assertEqual((await self.robot.look_at(1, 0, 0))["reason"], "look_timeout")

    async def test_feedback_converges_without_changing_neck(self):
        self.robot.config = replace(self.robot.config, look_feedback_period_s=0.01)
        angle = math.pi / 2 - 0.2
        self.transport.state["frames"]["camera"]["quat"] = [
            math.cos(angle / 2),
            0,
            math.sin(angle / 2),
            0,
        ]
        original = self.transport.request

        async def respond(method, params=None):
            result = await original(method, params)
            if method == "robot.look" and params["z"] < 0:
                self.transport.state["frames"]["camera"]["quat"] = [
                    math.sqrt(0.5),
                    0,
                    math.sqrt(0.5),
                    0,
                ]
            return result

        self.transport.request = respond
        outcome = await self.robot.look_at(1, 0, 0)
        self.assertTrue(outcome["completed"])
        self.assertEqual(outcome["corrections"], 1)
        self.assertEqual(outcome["aim_error_rad"], 0)
        requests = [p for method, p in self.transport.calls if method == "robot.look"]
        self.assertTrue(all(p["neck_pitch"] == 0.3491 for p in requests))
        self.assertEqual(self.transport.moving_pulses(), [])

    async def test_feedback_stops_at_correction_limit(self):
        self.robot.config = replace(self.robot.config, look_feedback_period_s=0.01)
        self.transport.state["frames"]["camera"]["quat"] = [1, 0, 0, 0]
        outcome = await self.robot.look_at(1, 0, 0)
        self.assertEqual(outcome["reason"], "look_correction_limit")
        self.assertTrue(outcome["stop"]["acknowledged"])
        self.assertEqual(outcome["corrections"], 0)

    def simulate_left_to_right_sweep(self, *, stall_fraction=1.0):
        self.robot.config = replace(
            self.robot.config, look_feedback_period_s=0.04, look_timeout_s=0.5
        )
        original_snapshot = self.transport.snapshot
        original_request = self.transport.request
        requested_at = None

        async def request(method, params):
            nonlocal requested_at
            result = await original_request(method, params)
            if method == "robot.look" and requested_at is None:
                requested_at = time.monotonic()
            return result

        def snapshot():
            elapsed = 0 if requested_at is None else time.monotonic() - requested_at
            fraction = min(stall_fraction, elapsed / 0.16)
            yaw = math.pi / 4 - math.pi / 2 * fraction
            # Camera +Z points along trunk +X after its fixed 90-degree pitch,
            # then sweeps from trunk left (+45) to trunk right (-45 degrees).
            c, s = math.cos(yaw / 2) / math.sqrt(2), math.sin(yaw / 2) / math.sqrt(2)
            self.transport.state["frames"]["camera"]["quat"] = [c, -s, c, s]
            return original_snapshot()

        self.transport.request = request
        self.transport.snapshot = snapshot

    async def test_wide_sweep_does_not_correct_while_camera_approaches_target(self):
        self.simulate_left_to_right_sweep()
        outcome = await self.robot.look_at(1, -1, 0)
        self.assertEqual(outcome["reason"], "gaze_settled")
        self.assertTrue(outcome["completed"])
        self.assertEqual(outcome["corrections"], 0)
        self.assertLessEqual(
            outcome["aim_error_rad"], self.robot.config.look_direction_tolerance_rad
        )
        self.assertEqual(len([call for call in self.transport.calls if call[0] == "robot.look"]), 1)
        self.assertEqual(self.transport.moving_pulses(), [])
        self.assertTrue(outcome["stop"]["acknowledged"])

    async def test_stalled_wide_sweep_still_enforces_correction_limit(self):
        self.simulate_left_to_right_sweep(stall_fraction=0.3)
        outcome = await self.robot.look_at(1, -1, 0)
        self.assertEqual(outcome["reason"], "look_correction_limit")
        self.assertFalse(outcome["completed"])
        self.assertEqual(outcome["corrections"], 0)
        self.assertLess(outcome["elapsed_s"], self.robot.config.look_timeout_s)
        self.assertEqual(self.transport.moving_pulses(), [])
        self.assertTrue(outcome["stop"]["acknowledged"])

    async def test_approaching_target_cannot_extend_look_timeout(self):
        self.simulate_left_to_right_sweep()
        self.robot.config = replace(self.robot.config, look_timeout_s=0.07)
        outcome = await self.robot.look_at(1, -1, 0)
        self.assertEqual(outcome["reason"], "look_timeout")
        self.assertFalse(outcome["completed"])
        self.assertEqual(outcome["corrections"], 0)
        self.assertLess(outcome["elapsed_s"], 0.12)
        self.assertTrue(outcome["stop"]["acknowledged"])

    async def test_look_requires_home_command_and_camera_contract(self):
        for field in ("joint_home", "head", "camera"):
            container = (
                self.transport.model
                if field == "joint_home"
                else self.transport.state
                if field == "head"
                else self.transport.state["frames"]
            )
            saved = container.pop(field)
            self.assertEqual((await self.robot.look_at(1, 0, 0))["reason"], "invalid_telemetry")
            container[field] = saved

    async def test_explicit_stop_cancels_head_hold(self):
        look = asyncio.create_task(self.robot.look_at(1, 0, 0))
        while not self.transport.head_pulses:
            await asyncio.sleep(0.002)
        await self.robot.stop()
        count = len(self.transport.head_pulses)
        self.assertEqual((await look)["reason"], "stopped")
        self.assertEqual(len(self.transport.head_pulses), count)

    async def test_missing_bad_and_unknown_depth_fail_closed(self):
        for change, expected in [
            (lambda d: d["status"].__setitem__(27, 4), "depth_quality"),
            (lambda d: d["distance_mm"].__setitem__(27, float("nan")), "invalid_depth"),
            (lambda d: d["status"].pop(), "invalid_depth"),
        ]:
            saved = copy.deepcopy(self.transport.depth)
            change(self.transport.depth)
            self.assertEqual((await self.robot.move_for(0.1, 0.1))["reason"], expected)
            json.dumps(await self.robot.observe(), allow_nan=False)
            self.transport.depth = saved
        self.transport.depth["status"] = [255] * 64
        self.assertTrue((await self.robot.observe())["ready"])
        self.transport.depth["status"] = [9] * 64
        self.assertTrue((await self.robot.observe())["ready"])

    async def test_live_health_failure_interrupts_movement(self):
        move = asyncio.create_task(self.robot.move_for(0.1, 0.3))
        await self.wait_for_movement()
        self.transport.health["bus"]["consecutive_errors"] = 1
        result = await move
        self.assertEqual(result["reason"], "bus_errors")
        self.assertLess(result["elapsed_s"], 0.15)

    async def test_slow_health_rpc_does_not_block_pulses_or_safety(self):
        self.transport.health_delay = 0.5
        result = await self.robot.move_for(0.1, 0.3)
        self.assertEqual(result["reason"], "unhealthy")
        times = [t for t, _ in self.transport.moving_pulses()]
        self.assertGreater(len(times), 3)
        self.assertLess(max(b - a for a, b in itertools.pairwise(times)), 0.04)

    async def test_connection_loss_and_send_failure_stop(self):
        move = asyncio.create_task(self.robot.move_for(0.1, 0.3))
        await self.wait_for_movement()
        self.transport.connected = False
        result = await move
        self.assertEqual(result["reason"], "disconnected")
        self.assertFalse(result["stop"]["acknowledged"])
        self.transport.connected = True
        self.transport.fail_notify = True
        self.assertEqual((await self.robot.move_for(0.1, 0.1))["reason"], "transport_error")

    async def test_task_cancellation_sends_stop(self):
        move = asyncio.create_task(self.robot.move_for(0.1, 0.3))
        await self.wait_for_movement()
        move.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await move
        self.assertEqual(self.transport.pulses[-1][1]["vx"], 0)
        self.assertEqual(self.transport.calls[-1][0], "robot.stop")

    async def test_explicit_stop_prevents_later_repulse_and_concurrent_action(self):
        move = asyncio.create_task(self.robot.move_for(0.1, 0.3))
        await self.wait_for_movement()
        with self.assertRaises(RuntimeError):
            await self.robot.turn_by(10)
        self.transport.stop_delay = 0.03
        stop = asyncio.create_task(self.robot.stop())
        await asyncio.sleep(0.005)
        count = len(self.transport.moving_pulses())
        with self.assertRaises(RuntimeError):
            await self.robot.move_for(0.1, 0.1)
        self.assertTrue((await stop)["completed"])
        self.assertEqual((await move)["reason"], "stopped")
        await asyncio.sleep(0.02)
        self.assertEqual(len(self.transport.moving_pulses()), count)

    async def test_turn_odometry_wrap_and_timeout(self):
        self.transport.state["odom"]["yaw"] = math.radians(179)
        self.transport.rotate = True
        result = await self.robot.turn_by(10)
        self.assertEqual(result["reason"], "angle_reached")
        self.assertGreater(result["turned_deg"], 8)
        self.assertGreater(result["before_odom"]["yaw"], 0)
        self.assertLess(result["after_odom"]["yaw"], 0)
        self.transport.rotate = False
        self.assertEqual((await self.robot.turn_by(-10))["reason"], "turn_timeout")

    async def test_small_turn_caps_command_budget_when_odom_is_frozen(self):
        result = await self.robot.turn_by(1)
        self.assertFalse(result["completed"])
        pulses = self.transport.moving_pulses()
        first_zero = next(t for t, params in self.transport.pulses if params["vyaw"] == 0)
        self.assertLess(first_zero - pulses[0][0], math.radians(1) / 0.25 + 0.02)
        self.assertGreater(result["elapsed_s"], 0.2)
        self.assertLess(len(pulses), 10)

    async def test_snapshot_failure_still_stops_and_releases_action(self):
        original = self.transport.snapshot

        def fail_once():
            self.transport.snapshot = original
            raise ValueError("bad snapshot")

        self.transport.snapshot = fail_once
        self.assertEqual((await self.robot.move_for(0.1, 0.1))["reason"], "transport_error")
        self.assertEqual(self.transport.calls[-1][0], "robot.stop")
        self.assertTrue((await self.robot.move_for(0.1, 0.02))["completed"])

    async def test_unsafe_robot_state_blocks_move(self):
        for key, value, expected in [
            ("fallen", True, "fallen"),
            ("limp", True, "not_enabled"),
            ("gravity", [0, 0, 0], "invalid_telemetry"),
        ]:
            saved = copy.deepcopy(self.transport.state["safety"])
            self.transport.state["safety"][key] = value
            self.assertEqual((await self.robot.move_for(0.1, 0.1))["reason"], expected)
            self.transport.state["safety"] = saved
        self.assertEqual(self.transport.moving_pulses(), [])


if __name__ == "__main__":
    unittest.main()
