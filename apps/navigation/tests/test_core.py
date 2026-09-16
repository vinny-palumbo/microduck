"""Safety regressions use a fake transport; no robot or network is contacted."""

import asyncio
import copy
import itertools
import json
import math
import time
import unittest

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
            "odom": {"position": [0, 0, 0.12], "yaw": 0},
            "frames": {"tof": {"pos": [0.03, 0, 0.05], "quat": [1, 0, 0, 0]}},
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

    async def test_look_waits_for_measured_joints_and_fresh_camera(self):
        self.transport.state["joints"][1] = 0.5
        self.assertEqual((await self.robot.look_at(1, 0, 0))["reason"], "look_timeout")
        self.transport.state["joints"][1] = 0.3491
        self.transport.frozen.add("camera")
        self.assertEqual((await self.robot.look_at(1, 0, 0))["reason"], "stale_camera")

    async def test_look_resends_offsets_but_settles_against_absolute_joints(self):
        # The simulator's head policy adds HOME pitch to the command. Zero
        # measured pitch is therefore not settled for a zero-offset command.
        self.transport.state["joints"][1] = 0
        self.assertEqual((await self.robot.look_at(1, 0, 0))["reason"], "look_timeout")
        self.transport.state["joints"][1] = 0.3491
        self.assertEqual((await self.robot.look_at(1, 0, 0))["reason"], "gaze_settled")
        self.assertTrue(self.transport.head_pulses)
        self.assertTrue(all(params["head_pitch"] == 0 for _, params in self.transport.head_pulses))

    async def test_look_requires_current_joint_target_contract(self):
        del self.transport.look_result["joint_targets"]
        self.assertEqual((await self.robot.look_at(1, 0, 0))["reason"], "invalid_telemetry")
        self.assertEqual(self.transport.head_pulses, [])

    async def test_look_preserves_measured_neck_posture(self):
        self.transport.state["joints"][0] = 0.27
        self.transport.look_result["joint_targets"]["neck_pitch"] = 0.27
        self.transport.look_result["head"]["neck_pitch"] = 0.27 - 0.3491
        self.assertTrue((await self.robot.look_at(1, 0.1, 0.05))["completed"])
        requested = next(
            params for method, params in self.transport.calls if method == "robot.look"
        )
        self.assertEqual(requested, {"x": 1, "y": 0.1, "z": 0.05, "neck_pitch": 0.27})

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
