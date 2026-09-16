"""Close side returns must survive a change in gaze; no robot is contacted."""

import copy
import math
import time
import unittest

from test_core import FakeRobot

from duck_nav.core import GuardConfig
from duck_nav.navigation import GaitNavigator


class SideHazardTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.transport = FakeRobot()
        self.robot = GaitNavigator(
            self.transport,
            GuardConfig(pulse_period_s=0.01, health_poll_s=0.02, rpc_timeout_s=0.1),
        )
        await self.robot.initialize()

    async def asyncTearDown(self):
        await self.robot.close()

    def scan(self, yaw, *, pitch=0, distance=2000, statuses=None):
        y, p = math.radians(yaw) / 2, math.radians(pitch) / 2
        self.transport.state["frames"]["tof"]["quat"] = [
            math.cos(y) * math.cos(p),
            -math.sin(y) * math.sin(p),
            math.cos(y) * math.sin(p),
            math.sin(y) * math.cos(p),
        ]
        self.transport.depth["distance_mm"] = [distance] * 64
        self.transport.depth["status"] = [5] * 64 if statuses is None else statuses

    async def latch(self, yaw=-45, distance=130):
        self.scan(yaw)
        self.transport.depth["distance_mm"][27] = distance
        observation = await self.robot.observe()
        self.assertEqual(len(observation["depth_summary"]["retained_side_hazards"]), 1)
        return observation

    async def assert_front_blocked(self, reason="obstacle"):
        self.scan(0)
        observation = await self.robot.observe()
        self.assertFalse(observation["ready"])
        self.assertEqual(observation["guard_reason"], reason)
        return observation

    async def test_right_hazard_then_clear_front_refuses_all_body_commands(self):
        # Reproduce the observed scan -> recenter -> curved-advance failure.
        self.scan(-45, distance=130)
        side = await self.robot.observe()
        self.assertEqual(side["guard_reason"], "head_not_forward")
        self.scan(0, distance=1092)
        front = await self.robot.observe()
        self.assertGreater(front["depth_summary"]["nearest_obstacle_m"], 1)
        self.assertEqual(front["guard_reason"], "obstacle")
        retained = front["depth_summary"]["retained_side_hazards"]
        self.assertEqual(retained[0]["direction"], "right")
        self.assertLess(retained[0]["nearest_obstacle_m"], 0.14)
        for operation in (
            self.robot.advance(0.2, -10),
            self.robot.move_for(0.1, 0.02),
            self.robot.turn_by(10),
        ):
            result = await operation
            self.assertFalse(result["completed"])
            self.assertEqual(result["reason"], "obstacle")
            self.assertTrue(result["stop"]["acknowledged"])
        self.assertEqual(self.transport.moving_pulses(), [])

    async def test_rescan_can_clear_with_actual_ranges_only_in_hazard_neighborhood(self):
        side = await self.latch()
        zones = side["depth_summary"]["retained_side_hazards"][0]["rescan_zones"]
        self.assertEqual(len(zones), 9)
        statuses = [255] * 64
        for index in zones:
            statuses[index] = 5
        self.scan(-45, statuses=statuses)
        clear = await self.robot.observe()
        self.assertEqual(clear["guard_reason"], "head_not_forward")
        self.assertEqual(clear["depth_summary"]["retained_side_hazards"], [])
        self.scan(0)
        self.assertTrue((await self.robot.move_for(0.05, 0.02))["completed"])

    async def test_unknown_or_poor_quality_rescan_cannot_clear(self):
        side = await self.latch()
        zones = side["depth_summary"]["retained_side_hazards"][0]["rescan_zones"]
        for missing_code in (255, 4):
            with self.subTest(missing_code=missing_code):
                self.scan(-45)
                self.transport.depth["status"][zones[0]] = missing_code
                await self.robot.observe()
                await self.assert_front_blocked()
        self.scan(-45)
        # Missing central data outside the original neighborhood still fails
        # the normal scan-quality gate, even with the close region now ranged.
        self.transport.depth["status"][45] = 4
        await self.robot.observe()
        await self.assert_front_blocked()

    async def test_unhealthy_stale_and_replayed_clear_scans_cannot_clear(self):
        await self.latch()
        self.scan(-45)
        snapshot = self.transport.snapshot()
        self.robot._health["healthy"] = False
        self.assertEqual(self.robot._guard(snapshot)[0], "unhealthy")
        self.robot._health["healthy"] = True
        stale = copy.deepcopy(snapshot)
        stale["depth"]["received_at"] = time.monotonic() - 1
        self.assertEqual(self.robot._guard(stale)[0], "stale_depth")
        replayed = copy.deepcopy(snapshot)
        hazard = self.robot._side_hazards[0]
        replayed["state"]["data"]["t_ns"] = hazard.depth_ns
        replayed["depth"]["data"]["t_ns"] = hazard.depth_ns
        self.robot._guard(replayed)
        await self.assert_front_blocked()

    async def test_elapsed_time_and_wrong_yaw_or_pitch_cannot_clear(self):
        await self.latch()
        # Age the stored observation; a timer is never clearance evidence.
        self.robot._side_hazards[0].received_at -= 3600
        for yaw, pitch in ((0, 0), (45, 0), (-45, -20)):
            with self.subTest(yaw=yaw, pitch=pitch):
                self.scan(yaw, pitch=pitch)
                await self.robot.observe()
                await self.assert_front_blocked()

    async def test_merging_cannot_walk_original_body_or_gaze_anchor(self):
        first = await self.latch()
        anchor = copy.deepcopy(first["depth_summary"]["retained_side_hazards"][0])
        for x, angle in ((0.01, -46), (0.02, -47)):
            self.transport.state["odom"]["position"][0] = x
            self.scan(angle)
            self.transport.depth["distance_mm"][28] = 120
            observation = await self.robot.observe()
            held = observation["depth_summary"]["retained_side_hazards"]
            self.assertEqual(len(held), 1)
            self.assertEqual(held[0]["body_pose"], anchor["body_pose"])
            self.assertEqual(held[0]["sensor_yaw_deg"], anchor["sensor_yaw_deg"])
            self.assertTrue(set(anchor["rescan_zones"]) <= set(held[0]["rescan_zones"]))
        self.transport.state["odom"]["position"][0] = 0.03
        moved = await self.assert_front_blocked("retained_hazard_pose_changed")
        self.assertIn("operator intervention", moved["depth_summary"]["retained_hazard_guidance"])
        # A matching gaze at the displaced pose still cannot erase the anchor.
        self.scan(-45)
        self.assertEqual(
            (await self.robot.observe())["guard_reason"], "retained_hazard_pose_changed"
        )

    async def test_body_yaw_change_cannot_clear_and_wrap_is_handled(self):
        self.transport.state["odom"]["yaw"] = math.radians(179)
        await self.latch()
        self.transport.state["odom"]["yaw"] = math.radians(-179)
        await self.assert_front_blocked()
        self.transport.state["odom"]["yaw"] = math.radians(-170)
        await self.assert_front_blocked("retained_hazard_pose_changed")

    async def test_translated_or_tilted_positive_rescan_cannot_clear_old_beams(self):
        await self.latch()
        for changed_field in ("body_origin", "tof_origin", "body_tilt", "body_yaw"):
            with self.subTest(changed_field=changed_field):
                original = copy.deepcopy(self.transport.state)
                if changed_field == "body_origin":
                    self.transport.state["odom"]["position"][1] = 0.02
                elif changed_field == "tof_origin":
                    self.transport.state["frames"]["tof"]["pos"][1] = 0.02
                elif changed_field == "body_tilt":
                    tilt = math.radians(4)
                    self.transport.state["safety"]["gravity"] = [0, math.sin(tilt), -math.cos(tilt)]
                else:
                    self.transport.state["odom"]["yaw"] = math.radians(4)
                self.scan(-45)
                observation = await self.robot.observe()
                held = observation["depth_summary"]["retained_side_hazards"]
                self.assertEqual(len(held), 1)
                self.assertTrue(held[0]["rescan_association_uncertain"])
                self.transport.state = original
                await self.assert_front_blocked()

    async def test_body_rotation_sensor_lever_arm_consumes_clearance_budget(self):
        await self.latch(distance=50)
        # Two degrees is below the angle-only bound, but also displaces the
        # sensor origin enough to make association with a 5 cm return unsafe.
        self.transport.state["odom"]["yaw"] = math.radians(2)
        self.scan(-45)
        observation = await self.robot.observe()
        self.assertEqual(observation["guard_reason"], "depth_too_close")
        self.assertTrue(observation["depth_summary"]["retained_side_hazards"])

    async def test_merged_pose_uncertainty_cannot_be_erased_by_returning_to_anchor(self):
        await self.latch()
        self.transport.state["odom"]["position"][1] = 0.02
        self.scan(-45)
        self.transport.depth["distance_mm"][27] = 120
        await self.robot.observe()
        self.transport.state["odom"]["position"][1] = 0
        self.scan(-45)
        observation = await self.robot.observe()
        held = observation["depth_summary"]["retained_side_hazards"]
        self.assertEqual(len(held), 1)
        self.assertTrue(held[0]["rescan_association_uncertain"])
        await self.assert_front_blocked()

    async def test_tiny_near_return_cannot_be_cleared_by_millimetre_origin_shift(self):
        await self.latch(distance=1)
        self.transport.state["odom"]["position"][1] = 0.001
        self.scan(-45)
        self.assertEqual((await self.robot.observe())["guard_reason"], "depth_too_close")

    async def test_hazard_store_has_bounded_explicit_overflow(self):
        for angle in (-30, -45, -60, 45):
            self.scan(angle)
            self.transport.depth["distance_mm"][27] = 130
            await self.robot.observe()
        observation = await self.assert_front_blocked("retained_hazard_capacity")
        summary = observation["depth_summary"]
        self.assertEqual(len(summary["retained_side_hazards"]), 3)
        self.assertTrue(summary["retained_hazard_capacity_exceeded"])
        self.scan(-30)
        await self.robot.observe()
        await self.assert_front_blocked("retained_hazard_capacity")

    async def test_side_near_contact_is_never_downgraded_by_recentring(self):
        observation = await self.latch(distance=50)
        self.assertEqual(observation["guard_reason"], "depth_too_close")
        await self.assert_front_blocked("depth_too_close")

    async def test_ordinary_retained_hazard_still_allows_head_recovery(self):
        await self.latch()
        self.scan(0)
        result = await self.robot.look_at(1, 0, 0)
        self.assertTrue(result["completed"])
        self.assertTrue(self.transport.head_pulses)
        self.assertEqual(self.transport.moving_pulses(), [])
        await self.assert_front_blocked()


if __name__ == "__main__":
    unittest.main()
