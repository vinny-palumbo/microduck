"""Short odometry-controlled walking arcs followed by measured settling.

The deployed gait steers while walking much better than it turns in place.
These commands intentionally describe arcs, not pivots or exact pose control.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections import deque
from dataclasses import dataclass

from .core import GuardedRobot, _number


def angle_delta(current, start):
    return math.atan2(math.sin(current - start), math.cos(current - start))


@dataclass(frozen=True)
class GaitConfig:
    command_speed_m_s: float = 0.3
    max_yaw_rate_rad_s: float = 0.5
    max_distance_m: float = 0.2
    max_heading_deg: float = 30
    drive_timeout_s: float = 3
    settle_timeout_s: float = 3
    settle_window_s: float = 0.4
    settle_min_s: float = 1.0
    stopping_lead_m: float = 0.02

    def __post_init__(self):
        for name, value in vars(self).items():
            if _number(value) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.command_speed_m_s > 0.4 or self.max_yaw_rate_rad_s > 0.5:
            raise ValueError("navigation commands exceed the measured gait envelope")
        if self.max_distance_m > 0.2 or self.max_heading_deg > 30:
            raise ValueError("navigation actions require at most 0.2 m and 30 degrees")
        if self.drive_timeout_s > 3 or self.settle_timeout_s > 5:
            raise ValueError("navigation actions must remain bounded")
        if self.settle_window_s >= self.settle_timeout_s:
            raise ValueError("settling window must fit inside the timeout")


class GaitNavigator(GuardedRobot):
    def __init__(self, transport, config=None, gait=None):
        super().__init__(transport, config)
        self.gait = gait or GaitConfig()
        self._course_target = None
        self._course_anchor = None
        self._course_reset_reason = "not_initialized"

    def _clear_course(self, reason):
        self._course_target = None
        self._course_anchor = None
        self._course_reset_reason = reason

    def _check_course_anchor(self, odom):
        if self._course_anchor is None:
            return
        if math.dist(odom["position"], self._course_anchor["position"]) > 0.025 or abs(
            angle_delta(odom["yaw"], self._course_anchor["yaw"])
        ) > math.radians(5):
            self._clear_course("external_pose_change")

    @staticmethod
    def _course_odom(snapshot):
        odom = GaitNavigator._odom(snapshot)
        if len(odom["position"]) != 3:
            raise ValueError("expected three odometry coordinates")
        return {
            "position": [_number(value) for value in odom["position"]],
            "yaw": _number(odom["yaw"]),
        }

    def course(self):
        """Describe retained heading; reset_reason explains an inactive course only."""
        error = None
        unavailable_reason = "telemetry_unavailable"
        try:
            snapshot = self.transport.snapshot()
            if snapshot.get("connected") is not True:
                unavailable_reason = "disconnected"
                raise ValueError("course telemetry is disconnected")
            odom = self._course_odom(snapshot)
            if not self._active:
                self._check_course_anchor(odom)
            if self._course_target is not None:
                error = math.degrees(angle_delta(self._course_target, odom["yaw"]))
        except Exception:  # noqa: BLE001 - an observation may race a connection/sensor transition
            # An active action owns its guard/cancellation lifecycle. Observing a
            # transient failure must not silently remove the target it is tracking.
            if not self._active:
                self._clear_course(unavailable_reason)
        return {
            "active": self._course_target is not None,
            "target_yaw_deg": (
                math.degrees(self._course_target) if self._course_target is not None else None
            ),
            "error_deg": error,
            "reset_reason": self._course_reset_reason,
        }

    async def observe(self):
        observation = await super().observe()
        observation["course"] = self.course()
        return observation

    async def initialize(self):
        self._clear_course("initialized")
        return await super().initialize()

    async def stop(self):
        self._clear_course("public_stop")
        return await super().stop()

    async def look_at(self, x, y, z):
        try:
            result = await super().look_at(x, y, z)
        except asyncio.CancelledError:
            self._clear_course("cancelled")
            raise
        if not result["completed"]:
            self._clear_course(result["reason"])
        return result

    async def _settle(self):
        """Verify continuous pose stability using new sensor samples, not wall time alone."""
        started = time.monotonic()
        samples = deque()
        last_stamp = None
        while time.monotonic() - started < self.gait.settle_timeout_s:
            snapshot = self.transport.snapshot()
            reason, _ = self._guard(snapshot)
            if reason:
                return False, reason
            state = snapshot["state"]["data"]
            stamp = state.get("t_ns", state.get("t"))
            now = time.monotonic()
            motion = state.get("move", {})
            requested, applied = motion.get("requested", []), motion.get("applied", [])
            zero = (
                len(requested) == len(applied) == 3
                and all(_number(v) == 0 for v in requested)
                and all(abs(_number(v)) < 0.001 for v in applied)
            )
            if not zero:
                samples.clear()
            elif stamp != last_stamp:
                odom = state["odom"]
                samples.append((now, odom["position"][:2], odom["yaw"]))
                while len(samples) > 2 and samples[1][0] < now - self.gait.settle_window_s:
                    samples.popleft()
                stable = (
                    len(samples) >= 3
                    and samples[-1][0] - samples[0][0] >= self.gait.settle_window_s
                    and all(math.dist(p, samples[0][1]) <= 0.004 for _, p, _ in samples)
                    and all(
                        abs(angle_delta(y, samples[0][2])) <= math.radians(1.5)
                        for _, _, y in samples
                    )
                )
                if stable and now - started >= self.gait.settle_min_s:
                    return True, "settled"
            last_stamp = stamp
            await asyncio.sleep(self.config.pulse_period_s)
        return False, "settle_timeout"

    async def advance(self, distance_m: float, heading_deg: float = 0):
        """Walk a short arc, stop, and return measured displacement and heading.

        Zero continues the last requested heading; a nonzero heading establishes
        a new target relative to measured current yaw. A positive heading curves left.
        Clearance must include the whole arc;
        the forward depth sensor does not certify side or rear clearance.
        """
        distance, heading = _number(distance_m), _number(heading_deg)
        if not 0.05 <= distance <= self.gait.max_distance_m:
            raise ValueError(f"distance must be 0.05–{self.gait.max_distance_m} metres")
        if abs(heading) > self.gait.max_heading_deg:
            raise ValueError(f"heading must be within +/-{self.gait.max_heading_deg} degrees")
        cancel = self._begin()
        started = time.monotonic()
        before = None
        effective_heading = target_yaw = None
        course_reset_reason = None
        reason, settled, motion_stopped = "drive_timeout", False, False
        stop = {"acknowledged": False, "physical_settling_verified": False}
        try:
            snapshot = self.transport.snapshot()
            before = self._course_odom(snapshot)
            self._check_course_anchor(before)
            course_reset_reason = self._course_reset_reason
            initial, _ = self._guard(snapshot)
            if initial:
                reason = initial
            else:
                target_yaw = (
                    angle_delta(before["yaw"] + math.radians(heading), 0)
                    if heading != 0 or self._course_target is None
                    else self._course_target
                )
                desired = angle_delta(target_yaw, before["yaw"])
                effective_heading = math.degrees(desired)
                correction_allowed = abs(desired) <= math.radians(self.gait.max_heading_deg) + 1e-12
                if not correction_allowed:
                    reason = "course_correction_limit"
                    self._clear_course(reason)
                else:
                    self._course_target = target_yaw
                    self._course_reset_reason = None
                while correction_allowed and time.monotonic() - started < self.gait.drive_timeout_s:
                    if cancel.is_set():
                        reason = "cancelled"
                        break
                    snapshot = self.transport.snapshot()
                    reason_now, _ = self._guard(snapshot)
                    if reason_now:
                        reason = reason_now
                        break
                    odom = snapshot["state"]["data"]["odom"]
                    travelled = math.dist(odom["position"][:2], before["position"][:2])
                    turned = angle_delta(odom["yaw"], before["yaw"])
                    if travelled >= max(0.03, distance - self.gait.stopping_lead_m):
                        reason = "distance_budget"
                        break
                    if abs(turned) > math.radians(40):
                        reason = "heading_excursion"
                        break
                    # Gait response is nonlinear. Close the loop on measured heading;
                    # never integrate a command as if it were achieved robot motion.
                    error = angle_delta(desired, turned)
                    yaw = max(
                        -self.gait.max_yaw_rate_rad_s,
                        min(self.gait.max_yaw_rate_rad_s, 2.0 * error),
                    )
                    self.transport.notify(
                        "robot.move", {"vx": self.gait.command_speed_m_s, "vy": 0, "vyaw": yaw}
                    )
                    try:
                        await asyncio.wait_for(cancel.wait(), self.config.pulse_period_s)
                    except TimeoutError:
                        pass
            stop = await self._send_stop()
            motion_stopped = True
            if stop["acknowledged"]:
                settled, settle_reason = await self._settle()
                if not settled:
                    reason = settle_reason
            else:
                reason = "stop_unacknowledged"
        except asyncio.CancelledError:
            cancel.set()
            self._clear_course("cancelled")
            raise
        except Exception as error:  # noqa: BLE001 - every navigation failure must stop the robot
            reason = f"navigation_error:{type(error).__name__}"
        finally:
            try:
                if not motion_stopped:
                    stop = await self._send_stop()
            finally:
                self._active = False
        try:
            after = self._course_odom(self.transport.snapshot())
        except Exception:  # noqa: BLE001 - malformed final telemetry cannot prove completion
            after, settled, reason = None, False, "telemetry_unavailable"
        moved, rotated = 0.0, 0.0
        if before and after:
            moved = math.dist(before["position"][:2], after["position"][:2])
            rotated = math.degrees(angle_delta(after["yaw"], before["yaw"]))
        if cancel.is_set():
            reason = "cancelled"
        elif settled and reason in {"distance_budget", "drive_timeout"} and moved < 0.01:
            reason = "no_progress"
        completed = settled and reason in {"distance_budget", "drive_timeout"} and moved >= 0.01
        if completed:
            self._course_anchor = after
        else:
            self._clear_course(reason)
            course_reset_reason = reason
        stop["physical_settling_verified"] = settled
        return {
            "action": "advance",
            "completed": completed,
            "reason": reason,
            "requested_distance_m": distance,
            "requested_heading_deg": heading,
            "effective_heading_deg": effective_heading,
            "target_yaw_deg": math.degrees(target_yaw) if target_yaw is not None else None,
            "course_reset_reason": course_reset_reason,
            "distance_m": moved,
            "heading_deg": rotated,
            "target_reached": (
                completed
                and abs(moved - distance) <= 0.03
                and abs(angle_delta(after["yaw"], target_yaw)) <= math.radians(10)
            ),
            "before_odom": before,
            "after_odom": after,
            "elapsed_s": time.monotonic() - started,
            "stop": stop,
            "source": "robot_odometry_not_simulator_ground_truth",
            "course": self.course(),
        }
