"""Bounded, fail-closed motion for a supervised, level-floor simulator.

This is an application guard, not a collision or cliff safety system. The head's
single depth sensor cannot certify clearance behind the robot or around a turn.
"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from typing import Any


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("expected a finite number")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError("expected a finite number") from exc
    if not math.isfinite(value):
        raise ValueError("expected a finite number")
    return value


def _json_safe(value: Any) -> Any:
    """Keep malformed telemetry inspectable without writing invalid JSON logs."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _vector(value: Any, size: int) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"expected {size} coordinates")
    return [_number(v) for v in value]


def _unit(value: Any, size: int) -> list[float]:
    values = _vector(value, size)
    norm = math.sqrt(sum(v * v for v in values))
    if not 0.75 <= norm <= 1.25:
        raise ValueError("invalid unit vector or quaternion")
    return [v / norm for v in values]


def _rotate(q: list[float], v: list[float]) -> list[float]:
    w, x, y, z = q
    tx, ty, tz = 2 * (y * v[2] - z * v[1]), 2 * (z * v[0] - x * v[2]), 2 * (x * v[1] - y * v[0])
    return [
        v[0] + w * tx + y * tz - z * ty,
        v[1] + w * ty + z * tx - x * tz,
        v[2] + w * tz + x * ty - y * tx,
    ]


@dataclass(frozen=True)
class GuardConfig:
    pulse_period_s: float = 0.05
    state_max_age_s: float = 0.35
    depth_max_age_s: float = 0.35
    camera_max_age_s: float = 1.0
    health_max_age_s: float = 0.6
    health_poll_s: float = 0.2
    rpc_timeout_s: float = 0.3
    model_timeout_s: float = 2.0
    initial_sensor_timeout_s: float = 5.0
    sensor_skew_s: float = 0.15
    max_speed_m_s: float = 0.10
    max_move_duration_s: float = 2.0
    max_turn_deg: float = 30.0
    turn_rate_rad_s: float = 0.25
    turn_timeout_s: float = 4.0
    turn_no_progress_s: float = 0.5
    turn_tolerance_deg: float = 2.0
    look_min_hold_s: float = 0.4
    look_timeout_s: float = 2.0
    look_feedback_period_s: float = 0.25
    look_feedback_gain: float = 0.5
    look_max_correction_ratio: float = 0.35
    look_settle_s: float = 0.15
    look_direction_tolerance_rad: float = 0.10
    obstacle_distance_m: float = 0.35
    min_loop_hz: float = 40.0

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if _number(value) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.pulse_period_s > 0.1:
            raise ValueError("pulse period must stay below the daemon's deadman")
        if self.max_speed_m_s > 0.1 or self.max_move_duration_s > 2 or self.max_turn_deg > 30:
            raise ValueError("this prototype permits at most 0.1 m/s, 2 seconds, and 30 degrees")
        if self.turn_rate_rad_s > 0.3 or self.turn_timeout_s > 5:
            raise ValueError("turn rate/duration exceeds the prototype bounds")
        if self.look_feedback_gain > 0.5 or self.look_max_correction_ratio > 0.35:
            raise ValueError("gaze feedback exceeds prototype correction bounds")
        if self.look_settle_s >= self.look_timeout_s:
            raise ValueError("gaze settling must fit inside timeout")
        if not self.look_min_hold_s < self.look_timeout_s <= 2:
            raise ValueError("look hold must fit within a bounded two-second timeout")


class GuardedRobot:
    def __init__(self, transport: Any, config: GuardConfig | None = None):
        self.transport = transport
        self.config = config or GuardConfig()
        self.model: dict[str, Any] | None = None
        self._beams: list[list[float]] = []
        self._health: dict[str, Any] | None = None
        self._health_at = 0.0
        self._health_task: asyncio.Task | None = None
        self._active = False
        self._stopping = 0
        self._cancel: asyncio.Event | None = None
        self._closed = False

    async def _request(self, method: str, params: dict, timeout: float | None = None) -> dict:
        return await asyncio.wait_for(
            self.transport.request(method, params), timeout or self.config.rpc_timeout_s
        )

    async def initialize(self) -> dict:
        """Fetch geometry and start health checks; never enable the robot's motors."""
        if self._closed:
            raise RuntimeError("guard is closed")
        if self.model is None:
            model = await self._request("robot.model", {}, self.config.model_timeout_s)
            beams = [_unit(v, 3) for v in model["tof_beams"]]
            if len(beams) != 64 or not 0.02 < _number(model["trunk_height_m"]) < 1:
                raise ValueError("robot.model did not provide valid 8x8 depth geometry")
            self._beams = beams
            self.model = model
        await self._refresh_health()
        if self._health_task is None:
            self._health_task = asyncio.create_task(self._poll_health())
        deadline = time.monotonic() + self.config.initial_sensor_timeout_s
        while time.monotonic() < deadline:
            snapshot = self.transport.snapshot()
            if not snapshot.get("connected") or all(
                snapshot.get(key) for key in ("camera", "state", "depth")
            ):
                break
            await asyncio.sleep(0.025)
        return await self.observe()

    async def _refresh_health(self) -> None:
        try:
            self._health = await self._request("robot.health", {})
        except Exception as exc:  # noqa: BLE001 - any failed health query closes the motion gate
            self._health = {
                "healthy": False,
                "reason": f"health request failed: {type(exc).__name__}: {exc}",
            }
        self._health_at = time.monotonic()

    async def _poll_health(self) -> None:
        while True:
            await asyncio.sleep(self.config.health_poll_s)
            await self._refresh_health()

    def _base_guard(self, snapshot: dict) -> str | None:
        if self._closed:
            return "closed"
        if self.model is None:
            return "not_initialized"
        if snapshot.get("connected") is not True:
            return "disconnected"
        now = time.monotonic()
        try:
            for key, limit in (
                ("state", self.config.state_max_age_s),
                ("depth", self.config.depth_max_age_s),
                ("camera", self.config.camera_max_age_s),
            ):
                sample = snapshot.get(key)
                if not sample:
                    return f"missing_{key}"
                age = now - _number(sample["received_at"])
                if age < -0.05 or age > limit:
                    return f"stale_{key}"
            if now - self._health_at > self.config.health_max_age_s:
                return "stale_health"
            health = self._health
            if not health or health.get("healthy") is not True:
                return "unhealthy"
            loop_health = health["control_loop"]
            if (
                _number(loop_health["last_tick_age_ms"]) > 200
                or _number(loop_health["achieved_hz"]) < self.config.min_loop_hz
            ):
                return "unhealthy_loop"
            if _number(health["bus"]["consecutive_errors"]) != 0:
                return "bus_errors"
            if (
                health["imu"]["ready"] is not True
                or _number(health["imu"]["consecutive_stale_blocks"]) >= 25
            ):
                return "unhealthy_imu"
            state = snapshot["state"]["data"]
            if state["safety"]["fallen"] is not False:
                return "fallen"
            if state["safety"]["limp"] is not False or state["policy"] == "held":
                return "not_enabled"
            if state["policy"] not in ("stand", "walk"):
                return "other_policy_active"
            if _number(state["loop"]["hz"]) < self.config.min_loop_hz:
                return "slow_loop"
            gravity = _unit(state["safety"]["gravity"], 3)
            if gravity[2] > -0.7:
                return "unsafe_tilt"
            position = _vector(state["odom"]["position"], 3)
            if not 0.02 < position[2] < 1:
                return "invalid_pose"
            _number(state["odom"]["yaw"])
            state_ns = _number(state["t_ns"])
            depth_ns = _number(snapshot["depth"]["data"]["t_ns"])
            if (
                min(state_ns, depth_ns) <= 0
                or abs(state_ns - depth_ns) / 1e9 > self.config.sensor_skew_s
            ):
                return "sensor_time_skew"
        except (KeyError, TypeError, ValueError, IndexError):
            return "invalid_telemetry"
        return None

    def _depth_guard(self, snapshot: dict) -> tuple[str | None, dict]:
        """Match kinematics::tof floor projection using wire-provided geometry.

        Dotting into measured gravity gives the same vertical component as its
        level_from_gravity rotation, without choosing an arbitrary world yaw.
        """
        details: dict[str, Any] = {}
        try:
            state = snapshot["state"]["data"]
            depth = snapshot["depth"]["data"]
            pose = state["frames"]["tof"]
            pos, quat = _vector(pose["pos"], 3), _unit(pose["quat"], 4)
            gravity = _unit(state["safety"]["gravity"], 3)
            axis = _rotate(quat, [1.0, 0.0, 0.0])
            sensor_yaw = math.atan2(axis[1], axis[0])
            downward_axis = sum(a * g for a, g in zip(axis, gravity))
            head_forward = (
                axis[0] > 0
                and abs(sensor_yaw) <= math.radians(20)
                and abs(downward_axis) <= math.sin(math.radians(35))
            )
            distances, statuses = depth["distance_mm"], depth["status"]
            if (
                depth["rows"] != 8
                or depth["cols"] != 8
                or len(distances) != 64
                or len(statuses) != 64
            ):
                return "invalid_depth", details
            above_floor = state["odom"]["position"][2] - sum(p * g for p, g in zip(pos, gravity))
            if above_floor <= 0:
                return "invalid_pose", details
            known, floors, hits, too_close, central_unknown = 0, 0, [], False, False
            sectors = {
                name: {"known_zones": 0, "floor_zones": 0, "nearest_obstacle_m": None}
                for name in ("left", "center", "right")
            }
            for index, (mm, code, beam) in enumerate(zip(distances, statuses, self._beams)):
                code = _number(code)
                mm = _number(mm)
                usable = code == 255 or (code in (5, 9) and mm > 0)
                known += int(usable)
                direction = _rotate(quat, beam)
                bearing = math.atan2(direction[1], direction[0])
                sector_name = (
                    "left"
                    if bearing > math.radians(7.5)
                    else ("right" if bearing < -math.radians(7.5) else "center")
                )
                sector = sectors[sector_name]
                sector["known_zones"] += int(usable)
                if not usable and 2 <= index // 8 <= 5 and 2 <= index % 8 <= 5:
                    central_unknown = True
                if code not in (5, 9) or mm <= 0:
                    continue
                downward = sum(d * g for d, g in zip(direction, gravity))
                r = mm / 1000.0
                if downward > 0 and r * downward >= above_floor * 0.85:
                    floors += 1
                    sector["floor_zones"] += 1
                    continue
                horizontal = r * math.sqrt(max(0.0, 1 - downward * downward))
                nearest = sector["nearest_obstacle_m"]
                sector["nearest_obstacle_m"] = (
                    horizontal if nearest is None else min(nearest, horizontal)
                )
                if horizontal < 0.10:
                    # Kinematics marks this noise; a guard cannot treat it as free space.
                    too_close = True
                else:
                    hits.append(horizontal)
            details = {
                "known_zones": known,
                "floor_zones": floors,
                "nearest_obstacle_m": min(hits) if hits else None,
                "sectors": sectors,
                "sector_frame": "trunk_left_center_right; null means no returned obstacle, not certified clearance",
                "sensor_yaw_deg": math.degrees(sensor_yaw),
            }
            # Side scans can inform planning, but never certify forward movement.
            if not head_forward:
                return "head_not_forward", details
            if too_close:
                return "depth_too_close", details
            if hits and min(hits) <= self.config.obstacle_distance_m:
                return "obstacle", details
            if known < 48 or central_unknown:
                return "depth_quality", details
        except (KeyError, TypeError, ValueError, IndexError):
            return "invalid_depth", details
        return None, details

    def _guard(self, snapshot: dict) -> tuple[str | None, dict]:
        reason = self._base_guard(snapshot)
        return (reason, {}) if reason else self._depth_guard(snapshot)

    async def observe(self) -> dict:
        snapshot = self.transport.snapshot()
        reason, depth_summary = self._guard(snapshot)
        now = time.monotonic()
        result = {
            "connected": bool(snapshot.get("connected")),
            "ready": reason is None,
            "guard_reason": reason,
            "depth_summary": depth_summary,
            "health": self._health,
            "transport_errors": snapshot.get("errors", {}),
        }
        for key in ("camera", "state", "depth"):
            sample = snapshot.get(key)
            if not sample:
                result[key] = None
                continue
            result[key] = {k: v for k, v in sample.items() if k not in ("image", "received_at")}
            try:
                result[key]["age_s"] = max(0, now - _number(sample["received_at"]))
            except (KeyError, TypeError, ValueError):
                result[key]["age_s"] = None
        return _json_safe(result)

    def _begin(self) -> asyncio.Event:
        if self._closed:
            raise RuntimeError("guard is closed")
        if self._active or self._stopping:
            raise RuntimeError("another action or stop is still running")
        self._active = True
        self._cancel = asyncio.Event()
        return self._cancel

    async def _send_stop(self) -> dict:
        # Zero on the same channel as motion first. The acknowledged request is
        # also needed: merely ceasing pulses leaves a full deadman interval.
        error = None
        try:
            self.transport.notify("robot.move", {"vx": 0.0, "vy": 0.0, "vyaw": 0.0})
        except Exception as exc:  # noqa: BLE001 - still try reliable stop after any notify failure
            error = f"{type(exc).__name__}: {exc}"
        try:
            reply = await self._request("robot.stop", {})
            if reply.get("accepted") is True:
                # robot.stop zeros the requested twist. Daemon smoothing and
                # gait dynamics continue briefly after its acknowledgement.
                return {"acknowledged": True, "physical_settling_verified": False}
            error = reply.get("reason", "stop refused")
        except Exception as exc:  # noqa: BLE001 - record stop failure without suppressing action cancellation
            error = f"{type(exc).__name__}: {exc}"
        return {"acknowledged": False, "physical_settling_verified": False, "error": error}

    @staticmethod
    def _odom(snapshot: dict) -> dict | None:
        return _json_safe((snapshot.get("state") or {}).get("data", {}).get("odom"))

    async def stop(self) -> dict:
        self._stopping += 1
        if self._cancel is not None:
            self._cancel.set()
        try:
            stopped = await self._send_stop()
            return {
                "action": "stop",
                "completed": stopped["acknowledged"],
                "reason": "stopped" if stopped["acknowledged"] else "stop_unacknowledged",
                "stop": stopped,
            }
        finally:
            self._stopping -= 1

    async def _motion(
        self,
        action: str,
        vx: float,
        yaw_rate: float,
        duration: float,
        target_angle: float | None = None,
    ) -> dict:
        cancel = self._begin()
        started = time.monotonic()
        before = None
        reason, details, completed = "duration_elapsed", {}, False
        travelled, previous_yaw = 0.0, None
        last_progress, progress_angle = started, 0.0
        command_budget = abs(target_angle / yaw_rate) if target_angle is not None else duration
        try:
            before = self._odom(self.transport.snapshot())
            while True:
                if cancel.is_set():
                    reason = "stopped"
                    break
                snapshot = self.transport.snapshot()
                reason, details = self._guard(snapshot)
                if reason:
                    break
                elapsed = time.monotonic() - started
                if target_angle is not None:
                    yaw = _number(snapshot["state"]["data"]["odom"]["yaw"])
                    if previous_yaw is not None:
                        travelled += math.atan2(
                            math.sin(yaw - previous_yaw), math.cos(yaw - previous_yaw)
                        )
                    previous_yaw = yaw
                    progress = travelled * math.copysign(1, target_angle)
                    if progress - progress_angle >= math.radians(0.5):
                        last_progress, progress_angle = time.monotonic(), progress
                    tolerance = min(
                        math.radians(self.config.turn_tolerance_deg), abs(target_angle) / 4
                    )
                    if progress >= abs(target_angle) - tolerance:
                        reason, completed = "angle_reached", True
                        break
                    if time.monotonic() - last_progress >= self.config.turn_no_progress_s:
                        reason = "turn_no_progress"
                        break
                if elapsed >= duration:
                    reason = "turn_timeout" if target_angle is not None else "duration_elapsed"
                    completed = target_angle is None
                    break
                # Fresh timestamps can conceal a frozen odometry estimate.
                # Never request more open-loop rotation than the requested
                # angle; after its budget, watch the gait settle with zero yaw.
                commanded_yaw = yaw_rate if elapsed < command_budget else 0.0
                self.transport.notify("robot.move", {"vx": vx, "vy": 0.0, "vyaw": commanded_yaw})
                wait = min(self.config.pulse_period_s, duration - elapsed)
                if elapsed < command_budget:
                    wait = min(wait, command_budget - elapsed)
                try:
                    await asyncio.wait_for(cancel.wait(), wait)
                except TimeoutError:
                    pass
        except asyncio.CancelledError:
            cancel.set()
            raise
        except Exception as exc:  # noqa: BLE001 - all execution failures must end in a stop attempt
            reason, details = "transport_error", {"error": f"{type(exc).__name__}: {exc}"}
        finally:
            cancel.set()
            try:
                stopped = await self._send_stop()
            finally:
                self._active = False
        if not stopped["acknowledged"] and completed:
            completed, reason = False, "stop_unacknowledged"
        return {
            "action": action,
            "completed": completed,
            "reason": reason,
            "elapsed_s": time.monotonic() - started,
            "before_odom": before,
            "after_odom": self._odom(self.transport.snapshot()),
            "depth_summary": details,
            "turned_deg": math.degrees(travelled) if target_angle is not None else None,
            "stop": stopped,
        }

    async def move_for(self, speed_m_s: float, duration_s: float) -> dict:
        speed, duration = _number(speed_m_s), _number(duration_s)
        if not 0 < speed <= self.config.max_speed_m_s:
            raise ValueError(f"forward speed must be in (0, {self.config.max_speed_m_s}] m/s")
        if not 0 < duration <= self.config.max_move_duration_s:
            raise ValueError(f"duration must be in (0, {self.config.max_move_duration_s}] seconds")
        return await self._motion("move_for", speed, 0.0, duration)

    async def turn_by(self, angle_deg: float) -> dict:
        angle = _number(angle_deg)
        if not 0 < abs(angle) <= self.config.max_turn_deg:
            raise ValueError(
                f"turn angle magnitude must be in (0, {self.config.max_turn_deg}] degrees"
            )
        return await self._motion(
            "turn_by",
            0.0,
            math.copysign(self.config.turn_rate_rad_s, angle),
            self.config.turn_timeout_s,
            math.radians(angle),
        )

    async def look_at(self, x: float, y: float, z: float) -> dict:
        point = [_number(v) for v in (x, y, z)]
        if point[0] <= 0 or math.sqrt(sum(v * v for v in point)) > 3:
            raise ValueError("look target must be ahead of the trunk and within 3 metres")
        cancel = self._begin()
        started = time.monotonic()
        before = None
        reason, completed, reply, camera_sequence = "look_timeout", False, None, None
        aim_error, corrections = None, 0
        try:
            before = self._odom(self.transport.snapshot())
            reason = self._base_guard(self.transport.snapshot())
            if reason:
                return {"action": "look_at", "completed": False, "reason": reason}
            stopped = await self._send_stop()
            if cancel.is_set() or not stopped["acknowledged"]:
                return {
                    "action": "look_at",
                    "completed": False,
                    "reason": "stopped" if cancel.is_set() else "stop_unacknowledged",
                    "stop": stopped,
                }
            snapshot = self.transport.snapshot()
            reason = self._base_guard(snapshot)
            if reason:
                return {"action": "look_at", "completed": False, "reason": reason}
            neck_index = self.model["joint_names"].index("neck_pitch")
            params = dict(zip(("x", "y", "z"), point))
            # Preserve the commanded posture. Reusing measured neck pitch would
            # accumulate the gait's tracking bias on every look and lower the neck.
            home = _vector(self.model["joint_home"], len(self.model["joint_names"]))
            command = _vector(snapshot["state"]["data"]["head"], 4)
            params["neck_pitch"] = home[neck_index] + command[0]
            reply = await self._request("robot.look", params)
            requested_at, settled_at = time.monotonic(), None
            last_adjusted = requested_at
            feedback_error = None
            virtual_point = point.copy()
            head = {
                key: _number(reply["head"][key])
                for key in ("neck_pitch", "head_pitch", "head_yaw", "head_roll")
            }
            for key in head:
                _number(reply["joint_targets"][key])
            while True:
                if cancel.is_set():
                    reason = "stopped"
                    break
                snapshot = self.transport.snapshot()
                reason = self._base_guard(snapshot)
                if reason:
                    break
                now = time.monotonic()
                state = snapshot["state"]["data"]
                # FK is based on measured joints. Stable optical alignment is
                # authoritative even when the gait has a joint tracking bias.
                camera = state["frames"]["camera"]
                optical = _rotate(_unit(camera["quat"], 4), [0.0, 0.0, 1.0])
                origin = _vector(camera["pos"], 3)
                delta = [target - start for target, start in zip(point, origin)]
                distance = math.sqrt(sum(v * v for v in delta))
                if distance <= 1e-6:
                    raise ValueError("target coincides with camera")
                desired = [v / distance for v in delta]
                cosine = sum(a * b for a, b in zip(optical, desired))
                aim_error = math.acos(max(-1.0, min(1.0, cosine)))
                if feedback_error is None:
                    feedback_error = aim_error
                aligned = aim_error <= self.config.look_direction_tolerance_rad
                if aligned:
                    settled_at = settled_at if settled_at is not None else now
                else:
                    settled_at = None
                if (
                    settled_at is not None
                    and now - settled_at >= self.config.look_settle_s
                    and now - requested_at >= self.config.look_min_hold_s
                    and snapshot["camera"]["received_at"] > settled_at
                ):
                    completed = reply.get("clamped") is False
                    reason = "gaze_settled" if completed else "look_clamped"
                    camera_sequence = snapshot["camera"]["sequence"]
                    break
                if now - requested_at >= self.config.look_timeout_s:
                    reason = "look_timeout"
                    break
                if not aligned and now - last_adjusted >= self.config.look_feedback_period_s:
                    # A wide sweep can still be approaching its requested target
                    # at the first feedback tick. Correct only the remaining bias
                    # once improvement slows, rather than winding up against that
                    # normal transient. The original timeout still bounds waiting.
                    approaching = (
                        feedback_error - aim_error > self.config.look_direction_tolerance_rad * 0.5
                    )
                    feedback_error, last_adjusted = aim_error, now
                    if not approaching:
                        # Move a virtual IK target opposite the measured optical
                        # error. IK and travel limits stay daemon-owned; only the
                        # original target can establish success. Preserve posture.
                        proposed = [
                            v + self.config.look_feedback_gain * distance * (d - o)
                            for v, d, o in zip(virtual_point, desired, optical)
                        ]
                        correction = math.dist(proposed, point)
                        limit = self.config.look_max_correction_ratio * math.sqrt(
                            sum(v * v for v in point)
                        )
                        if correction > limit:
                            reason = "look_correction_limit"
                            break
                        virtual_point = proposed
                        params.update(zip(("x", "y", "z"), virtual_point))
                        reply = await self._request("robot.look", params)
                        head = {key: _number(reply["head"][key]) for key in head}
                        for key in head:
                            _number(reply["joint_targets"][key])
                        if reply.get("clamped") is not False:
                            reason = "look_clamped"
                            break
                        corrections += 1
                        last_adjusted, settled_at = time.monotonic(), None
                self.transport.notify("robot.head", head)
                try:
                    await asyncio.wait_for(cancel.wait(), self.config.pulse_period_s)
                except TimeoutError:
                    pass
        except asyncio.CancelledError:
            cancel.set()
            raise
        except (KeyError, TypeError, ValueError, IndexError):
            reason = "invalid_telemetry"
        except Exception as exc:  # noqa: BLE001 - head/control transport failures still require stop
            reason, reply = "transport_error", {"error": f"{type(exc).__name__}: {exc}"}
        finally:
            cancel.set()
            try:
                stopped = await self._send_stop()
            finally:
                self._active = False
        if completed and not stopped["acknowledged"]:
            completed, reason = False, "stop_unacknowledged"
        return {
            "action": "look_at",
            "completed": completed,
            "reason": reason,
            "elapsed_s": time.monotonic() - started,
            "before_odom": before,
            "after_odom": self._odom(self.transport.snapshot()),
            "result": reply,
            "camera_sequence": camera_sequence,
            "aim_error_rad": aim_error,
            "corrections": corrections,
            "stop": stopped,
        }

    async def close(self) -> None:
        self._closed = True
        await self.stop()
        if self._health_task is not None:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
            self._health_task = None
