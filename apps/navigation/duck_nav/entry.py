"""Observed-doorway entry evidence from settled body odometry, never room identity.

The reference's initial half-plane is outside. This does not establish which
room is on either side, certify the projected contacts, or replace live guards.
Unknown tracking is latched; only a new DoorwayEntry can establish a new frame.
"""

from __future__ import annotations

import math

from .core import _number, _vector

REQUIRED_INSIDE_M = 0.25
JAMB_MARGIN_M = 0.15
MAX_AGE_S = 240.0
STATE_MAX_AGE_S = 0.35  # Matches the default guard; callers still apply their current guard.
MAX_IDLE_TRANSLATION_M = 0.025
MAX_IDLE_YAW_RAD = math.radians(5)
MAX_MOTION_M = 0.30
MAX_MOTION_YAW_RAD = math.radians(50)
EPS = 1e-9


def _angle_difference(left, right):
    return math.atan2(math.sin(left - right), math.cos(left - right))


def _pose(snapshot, now, *, fresh):
    if not isinstance(snapshot, dict) or snapshot.get("connected") is not True:
        raise ValueError("entry_disconnected")
    try:
        state = snapshot["state"]
        received = _number(state["received_at"])
        odom = state["data"]["odom"]
        position = tuple(_vector(odom["position"], 3))
        yaw = _number(odom["yaw"])
    except (KeyError, TypeError, ValueError, IndexError):
        raise ValueError("entry_pose_unknown") from None
    if received < 0 or received > now + EPS:
        raise ValueError("entry_state_time_invalid")
    if fresh and now - received > STATE_MAX_AGE_S + EPS:
        raise ValueError("entry_state_stale")
    return position, math.atan2(math.sin(yaw), math.cos(yaw)), received


class DoorwayEntry:
    """Track one observed plane across explicitly reported settled body actions."""

    def __init__(self, reference, source_view_id, snapshot, *, now):
        try:
            now = _number(now)
            if now < 0:
                raise ValueError("entry_time_invalid")
            if (
                not isinstance(source_view_id, str)
                or not source_view_id.strip()
                or len(source_view_id) > 200
            ):
                raise ValueError("entry_source_view_invalid")
            self.gap_center = tuple(_vector(reference["gap_center"], 2))
            self.gap_normal = tuple(_vector(reference["gap_normal"], 2))
            self.gap_tangent = tuple(_vector(reference["gap_tangent"], 2))
            self.gap_width_m = _number(reference["gap_width_m"])
            if (
                self.gap_width_m <= 2 * JAMB_MARGIN_M
                or abs(math.hypot(*self.gap_normal) - 1) > 1e-6
                or abs(math.hypot(*self.gap_tangent) - 1) > 1e-6
                or abs(sum(a * b for a, b in zip(self.gap_normal, self.gap_tangent))) > 1e-6
            ):
                raise ValueError("entry_reference_invalid")
            position, yaw, received = _pose(snapshot, now, fresh=True)
            outside, _ = self._offset(position)
            if outside <= 0:
                raise ValueError("entry_initial_side_not_outside")
        except (KeyError, TypeError, ValueError, IndexError, OverflowError) as error:
            raise ValueError("invalid observed doorway entry reference or initial pose") from error
        self.source_view_id = source_view_id
        self.created_at = now
        self._last_now = now
        self._anchor_position = position
        self._anchor_yaw = yaw
        self._anchor_received_at = received
        self._crossing_observed = False
        self._invalid_reason = None

    def _offset(self, position):
        delta = [position[i] - self.gap_center[i] for i in range(2)]
        outside = sum(a * b for a, b in zip(delta, self.gap_normal))
        lateral = sum(a * b for a, b in zip(delta, self.gap_tangent))
        if not math.isfinite(outside) or not math.isfinite(lateral):
            raise ValueError("entry_geometry_unknown")
        return outside, lateral

    def _summary(self, status, reason, outside=None):
        return {
            "status": status,
            "reason": reason,
            "signed_outside_m": outside,
            "required_inside_m": REQUIRED_INSIDE_M,
            "source_view_id": self.source_view_id,
            "source": "observed_doorway_and_robot_odometry",
            "crossing_evidence": (
                "settled_odometry_segment_estimate_not_trajectory_certification"
                if self._crossing_observed
                else None
            ),
            "arrival_verified": False,
        }

    def _invalidate(self, reason):
        if self._invalid_reason is None:
            self._invalid_reason = reason
        return self._summary("unknown", self._invalid_reason)

    def _time(self, now):
        now = _number(now)
        if now < self._last_now - EPS:
            raise ValueError("entry_time_regressed")
        if now - self.created_at > MAX_AGE_S + EPS:
            raise ValueError("entry_reference_expired")
        self._last_now = now
        return now

    def _check_anchor(self, position, yaw, received):
        if received < self._anchor_received_at - EPS:
            raise ValueError("entry_state_replayed")
        if (
            math.dist(position, self._anchor_position) > MAX_IDLE_TRANSLATION_M + EPS
            or abs(_angle_difference(yaw, self._anchor_yaw)) > MAX_IDLE_YAW_RAD + EPS
        ):
            raise ValueError("entry_unexpected_pose_change")

    def _classify(self, position):
        outside, _ = self._offset(position)
        if outside > EPS:
            self._crossing_observed = False
        if outside > -REQUIRED_INSIDE_M + EPS:
            return self._summary("outside", "body_not_far_enough_inside", outside)
        if not self._crossing_observed:
            return self._summary("outside", "doorway_crossing_not_established", outside)
        return self._summary("inside", "observed_doorway_body_entry", outside)

    def observation(self, snapshot, *, now):
        """Classify a stationary observation; it never moves the body-pose anchor."""
        if self._invalid_reason is not None:
            return self._summary("unknown", self._invalid_reason)
        try:
            now = self._time(now)
            position, yaw, received = _pose(snapshot, now, fresh=True)
            self._check_anchor(position, yaw, received)
            return self._classify(position)
        except (KeyError, TypeError, ValueError, IndexError, OverflowError) as error:
            return self._invalidate(str(error))

    def note_motion(self, before, after, result, *, now):
        """Advance the anchor only after a completed, settled and bounded body action.

        The before sample is historical by the end of an action. Its timestamp
        must follow the anchor sample and precede the fresh after sample; applying
        end-time freshness to it would reject legitimate multi-second walking.
        """
        if self._invalid_reason is not None:
            return self._summary("unknown", self._invalid_reason)
        try:
            now = self._time(now)
            before_position, before_yaw, before_at = _pose(before, now, fresh=False)
            self._check_anchor(before_position, before_yaw, before_at)
            if (
                not isinstance(result, dict)
                or result.get("completed") is not True
                or not isinstance(result.get("stop"), dict)
                or result["stop"].get("physical_settling_verified") is not True
            ):
                raise ValueError("entry_motion_not_completed_and_settled")
            position, yaw, received = _pose(after, now, fresh=True)
            if received <= before_at:
                raise ValueError("entry_motion_state_not_newer")
            if (
                math.dist(position, before_position) > MAX_MOTION_M + EPS
                or abs(_angle_difference(yaw, before_yaw)) > MAX_MOTION_YAW_RAD + EPS
            ):
                raise ValueError("entry_motion_out_of_bounds")
            # Only reported settled body motion can establish a crossing. A
            # later in-room lateral move, or head-scan body drift, cannot create
            # missing evidence that the observed doorway itself was traversed.
            before_outside, before_lateral = self._offset(before_position)
            outside, lateral = self._offset(position)
            if before_outside > EPS and outside <= EPS:
                fraction = min(1.0, before_outside / (before_outside - outside))
                crossing_lateral = before_lateral + fraction * (lateral - before_lateral)
                # This discrete odometry-segment interpolation is evidence only;
                # it does not reconstruct or certify the gait between samples.
                self._crossing_observed = (
                    abs(crossing_lateral) <= self.gap_width_m / 2 - JAMB_MARGIN_M + EPS
                )
            self._anchor_position = position
            self._anchor_yaw = yaw
            self._anchor_received_at = received
            return self._classify(position)
        except (KeyError, TypeError, ValueError, IndexError, OverflowError) as error:
            return self._invalidate(str(error))
