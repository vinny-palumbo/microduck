"""Bounded doorway reference paths from two observed floor endpoints.

This pure geometry does not certify the floor, the inferred wall, unseen space,
or the gait's ability to follow a curve. The initial robot side defines outside;
unordered endpoints alone cannot identify which side is a room interior.
Execution still needs fresh observations, all local guards, and measured stops.
"""

from __future__ import annotations

import math
from itertools import pairwise

RADII = (0.35, 0.45, 0.60)
STAGE_DISTANCES = (0.4, 0.6, 0.8, 1.0)
SAMPLE_STEP = 0.02
MAX_LENGTH = 4.0
REFERENCE_CLEARANCE = 0.37
CURRENT_CLEARANCE = 0.35
WALL_HALF_THICKNESS = 0.03
INSIDE_DISTANCE = 0.35
MAX_PROGRESS_STEP = 0.30
LOOKAHEAD = 0.20
MAX_CROSS_TRACK = 0.10
MAX_TANGENT_ERROR = math.radians(60)
COMPLETE_DISTANCE = 0.06
COMPLETE_HEADING = math.radians(10)
EPS = 1e-9


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("geometry values must be finite numbers")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("geometry values must be finite numbers")
    return value


def _xy(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("expected two finite coordinates")
    return [_number(v) for v in value]


def _wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def _positive_angle(angle):
    result = angle % math.tau
    return 0.0 if min(result, math.tau - result) < EPS else result


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def _subtract(a, b):
    return [a[0] - b[0], a[1] - b[1]]


def _local(point, center, tangent, normal):
    delta = _subtract(point, center)
    return [_dot(delta, tangent), _dot(delta, normal)]


def _point_segment_distance(point, start, end):
    delta = _subtract(end, start)
    squared = _dot(delta, delta)
    fraction = max(0.0, min(1.0, _dot(_subtract(point, start), delta) / squared)) if squared else 0
    return math.dist(point, [start[i] + fraction * delta[i] for i in range(2)])


def _wall_distance(start, end, width):
    """Exact segment distance to the two unbounded wall strips in gap coordinates."""
    best = math.inf
    for side in (-1, 1):
        x, y = side * start[0], start[1]
        dx, dy = side * (end[0] - start[0]), end[1] - start[1]
        half = width / 2
        breaks = [0.0, 1.0]
        for boundary, value, slope in (
            (half, x, dx),
            (WALL_HALF_THICKNESS, y, dy),
            (-WALL_HALF_THICKNESS, y, dy),
        ):
            if slope:
                fraction = (boundary - value) / slope
                if 0 < fraction < 1:
                    breaks.append(fraction)
        breaks.sort()

        def distance(fraction, half=half, x=x, y=y, dx=dx, dy=dy):
            return math.hypot(
                max(0.0, half - x - fraction * dx),
                max(0.0, abs(y + fraction * dy) - WALL_HALF_THICKNESS),
            )

        for lo, hi in pairwise(breaks):
            middle = (lo + hi) / 2
            cx, mx = (half - x, -dx) if x + middle * dx < half else (0.0, 0.0)
            yy = y + middle * dy
            sign = 1 if yy > WALL_HALF_THICKNESS else -1 if yy < -WALL_HALF_THICKNESS else 0
            cy, my = (sign * y - WALL_HALF_THICKNESS, sign * dy) if sign else (0.0, 0.0)
            denominator = mx * mx + my * my
            minimum = max(lo, min(hi, -(cx * mx + cy * my) / denominator)) if denominator else lo
            best = min(best, distance(lo), distance(hi), distance(minimum))
    return best


def _csc(start, yaw, goal, goal_yaw, radius, first, last):
    """Common forward tangent between signed start/end turning circles."""
    c1 = [start[0] - first * radius * math.sin(yaw), start[1] + first * radius * math.cos(yaw)]
    c2 = [
        goal[0] - last * radius * math.sin(goal_yaw),
        goal[1] + last * radius * math.cos(goal_yaw),
    ]
    vector = _subtract(c2, c1)
    distance = math.hypot(*vector)
    offset = (last - first) * radius
    if distance < EPS:
        if first != last:
            return None
        theta, straight = yaw, 0.0
    else:
        if distance < abs(offset) - EPS:
            return None
        theta = math.atan2(vector[1], vector[0]) - math.asin(max(-1.0, min(1.0, offset / distance)))
        straight = math.sqrt(max(0.0, distance * distance - offset * offset))
    arcs = [_positive_angle(first * (theta - yaw)), _positive_angle(last * (goal_yaw - theta))]
    if max(arcs) > math.pi + EPS:
        return None
    return [(radius * arcs[0], first / radius), (straight, 0.0), (radius * arcs[1], last / radius)]


def _append_segment(samples, length, curvature):
    if length <= EPS:
        return
    start, yaw, initial_s = samples[-1]["position"], samples[-1]["yaw"], samples[-1]["s"]
    count = math.ceil(length / SAMPLE_STEP)
    for index in range(1, count + 1):
        travel = length * index / count
        angle = yaw + curvature * travel
        if curvature:
            position = [
                start[0] + (math.sin(angle) - math.sin(yaw)) / curvature,
                start[1] - (math.cos(angle) - math.cos(yaw)) / curvature,
            ]
        else:
            position = [start[0] + travel * math.cos(yaw), start[1] + travel * math.sin(yaw)]
        samples.append({"position": position, "yaw": _wrap(angle), "s": initial_s + travel})


def _reference_clear(samples, endpoints, center, tangent, normal, width, stage_s):
    for before, after in pairwise(samples):
        p, q = before["position"], after["position"]
        local_p, local_q = [_local(v, center, tangent, normal) for v in (p, q)]
        # The approach phase may not cross the plane before reaching its staging pose.
        if after["s"] <= stage_s + EPS and min(local_p[1], local_q[1]) < WALL_HALF_THICKNESS:
            return False
        angle = abs(_wrap(after["yaw"] - before["yaw"]))
        ds = after["s"] - before["s"]
        sagitta = ds / angle * (1 - math.cos(angle / 2)) if angle > EPS else 0.0
        clearance = min(
            _wall_distance(local_p, local_q, width),
            *(_point_segment_distance(e, p, q) for e in endpoints),
        )
        # The curved reference deviates from its sampled chord by at most sagitta.
        if clearance + EPS < REFERENCE_CLEARANCE + sagitta:
            return False
    return True


def build_gap_path(start_xy, yaw, endpoints):
    """Build the shortest admissible CSC approach followed by normal entry.

    Coordinates are measured odometry metres; yaw is radians. Endpoints are the
    two projected, observed jamb-floor points, in either order. Raises ValueError
    for malformed input or when no candidate meets the reference geometry limits.
    The initial half-plane defines outside, not the room's semantic interior.
    """
    try:
        start, yaw = _xy(start_xy), _wrap(_number(yaw))
        if not isinstance(endpoints, (list, tuple)) or len(endpoints) != 2:
            raise ValueError("expected two observed floor endpoints")
        endpoints = [_xy(point) for point in endpoints]
        width = math.dist(*endpoints)
        if not 2 * REFERENCE_CLEARANCE < width <= 3:
            raise ValueError("gap width must exceed 0.74 m and be at most 3 m")
        center = [(a + b) / 2 for a, b in zip(*endpoints)]
        tangent = [(b - a) / width for a, b in zip(*endpoints)]
        normal = [-tangent[1], tangent[0]]
        if _dot(_subtract(start, center), normal) < 0:
            normal = [-v for v in normal]
        local_start = _local(start, center, tangent, normal)
        if local_start[1] <= 0.10:
            raise ValueError("initial side is ambiguous or too close to the doorway plane")
        if _wall_distance(local_start, local_start, width) + EPS < REFERENCE_CLEARANCE:
            raise ValueError("initial pose is too close to the inferred doorway wall")
        goal_yaw = math.atan2(-normal[1], -normal[0])
        final = [center[i] - INSIDE_DISTANCE * normal[i] for i in range(2)]
        best = None
        for radius in RADII:
            for stage_distance in STAGE_DISTANCES:
                stage = [center[i] + stage_distance * normal[i] for i in range(2)]
                for first, last in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                    segments = _csc(start, yaw, stage, goal_yaw, radius, first, last)
                    if segments is None:
                        continue
                    stage_s = sum(length for length, _ in segments)
                    total = stage_s + stage_distance + INSIDE_DISTANCE
                    if total > MAX_LENGTH + EPS or (best and total >= best["total_length"] - EPS):
                        continue
                    samples = [{"position": start[:], "yaw": yaw, "s": 0.0}]
                    for length, curvature in segments:
                        _append_segment(samples, length, curvature)
                    if (
                        math.dist(samples[-1]["position"], stage) > 1e-7
                        or abs(_wrap(samples[-1]["yaw"] - goal_yaw)) > 1e-7
                    ):
                        continue
                    samples[-1] = {"position": stage[:], "yaw": goal_yaw, "s": stage_s}
                    _append_segment(samples, stage_distance + INSIDE_DISTANCE, 0.0)
                    samples[-1] = {"position": final[:], "yaw": goal_yaw, "s": total}
                    if not _reference_clear(
                        samples, endpoints, center, tangent, normal, width, stage_s
                    ):
                        continue
                    best = {
                        "status": "ready",
                        "samples": samples,
                        "endpoints": endpoints,
                        "gap_center": center,
                        "gap_tangent": tangent,
                        "gap_normal": normal,
                        "gap_width_m": width,
                        "stage_s": stage_s,
                        "stage_distance_m": stage_distance,
                        "stage_pose": {"position": stage, "yaw": goal_yaw},
                        "total_length": total,
                        "reference_radius": radius,
                        "segments": [
                            {"length": length, "curvature": curvature}
                            for length, curvature in segments
                            + [(stage_distance + INSIDE_DISTANCE, 0.0)]
                        ],
                        "limits": {
                            "sample_step_m": SAMPLE_STEP,
                            "max_length_m": MAX_LENGTH,
                            "max_arc_rad": math.pi,
                            "reference_clearance_m": REFERENCE_CLEARANCE,
                            "wall_half_thickness_m": WALL_HALF_THICKNESS,
                            "max_cross_track_m": MAX_CROSS_TRACK,
                            "max_progress_step_m": MAX_PROGRESS_STEP,
                            "lookahead_m": LOOKAHEAD,
                        },
                        "source": "observed_gap_reference_geometry_not_gait_or_free_space_certification",
                    }
        if best is None:
            raise ValueError("no bounded forward approach path satisfies the observed gap geometry")
        return best
    except (TypeError, OverflowError) as error:
        raise ValueError("invalid finite gap geometry") from error


def _validated_plan(plan):
    samples = plan["samples"]
    if not isinstance(samples, list) or not 2 <= len(samples) <= 1000:
        raise ValueError("invalid reference samples")
    center, tangent, normal = [
        _xy(plan[key]) for key in ("gap_center", "gap_tangent", "gap_normal")
    ]
    width, total, stage = [_number(plan[key]) for key in ("gap_width_m", "total_length", "stage_s")]
    if not 0.74 < width <= 3 or not 0 < total <= MAX_LENGTH + EPS or not 0 <= stage < total:
        raise ValueError("invalid reference dimensions")
    if (
        abs(math.hypot(*normal) - 1) > EPS
        or abs(math.hypot(*tangent) - 1) > EPS
        or abs(_dot(normal, tangent)) > EPS
    ):
        raise ValueError("invalid reference frame")
    checked = [
        {
            "position": _xy(sample["position"]),
            "yaw": _wrap(_number(sample["yaw"])),
            "s": _number(sample["s"]),
        }
        for sample in samples
    ]
    if abs(checked[0]["s"]) > EPS or abs(checked[-1]["s"] - total) > EPS:
        raise ValueError("invalid reference progress endpoints")
    for left, right in pairwise(checked):
        ds = right["s"] - left["s"]
        if (
            not 0 < ds <= SAMPLE_STEP + EPS
            or math.dist(left["position"], right["position"]) > ds + EPS
        ):
            raise ValueError("invalid reference sample spacing")
    expected_final = [center[i] - INSIDE_DISTANCE * normal[i] for i in range(2)]
    final_yaw = math.atan2(-normal[1], -normal[0])
    if (
        math.dist(checked[-1]["position"], expected_final) > EPS
        or abs(_wrap(checked[-1]["yaw"] - final_yaw)) > EPS
    ):
        raise ValueError("invalid reference final pose")
    return checked, center, tangent, normal, width, total, stage


def _at(samples, s):
    for before, after in pairwise(samples):
        if s <= after["s"] + EPS:
            fraction = max(0.0, min(1.0, (s - before["s"]) / (after["s"] - before["s"])))
            return {
                "position": [
                    before["position"][i]
                    + fraction * (after["position"][i] - before["position"][i])
                    for i in range(2)
                ],
                "yaw": _wrap(before["yaw"] + fraction * _wrap(after["yaw"] - before["yaw"])),
                "s": s,
            }
    return samples[-1]


def path_step(plan, current_xy, yaw, progress_s):
    """Return one bounded reference-following proposal, completion, or refusal.

    Progress is monotonic and may advance at most 0.30 m per call. The caller owns
    that progress token and must not reset it to bypass a refusal. A proposal is
    not movement permission: inspect current free space and retain all guards.
    """
    try:
        current, yaw, progress = _xy(current_xy), _wrap(_number(yaw)), _number(progress_s)
        samples, center, tangent, normal, width, total, stage_s = _validated_plan(plan)
        if not 0 <= progress <= total:
            raise ValueError("progress is outside the reference path")
        upper = min(total, progress + MAX_PROGRESS_STEP)
        nearest = None
        for before, after in pairwise(samples):
            lo, hi = max(progress, before["s"]), min(upper, after["s"])
            if hi < lo:
                continue
            a, b = _at(samples, lo), _at(samples, hi)
            delta = _subtract(b["position"], a["position"])
            squared = _dot(delta, delta)
            fraction = (
                max(0.0, min(1.0, _dot(_subtract(current, a["position"]), delta) / squared))
                if squared
                else 0.0
            )
            candidate = _at(samples, lo + fraction * (hi - lo))
            distance = math.dist(current, candidate["position"])
            if nearest is None or distance < nearest[0] - EPS:
                nearest = (distance, candidate)
        cross_track, reference = nearest
        progress = reference["s"]
        heading_error = _wrap(yaw - reference["yaw"])
        result = {
            "status": "refused",
            "reason": None,
            "target": None,
            "heading_deg": None,
            "distance_m": 0.0,
            "progress_s": progress,
            "cross_track_m": cross_track,
            "reference_heading_error_deg": math.degrees(heading_error),
        }
        if cross_track > MAX_CROSS_TRACK + EPS:
            return {**result, "reason": "cross_track_limit"}
        if abs(heading_error) > MAX_TANGENT_ERROR + EPS:
            return {**result, "reason": "reference_heading_limit"}
        lateral, outside = _local(current, center, tangent, normal)
        if _wall_distance([lateral, outside], [lateral, outside], width) + EPS < CURRENT_CLEARANCE:
            return {**result, "reason": "current_gap_clearance"}
        if outside < -WALL_HALF_THICKNESS and progress + COMPLETE_DISTANCE < stage_s:
            return {**result, "reason": "crossed_before_staging"}
        if progress >= stage_s - EPS or abs(outside) <= 0.20 + EPS:
            direction = [math.cos(yaw), math.sin(yaw)]
            inward = -_dot(direction, normal)
            if inward <= 1e-6:
                return {**result, "reason": "not_facing_through_gap"}
            crossing = lateral + outside * _dot(direction, tangent) / inward
            clearance = (width / 2 - abs(crossing)) * inward
            result["crossing_clearance_m"] = clearance
            if clearance + EPS < CURRENT_CLEARANCE:
                return {**result, "reason": "crossing_alignment_clearance"}
        final = samples[-1]
        final_heading_error = abs(_wrap(yaw - final["yaw"]))
        if (
            outside < -WALL_HALF_THICKNESS
            and progress >= total - COMPLETE_DISTANCE - EPS
            and math.dist(current, final["position"]) <= COMPLETE_DISTANCE + EPS
            and final_heading_error <= COMPLETE_HEADING + EPS
        ):
            return {
                **result,
                "status": "complete",
                "reason": "reference_final_pose_reached",
                "target": final["position"][:],
            }
        target = _at(samples, min(total, progress + LOOKAHEAD))["position"]
        delta = _subtract(target, current)
        bearing = _wrap(math.atan2(delta[1], delta[0]) - yaw)
        distance = min(0.1, math.hypot(*delta), total - progress)
        if distance < 0.05 - EPS:
            return {**result, "reason": "remaining_distance_below_action_minimum"}
        if abs(bearing) > math.pi / 2:
            return {**result, "reason": "lookahead_not_ahead"}
        return {
            **result,
            "status": "advance",
            "reason": "reference_step",
            "target": target,
            "heading_deg": max(-30.0, min(30.0, math.degrees(bearing))),
            "heading_limited": abs(bearing) > math.radians(30),
            "distance_m": distance,
        }
    except (KeyError, TypeError, IndexError, AttributeError, OverflowError) as error:
        raise ValueError("invalid finite path following input") from error
