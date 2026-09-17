"""Project a point in an exact stored upright image onto the local level floor.

This is sensor geometry, not obstacle clearance or permission to move. The caller
must capture a stopped, settled image with its matching state and retain that
snapshot unchanged. Receive-time proximity alone cannot establish capture-time
synchronization. Never combine a stored pixel with a later head pose.
"""

from __future__ import annotations

import math

from .core import _rotate

MIN_DISTANCE_M = 0.15
MAX_DISTANCE_M = 4.0
MIN_DOWNWARD_COSINE = 0.05
MAX_RECEIVE_SKEW_S = 0.15
MIN_STATIONARY_INTERVAL_S = 0.4
MAX_STATIONARY_INTERVAL_S = 30.0
STATIONARY_POSITION_TOLERANCE_M = 0.004
STATIONARY_ANGLE_TOLERANCE_RAD = math.radians(1.5)


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    try:
        result = float(value)
    except OverflowError:
        raise ValueError(f"{name} must be a finite number") from None
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _vector(value, size, name):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"{name} must contain {size} numbers")
    return [_finite(item, name) for item in value]


def _unit(value, size, name):
    vector = _vector(value, size, name)
    norm = math.hypot(*vector)
    if abs(norm - 1) > 0.01:
        raise ValueError(f"{name} must have unit norm within 0.01")
    return [item / norm for item in vector]


def _dimensions(value, name):
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(isinstance(v, bool) or not isinstance(v, int) or v < 2 for v in value)
    ):
        raise ValueError(f"{name} must contain integer width and height of at least two pixels")
    return value


def _dot(left, right):
    return sum(a * b for a, b in zip(left, right))


def _stored_geometry(snapshot):
    """Validate fields shared by floor projection and stopped-capture comparison."""
    if snapshot["connected"] is not True:
        raise ValueError("stored view was disconnected")
    camera, state_sample = snapshot["camera"], snapshot["state"]
    metadata, state = camera["metadata"], state_sample["data"]
    camera_at = _finite(camera["received_at"], "camera received_at")
    state_at = _finite(state_sample["received_at"], "state received_at")
    if min(camera_at, state_at) < 0 or abs(camera_at - state_at) > MAX_RECEIVE_SKEW_S:
        raise ValueError("camera and state receive times do not match")
    requested = _vector(state["move"]["requested"], 3, "requested motion")
    applied = _vector(state["move"]["applied"], 3, "applied motion")
    if any(v != 0 for v in requested) or any(abs(v) >= 0.001 for v in applied):
        raise ValueError("floor projection requires a stopped stored view")

    width, height = _dimensions([metadata["width"], metadata["height"]], "raw dimensions")
    rotation = metadata["rotate"]
    if (
        isinstance(rotation, bool)
        or not isinstance(rotation, int)
        or rotation not in (0, 90, 180, 270)
    ):
        raise ValueError("camera rotation must be 0, 90, 180 or 270 clockwise")
    upright_width, upright_height = (height, width) if rotation in (90, 270) else (width, height)
    if tuple(camera["image"].shape) != (upright_height, upright_width, 3):
        raise ValueError("upright image dimensions do not match raw metadata and rotation")

    intrinsics = metadata["intrinsics"]
    source = intrinsics["source"]
    calibrated = intrinsics["calibrated"]
    if not isinstance(calibrated, bool) or source not in ("sim", "robot", "family"):
        raise ValueError("unsupported or nominal camera calibration")
    if source != "sim" and calibrated is not True:
        raise ValueError("physical camera intrinsics must be calibrated")
    distortion = intrinsics.get("distortion", [])
    if not isinstance(distortion, (list, tuple)) or len(distortion) not in (0, 5):
        raise ValueError("unsupported distortion coefficient shape")
    distortion = [_finite(value, "distortion") for value in distortion]
    if any(value != 0 for value in distortion):
        raise ValueError("nonzero camera distortion is unsupported")
    if source != "sim" and len(distortion) != 5:
        raise ValueError("physical camera requires an explicit zero-distortion model")
    fx, fy, cx, cy = [_finite(intrinsics[k], k) for k in ("fx", "fy", "cx", "cy")]
    if fx <= 0 or fy <= 0 or not 0 <= cx < width or not 0 <= cy < height:
        raise ValueError(
            "intrinsics require positive focal lengths and an in-frame principal point"
        )
    pose = state["frames"]["camera"]
    origin = _vector(pose["pos"], 3, "camera position")
    quaternion = _unit(pose["quat"], 4, "camera quaternion")
    gravity = _unit(state["safety"]["gravity"], 3, "gravity")
    odometry = state["odom"]
    odom_position = _vector(odometry["position"], 3, "odometry position")
    yaw = _finite(odometry["yaw"], "odometry yaw")
    if odom_position[2] <= 0:
        raise ValueError("odometry height must be positive above the assumed floor")
    camera_height = odom_position[2] - _dot(gravity, origin)
    if not math.isfinite(camera_height) or camera_height <= 0:
        raise ValueError("camera must be above the assumed floor")
    return {
        "camera_at": camera_at,
        "calibration": (
            width,
            height,
            rotation,
            source,
            calibrated,
            fx,
            fy,
            cx,
            cy,
            tuple(distortion),
        ),
        "raw_size": (width, height),
        "upright_size": (upright_width, upright_height),
        "rotation": rotation,
        "intrinsics": (fx, fy, cx, cy),
        "source": source,
        "origin": origin,
        "quaternion": quaternion,
        "gravity": gravity,
        "odom_position": odom_position,
        "yaw": yaw,
        "camera_height": camera_height,
    }


def captures_stationary(previous_snapshot, current_snapshot) -> bool:
    """Whether two stopped captures agree within 4 mm and 1.5 degrees.

    Images must be received 0.4–30 seconds apart with unchanged calibration,
    rotation and dimensions. Each image needs state received within 0.15 seconds.
    Compare full 3D odometry position, wrapped yaw, gravity, and the camera's
    position and quaternion in the trunk. Quaternion signs are interchangeable.

    Missing or invalid data returns False. This verifies two sampled poses only;
    it does not establish continuous stillness or synchronize capture clocks.
    The caller must preserve each image and state together unchanged.
    """
    try:
        previous = _stored_geometry(previous_snapshot)
        current = _stored_geometry(current_snapshot)
        interval = current["camera_at"] - previous["camera_at"]
        if not MIN_STATIONARY_INTERVAL_S - 1e-12 <= interval <= MAX_STATIONARY_INTERVAL_S + 1e-12:
            return False
        if previous["calibration"] != current["calibration"]:
            return False
        for key in ("odom_position", "origin"):
            if math.dist(previous[key], current[key]) > STATIONARY_POSITION_TOLERANCE_M + 1e-12:
                return False
        yaw_difference = current["yaw"] - previous["yaw"]
        yaw_angle = abs(math.atan2(math.sin(yaw_difference), math.cos(yaw_difference)))
        gravity_angle = math.acos(
            max(-1.0, min(1.0, _dot(previous["gravity"], current["gravity"])))
        )
        quaternion_angle = 2 * math.acos(
            min(1.0, abs(_dot(previous["quaternion"], current["quaternion"])))
        )
        return max(yaw_angle, gravity_angle, quaternion_angle) <= (
            STATIONARY_ANGLE_TOLERANCE_RAD + 1e-12
        )
    except (KeyError, TypeError, ValueError, AttributeError, IndexError, OverflowError):
        return False


def project_floor_point(snapshot, jpeg_size, point):
    """Return a floor target from native normalized ``[y, x]`` image coordinates.

    ``jpeg_size`` is ``(width, height)`` of the exact planner JPEG. Coordinates
    0/1000 denote first/last pixel centers. The JPEG must be an aspect-preserving
    thumbnail of ``snapshot.camera.image``, already upright as transport.snapshot
    supplies it. Metadata intrinsics describe the raw, unrotated decoded frame.

    ``position`` is [forward, left] in the gravity-leveled capture-body frame;
    ``odometry_position`` is the same point on the odometry horizontal plane.
    Every transform uses this snapshot only. Movement code must revalidate its
    captured body pose and transform the fixed target after any head recentering.

    Raises ValueError for unavailable, unsupported or inconsistent geometry.
    Hardware requires calibrated robot/family intrinsics and an explicit five-
    coefficient zero-distortion model. Exact simulation intrinsics need no
    physical calibration. No nonzero distortion model is supported here.
    """
    try:
        geometry = _stored_geometry(snapshot)
        width, height = geometry["raw_size"]
        upright_width, upright_height = geometry["upright_size"]
        rotation = geometry["rotation"]
        fx, fy, cx, cy = geometry["intrinsics"]
        jpeg_width, jpeg_height = _dimensions(jpeg_size, "JPEG dimensions")
        if (
            jpeg_width > upright_width
            or jpeg_height > upright_height
            or abs(jpeg_width * upright_height - jpeg_height * upright_width)
            > max(upright_width, upright_height)
        ):
            raise ValueError("JPEG must be an aspect-preserving thumbnail without cropping")

        y, x = _vector(point, 2, "image point [y, x]")
        if not 0 <= x <= 1000 or not 0 <= y <= 1000:
            raise ValueError("image point must be within 0..1000")
        # Invert the thumbnail's pixel-center affine transform, then the exact
        # np.rot90 pixel permutation. This also handles off-center principal points.
        u = (x / 1000 * (jpeg_width - 1) + 0.5) * upright_width / jpeg_width - 0.5
        v = (y / 1000 * (jpeg_height - 1) + 0.5) * upright_height / jpeg_height - 0.5
        raw_u, raw_v = {
            0: (u, v),
            90: (v, height - 1 - u),
            180: (width - 1 - u, height - 1 - v),
            270: (width - 1 - v, u),
        }[rotation]
        raw_x, raw_y = (raw_u - cx) / fx, (raw_v - cy) / fy
        camera_x, camera_y = {
            0: (raw_x, raw_y),
            90: (-raw_y, raw_x),
            180: (-raw_x, -raw_y),
            270: (raw_y, -raw_x),
        }[rotation]
        ray_norm = math.hypot(camera_x, camera_y, 1)
        if not math.isfinite(ray_norm):
            raise ValueError("image ray is not finite")
        camera_ray = [camera_x / ray_norm, camera_y / ray_norm, 1 / ray_norm]
        origin, quaternion = geometry["origin"], geometry["quaternion"]
        direction = _rotate(quaternion, camera_ray)
        gravity = geometry["gravity"]
        odom_position, yaw = geometry["odom_position"], geometry["yaw"]
        camera_height = geometry["camera_height"]
        downward = _dot(gravity, direction)
        if downward < MIN_DOWNWARD_COSINE:
            raise ValueError("image ray is above or too close to the floor horizon")
        distance_along_ray = camera_height / downward
        body_point = [c + distance_along_ray * d for c, d in zip(origin, direction)]
        if not all(math.isfinite(v) for v in body_point):
            raise ValueError("floor intersection is not finite")

        # Level with the horizontal projection of trunk +x as forward, then apply
        # measured odometry yaw. A generic shortest gravity rotation can add a
        # spurious yaw when pitch and roll coexist.
        up = [-v for v in gravity]
        forward = [float(i == 0) - up[0] * up[i] for i in range(3)]
        forward_norm = math.hypot(*forward)
        if forward_norm < 0.1:
            raise ValueError("trunk forward axis is too vertical to define a floor heading")
        forward = [v / forward_norm for v in forward]
        left = [
            up[1] * forward[2] - up[2] * forward[1],
            up[2] * forward[0] - up[0] * forward[2],
            up[0] * forward[1] - up[1] * forward[0],
        ]
        position = [_dot(forward, body_point), _dot(left, body_point)]
        distance = math.hypot(*position)
        if not MIN_DISTANCE_M <= distance <= MAX_DISTANCE_M:
            raise ValueError("floor target must be between 0.15 and 4 metres from the body")
        cosine, sine = math.cos(yaw), math.sin(yaw)
        odom_target = [
            odom_position[0] + cosine * position[0] - sine * position[1],
            odom_position[1] + sine * position[0] + cosine * position[1],
        ]
        if not all(math.isfinite(v) for v in odom_target):
            raise ValueError("odometry floor target is not finite")
        return {
            "position": position,
            "bearing_deg": math.degrees(math.atan2(position[1], position[0])),
            "distance_m": distance,
            "odometry_position": odom_target,
            "body_point": body_point,
            "raw_pixel": [raw_u, raw_v],
            "upright_pixel": [u, v],
            "camera_height_m": camera_height,
            "downward_cosine": downward,
            "intrinsics_source": geometry["source"],
            "source": "stored_camera_and_robot_odometry_assuming_level_floor",
        }
    except (KeyError, TypeError, AttributeError, IndexError, OverflowError) as exc:
        raise ValueError("stored view has missing or malformed projection data") from exc
