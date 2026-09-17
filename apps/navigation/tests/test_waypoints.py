"""Floor geometry regressions use synthetic sensors, never a robot or scene map."""

import copy
import json
import math

import numpy as np
import pytest

from duck_nav.core import _rotate
from duck_nav.waypoints import captures_stationary, project_floor_point


def snapshot(rotation=0):
    width, height = (480, 640) if rotation in (90, 270) else (640, 480)
    return {
        "connected": True,
        "camera": {
            "received_at": 100.0,
            "image": np.zeros((height, width, 3), dtype=np.uint8),
            "metadata": {
                "width": 640,
                "height": 480,
                "rotate": rotation,
                "intrinsics": {
                    "fx": 200.0,
                    "fy": 300.0,
                    "cx": 310.0,
                    "cy": 200.0,
                    "source": "sim",
                    "calibrated": False,
                },
            },
        },
        "state": {
            "received_at": 100.01,
            "data": {
                "move": {"requested": [0, 0, 0], "applied": [0, 0, 0]},
                "frames": {
                    # Upright CV2 +x right, +y down, +z forward in the trunk.
                    "camera": {"pos": [0.05, 0.01, 0.1], "quat": [0.5, -0.5, 0.5, -0.5]}
                },
                "safety": {"gravity": [0, 0, -1]},
                "odom": {"position": [1, 2, 0.1], "yaw": 0},
            },
        },
    }


def normalized(u, v, size=(640, 480)):
    return [1000 * v / (size[1] - 1), 1000 * u / (size[0] - 1)]


def project(sample=None, point=None, size=(640, 480)):
    return project_floor_point(
        snapshot() if sample is None else sample,
        size,
        normalized(330, 350, size) if point is None else point,
    )


def capture_pair(interval=1.0):
    previous = snapshot()
    current = copy.deepcopy(previous)
    current["camera"]["received_at"] += interval
    current["state"]["received_at"] += interval
    return previous, current


@pytest.mark.parametrize(
    "rotation,size,upright,raw",
    [
        (0, (640, 480), (330, 350), (330, 350)),
        (90, (480, 640), (309, 410), (410, 170)),
        (180, (640, 480), (349, 429), (290, 50)),
        (270, (480, 640), (230, 429), (210, 230)),
    ],
)
def test_raw_pixel_rotation_and_upright_cv2_pose(rotation, size, upright, raw):
    # Unequal focal lengths and an off-center principal point catch swapped K
    # values and the one-pixel offsets in clockwise np.rot90 permutations.
    result = project(snapshot(rotation), normalized(*upright, size), size)
    assert result["raw_pixel"] == pytest.approx(raw)
    assert result["upright_pixel"] == pytest.approx(upright)
    assert result["body_point"] == pytest.approx([0.45, -0.03, -0.1])
    assert result["position"] == pytest.approx([0.45, -0.03])
    assert result["odometry_position"] == pytest.approx([1.45, 1.97])
    assert result["distance_m"] == pytest.approx(math.hypot(0.45, 0.03))
    assert result["bearing_deg"] == pytest.approx(math.degrees(math.atan2(-0.03, 0.45)))
    assert result["camera_height_m"] == pytest.approx(0.2)
    json.dumps(result, allow_nan=False)


def test_thumbnail_uses_pixel_centers_before_inverse_rotation():
    sample = snapshot(90)
    full = project(sample, normalized(309, 410, (480, 640)), (480, 640))
    # A half-size thumbnail maps full pixel center (309,410) to (154.25,204.75).
    small = project(sample, normalized(154.25, 204.75, (240, 320)), (240, 320))
    assert small["raw_pixel"] == pytest.approx(full["raw_pixel"])
    assert small["odometry_position"] == pytest.approx(full["odometry_position"])


def quaternion_product(a, b):
    w, x, y, z = a
    v, i, j, k = b
    return [
        w * v - x * i - y * j - z * k,
        w * i + x * v + y * k - z * j,
        w * j - x * k + y * v + z * i,
        w * k + x * j - y * i + z * v,
    ]


def test_combined_body_tilt_uses_measured_gravity_and_odometry_heading():
    sample = snapshot()
    state = sample["state"]["data"]
    roll, pitch, yaw = map(math.radians, [15, 20, 30])
    # Body-to-world is Rz(yaw) Ry(pitch) Rx(roll). Transform the same physical
    # camera and gravity into that tilted trunk, while keeping the floor fixed.
    body_from_level = quaternion_product(
        [math.cos(roll / 2), -math.sin(roll / 2), 0, 0],
        [math.cos(pitch / 2), 0, -math.sin(pitch / 2), 0],
    )
    camera = state["frames"]["camera"]
    camera["pos"] = _rotate(body_from_level, camera["pos"])
    camera["quat"] = quaternion_product(body_from_level, camera["quat"])
    state["safety"]["gravity"] = _rotate(body_from_level, [0, 0, -1])
    state["odom"]["yaw"] = yaw
    result = project(sample)
    assert result["position"] == pytest.approx([0.45, -0.03])
    assert result["odometry_position"] == pytest.approx(
        [
            1 + math.cos(yaw) * 0.45 + math.sin(yaw) * 0.03,
            2 + math.sin(yaw) * 0.45 - math.cos(yaw) * 0.03,
        ]
    )
    assert sum(a * b for a, b in zip(state["safety"]["gravity"], result["body_point"])) == (
        pytest.approx(state["odom"]["position"][2])
    )


def test_captured_head_pose_is_used_and_later_head_state_cannot_replace_it():
    stored = snapshot()
    current = copy.deepcopy(stored)
    original = project(stored)
    yaw = math.radians(45)
    head_turn = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    current["state"]["data"]["frames"]["camera"]["quat"] = quaternion_product(
        head_turn, current["state"]["data"]["frames"]["camera"]["quat"]
    )
    # A stored view does not consult the current robot/head. An image supplied
    # with a detectably later state fails, rather than reprojecting its old pixel.
    assert project(stored) == original
    current["state"]["received_at"] += 1
    with pytest.raises(ValueError, match="receive times"):
        project(current)
    # An independently matched turned view uses its own extrinsics.
    current["camera"]["received_at"] += 1
    turned = project(current)
    assert turned["position"] == pytest.approx(
        [0.05 + (0.4 + 0.04) / math.sqrt(2), 0.01 + (0.4 - 0.04) / math.sqrt(2)]
    )


@pytest.mark.parametrize("v", [170, 200, 203])
def test_upward_and_near_horizon_rays_are_rejected(v):
    with pytest.raises(ValueError, match="horizon"):
        project(point=normalized(310, v))


@pytest.mark.parametrize("height,v", [(0.02, 350), (0.2, 224)])
def test_target_distance_limits(height, v):
    sample = snapshot()
    sample["state"]["data"]["odom"]["position"][2] = height
    sample["state"]["data"]["frames"]["camera"]["pos"][2] = height
    with pytest.raises(ValueError, match="between 0.15 and 4"):
        project(sample, normalized(310, v))


@pytest.mark.parametrize("source", ["robot", "family"])
def test_physical_calibration_requires_explicit_zero_distortion(source):
    sample = snapshot()
    calibration = sample["camera"]["metadata"]["intrinsics"]
    calibration.update(source=source, calibrated=True)
    with pytest.raises(ValueError, match="explicit zero-distortion"):
        project(sample)
    calibration["distortion"] = [0, 0, 0, 0, 0]
    assert project(sample)["intrinsics_source"] == source
    calibration["distortion"][0] = 0.000001
    with pytest.raises(ValueError, match="nonzero camera distortion"):
        project(sample)


@pytest.mark.parametrize(
    "path,value",
    [
        (("connected",), False),
        (("camera", "received_at"), float("nan")),
        (("state", "received_at"), 100.2),
        (("camera", "metadata", "width"), 640.0),
        (("camera", "metadata", "height"), 0),
        (("camera", "metadata", "rotate"), 45),
        (("camera", "metadata", "rotate"), True),
        (("camera", "metadata", "intrinsics", "source"), "nominal"),
        (("camera", "metadata", "intrinsics", "source"), None),
        (("camera", "metadata", "intrinsics", "calibrated"), 1),
        (("camera", "metadata", "intrinsics", "fx"), 0),
        (("camera", "metadata", "intrinsics", "fx"), float("nan")),
        (("camera", "metadata", "intrinsics", "fy"), -1),
        (("camera", "metadata", "intrinsics", "cx"), 640),
        (("camera", "metadata", "intrinsics", "cy"), -1),
        (("camera", "metadata", "intrinsics", "distortion"), [0, 0]),
        (("camera", "metadata", "intrinsics", "distortion"), [0, 0, 0, 0, 0.001]),
        (("camera", "metadata", "intrinsics", "distortion"), [0, 0, 0, 0, float("inf")]),
        (("state", "data", "move", "requested"), [0.01, 0, 0]),
        (("state", "data", "move", "applied"), [0, 0, 0.002]),
        (("state", "data", "frames", "camera", "pos"), [0, 0]),
        (("state", "data", "frames", "camera", "quat"), [0, 0, 0, 0]),
        (("state", "data", "frames", "camera", "quat"), [2, 0, 0, 0]),
        (("state", "data", "frames", "camera", "quat"), [1, 0, 0]),
        (("state", "data", "frames", "camera", "quat"), [float("nan"), 0, 0, 0]),
        (("state", "data", "safety", "gravity"), [0, 0, 0]),
        (("state", "data", "safety", "gravity"), [0, 0, -9.81]),
        (("state", "data", "safety", "gravity"), [0, 0, float("nan")]),
        (("state", "data", "odom", "position"), [1, 2, 0]),
        (("state", "data", "odom", "yaw"), float("inf")),
    ],
)
def test_invalid_sensor_metadata_is_refused(path, value):
    sample = snapshot()
    parent = sample
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value
    with pytest.raises(ValueError):
        project(sample)
    previous = snapshot()
    previous["camera"]["received_at"] -= 1
    previous["state"]["received_at"] -= 1
    _, future = capture_pair(1)
    assert not captures_stationary(previous, sample)
    assert not captures_stationary(sample, future)


@pytest.mark.parametrize(
    "point",
    [
        [500],
        [500, 500, 500],
        [-1, 500],
        [500, 1001],
        [True, 500],
        [float("nan"), 500],
        [500, "400"],
    ],
)
def test_invalid_native_point_is_refused(point):
    with pytest.raises(ValueError):
        project(point=point)


@pytest.mark.parametrize("size", [(640, 400), (1280, 960), (1, 1), (640.0, 480), None])
def test_cropped_upscaled_or_malformed_jpeg_dimensions_are_refused(size):
    with pytest.raises(ValueError):
        project(size=size, point=[500, 500])


def test_missing_metadata_bad_image_and_unknown_physical_calibration_are_refused():
    sample = snapshot()
    del sample["camera"]["metadata"]["intrinsics"]
    with pytest.raises(ValueError, match="missing or malformed"):
        project(sample)
    sample = snapshot()
    sample["camera"]["image"] = np.zeros((640, 480, 3))
    with pytest.raises(ValueError, match="upright image dimensions"):
        project(sample)
    sample = snapshot()
    sample["camera"]["metadata"]["intrinsics"]["source"] = "robot"
    with pytest.raises(ValueError, match="must be calibrated"):
        project(sample)


def test_camera_below_assumed_floor_is_refused():
    sample = snapshot()
    sample["state"]["data"]["frames"]["camera"]["pos"][2] = -0.2
    with pytest.raises(ValueError, match="camera must be above"):
        project(sample)


def test_vertical_trunk_has_no_defined_level_heading():
    sample = snapshot()
    sample["state"]["data"]["safety"]["gravity"] = [1, 0, 0]
    with pytest.raises(ValueError, match="too vertical"):
        project(sample)


def test_projection_does_not_modify_the_stored_snapshot():
    sample = snapshot()
    before = copy.deepcopy(sample)
    project(sample)
    assert np.array_equal(sample["camera"].pop("image"), before["camera"].pop("image"))
    assert sample == before


@pytest.mark.parametrize("interval", [0.4, 1, 5.068, 30])
def test_stationary_capture_interval_and_unchanged_geometry(interval):
    previous, current = capture_pair(interval)
    assert captures_stationary(previous, current)


@pytest.mark.parametrize("interval", [-1, 0, 0.399, 30.001])
def test_stationary_captures_need_separated_recent_images(interval):
    assert not captures_stationary(*capture_pair(interval))


@pytest.mark.parametrize("bad", [None, {}, False, [], "snapshot"])
def test_stationary_capture_missing_or_malformed_snapshot_is_false(bad):
    previous, current = capture_pair()
    assert not captures_stationary(bad, current)
    assert not captures_stationary(previous, bad)


def test_stationary_comparison_ignores_unrelated_stream_clock_fields():
    previous, current = capture_pair()
    previous["camera"]["metadata"].update(pts=17, mono_ns=1, real_ns=2)
    current["camera"]["metadata"].update(pts=99, mono_ns=3, real_ns=4, sequence=10)
    assert captures_stationary(previous, current)


@pytest.mark.parametrize(
    "key,value",
    [
        ("fx", 201),
        ("fy", 301),
        ("cx", 309),
        ("cy", 201),
        ("calibrated", True),
        ("distortion", [0] * 5),
    ],
)
def test_stationary_captures_require_identical_calibration(key, value):
    previous, current = capture_pair()
    current["camera"]["metadata"]["intrinsics"][key] = value
    assert not captures_stationary(previous, current)


def test_stationary_captures_require_identical_rotation_and_dimensions():
    previous, current = capture_pair()
    current["camera"]["metadata"]["rotate"] = 180
    assert not captures_stationary(previous, current)
    previous, current = capture_pair()
    current["camera"]["metadata"]["height"] = 481
    current["camera"]["image"] = np.zeros((481, 640, 3), dtype=np.uint8)
    assert not captures_stationary(previous, current)


@pytest.mark.parametrize("key", ["body", "camera"])
@pytest.mark.parametrize(
    "offset,accepted",
    [
        ([0.004, 0, 0], True),
        ([0, 0, 0.004], True),
        ([0, 0, 0.0041], False),
        ([0.003, 0.003, 0], False),
    ],
)
def test_stationary_captures_bound_full_3d_translation(key, offset, accepted):
    previous, current = capture_pair()
    state = current["state"]["data"]
    position = state["odom"]["position"] if key == "body" else state["frames"]["camera"]["pos"]
    position[:] = [a + b for a, b in zip(position, offset)]
    assert captures_stationary(previous, current) is accepted


@pytest.mark.parametrize("key", ["yaw", "gravity", "camera"])
@pytest.mark.parametrize("degrees,accepted", [(1.5, True), (1.501, False), (30, False)])
def test_stationary_captures_bound_body_and_head_rotation(key, degrees, accepted):
    previous, current = capture_pair()
    angle = math.radians(degrees)
    state = current["state"]["data"]
    if key == "yaw":
        state["odom"]["yaw"] = angle
    elif key == "gravity":
        state["safety"]["gravity"] = [math.sin(angle), 0, -math.cos(angle)]
    else:
        state["frames"]["camera"]["quat"] = quaternion_product(
            [math.cos(angle / 2), 0, 0, math.sin(angle / 2)],
            state["frames"]["camera"]["quat"],
        )
    assert captures_stationary(previous, current) is accepted


def test_stationary_yaw_wrap_and_quaternion_sign_are_equivalent():
    previous, current = capture_pair()
    previous["state"]["data"]["odom"]["yaw"] = math.radians(179.5)
    current["state"]["data"]["odom"]["yaw"] = math.radians(-179.5)
    camera = current["state"]["data"]["frames"]["camera"]
    camera["quat"] = [-v for v in camera["quat"]]
    assert captures_stationary(previous, current)


def test_zero_commands_do_not_hide_initial_coast_or_moving_head():
    previous, current = capture_pair()
    # The motor command has reached zero but measured body position is still
    # changing. A zero command at either endpoint alone is insufficient.
    current["state"]["data"]["odom"]["position"][0] += 0.01
    assert not captures_stationary(previous, current)
    previous, current = capture_pair()
    current["state"]["data"]["frames"]["camera"]["quat"] = [1, 0, 0, 0]
    assert not captures_stationary(previous, current)


def test_stationary_comparison_preserves_its_inputs():
    previous, current = capture_pair()
    before_previous, before_current = copy.deepcopy(previous), copy.deepcopy(current)
    assert captures_stationary(previous, current)
    for before, after in [(before_previous, previous), (before_current, current)]:
        assert np.array_equal(before["camera"].pop("image"), after["camera"].pop("image"))
        assert before == after
