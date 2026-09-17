"""Point steering preserves image provenance and the guarded movement lifecycle."""

import asyncio
import copy
import math
import time

import pytest
from test_live import Robot, Session, VisualPlanner, camera_snapshot, run

from duck_nav.cli import Recorder
from duck_nav.live import LiveConfig, LiveMission, _jpeg, connect_config


class FloorRobot(Robot):
    def __init__(self):
        super().__init__()
        self.yaw = 0
        self.gaze = 0
        self.headings = []
        self.guard = None

    def snapshot(self):
        sample = camera_snapshot(self.gaze, body_yaw=self.yaw, position=(self.distance, 0, 0.12))
        sample["connected"] = self.connected
        sample["state"]["data"]["safety"] = {"gravity": [0, 0, -1]}
        sample["state"]["data"]["frames"]["camera"]["pos"] = [0.05, 0, 0.2]
        sample["camera"]["metadata"] = {
            "width": 32,
            "height": 24,
            "rotate": 0,
            "intrinsics": {
                "fx": 20,
                "fy": 20,
                "cx": 16,
                "cy": 12,
                "source": "sim",
                "calibrated": False,
                "distortion": [],
            },
        }
        return sample

    async def observe(self):
        return {
            **await super().observe(),
            "ready": self.guard is None,
            "guard_reason": self.guard,
        }

    async def advance(self, distance_m, heading_deg=0, *, new_course=False):
        self.headings.append((distance_m, heading_deg, new_course))
        return await super().advance(distance_m, heading_deg)


class ApprovingGapReviewer:
    async def review(self, jpeg, *, camera, point, opposite_point):
        return {
            "doorway_visible": True,
            "both_contacts_visible": True,
            "points_match_contacts": True,
            "evidence": "Fixture shows the two selected jamb-floor contacts.",
        }


@pytest.fixture(autouse=True)
def fast_images(monkeypatch):
    original = LiveMission.image

    async def fast(mission):
        mission.last_image = -math.inf
        return await original(mission)

    monkeypatch.setattr(LiveMission, "image", fast)


def mission(tmp_path, robot=None):
    robot = robot or FloorRobot()
    return LiveMission(
        robot,
        robot,
        Session(),
        Recorder(tmp_path),
        audio=None,
        goal="Kitchen",
        config=LiveConfig(),
        speak=None,
        emit=lambda _: None,
        navigation_planner=VisualPlanner([]),
        gap_reviewer=ApprovingGapReviewer(),
    )


def retain(worker, snapshot=None):
    snapshot = snapshot or worker.transport.snapshot()
    previous = copy.deepcopy(snapshot)
    previous["camera"]["received_at"] -= 0.5
    previous["state"]["received_at"] -= 0.5
    worker.stationary_views.retain(previous, _jpeg(previous))
    camera, _ = worker.stationary_views.retain(snapshot, _jpeg(snapshot))
    worker.supplied_view_ids.add(camera["view_id"])
    return camera["view_id"]


def arguments(view_id, **changes):
    return {
        "view_id": view_id,
        "point": [600, 400],
        "max_distance_m": 0.1,
        "reason": "Clear floor with margin from the doorway edges",
        **changes,
    }


@pytest.mark.parametrize("bearing", [0, 20, -20, 60, -60])
async def test_point_uses_exact_capture_but_current_body_heading(tmp_path, monkeypatch, bearing):
    worker = mission(tmp_path)
    robot = worker.robot
    captured = robot.snapshot()
    captured["state"]["data"]["frames"]["camera"]["pos"] = [0.2, 0.1, 0.1]
    view_id = retain(worker, captured)
    target = [math.cos(math.radians(bearing)), math.sin(math.radians(bearing))]
    seen = []

    def project(snapshot, size, point):
        seen.append((copy.deepcopy(snapshot), size, point))
        return {"odometry_position": target}

    monkeypatch.setattr("duck_nav.live.project_floor_point", project)
    # A later head pose must never replace capture extrinsics.
    captured["state"]["data"]["frames"]["camera"]["pos"][0] = 9
    robot.gaze = 0
    result = await worker.execute_tool("advance_to_floor", arguments(view_id))
    assert seen[0][0]["state"]["data"]["frames"]["camera"]["pos"][0] == 0.2
    assert seen[0][1:] == ((32, 24), [600.0, 400.0])
    assert robot.headings == [(0.1, pytest.approx(max(-30, min(30, bearing))), True)]
    assert result["floor_target"]["bearing_deg"] == pytest.approx(bearing)
    assert result["floor_target"]["point_reached"] is False
    assert result["progress"]["negligible"] is False
    assert all(view["camera"]["view_id"] != view_id for view in worker.stationary_views.views)


async def test_gap_midpoint_is_computed_after_projection(tmp_path, monkeypatch):
    worker = mission(tmp_path)
    view_id = retain(worker)
    projected = []

    def project(snapshot, size, point):
        projected.append(point)
        return {"odometry_position": [1, -0.4] if point[1] == 100 else [1, 0.4]}

    monkeypatch.setattr("duck_nav.live.project_floor_point", project)
    result = await worker.execute_tool(
        "advance_to_floor", arguments(view_id, point=[700, 100], opposite_point=[500, 900])
    )
    assert projected == [[700, 100], [500, 900]]
    assert result["floor_target"]["odometry_target"] == [1, 0]
    assert result["floor_target"]["projected_gap_width_m"] == pytest.approx(0.8)
    assert worker.robot.headings == [(0.1, 0, True)]


@pytest.mark.parametrize(
    "failure",
    [
        "unsupplied",
        "expired",
        "moved",
        "yaw_changed",
        "unstable",
        "bad_geometry",
        "behind",
        "near",
        "narrow_gap",
    ],
)
async def test_unusable_points_stop_without_dispatch(tmp_path, monkeypatch, failure):
    worker = mission(tmp_path)
    view_id = retain(worker)
    target = [1, 0]
    args = arguments(view_id)
    if failure == "unsupplied":
        worker.supplied_view_ids.clear()
    elif failure == "expired":
        worker.stationary_views.views[0]["camera"]["received_at"] = time.monotonic() - 31
    elif failure == "moved":
        worker.robot.distance = 0.03
    elif failure == "yaw_changed":
        worker.robot.yaw = math.radians(6)
    elif failure == "unstable":
        worker.stationary_views.views[0]["projection_stationary"] = False
    elif failure == "behind":
        target = [-1, 0]
    elif failure == "near":
        target = [0.06, 0]
    elif failure == "narrow_gap":
        args["opposite_point"] = [650, 500]

    def project(*_):
        if failure == "bad_geometry":
            raise ValueError("unsupported calibration")
        return {"odometry_position": target}

    monkeypatch.setattr("duck_nav.live.project_floor_point", project)
    result = await worker.execute_tool("advance_to_floor", args)
    assert result["result"]["reason"] == "floor_target_refused"
    assert worker.robot.calls == ["stop"]
    assert not worker.robot.headings


async def test_recovery_gate_still_precedes_point_movement(tmp_path, monkeypatch):
    worker = mission(tmp_path)
    view_id = retain(worker)
    monkeypatch.setattr(
        "duck_nav.live.project_floor_point", lambda *_: {"odometry_position": [1, 0]}
    )
    worker.recovery = {
        "failed_arguments": {"distance_m": 0.1, "heading_deg": 0},
        "clearance_was_blocked": True,
        "inspected": False,
    }
    result = await worker.execute_tool("advance_to_floor", arguments(view_id))
    assert result["result"]["reason"] == "recovery_inspection_required"
    assert worker.robot.calls == ["stop"]
    assert not worker.robot.headings


async def test_fatal_guard_wins_even_when_view_is_invalid(tmp_path, monkeypatch):
    worker = mission(tmp_path)
    worker.robot.guard = "state_stale"

    def unexpected(*_):
        raise AssertionError("projection must not run after a fatal guard")

    monkeypatch.setattr("duck_nav.live.project_floor_point", unexpected)
    result = await worker.execute_tool("advance_to_floor", arguments("unknown-view"))
    assert result["result"]["reason"] == "state_stale"
    assert result["result"]["fatal_guard"] is True
    assert worker.result["status"] == "blocked"
    assert worker.robot.calls == ["stop"]


async def test_repeated_point_refusals_exhaust_observation_budget(tmp_path, monkeypatch):
    class RefusingPlanner:
        async def decide(self, context, jpeg, *, views):
            return {"name": "advance_to_floor", "args": arguments(context["camera"]["view_id"])}

    def refuse(*_):
        raise ValueError("unsupported calibration")

    monkeypatch.setattr("duck_nav.live.project_floor_point", refuse)
    result, robot, _ = await run(
        tmp_path, Session(), FloorRobot(), goal="Kitchen", navigation_planner=RefusingPlanner()
    )
    assert result["status"] == "blocked"
    assert "Observation budget exhausted" in result["reason"]
    assert not robot.headings


async def test_point_movement_can_be_cancelled_by_spoken_stop(tmp_path, monkeypatch):
    class PointPlanner:
        async def decide(self, context, jpeg, *, views):
            return {"name": "advance_to_floor", "args": arguments(context["camera"]["view_id"])}

    monkeypatch.setattr(
        "duck_nav.live.project_floor_point", lambda *_: {"odometry_position": [1, 0]}
    )
    # This test compresses capture waits to exercise interruption during dispatch.
    # Independent tests check stationary-pose acceptance using measured samples.
    monkeypatch.setattr("duck_nav.live.captures_stationary", lambda *_: True)
    robot, session = FloorRobot(), Session()
    robot.advance_release = asyncio.Event()
    task = asyncio.create_task(
        run(tmp_path, session, robot, goal="Kitchen", navigation_planner=PointPlanner())
    )
    await asyncio.wait_for(robot.advance_entered.wait(), 1)
    session.push({"server_content": {"input_transcription": {"text": "Stop"}}})
    result, _, _ = await asyncio.wait_for(task, 1)
    assert result["status"] == "cancelled"
    assert robot.calls[-1] == "stop"


async def test_completed_point_step_resets_observation_budget(tmp_path, monkeypatch):
    class PointPlanner:
        def __init__(self):
            self.calls = 0

        async def decide(self, context, jpeg, *, views):
            self.calls += 1
            if self.calls == 12:
                return {"name": "advance_to_floor", "args": arguments(context["camera"]["view_id"])}
            if self.calls == 24:
                return {"name": "finish", "args": {"status": "blocked", "reason": "End fixture"}}
            return {"name": "observe", "args": {"reason": "Inspect clear floor"}}

    monkeypatch.setattr(
        "duck_nav.live.project_floor_point", lambda *_: {"odometry_position": [1, 0]}
    )
    monkeypatch.setattr("duck_nav.live.captures_stationary", lambda *_: True)
    planner = PointPlanner()
    result, robot, _ = await run(
        tmp_path, Session(), FloorRobot(), goal="Kitchen", navigation_planner=planner
    )
    assert result["reason"] == "End fixture"
    assert planner.calls == 24
    assert len(robot.headings) == 1


def test_point_tool_is_only_exposed_to_stateless_planner():
    for mode in ("streaming", "standard"):
        tools = connect_config(mode)["tools"][0]["function_declarations"]
        assert "advance_to_floor" not in {tool["name"] for tool in tools}
