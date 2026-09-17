"""Observed doorway paths retain provenance but never bypass movement guards."""

import asyncio
import copy
import json
import math

import pytest
from test_floor_target_live import FloorRobot, arguments, retain
from test_floor_target_live import mission as floor_mission
from test_live import Session

from duck_nav.live import LiveMission, connect_config


class GapRobot(FloorRobot):
    """Ideal measured motion; guards refuse before any displacement."""

    async def advance(self, *args, **kwargs):
        if self.guard is not None:
            await self.stop()
            return {
                "completed": False,
                "reason": self.guard,
                "stop": {"acknowledged": True, "physical_settling_verified": True},
            }
        result = await super().advance(*args, **kwargs)
        return {**result, "stop": {"acknowledged": True, "physical_settling_verified": True}}


def mission(tmp_path, robot=None):
    return floor_mission(tmp_path, robot or GapRobot())


@pytest.fixture(autouse=True)
def fast_images(monkeypatch):
    original = LiveMission.image

    async def fast(worker):
        worker.last_image = -math.inf
        return await original(worker)

    monkeypatch.setattr(LiveMission, "image", fast)


@pytest.fixture
def projected_gap(monkeypatch):
    calls = []

    def project(snapshot, size, point):
        calls.append((copy.deepcopy(snapshot), size, point))
        return {"odometry_position": [1, -0.4] if point[1] == 100 else [1, 0.4]}

    monkeypatch.setattr("duck_nav.live.project_floor_point", project)
    return calls


def gap_arguments(view_id):
    return arguments(view_id, point=[700, 100], opposite_point=[500, 900])


async def start_gap(worker):
    view_id = retain(worker)
    result = await worker.execute_tool("advance_to_floor", gap_arguments(view_id))
    assert result["result"]["completed"] is True
    assert worker.gap_plan is not None
    return view_id, result


async def test_follow_uses_retained_reference_after_source_image_cache_is_cleared(
    tmp_path, projected_gap, monkeypatch
):
    worker = mission(tmp_path)
    view_id, _ = await start_gap(worker)
    assert worker.gap_plan["source_view_id"] == view_id
    reference = copy.deepcopy(worker.gap_plan["reference"])
    worker.stationary_views.clear()
    worker.supplied_view_ids.clear()

    def no_reprojection(*_):
        raise AssertionError("following a retained path must not reproject an unavailable frame")

    monkeypatch.setattr("duck_nav.live.project_floor_point", no_reprojection)
    result = await worker.execute_tool("follow_gap", {})
    assert result["result"]["completed"] is True
    assert worker.robot.headings == [(0.1, 0, True), (0.1, 0, True)]
    assert worker.gap_plan["reference"] == reference
    assert worker.gap_plan["source_view_id"] == view_id
    assert worker.gap_plan["progress_s"] > 0
    assert len(projected_gap) == 2


async def test_gap_context_is_sanitized_and_does_not_alias_private_reference(
    tmp_path, projected_gap
):
    worker = mission(tmp_path)
    await start_gap(worker)
    private = copy.deepcopy(worker.gap_plan)
    context = worker.context(await worker.robot.observe())
    assert context["gap_plan"] is not None
    serialized = json.dumps(context["gap_plan"])
    for forbidden in ("samples", "geometry_snapshot", "simulator_truth", "qpos", "qvel"):
        assert forbidden not in serialized
    context["gap_plan"].clear()
    assert worker.gap_plan == private


@pytest.mark.parametrize(
    "name,args",
    [
        ("observe", {"reason": "Inspect current floor"}),
        ("look_at", {"x": 1, "y": 1, "z": 0, "reason": "Inspect left jamb"}),
        (
            "remember_place",
            {"name": "opening", "observation": "Two visible jambs", "explored": False},
        ),
    ],
)
async def test_stationary_success_preserves_gap_reference(tmp_path, projected_gap, name, args):
    worker = mission(tmp_path)
    await start_gap(worker)
    saved = copy.deepcopy(worker.gap_plan)
    await worker.execute_tool(name, args)
    assert worker.gap_plan == saved


@pytest.mark.parametrize("action", ["advance", "single_point", "finish", "stop"])
async def test_new_route_or_terminal_tool_clears_gap_reference(tmp_path, projected_gap, action):
    worker = mission(tmp_path)
    await start_gap(worker)
    if action == "advance":
        await worker.execute_tool(
            "advance", {"distance_m": 0.1, "heading_deg": 0, "reason": "New route"}
        )
    elif action == "single_point":
        await worker.execute_tool("advance_to_floor", arguments(retain(worker)))
    elif action == "finish":
        await worker.execute_tool("finish", {"status": "blocked", "reason": "No clear route"})
    else:
        await worker.execute_tool("stop", {"reason": "User stop"})
    assert worker.gap_plan is None


async def test_arrival_claim_preserves_reference_until_body_enters(tmp_path, projected_gap):
    worker = mission(tmp_path)
    await start_gap(worker)
    reference = copy.deepcopy(worker.gap_plan)
    worker.arrival_reviewer = object()

    async def review(**_):
        raise AssertionError("Visual review cannot establish body entry")

    worker.review_arrival = review
    result = await worker.execute_tool(
        "finish", {"status": "goal_observed", "reason": "Candidate destination"}
    )
    assert result["status"] == "entry_not_confirmed"
    assert result["continue_navigation"] is True
    assert worker.gap_plan == reference


@pytest.mark.parametrize("failure", ["expired", "translation", "yaw"])
async def test_stale_or_externally_moved_reference_stops_without_advance(
    tmp_path, projected_gap, failure
):
    worker = mission(tmp_path)
    await start_gap(worker)
    before = len(worker.robot.headings)
    if failure == "expired":
        worker.gap_plan["created_at"] -= 241
    elif failure == "translation":
        worker.robot.distance += 0.026
    else:
        worker.robot.yaw += math.radians(5.1)
    result = await worker.execute_tool("follow_gap", {})
    assert result["result"]["completed"] is False
    assert worker.gap_plan is None
    assert len(worker.robot.headings) == before
    assert worker.robot.calls[-1] == "stop"


@pytest.mark.parametrize("failure", ["body", "gaze"])
async def test_failed_physical_action_discards_reference(tmp_path, projected_gap, failure):
    worker = mission(tmp_path)
    await start_gap(worker)
    if failure == "body":
        worker.robot.advance_ok = False
        result = await worker.execute_tool("follow_gap", {})
    else:

        async def failed_gaze(**_):
            return {"completed": False, "reason": "look_correction_limit"}

        worker.robot.look_at = failed_gaze
        result = await worker.execute_tool(
            "look_at", {"x": 1, "y": 1, "z": 0, "reason": "Inspect jamb"}
        )
    assert result["result"]["completed"] is False
    assert worker.gap_plan is None
    assert worker.result["status"] == "blocked"


@pytest.mark.parametrize("settling", [None, False, 1])
async def test_command_completion_without_verified_settling_cannot_retain_path(
    tmp_path, projected_gap, settling
):
    class UnsettledRobot(GapRobot):
        async def advance(self, *args, **kwargs):
            result = await super().advance(*args, **kwargs)
            result["stop"]["physical_settling_verified"] = settling
            return result

    worker = mission(tmp_path, UnsettledRobot())
    result = await worker.execute_tool("advance_to_floor", gap_arguments(retain(worker)))
    assert result["result"]["completed"] is True
    assert worker.robot.distance == pytest.approx(0.1)
    assert worker.gap_plan is None


async def test_post_motion_tracking_refusal_discards_plan_but_reports_actual_movement(
    tmp_path, projected_gap
):
    class DriftingRobot(GapRobot):
        def snapshot(self):
            snapshot = super().snapshot()
            snapshot["state"]["data"]["odom"]["position"][1] = 0.11 if self.distance else 0
            return snapshot

    worker = mission(tmp_path, DriftingRobot())
    result = await worker.execute_tool("advance_to_floor", gap_arguments(retain(worker)))
    assert result["result"]["completed"] is True
    assert result["progress"]["negligible"] is False
    assert worker.gap_plan is None
    assert worker.context(await worker.robot.observe())["gap_plan"]["status"] == "refused"
    await worker.execute_tool("follow_gap", {})
    assert len(worker.robot.headings) == 1


async def test_local_reference_completion_never_claims_destination_arrival(tmp_path, projected_gap):
    worker = mission(tmp_path)
    await start_gap(worker)
    for _ in range(20):
        if worker.gap_plan is None:
            break
        result = await worker.execute_tool("follow_gap", {})
        assert result["result"]["completed"] is True
    assert worker.gap_plan is None
    assert worker.gap_summary["status"] == "complete"
    assert worker.gap_summary["arrival_verified"] is False
    assert worker.robot.distance == pytest.approx(1.3)
    assert worker.result["status"] != "goal_observed"
    assert worker.result["goal_verified"] is False
    assert not worker.done.is_set()


async def test_refusal_with_failed_stop_terminates_mission(tmp_path, projected_gap):
    worker = mission(tmp_path)
    await start_gap(worker)
    worker.gap_plan["created_at"] -= 241
    worker.robot.stop_ok = False
    result = await worker.execute_tool("follow_gap", {})
    assert result["result"]["stop"]["completed"] is False
    assert worker.result["status"] == "blocked"
    assert worker.gap_plan is None
    assert len(worker.robot.headings) == 1


async def test_fatal_guard_wins_over_expired_reference(tmp_path, projected_gap):
    worker = mission(tmp_path)
    await start_gap(worker)
    worker.gap_plan["created_at"] -= 241
    worker.robot.guard = "state_stale"
    result = await worker.execute_tool("follow_gap", {})
    assert result["result"]["reason"] == "gap_plan_refused"
    assert result["result"]["detail"] == "state_stale"
    assert result["result"]["fatal_guard"] is True
    assert worker.result["status"] == "blocked"
    assert len(worker.robot.headings) == 1
    assert worker.robot.calls[-1] == "stop"
    assert worker.gap_plan is None


@pytest.mark.parametrize("guard", ["obstacle", "head_not_forward", "depth_quality"])
async def test_current_nonready_guard_prevents_reference_step(tmp_path, projected_gap, guard):
    worker = mission(tmp_path)
    await start_gap(worker)
    worker.robot.guard = guard
    result = await worker.execute_tool("follow_gap", {})
    assert result["result"]["completed"] is False
    assert len(worker.robot.headings) == 1
    assert worker.robot.calls[-1] == "stop"


async def test_existing_recovery_inspection_gate_prevents_follow_dispatch(tmp_path, projected_gap):
    worker = mission(tmp_path)
    await start_gap(worker)
    worker.recovery = {
        "failed_arguments": {"distance_m": 0.1, "heading_deg": 0},
        "clearance_was_blocked": True,
        "inspected": False,
    }
    result = await worker.execute_tool("follow_gap", {})
    assert result["result"]["reason"] == "recovery_inspection_required"
    assert len(worker.robot.headings) == 1
    assert worker.robot.calls[-1] == "stop"
    assert worker.gap_plan is None


@pytest.mark.parametrize(
    "args", [{"reason": "continue"}, {"distance_m": 0.1}, {"progress_s": 1}, None, []]
)
async def test_follow_gap_has_strict_empty_arguments(tmp_path, args):
    worker = mission(tmp_path)
    with pytest.raises((TypeError, ValueError)):
        await worker.execute_tool("follow_gap", args)
    assert not worker.robot.calls


@pytest.mark.parametrize("missing", ["goal", "planner"])
async def test_follow_requires_accepted_goal_and_visual_planner(tmp_path, missing):
    worker = mission(tmp_path)
    if missing == "goal":
        worker.goal = None
    else:
        worker.navigation_planner = None
    with pytest.raises(ValueError):
        await worker.execute_tool("follow_gap", {})
    assert not worker.robot.calls


async def test_follow_without_reference_refuses_without_motion(tmp_path):
    worker = mission(tmp_path)
    result = await worker.execute_tool("follow_gap", {})
    assert result["result"]["completed"] is False
    assert not worker.robot.headings
    assert worker.robot.calls[-1] == "stop"


@pytest.mark.parametrize("mode", ["streaming", "standard"])
def test_follow_gap_cannot_be_requested_by_voice_model(mode):
    tools = connect_config(mode)["tools"][0]["function_declarations"]
    assert "follow_gap" not in {tool["name"] for tool in tools}


class GapPlanner:
    def __init__(self, actions):
        self.actions = iter(actions)
        self.calls = 0

    async def decide(self, context, jpeg, *, views):
        self.calls += 1
        name = next(self.actions)
        if name == "advance_to_floor":
            args = gap_arguments(context["camera"]["view_id"])
        elif name == "finish":
            args = {"status": "blocked", "reason": "End fixture"}
        elif name == "observe":
            args = {"reason": "Inspect current floor"}
        else:
            args = {}
        return {"name": name, "args": args}


async def test_completed_follow_step_resets_nonprogress_budget(
    tmp_path, projected_gap, monkeypatch
):
    monkeypatch.setattr("duck_nav.live.captures_stationary", lambda *_: True)
    worker = mission(tmp_path)
    worker.navigation_planner = GapPlanner(
        ["advance_to_floor", *["observe"] * 11, "follow_gap", *["observe"] * 11, "finish"]
    )
    result = await worker.run()
    assert result["reason"] == "End fixture"
    assert worker.navigation_planner.calls == 25
    assert len(worker.robot.headings) == 2


async def test_missing_reference_refusals_exhaust_nonprogress_budget(tmp_path):
    worker = mission(tmp_path)
    worker.navigation_planner = GapPlanner(["follow_gap"] * 20)
    result = await worker.run()
    assert result["status"] == "blocked"
    assert "Observation budget exhausted" in result["reason"]
    assert worker.navigation_planner.calls == 12
    assert not worker.robot.headings


@pytest.mark.parametrize("event", ["spoken", "operator"])
async def test_inflight_follow_is_cancelled_and_reference_cleared(
    tmp_path, projected_gap, monkeypatch, event
):
    monkeypatch.setattr("duck_nav.live.captures_stationary", lambda *_: True)

    class PausingRobot(GapRobot):
        def __init__(self):
            super().__init__()
            self.follow_entered = asyncio.Event()

        async def advance(self, *args, **kwargs):
            if self.headings:
                self.advance_release = asyncio.Event()
                self.follow_entered.set()
            return await super().advance(*args, **kwargs)

    worker = mission(tmp_path, PausingRobot())
    worker.session = Session()
    worker.navigation_planner = GapPlanner(["advance_to_floor", "follow_gap"])
    task = asyncio.create_task(worker.run())
    try:
        await asyncio.wait_for(worker.robot.follow_entered.wait(), 2)
        assert worker.gap_plan is not None
        if event == "operator":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 1)
        else:
            worker.session.push({"server_content": {"input_transcription": {"text": "Stop"}}})
            result = await asyncio.wait_for(task, 1)
            assert result["status"] == "cancelled"
        assert worker.gap_plan is None
        assert worker.robot.calls[-1] == "stop"
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
