"""A semantic doorway veto cannot replace capture provenance or physical guards."""

import asyncio
import copy
import json
import math

import pytest
from test_floor_target_live import ApprovingGapReviewer, arguments, retain
from test_live import Session, run
from test_live_gap import GapPlanner, GapRobot, gap_arguments, mission

from duck_nav.live import LiveMission


class Reviewer(ApprovingGapReviewer):
    def __init__(self, changes=None, during=None):
        self.changes = changes or {}
        self.during = during
        self.calls = []
        self.entered = asyncio.Event()
        self.release = None
        self.cancelled = False

    async def review(self, jpeg, *, camera, point, opposite_point):
        self.calls.append(
            {
                "jpeg": jpeg,
                "camera": copy.deepcopy(camera),
                "point": point[:],
                "opposite_point": opposite_point[:],
            }
        )
        self.entered.set()
        try:
            if self.release is not None:
                await self.release.wait()
            if self.during is not None:
                self.during()
            result = await super().review(
                jpeg, camera=camera, point=point, opposite_point=opposite_point
            )
            return {**result, **self.changes}
        except asyncio.CancelledError:
            self.cancelled = True
            raise


@pytest.fixture(autouse=True)
def fast_images_and_geometry(monkeypatch):
    original = LiveMission.image

    async def fast(worker):
        worker.last_image = -math.inf
        return await original(worker)

    monkeypatch.setattr(LiveMission, "image", fast)
    monkeypatch.setattr(
        "duck_nav.live.project_floor_point",
        lambda snapshot, size, point: {
            "odometry_position": [1, -0.4] if point[1] == 100 else [1, 0.4]
        },
    )


def events(worker):
    return [
        json.loads(line)
        for line in (worker.recorder.path / "events.jsonl").read_text().splitlines()
    ]


async def test_approval_binds_exact_prior_jpeg_camera_and_original_points(tmp_path):
    worker = mission(tmp_path)
    reviewer = worker.gap_reviewer = Reviewer()
    worker.robot.gaze = 45
    capture = worker.robot.snapshot()
    capture["camera"]["image"][:] = [220, 30, 10]
    view_id = retain(worker, capture)
    view = copy.deepcopy(worker.stationary_views.views[-1])
    worker.robot.gaze = 0
    retain(worker)
    args = gap_arguments(view_id)
    saved = copy.deepcopy(args)
    result = await worker.execute_tool("advance_to_floor", args)
    assert reviewer.calls == [
        {
            "jpeg": view["jpeg"],
            "camera": {
                key: view["camera"][key] for key in ("view_id", "label", "yaw_deg", "pitch_deg")
            },
            "point": saved["point"],
            "opposite_point": saved["opposite_point"],
        }
    ]
    assert args == saved
    assert result["result"]["completed"] is True
    assert worker.gap_plan["source_view_id"] == view_id
    assert len(worker.robot.headings) == 1
    assert worker.result["status"] != "goal_observed"
    assert worker.result["goal_verified"] is False
    review_events = [entry for entry in events(worker) if entry["event"].startswith("gap_review_")]
    assert [entry["event"] for entry in review_events] == [
        "gap_review_started",
        "gap_review_finished",
    ]
    assert review_events[-1]["accepted"] is True
    assert all(entry["view_id"] == view_id for entry in review_events)
    assert "simulator_truth" not in json.dumps(review_events)
    await worker.execute_tool("follow_gap", {})
    assert len(reviewer.calls) == 1


@pytest.mark.parametrize(
    "field", ["doorway_visible", "both_contacts_visible", "points_match_contacts"]
)
@pytest.mark.parametrize("value", [False, None, 1, "true"])
async def test_every_review_predicate_requires_literal_true(tmp_path, field, value):
    worker = mission(tmp_path)
    reviewer = worker.gap_reviewer = Reviewer(
        {field: value, "evidence": "Selected feature is a wall."}
    )
    result = await worker.execute_tool("advance_to_floor", gap_arguments(retain(worker)))
    assert len(reviewer.calls) == 1
    assert result["result"]["reason"] == "floor_target_refused"
    assert result["result"]["detail"] == "gap_review_not_confirmed"
    assert result["gap_review"][field] == value
    assert worker.history[-1]["gap_review"]["evidence"] == "Selected feature is a wall."
    assert worker.gap_plan is None
    assert not worker.robot.headings
    assert worker.robot.calls[-1] == "stop"
    assert [entry for entry in events(worker) if entry["event"] == "gap_review_finished"][-1][
        "accepted"
    ] is False


async def test_paired_floor_target_requires_reviewer(tmp_path):
    worker = mission(tmp_path)
    worker.gap_reviewer = None
    result = await worker.execute_tool("advance_to_floor", gap_arguments(retain(worker)))
    assert result["result"]["detail"] == "gap_review_unavailable"
    assert worker.gap_plan is None
    assert not worker.robot.headings
    assert worker.robot.calls[-1] == "stop"


@pytest.mark.parametrize("reviewer_present", [False, True])
async def test_single_floor_point_does_not_call_doorway_reviewer(tmp_path, reviewer_present):
    worker = mission(tmp_path)
    reviewer = Reviewer({"doorway_visible": False})
    worker.gap_reviewer = reviewer if reviewer_present else None
    result = await worker.execute_tool("advance_to_floor", arguments(retain(worker)))
    assert result["result"]["completed"] is True
    assert not reviewer.calls
    assert worker.gap_plan is None


@pytest.mark.parametrize("bad_input", ["geometry", "expired", "unsupplied"])
async def test_invalid_geometry_or_source_is_refused_before_review(
    tmp_path, monkeypatch, bad_input
):
    worker = mission(tmp_path)
    reviewer = worker.gap_reviewer = Reviewer()
    view_id = retain(worker)
    if bad_input == "geometry":
        monkeypatch.setattr(
            "duck_nav.live.project_floor_point", lambda *_: {"odometry_position": [1, 0]}
        )
    elif bad_input == "expired":
        worker.stationary_views.views[-1]["camera"]["received_at"] -= 31
    else:
        worker.supplied_view_ids.clear()
    result = await worker.execute_tool("advance_to_floor", gap_arguments(view_id))
    assert result["result"]["reason"] == "floor_target_refused"
    assert not reviewer.calls
    assert not worker.robot.headings
    assert worker.gap_plan is None


@pytest.mark.parametrize("change", ["expired", "translation", "yaw", "unsupplied"])
async def test_approval_cannot_override_source_change_during_review(tmp_path, change):
    worker = mission(tmp_path)
    view_id = retain(worker)

    def changed_source():
        if change == "expired":
            worker.stationary_views.views[-1]["camera"]["received_at"] -= 31
        elif change == "translation":
            worker.robot.distance += 0.026
        elif change == "yaw":
            worker.robot.yaw += math.radians(5.1)
        else:
            worker.supplied_view_ids.clear()

    reviewer = worker.gap_reviewer = Reviewer(during=changed_source)
    result = await worker.execute_tool("advance_to_floor", gap_arguments(view_id))
    assert len(reviewer.calls) == 1
    assert result["result"]["reason"] == "floor_target_refused"
    assert result["result"]["completed"] is False
    assert worker.gap_plan is None
    assert not worker.robot.headings
    assert worker.robot.calls[-1] == "stop"


@pytest.mark.parametrize(
    "field,value",
    [("requested", [0.01, 0, 0]), ("applied", [0, 0, 0.001]), ("applied", [float("nan"), 0, 0])],
)
async def test_approval_cannot_override_unstopped_or_invalid_motion_state(tmp_path, field, value):
    worker = mission(tmp_path)
    view_id = retain(worker)
    snapshot = worker.robot.snapshot

    def changed_motion():
        def moving_snapshot():
            result = snapshot()
            result["state"]["data"]["move"][field] = value
            return result

        worker.robot.snapshot = moving_snapshot

    worker.gap_reviewer = Reviewer(during=changed_motion)
    # Keep the fixture's result image valid after observing the invalid state;
    # production transport validation is independent of the post-review gate.
    original_stop = worker.robot.stop

    async def stop():
        worker.robot.snapshot = snapshot
        return await original_stop()

    worker.robot.stop = stop
    result = await worker.execute_tool("advance_to_floor", gap_arguments(view_id))
    assert result["result"]["reason"] == "floor_target_refused"
    assert worker.gap_plan is None
    assert not worker.robot.headings
    assert worker.robot.calls[-1] == "stop"


async def test_fatal_guard_after_review_precedes_new_source_expiry(tmp_path):
    worker = mission(tmp_path)
    view_id = retain(worker)

    def changed_source():
        worker.robot.guard = "state_stale"
        worker.stationary_views.views[-1]["camera"]["received_at"] -= 31

    worker.gap_reviewer = Reviewer(during=changed_source)
    result = await worker.execute_tool("advance_to_floor", gap_arguments(view_id))
    assert result["result"]["fatal_guard"] is True
    assert "state_stale" in json.dumps(result)
    assert worker.result["status"] == "blocked"
    assert not worker.robot.headings
    assert worker.robot.calls[-1] == "stop"
    assert worker.gap_plan is None


@pytest.mark.parametrize("guard", ["obstacle", "head_not_forward", "depth_quality"])
async def test_approval_does_not_override_new_physical_guard(tmp_path, guard):
    worker = mission(tmp_path)
    worker.gap_reviewer = Reviewer(during=lambda: setattr(worker.robot, "guard", guard))
    result = await worker.execute_tool("advance_to_floor", gap_arguments(retain(worker)))
    assert result["result"]["completed"] is False
    assert not worker.robot.headings
    assert worker.robot.calls[-1] == "stop"
    assert worker.gap_plan is None


async def test_completion_during_review_cannot_start_motion(tmp_path):
    worker = mission(tmp_path)
    worker.gap_reviewer = Reviewer(during=lambda: worker.finish("cancelled", "User stop"))
    with pytest.raises(asyncio.CancelledError):
        await worker.execute_tool("advance_to_floor", gap_arguments(retain(worker)))
    assert not worker.robot.headings
    assert worker.gap_plan is None


@pytest.mark.parametrize("event", ["spoken", "operator"])
async def test_pending_review_is_cancelled_with_final_stop(tmp_path, monkeypatch, event):
    monkeypatch.setattr("duck_nav.live.captures_stationary", lambda *_: True)
    worker = mission(tmp_path)
    worker.session = Session()
    worker.navigation_planner = GapPlanner(["advance_to_floor"])
    reviewer = worker.gap_reviewer = Reviewer()
    reviewer.release = asyncio.Event()
    task = asyncio.create_task(worker.run())
    try:
        await asyncio.wait_for(reviewer.entered.wait(), 2)
        assert not worker.robot.headings
        if event == "operator":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 1)
        else:
            worker.session.push({"server_content": {"input_transcription": {"text": "Stop"}}})
            result = await asyncio.wait_for(task, 1)
            assert result["status"] == "cancelled"
        assert reviewer.cancelled
        assert not worker.robot.headings
        assert worker.robot.calls[-1] == "stop"
        assert worker.gap_plan is None
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def test_repeated_review_vetoes_consume_nonprogress_budget(tmp_path, monkeypatch):
    monkeypatch.setattr("duck_nav.live.captures_stationary", lambda *_: True)
    worker = mission(tmp_path)
    worker.navigation_planner = GapPlanner(["advance_to_floor"] * 20)
    reviewer = worker.gap_reviewer = Reviewer({"doorway_visible": False})
    result = await worker.run()
    assert result["status"] == "blocked"
    assert "Observation budget exhausted" in result["reason"]
    assert worker.navigation_planner.calls == 12
    assert len(reviewer.calls) == 12
    assert not worker.robot.headings
    assert worker.gap_plan is None


async def test_run_live_forwards_injected_reviewer(tmp_path, monkeypatch):
    monkeypatch.setattr("duck_nav.live.captures_stationary", lambda *_: True)
    reviewer = Reviewer({"doorway_visible": False})
    result, robot, _ = await run(
        tmp_path,
        Session(),
        GapRobot(),
        goal="Kitchen",
        navigation_planner=GapPlanner(["advance_to_floor", "finish"]),
        gap_reviewer=reviewer,
    )
    assert result["reason"] == "End fixture"
    assert len(reviewer.calls) == 1
    assert not robot.headings
