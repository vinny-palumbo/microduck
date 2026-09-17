"""Observed body entry is an independent gate on visual arrival claims."""

import asyncio
import copy
import json
import math

import pytest
from test_floor_target_live import arguments, retain
from test_live import ArrivalReviewer
from test_live_gap import GapPlanner, gap_arguments, mission, start_gap

from duck_nav.gap_review import InvalidGapAssessment
from duck_nav.live import LiveMission


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


async def finish(worker):
    return await worker.execute_tool(
        "finish", {"status": "goal_observed", "reason": "Candidate destination"}
    )


async def move_inside(worker):
    # Exercise real runtime tracking: settled increments straddle the observed
    # plane between its jambs, then carry the body beyond the required margin.
    for _ in range(6):
        result = await worker.execute_tool(
            "advance", {"distance_m": 0.2, "heading_deg": 0, "reason": "Visible clear floor"}
        )
        assert result["result"]["completed"] is True
    assert worker.entry_observation()["status"] == "inside"


async def test_reviewed_pair_installs_entry_before_first_motion_and_exposes_safe_context(tmp_path):
    worker = mission(tmp_path)
    view_id, result = await start_gap(worker)
    summary = worker.entry_observation()
    assert summary["status"] == "outside"
    assert summary["signed_outside_m"] == pytest.approx(0.9)
    assert summary["source_view_id"] == view_id
    assert summary["arrival_verified"] is False
    assert result["observation"]["doorway_entry"] == summary
    recorded = events(worker)
    created = next(
        i for i, event in enumerate(recorded) if event["event"] == "doorway_entry_created"
    )
    moved = next(i for i, event in enumerate(recorded) if event["event"] == "doorway_entry_motion")
    assert created < moved
    assert recorded[created]["doorway_entry"]["signed_outside_m"] == pytest.approx(1)
    serialized = json.dumps(summary)
    assert all(key not in serialized for key in ("simulator_truth", "samples", "gap_center"))
    result["observation"]["doorway_entry"].clear()
    assert worker.entry_observation() == summary


@pytest.mark.parametrize("motion", ["follow_gap", "advance", "single_point"])
async def test_each_body_dispatch_tracks_entry_even_when_it_abandons_gap(tmp_path, motion):
    worker = mission(tmp_path)
    await start_gap(worker)
    reference = worker.entry_reference
    if motion == "single_point":
        result = await worker.execute_tool("advance_to_floor", arguments(retain(worker)))
    elif motion == "advance":
        result = await worker.execute_tool(
            "advance", {"distance_m": 0.1, "reason": "Visible clear floor"}
        )
    else:
        result = await worker.execute_tool("follow_gap", {})
    assert result["result"]["completed"] is True
    assert worker.entry_reference is reference
    assert worker.entry_observation()["signed_outside_m"] == pytest.approx(0.8)
    assert (worker.gap_plan is not None) is (motion == "follow_gap")
    assert len([event for event in events(worker) if event["event"] == "doorway_entry_motion"]) == 2


@pytest.mark.parametrize("entry_state", ["outside", "unknown"])
async def test_early_claim_keeps_goal_and_valid_gap_without_visual_review(tmp_path, entry_state):
    worker = mission(tmp_path)
    await start_gap(worker)
    reviewer = worker.arrival_reviewer = ArrivalReviewer([True])
    reference, plan = worker.entry_reference, copy.deepcopy(worker.gap_plan)
    if entry_state == "unknown":
        # Untracked displacement invalidates the crossing evidence permanently.
        worker.robot.distance += 0.03
    before_calls = worker.robot.calls[:]
    result = await finish(worker)
    assert result["status"] == "entry_not_confirmed"
    assert result["continue_navigation"] is True
    assert result["goal_verified"] is False
    assert result["doorway_entry"]["status"] == entry_state
    assert result["goal"] == worker.goal == "Kitchen"
    assert worker.entry_reference is reference
    assert worker.gap_plan == plan
    assert worker.robot.calls == [*before_calls, "stop"]
    assert not reviewer.calls
    assert worker.arrival_claims == 1
    assert not worker.done.is_set()
    assert not any(event["event"] == "arrival_review_started" for event in events(worker))


async def test_three_premature_claims_block_without_extra_motion_or_reviewer(tmp_path):
    worker = mission(tmp_path)
    await start_gap(worker)
    worker.arrival_reviewer = ArrivalReviewer([True])
    for claim in range(1, 4):
        result = await finish(worker)
        assert worker.arrival_claims == claim
        assert result["status"] == ("blocked" if claim == 3 else "entry_not_confirmed")
    assert worker.result["status"] == "blocked"
    assert worker.goal == "Kitchen"
    assert not worker.arrival_reviewer.calls
    assert len(worker.robot.headings) == 1
    assert [event["claim"] for event in events(worker) if event["event"] == "entry_rejected"] == [
        1,
        2,
        3,
    ]


@pytest.mark.parametrize("visual_accepts", [False, True])
async def test_measured_entry_still_requires_visual_identity_and_counts_claim_once(
    tmp_path, visual_accepts
):
    worker = mission(tmp_path)
    await start_gap(worker)
    await move_inside(worker)
    reviewer = worker.arrival_reviewer = ArrivalReviewer([visual_accepts])
    result = await finish(worker)
    assert result["status"] == ("goal_observed" if visual_accepts else "arrival_not_confirmed")
    assert result["goal_verified"] is False
    assert worker.arrival_claims == 1
    assert len(reviewer.calls) == 1
    assert reviewer.calls[0][0] == "Kitchen"
    assert len(reviewer.calls[0][1]) == 4
    assert worker.entry_observation()["status"] == "inside"
    assert worker.gap_plan is None


async def test_without_observed_doorway_legacy_arrival_review_remains_available(tmp_path):
    worker = mission(tmp_path)
    worker.arrival_reviewer = ArrivalReviewer([True])
    result = await finish(worker)
    assert result["status"] == "goal_observed"
    assert worker.entry_reference is None
    assert worker.arrival_claims == 1
    assert len(worker.arrival_reviewer.calls) == 1


@pytest.mark.parametrize("failure", ["failed_motion", "external_displacement"])
async def test_unknown_entry_cannot_disappear_after_gap_clear_and_recenter(tmp_path, failure):
    worker = mission(tmp_path)
    await start_gap(worker)
    reference = worker.entry_reference
    if failure == "failed_motion":
        worker.robot.advance_ok = False
        await worker.execute_tool("follow_gap", {})
        worker.robot.advance_ok = True
    else:
        worker.robot.distance += 0.03
        assert worker.entry_observation()["status"] == "unknown"
        worker.robot.distance -= 0.03
    worker.clear_gap("new_view")
    await worker.execute_tool("look_at", {"x": 1, "y": 0, "z": 0, "reason": "Recenter"})
    worker.arrival_reviewer = ArrivalReviewer([True])
    result = await finish(worker)
    assert result["status"] == "entry_not_confirmed"
    assert result["doorway_entry"]["status"] == "unknown"
    assert worker.entry_reference is reference
    assert worker.gap_plan is None
    assert not worker.arrival_reviewer.calls


async def test_body_change_during_positive_visual_review_rejects_arrival(tmp_path):
    worker = mission(tmp_path)
    await start_gap(worker)
    await move_inside(worker)

    class MovedDuringReview(ArrivalReviewer):
        async def review(self, goal, views):
            result = await super().review(goal, views)
            worker.robot.distance += 0.03
            return result

    worker.arrival_reviewer = MovedDuringReview([True])
    result = await finish(worker)
    assert result["status"] == "entry_not_confirmed"
    assert result["doorway_entry"]["status"] == "unknown"
    assert worker.arrival_claims == 1
    assert worker.result["status"] != "goal_observed"
    reviewed = [event for event in events(worker) if event["event"] == "arrival_review_finished"]
    assert len(reviewed) == 1
    assert reviewed[0]["accepted"] is False
    assert reviewed[0]["doorway_entry"]["status"] == "unknown"


async def test_transient_pose_after_cloud_review_cannot_clear_before_fresh_frame(tmp_path):
    worker = mission(tmp_path)
    await start_gap(worker)
    await move_inside(worker)
    anchor = worker.robot.distance
    phase = 0
    observe = worker.robot.observe

    async def transient_observation():
        nonlocal phase
        if phase == 2:
            worker.robot.distance = anchor
        observed = await observe()
        if phase == 1:
            phase = 2
        return observed

    class TransientReview(ArrivalReviewer):
        async def review(self, goal, views):
            nonlocal phase
            verdict = await super().review(goal, views)
            worker.robot.distance += 0.03
            phase = 1
            return verdict

    worker.robot.observe = transient_observation
    worker.arrival_reviewer = TransientReview([True])
    result = await finish(worker)
    assert result["status"] == "entry_not_confirmed"
    assert result["doorway_entry"]["status"] == "unknown"
    worker.robot.distance = anchor
    assert worker.entry_observation()["status"] == "unknown"
    assert not any(
        event["event"] == "arrival_review_finished" and event["accepted"]
        for event in events(worker)
    )


@pytest.mark.parametrize("drift", [0, 0.03])
async def test_motion_tracking_precedes_delayed_image_send_and_rejects_later_drift(tmp_path, drift):
    worker = mission(tmp_path)
    await start_gap(worker)
    send = worker.session.send_realtime_input

    async def delayed_send(**kwargs):
        if "video" in kwargs:
            await asyncio.sleep(0.36)
            worker.robot.distance += drift
        await send(**kwargs)

    worker.session.send_realtime_input = delayed_send
    result = await worker.execute_tool("follow_gap", {})
    assert result["result"]["completed"] is True
    assert result["observation"]["doorway_entry"]["status"] == ("unknown" if drift else "outside")
    moved = [event for event in events(worker) if event["event"] == "doorway_entry_motion"]
    assert moved[-1]["doorway_entry"]["signed_outside_m"] == pytest.approx(0.8)


async def test_unreported_body_drift_during_head_sweep_prevents_visual_review(tmp_path):
    worker = mission(tmp_path)
    await start_gap(worker)
    await move_inside(worker)
    worker.arrival_reviewer = ArrivalReviewer([True])
    look = worker.robot.look_at

    async def untracked_drift(**kwargs):
        worker.robot.distance += 0.01
        return await look(**kwargs)

    worker.robot.look_at = untracked_drift
    result = await finish(worker)
    assert result["status"] == "entry_not_confirmed"
    assert result["doorway_entry"]["status"] == "unknown"
    assert not worker.arrival_reviewer.calls
    assert worker.arrival_claims == 1


async def test_transient_left_scan_drift_is_latched_before_later_head_scan_returns(tmp_path):
    worker = mission(tmp_path)
    await start_gap(worker)
    await move_inside(worker)
    worker.arrival_reviewer = ArrivalReviewer([True])
    look = worker.robot.look_at
    anchor = worker.robot.distance

    async def transient_drift(**kwargs):
        # A later scan returning to the anchor must not erase already observed
        # untracked movement. Only the left scan reports the displacement.
        worker.robot.distance = anchor + (0.03 if kwargs["y"] == 1 else 0)
        return await look(**kwargs)

    worker.robot.look_at = transient_drift
    result = await finish(worker)
    assert result["status"] == "entry_not_confirmed"
    assert result["doorway_entry"]["status"] == "unknown"
    assert not worker.arrival_reviewer.calls
    worker.robot.distance = anchor
    assert worker.entry_observation()["status"] == "unknown"


async def test_refused_replacement_geometry_keeps_preexisting_unknown_gate(tmp_path, monkeypatch):
    worker = mission(tmp_path)
    await start_gap(worker)
    reference = worker.entry_reference
    worker.robot.distance += 0.03
    assert worker.entry_observation()["status"] == "unknown"
    monkeypatch.setattr(
        "duck_nav.live.project_floor_point", lambda *_: {"odometry_position": [1, 0]}
    )
    result = await worker.execute_tool("advance_to_floor", gap_arguments(retain(worker)))
    assert result["result"]["completed"] is False
    assert worker.entry_reference is reference
    assert worker.entry_observation()["status"] == "unknown"
    assert (await finish(worker))["status"] == "entry_not_confirmed"


@pytest.mark.parametrize("tool", ["follow_gap", "advance_to_floor"])
async def test_cancellation_observed_with_fatal_guard_keeps_cancelled_status(tmp_path, tool):
    worker = mission(tmp_path)
    await start_gap(worker)
    reference = worker.entry_reference
    observe = worker.robot.observe
    observe_calls = 0

    async def cancel_during_observation():
        nonlocal observe_calls
        observe_calls += 1
        # follow_gap samples once before motion; paired floor targets sample once
        # before their cloud review and once immediately after the review.
        if observe_calls == (1 if tool == "follow_gap" else 2):
            worker.finish("cancelled", "Spoken stop")
            worker.robot.guard = "depth_too_close"
        return await observe()

    worker.robot.observe = cancel_during_observation
    args = {} if tool == "follow_gap" else gap_arguments(retain(worker))
    with pytest.raises(asyncio.CancelledError):
        await worker.execute_tool(tool, args)
    assert worker.result["status"] == "cancelled"
    assert worker.entry_reference is reference
    assert len(worker.robot.headings) == 1


class ClaimPlanner:
    async def decide(self, context, jpeg, *, views):
        return {"name": "finish", "args": {"status": "goal_observed", "reason": "Candidate room"}}


@pytest.mark.parametrize("guard", ["actuator_unhealthy", "state_stale", "depth_too_close"])
async def test_fatal_guard_precedes_entry_refusal_and_ends_with_stop(tmp_path, guard):
    worker = mission(tmp_path)
    await start_gap(worker)
    worker.robot.guard = guard
    worker.navigation_planner = ClaimPlanner()
    worker.arrival_reviewer = ArrivalReviewer([True])
    result = await worker.run()
    assert result["status"] != "goal_observed"
    assert worker.robot.calls[-1] == "stop"
    assert not worker.arrival_reviewer.calls
    assert not any(event["event"] == "entry_rejected" for event in events(worker))
    assert result["doorway_entry"] is not None


async def test_spoken_stop_cancels_inflight_motion_and_preserves_unknown_entry(tmp_path):
    worker = mission(tmp_path)
    await start_gap(worker)
    reference = worker.entry_reference
    worker.robot.advance_entered.clear()
    worker.robot.advance_release = asyncio.Event()
    worker.navigation_planner = GapPlanner(["follow_gap"])
    task = asyncio.create_task(worker.run())
    try:
        await asyncio.wait_for(worker.robot.advance_entered.wait(), 2)
        worker.session.push({"server_content": {"input_transcription": {"text": "Stop"}}})
        result = await asyncio.wait_for(task, 1)
        assert result["status"] == "cancelled"
        assert result["doorway_entry"]["status"] == "unknown"
        assert worker.entry_reference is reference
        assert worker.gap_plan is None
        assert worker.robot.calls[-1] == "stop"
        assert not any(event["event"] == "entry_rejected" for event in events(worker))
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def test_invalid_gap_review_records_only_safe_diagnostics_without_installing_entry(tmp_path):
    worker = mission(tmp_path)
    secret = "UNTRUSTED_RAW_PROVIDER_TEXT_MUST_NOT_BE_RECORDED"
    bad = InvalidGapAssessment(
        "report_keys",
        response={
            "candidates": [
                {
                    "finishReason": "STOP",
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "report_gap_review",
                                    "args": {
                                        "doorway_visible": True,
                                        "both_contacts_visible": True,
                                        "points_match_contacts": True,
                                        "evidence": secret,
                                        "best_supported_endpoints": [[321.987654, 222], [444, 555]],
                                        secret: "private",
                                    },
                                }
                            }
                        ]
                    },
                }
            ]
        },
    )

    class InvalidReviewer:
        async def review(self, *args, **kwargs):
            raise bad

    worker.gap_reviewer = InvalidReviewer()
    result = await worker.execute_tool("advance_to_floor", gap_arguments(retain(worker)))
    assert result["result"]["completed"] is False
    assert result["result"]["reason"] == "floor_target_refused"
    assert worker.robot.calls[-1] == "stop"
    recorded = events(worker)
    rejected = [event for event in recorded if event["event"] == "gap_review_rejected"]
    assert len(rejected) == 1
    assert rejected[0]["diagnostics"] == bad.diagnostics
    assert bad.diagnostics["unknown_report_key_count"] == 1
    serialized = json.dumps(recorded)
    assert secret not in serialized
    assert "321.987654" not in serialized
    assert worker.entry_reference is None
    assert worker.gap_plan is None
    assert not worker.robot.headings
