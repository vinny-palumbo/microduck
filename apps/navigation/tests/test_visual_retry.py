"""A malformed visual reply may refresh once, but cannot trigger hidden motion."""

import asyncio
import copy
import json
import math
from pathlib import Path

import pytest
from test_floor_target_live import FloorRobot, arguments
from test_live import Session, run

from duck_nav.live import LiveConfig, LiveMission
from duck_nav.planning import InvalidVisualDecision

UNTRUSTED_REPLY = "RAW_REJECTED_REPLY_MUST_NOT_BECOME_AN_INSTRUCTION"


def invalid_reply():
    return InvalidVisualDecision(
        "candidate_count",
        response={
            "candidates": [
                {"finishReason": "STOP", "content": {"parts": [{"text": UNTRUSTED_REPLY}]}}
            ]
        },
    )


class ChangingCameraRobot(FloorRobot):
    def __init__(self):
        super().__init__()
        self.frames = 0
        self.motion = {"requested": [0, 0, 0], "applied": [0, 0, 0]}

    def snapshot(self):
        sample = super().snapshot()
        self.frames += 1
        sample["camera"]["image"][:, :, 0] = (self.frames * 25) % 256
        sample["state"]["data"]["move"] = copy.deepcopy(self.motion)
        return sample


class Planner:
    model = "visual-retry-fixture"

    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.contexts, self.images, self.views = [], [], []

    async def decide(self, context, jpeg, *, views=None):
        self.contexts.append(copy.deepcopy(context))
        self.images.append(jpeg)
        self.views.append(copy.deepcopy(views))
        outcome = next(self.outcomes)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


@pytest.fixture(autouse=True)
def fast_images(monkeypatch):
    original = LiveMission.image

    async def fresh(worker):
        # Keep actual snapshots, view retention, JPEG creation and recording.
        # Separate tests exercise the production one-second throttle.
        worker.last_image = -math.inf
        return await original(worker)

    monkeypatch.setattr(LiveMission, "image", fresh)


def recorded_events(recorder):
    return [json.loads(line) for line in (recorder.path / "events.jsonl").read_text().splitlines()]


async def test_invalid_reply_refreshes_images_and_context_without_spending_another_action(tmp_path):
    bad = invalid_reply()
    planner = Planner(
        [bad, {"name": "advance", "args": {"distance_m": 0.1, "reason": "Visible clear floor"}}]
    )
    robot = ChangingCameraRobot()
    result, _, recorder = await run(
        tmp_path,
        Session(),
        robot,
        goal="Kitchen",
        navigation_planner=planner,
        config=LiveConfig(max_actions=1),
    )
    assert result["status"] == "action_limit"
    assert result["actions"] == result["navigation_actions"] == 1
    assert robot.calls == ["stop", "advance", "stop"]
    assert len(planner.contexts) == 2
    assert planner.images[0] != planner.images[1]
    first, second = [context["camera"] for context in planner.contexts]
    assert first["view_id"] != second["view_id"]
    assert second["received_at"] > first["received_at"]
    assert all(context["recent_actions"] == [] for context in planner.contexts)
    assert [context["step"] for context in planner.contexts] == [1, 1]
    assert UNTRUSTED_REPLY not in json.dumps(planner.contexts)
    assert "invalid Gemini" not in json.dumps(planner.contexts)
    events = recorded_events(recorder)
    rejected = [e for e in events if e["event"] == "visual_reply_rejected"]
    assert len(rejected) == 1
    assert rejected[0]["diagnostics"] == bad.diagnostics
    assert rejected[0]["will_retry"] is True
    selected = [e for e in events if e["event"] == "visual_views_selected"]
    assert [e["attempt"] for e in selected] == [1, 2]
    assert [e["context"] for e in selected] == planner.contexts
    for event, image in zip(selected, planner.images, strict=True):
        assert Path(event["current"]["image_path"]).read_bytes() == image
    assert sum(e["event"] == "visual_decision" for e in events) == 1
    assert sum(e["event"] == "tool_requested" for e in events) == 1


async def test_rejected_front_view_cannot_be_reused_as_a_hidden_floor_source(tmp_path, monkeypatch):
    class StalePointPlanner(Planner):
        async def decide(self, context, jpeg, *, views=None):
            if self.contexts:
                self.outcomes = iter(
                    [
                        {
                            "name": "advance_to_floor",
                            "args": arguments(self.contexts[0]["camera"]["view_id"]),
                        }
                    ]
                )
            return await super().decide(context, jpeg, views=views)

    def forbidden_projection(*_):
        raise AssertionError("An unsupplied rejected frame must fail before pixel projection")

    monkeypatch.setattr("duck_nav.live.project_floor_point", forbidden_projection)
    planner = StalePointPlanner([invalid_reply()])
    result, robot, recorder = await run(
        tmp_path,
        Session(),
        ChangingCameraRobot(),
        goal="Kitchen",
        navigation_planner=planner,
        config=LiveConfig(max_actions=1),
    )
    assert result["status"] == "action_limit"
    assert "advance" not in robot.calls
    assert len(planner.contexts) == 2
    first_id = planner.contexts[0]["camera"]["view_id"]
    assert first_id != planner.contexts[1]["camera"]["view_id"]
    assert all(view["camera"]["view_id"] != first_id for view in planner.views[1])
    refusal = next(e for e in recorded_events(recorder) if e["event"] == "floor_target_refused")
    assert refusal["result"]["detail"] == "floor_view_not_supplied"


async def test_second_invalid_reply_ends_mission_and_stops_without_third_request(tmp_path):
    planner = Planner([invalid_reply(), invalid_reply()])
    result, robot, recorder = await run(
        tmp_path, Session(), ChangingCameraRobot(), goal="Kitchen", navigation_planner=planner
    )
    assert result["status"] == "error"
    assert result["actions"] == result["navigation_actions"] == 1
    assert robot.calls == ["stop", "stop"]
    assert len(planner.contexts) == 2
    rejected = [e for e in recorded_events(recorder) if e["event"] == "visual_reply_rejected"]
    assert [(e["attempt"], e["will_retry"]) for e in rejected] == [(1, True), (2, False)]


@pytest.mark.parametrize(
    "error",
    [
        ValueError("ordinary validation failure"),
        ConnectionError("network unavailable"),
        TimeoutError("network timeout"),
    ],
)
async def test_non_reply_errors_are_not_retried(tmp_path, error):
    planner = Planner([error])
    result, robot, recorder = await run(
        tmp_path, Session(), ChangingCameraRobot(), goal="Kitchen", navigation_planner=planner
    )
    assert result["status"] == "error"
    assert robot.calls == ["stop", "stop"]
    assert len(planner.contexts) == 1
    assert not any(e["event"] == "visual_reply_rejected" for e in recorded_events(recorder))


async def test_successful_observation_after_retry_uses_normal_no_progress_accounting(tmp_path):
    planner = Planner(
        [invalid_reply(), {"name": "observe", "args": {"reason": "Inspect fresh floor"}}]
    )
    result, robot, _ = await run(
        tmp_path,
        Session(),
        ChangingCameraRobot(),
        goal="Kitchen",
        navigation_planner=planner,
        config=LiveConfig(max_actions=1),
    )
    assert result["status"] == "action_limit"
    assert result["progress_budget"]["observations_without_progress"] == 1
    assert result["actions"] == 1
    assert robot.calls == ["stop", "stop"]


@pytest.mark.parametrize("unsafe", ["fallen", "requested_motion", "applied_motion"])
async def test_new_fatal_guard_or_motion_prevents_second_planner_request(tmp_path, unsafe):
    robot = ChangingCameraRobot()

    class UnsafeRetryPlanner(Planner):
        async def decide(self, *args, **kwargs):
            if unsafe == "fallen":
                robot.guard = "fallen"
            elif unsafe == "requested_motion":
                robot.motion["requested"] = [0.1, 0, 0]
            else:
                robot.motion["applied"] = [0, 0, 0.001]
            return await super().decide(*args, **kwargs)

    planner = UnsafeRetryPlanner([invalid_reply()])
    result, _, _ = await run(tmp_path, Session(), robot, goal="Kitchen", navigation_planner=planner)
    assert result["status"] == "error"
    assert len(planner.contexts) == 1
    assert robot.calls[-1] == "stop"
    assert all(call == "stop" for call in robot.calls)


@pytest.mark.parametrize("cancellation", ["spoken", "stop_tool", "disconnect"])
async def test_stop_cancels_pending_retry_model_without_dispatch(tmp_path, cancellation):
    entered, cancelled = asyncio.Event(), asyncio.Event()

    class WaitingRetryPlanner(Planner):
        async def decide(self, context, jpeg, *, views=None):
            if self.contexts:
                entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()
            return await super().decide(context, jpeg, views=views)

    planner, session, robot = (
        WaitingRetryPlanner([invalid_reply()]),
        Session(),
        ChangingCameraRobot(),
    )
    task = asyncio.create_task(
        run(tmp_path, session, robot, goal="Kitchen", navigation_planner=planner)
    )
    await asyncio.wait_for(entered.wait(), 1)
    if cancellation == "spoken":
        session.push({"server_content": {"input_transcription": {"text": "Stop"}}})
    elif cancellation == "stop_tool":
        session.call("stop", reason="User asked to stop")
    else:
        session.push(ConnectionError("Voice connection lost"))
    result, _, _ = await asyncio.wait_for(task, 0.5)
    assert result["status"] == ("error" if cancellation == "disconnect" else "cancelled")
    assert cancelled.is_set()
    assert robot.calls == ["stop", "stop"]


async def test_stop_cancels_retry_while_waiting_for_its_fresh_image(tmp_path, monkeypatch):
    entered, cancelled = asyncio.Event(), asyncio.Event()
    planner = Planner([invalid_reply()])
    original = LiveMission.image

    async def blocked_retry_image(worker):
        if planner.contexts:
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        return await original(worker)

    monkeypatch.setattr(LiveMission, "image", blocked_retry_image)
    session, robot = Session(), ChangingCameraRobot()
    task = asyncio.create_task(
        run(tmp_path, session, robot, goal="Kitchen", navigation_planner=planner)
    )
    await asyncio.wait_for(entered.wait(), 1)
    session.push({"server_content": {"input_transcription": {"text": "Stop"}}})
    result, _, _ = await asyncio.wait_for(task, 0.5)
    assert result["status"] == "cancelled"
    assert cancelled.is_set()
    assert robot.calls == ["stop", "stop"]
    assert len(planner.contexts) == 1
