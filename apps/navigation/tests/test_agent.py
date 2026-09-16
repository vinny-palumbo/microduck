"""Mission failures must stop; fixtures never count as visual-model validation."""

import asyncio
import base64
import copy
import json
import time
from types import SimpleNamespace

import numpy as np
import pytest

from duck_nav.agent import GeminiPlanner, fresh_observation, mission, validate_decision
from duck_nav.cli import TOOLS, Recorder


def decision(tool="observe", **arguments):
    return {"tool": tool, "arguments": arguments, "reason": "Fixture decision"}


class Robot:
    def __init__(self):
        self.config = SimpleNamespace(camera_max_age_s=1)
        self.calls = []
        self.distance = 0
        self.advance = 0
        self.reason = None
        self.completed = True
        self.stop_ok = True
        self.frozen = False
        self.created = time.monotonic() - 10

    def snapshot(self):
        now = self.created if self.frozen else time.monotonic()
        return {
            "connected": True,
            "camera": {"received_at": now, "image": np.zeros((24, 32, 3), dtype=np.uint8)},
            "state": {
                "received_at": now,
                "data": {
                    "odom": {"position": [self.distance, 0, 0.12], "yaw": 0},
                    "move": {"requested": [0, 0, 0], "applied": [0, 0, 0]},
                    "simulator_truth": "must never reach the planner",
                },
            },
        }

    async def observe(self):
        return {
            "ready": self.reason is None,
            "guard_reason": self.reason,
            "depth_summary": {},
            "state": self.snapshot()["state"],
        }

    async def stop(self):
        self.calls.append("stop")
        return {"completed": self.stop_ok, "reason": "stopped"}

    async def move_for(self, speed_m_s, duration_s):
        self.calls.append("move_for")
        self.distance += self.advance
        return {"completed": self.completed, "reason": "duration_elapsed"}

    async def look_at(self, x, y, z):
        self.calls.append("look_at")
        return {"completed": True, "reason": "gaze_settled"}


class Planner:
    model = "scripted-fixture"

    def __init__(self, decisions):
        self.decisions = iter(decisions)
        self.contexts = []
        self.images = []

    async def decide(self, context, jpeg):
        self.contexts.append(copy.deepcopy(context))
        self.images.append(jpeg)
        return next(self.decisions)


async def execute(tmp_path, decisions, robot=None, **kwargs):
    robot = robot or Robot()
    planner = Planner(decisions)
    recorder = Recorder(tmp_path)
    result = await mission(robot, robot, planner, recorder, "Inspect the room", **kwargs)
    return result, robot, planner, recorder


@pytest.mark.asyncio
async def test_serial_image_result_and_unverified_finish(tmp_path):
    robot = Robot()
    robot.advance = 0.02
    result, robot, planner, recorder = await execute(
        tmp_path,
        [
            decision("move_for", speed_m_s=0.05, duration_s=0.5),
            decision("finish", status="goal_observed"),
        ],
        robot,
    )
    assert result["status"] == "goal_observed"
    assert result["goal_verified"] is False
    assert result["decisions"] == 2
    assert robot.calls == ["stop", "move_for", "stop"]
    assert all(image.startswith(b"\xff\xd8") for image in planner.images)
    assert planner.contexts[1]["recent_actions"][0]["progress"]["estimated_displacement_m"] == 0.02
    assert "simulator_truth" not in json.dumps(planner.contexts)
    assert json.loads((recorder.path / "mission.json").read_text()) == result
    assert len(list(recorder.path.glob("frame-*.jpg"))) >= 3


@pytest.mark.asyncio
@pytest.mark.parametrize("completed", [True, False])
async def test_two_stalls_stop_even_with_intervening_look(tmp_path, completed):
    robot = Robot()
    robot.completed = completed
    result, robot, planner, _ = await execute(
        tmp_path,
        [
            decision("move_for", speed_m_s=0.05, duration_s=0.5),
            decision("look_at", x=1, y=0, z=0),
            decision("move_for", speed_m_s=0.05, duration_s=0.5),
            decision("finish", status="goal_observed"),
        ],
        robot,
    )
    assert result["status"] == "blocked"
    assert result["decisions"] == 3
    assert len(planner.contexts) == 3
    assert robot.calls[-1] == "stop"


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["stale_camera", "disconnected", "invalid_depth"])
async def test_bad_observation_never_calls_model(tmp_path, reason):
    robot = Robot()
    robot.reason = reason
    result, robot, planner, _ = await execute(tmp_path, [], robot)
    assert result["status"] == "error"
    assert not planner.contexts
    assert robot.calls == ["stop", "stop"]


@pytest.mark.asyncio
async def test_obstacle_remains_visible_to_planner(tmp_path):
    robot = Robot()
    robot.reason = "obstacle"
    result, _, planner, _ = await execute(tmp_path, [decision("finish", status="blocked")], robot)
    assert result["status"] == "blocked"
    assert planner.contexts[0]["guard_reason"] == "obstacle"


@pytest.mark.asyncio
async def test_frozen_frame_is_not_reused():
    robot = Robot()
    robot.frozen = True
    with pytest.raises(TimeoutError):
        await fresh_observation(robot, robot, time.monotonic(), timeout=0.03)


@pytest.mark.asyncio
async def test_bad_tool_stops_without_dispatch(tmp_path):
    result, robot, _, _ = await execute(tmp_path, [decision("enable_motors")])
    assert result["status"] == "error"
    assert robot.calls == ["stop", "stop"]


@pytest.mark.asyncio
async def test_step_budget_stops(tmp_path):
    result, robot, planner, _ = await execute(tmp_path, [decision(), decision()], max_steps=1)
    assert result["status"] == "step_limit"
    assert len(planner.contexts) == 1
    assert robot.calls[-1] == "stop"


@pytest.mark.asyncio
async def test_failed_stop_cannot_be_reported_as_success(tmp_path):
    robot = Robot()
    robot.stop_ok = False
    result, _, planner, _ = await execute(
        tmp_path, [decision("finish", status="goal_observed")], robot
    )
    assert result["status"] == "error"
    assert not planner.contexts


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [True, False])
async def test_cancel_or_deadline_during_model_requests_stop(tmp_path, cancel):
    entered = asyncio.Event()

    class WaitingPlanner:
        model = "scripted-fixture"

        async def decide(self, context, jpeg):
            entered.set()
            await asyncio.Event().wait()

    robot = Robot()
    recorder = Recorder(tmp_path)
    task = asyncio.create_task(
        mission(robot, robot, WaitingPlanner(), recorder, "Inspect", max_seconds=1)
    )
    await entered.wait()
    if cancel:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        assert (await task)["status"] == "timeout"
    assert robot.calls == ["stop", "stop"]
    saved = json.loads((recorder.path / "mission.json").read_text())
    assert saved["status"] == ("cancelled" if cancel else "timeout")


@pytest.mark.parametrize(
    "bad",
    [
        decision("move_for", speed_m_s=True, duration_s=1),
        decision("turn_by", angle_deg=float("nan")),
        decision("observe", injected="text"),
        decision("finish", status=[]),
        decision("look_at", x=1),
    ],
)
def test_malformed_decisions_rejected(bad):
    with pytest.raises((ValueError, TypeError)):
        validate_decision(bad)


def response(calls, finish="STOP"):
    return {
        "candidates": [
            {"finishReason": finish, "content": {"parts": [{"functionCall": c} for c in calls]}}
        ]
    }


def test_provider_image_schema_and_response():
    original = copy.deepcopy(TOOLS)
    planner = GeminiPlanner("test-only-key")
    payload = planner.payload({"goal": "inspect"}, b"jpeg")
    assert base64.b64decode(payload["contents"][0]["parts"][1]["inlineData"]["data"]) == b"jpeg"
    assert "test-only-key" not in json.dumps(payload)
    assert payload["toolConfig"]["functionCallingConfig"]["mode"] == "ANY"
    assert TOOLS == original
    parsed = planner.parse(
        response(
            [{"name": "finish", "args": {"status": "blocked", "reason": "Doorway is obstructed"}}]
        )
    )
    assert parsed["arguments"] == {"status": "blocked"}
    assert parsed["reason"] == "Doorway is obstructed"


@pytest.mark.parametrize(
    "calls, finish",
    [
        ([], "STOP"),
        ([{"name": "observe"}, {"name": "move_for"}], "STOP"),
        ([{"name": "observe", "args": {"reason": "inspect"}}], "MAX_TOKENS"),
    ],
)
def test_provider_rejects_missing_parallel_or_truncated_calls(calls, finish):
    with pytest.raises(ValueError):
        GeminiPlanner.parse(response(calls, finish))
