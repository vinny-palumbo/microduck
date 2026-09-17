"""Exercise actual receive-turn lifetimes and concurrent interruption of movement."""

import asyncio
import copy
import io
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from duck_nav.cli import Recorder
from duck_nav.live import (
    LiveConfig,
    LiveMission,
    StationaryViews,
    connect_config,
    declarations,
    is_spoken_stop,
    parser,
    run_live,
    visual_declarations,
)


class Robot:
    def __init__(self):
        self.config = SimpleNamespace(camera_max_age_s=1)
        self.calls = []
        self.distance = 0
        self.connected = True
        self.stop_ok = True
        self.advance_ok = True
        self.advance_entered = asyncio.Event()
        self.advance_release = None

    def snapshot(self):
        now = time.monotonic()
        return {
            "connected": self.connected,
            "camera": {"received_at": now, "image": np.zeros((24, 32, 3), dtype=np.uint8)},
            "state": {
                "received_at": now,
                "data": {
                    "odom": {"position": [self.distance, 0, 0.12], "yaw": 0},
                    "move": {"requested": [0, 0, 0], "applied": [0, 0, 0]},
                    "simulator_truth": "private position must not reach the model",
                },
            },
        }

    async def observe(self):
        return {
            "ready": True,
            "guard_reason": None,
            "depth_summary": {},
            "state": self.snapshot()["state"],
        }

    async def stop(self):
        self.calls.append("stop")
        return {"completed": self.stop_ok}

    async def advance(self, distance_m, heading_deg=0):
        self.calls.append("advance")
        self.advance_entered.set()
        if self.advance_release:
            await self.advance_release.wait()
        if self.advance_ok:
            self.distance += distance_m
        return {"completed": self.advance_ok, "reason": "distance_reached"}

    async def look_at(self, x, y, z):
        self.calls.append("look_at")
        return {"completed": True}


class Session:
    """Mimic google-genai: receive() returns after each model turn, not on tool calls."""

    def __init__(self):
        self.messages = asyncio.Queue()
        self.responses = []
        self.inputs = []
        self.prompts = []
        self.receive_count = 0
        self.next_id = 0
        self.on_prompt = None
        self.on_response = None
        self.on_audio = None

    def push(self, message):
        self.messages.put_nowait(message)

    def call(self, tool_name, **arguments):
        self.next_id += 1
        self.push(
            {
                "tool_call": {
                    "function_calls": [
                        {
                            "id": str(self.next_id),
                            "name": tool_name,
                            "args": arguments,
                        }
                    ]
                }
            }
        )

    def complete(self):
        self.push({"server_content": {"turn_complete": True}})

    async def receive(self):
        self.receive_count += 1
        while True:
            message = await self.messages.get()
            if isinstance(message, Exception):
                raise message
            yield message
            if message.get("server_content", {}).get("turn_complete"):
                return

    async def send_realtime_input(self, **input):
        self.inputs.append(input)
        if "audio" in input and self.on_audio:
            await self.on_audio(input)

    async def send_client_content(self, **content):
        self.prompts.append(content)
        if self.on_prompt:
            await self.on_prompt(content)

    async def send_tool_response(self, **response):
        self.responses.append(copy.deepcopy(response))
        if self.on_response:
            await self.on_response(response["function_responses"][0])


async def silent_audio():
    yield bytes(3200)


async def run(tmp_path, session, robot=None, **kwargs):
    robot = robot or Robot()
    recorder = Recorder(tmp_path)
    result = await run_live(robot, robot, session, recorder, **kwargs)
    saved = json.loads((recorder.path / "mission.json").read_text())
    assert result == saved
    return result, robot, recorder


async def test_spoken_goal_then_multiple_turns_and_entered_goal(tmp_path):
    session = Session()

    async def audio(_):
        session.on_audio = None
        session.push(
            {
                "server_content": {
                    "input_transcription": {
                        "text": "Go to the kitchen",
                        "finished": True,
                    }
                }
            }
        )
        session.call("start_navigation", goal="Go to the kitchen")

    async def prompt(_):
        session.call("advance", distance_m=0.1, heading_deg=0, reason="Clear doorway ahead")

    async def response(response):
        if response["name"] == "start_navigation":
            session.complete()
        elif response["name"] == "advance":
            session.push({"server_content": {"output_transcription": {"text": "I see a sink."}}})
            session.call(
                "remember_place", name="kitchen", observation="Sink beyond threshold", explored=True
            )
        elif response["name"] == "remember_place":
            session.call(
                "finish", status="goal_observed", reason="Inside the room beside a sink and stove"
            )

    session.on_audio, session.on_prompt, session.on_response = audio, prompt, response
    result, robot, recorder = await run(tmp_path, session, audio=silent_audio())
    assert result["status"] == "goal_observed"
    assert result["goal"] == "Go to the kitchen"
    assert result["goal_verified"] is False
    assert result["remembered_places"]["kitchen"]["explored"] is True
    assert session.receive_count >= 2
    assert robot.calls == ["stop", "advance", "stop"]
    assert "simulator_truth" not in json.dumps(session.responses + session.prompts)
    assert "I see a sink" in (recorder.path / "events.jsonl").read_text()
    images = [i["video"] for i in session.inputs if "video" in i]
    assert all(i["data"].startswith(b"\xff\xd8") for i in images)
    assert any(i.get("audio_stream_end") for i in session.inputs)


@pytest.mark.parametrize(
    "event", ["spoken", "tool_cancel", "interrupted", "disconnect", "operator"]
)
async def test_pending_action_cannot_block_stop(tmp_path, event):
    session, robot = Session(), Robot()
    robot.advance_release = asyncio.Event()

    async def prompt(_):
        session.call("advance", distance_m=0.1, reason="Clear floor")

    session.on_prompt = prompt
    task = asyncio.create_task(run(tmp_path, session, robot, goal="Go to kitchen"))
    await asyncio.wait_for(robot.advance_entered.wait(), 1)
    if event == "spoken":
        session.push({"server_content": {"input_transcription": {"text": "Stop!"}}})
    elif event == "tool_cancel":
        session.push({"tool_call_cancellation": {"ids": ["1"]}})
    elif event == "interrupted":
        session.push({"server_content": {"interrupted": True}})
    elif event == "disconnect":
        robot.connected = False
    else:
        task.cancel()
    if event == "operator":
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 0.5)
    else:
        result, _, _ = await asyncio.wait_for(task, 0.5)
        assert result["status"] == ("disconnected" if event == "disconnect" else "cancelled")
    assert robot.calls[-1] == "stop"
    assert not session.responses


async def test_heartbeat_waits_for_model_turn_complete(tmp_path):
    session = Session()
    task = asyncio.create_task(
        run(
            tmp_path,
            session,
            goal="Kitchen",
            config=LiveConfig(model_timeout_s=3),
        )
    )
    await asyncio.sleep(1.15)
    assert len(session.prompts) == 1  # More images must not barge into the model's current turn.
    session.complete()

    async def finish(_):
        session.call("finish", status="blocked", reason="No clear route")

    session.on_prompt = finish
    result, _, _ = await asyncio.wait_for(task, 2)
    assert result["status"] == "blocked"
    assert len(session.prompts) == 2


@pytest.mark.parametrize("cause", ["silent_model", "closed_model", "no_instruction", "budget"])
async def test_deadlines_and_connection_failure_stop(tmp_path, cause):
    session = Session()
    config = LiveConfig(model_timeout_s=0.1, instruction_timeout_s=0.1, max_seconds=0.2)
    kwargs = {"goal": "Kitchen", "config": config}
    if cause == "closed_model":
        session.push(ConnectionError("Disconnected"))
    elif cause == "no_instruction":
        kwargs = {"audio": silent_audio(), "config": config}
    elif cause == "budget":
        kwargs["config"] = LiveConfig(model_timeout_s=3, max_seconds=0.1)
    result, robot, _ = await run(tmp_path, session, **kwargs)
    assert (
        result["status"]
        == {
            "silent_model": "model_timeout",
            "closed_model": "error",
            "no_instruction": "instruction_timeout",
            "budget": "timeout",
        }[cause]
    )
    assert robot.calls == ["stop", "stop"]


async def test_failed_motion_ends_without_second_attempt(tmp_path):
    session, robot = Session(), Robot()
    robot.advance_ok = False

    async def prompt(_):
        session.call("advance", distance_m=0.1, reason="Clear floor")

    session.on_prompt = prompt
    result, robot, _ = await run(tmp_path, session, robot, goal="Kitchen")
    assert result["status"] == "blocked"
    assert robot.calls.count("advance") == 1


class RecoveryRobot(Robot):
    def __init__(self, failures, *, stop_acknowledged=True, keep_obstacle=False):
        super().__init__()
        self.failures = list(failures)
        self.guard_reason = None
        self.stop_acknowledged = stop_acknowledged
        self.keep_obstacle = keep_obstacle

    async def observe(self):
        return {
            **await super().observe(),
            "ready": self.guard_reason is None,
            "guard_reason": self.guard_reason,
        }

    async def advance(self, distance_m, heading_deg=0):
        cause = self.failures.pop(0) if self.failures else None
        if cause is None:
            return await super().advance(distance_m, heading_deg)
        self.calls.append("advance")
        self.advance_entered.set()
        self.guard_reason = (
            cause if cause in {"obstacle", "head_not_forward", "depth_quality"} else None
        )
        return {
            "completed": False,
            "reason": cause,
            "stop": {
                "acknowledged": self.stop_acknowledged,
                "physical_settling_verified": cause == "no_progress",
            },
        }

    async def look_at(self, x, y, z):
        result = await super().look_at(x, y, z)
        self.guard_reason = "head_not_forward" if y else "obstacle" if self.keep_obstacle else None
        return result


async def test_recoverable_stop_inspect_recenter_then_alternative_movement(tmp_path):
    session = Session()
    robot = RecoveryRobot(["obstacle"])
    sequence = iter(
        [
            ("advance", {"distance_m": 0.1, "heading_deg": 0, "reason": "Possible doorway"}),
            (
                "look_at",
                {"x": 1, "y": -0.5, "z": 0, "reason": "Inspect floor through right opening"},
            ),
            ("look_at", {"x": 1, "y": 0, "z": 0, "reason": "Recenter before the clear arc"}),
            (
                "advance",
                {"distance_m": 0.1, "heading_deg": -15, "reason": "Clear floor through doorway"},
            ),
            ("finish", {"status": "goal_observed", "reason": "Inside kitchen beside sink"}),
        ]
    )

    async def next_call(_):
        tool, args = next(sequence)
        session.call(tool, **args)

    async def response(response):
        if response["name"] != "finish":
            await next_call(None)

    session.on_prompt, session.on_response = next_call, response
    result, robot, _ = await run(tmp_path, session, robot, goal="Kitchen")
    assert result["status"] == "goal_observed"
    assert robot.calls == ["stop", "advance", "look_at", "look_at", "advance", "stop"]
    first = session.responses[0]["function_responses"][0]["response"]
    assert first["result"]["reason"] == "obstacle"
    assert first["recovery"]["consecutive_refusals"] == 1
    assert first["observation"]["ready"] is False
    recovered = session.responses[3]["function_responses"][0]["response"]
    assert recovered["observation"]["recovery"] is None


async def recovery_mission(tmp_path, robot):
    mission = LiveMission(
        robot,
        robot,
        Session(),
        Recorder(tmp_path),
        audio=None,
        goal="Kitchen",
        config=LiveConfig(),
        speak=None,
        emit=lambda event: None,
    )

    async def no_image():
        return robot.snapshot(), await robot.observe()

    mission.image = no_image
    return mission


@pytest.mark.parametrize("cause", ["obstacle", "head_not_forward", "depth_quality", "no_progress"])
async def test_recoverable_refusals_require_inspection_and_have_finite_budget(tmp_path, cause):
    robot = RecoveryRobot([cause])
    mission = await recovery_mission(tmp_path, robot)
    args = {"distance_m": 0.1, "reason": "Check route"}
    first = await mission.execute_tool("advance", args)
    assert first["recovery"]["remaining_refusals_before_stop"] == 2
    # Changing arguments is not permission to skip inspection after a refusal.
    second = await mission.execute_tool("advance", {**args, "heading_deg": 10})
    assert second["result"]["reason"] == "recovery_inspection_required"
    await mission.execute_tool("advance", {**args, "heading_deg": -10})
    assert mission.result["status"] == "blocked"
    assert mission.recoverable_failures == 3
    assert robot.calls.count("advance") == 1


async def test_same_failed_action_needs_cleared_guard_not_just_new_look(tmp_path):
    robot = RecoveryRobot(["no_progress"])
    mission = await recovery_mission(tmp_path, robot)
    args = {"distance_m": 0.1, "reason": "Inspect route"}
    await mission.execute_tool("advance", args)
    await mission.execute_tool("look_at", {"x": 1, "y": 0, "z": 0, "reason": "Reassess floor"})
    refused = await mission.execute_tool("advance", args)
    assert refused["result"]["reason"] == "recovery_unchanged_action"
    assert robot.calls.count("advance") == 1
    result = await mission.execute_tool("advance", {**args, "heading_deg": 10})
    assert result["result"]["completed"] is True
    assert mission.recoverable_failures == 0


async def test_inspection_cannot_override_current_obstacle_guard(tmp_path):
    robot = RecoveryRobot(["obstacle"], keep_obstacle=True)
    mission = await recovery_mission(tmp_path, robot)
    args = {"distance_m": 0.1, "reason": "Possible doorway"}
    await mission.execute_tool("advance", args)
    await mission.execute_tool("look_at", {"x": 1, "y": 0, "z": 0, "reason": "Inspect wall"})
    refused = await mission.execute_tool("advance", {**args, "heading_deg": 15})
    assert refused["result"]["reason"] == "recovery_guard_not_ready"
    assert refused["observation"]["ready"] is False
    assert robot.calls.count("advance") == 1


@pytest.mark.parametrize("guard_reason", ["depth_too_close", "stale_health", "unhealthy"])
async def test_fatal_recovery_observation_is_preserved_even_if_it_clears(
    tmp_path,
    monkeypatch,
    guard_reason,
):
    robot = RecoveryRobot(["obstacle"])
    mission = await recovery_mission(tmp_path, robot)
    args = {"distance_m": 0.1, "reason": "Possible doorway"}
    await mission.execute_tool("advance", args)
    mission.recovery["inspected"] = True
    robot.guard_reason = guard_reason
    original_observe = robot.observe

    async def transient_observation():
        observed = await original_observe()
        robot.guard_reason = None
        return observed

    async def must_not_wait_for_another_observation(*args, **kwargs):
        pytest.fail("A fatal guard must terminate after stop, before fresh observation")

    robot.observe = transient_observation
    monkeypatch.setattr("duck_nav.live.fresh_observation", must_not_wait_for_another_observation)
    refused = await mission.execute_tool("advance", {**args, "heading_deg": 15})
    assert refused["result"]["reason"] == guard_reason
    assert refused["result"]["guard_reason"] == guard_reason
    assert refused["result"]["fatal_guard"] is True
    assert refused["result"]["retry_refused"] is False
    assert refused["result"]["stop"]["acknowledged"] is True
    assert mission.result["status"] == "blocked"
    assert guard_reason in mission.result["reason"]
    assert mission.recovery is None
    assert robot.calls == ["advance", "stop"]


@pytest.mark.parametrize("reported_reason", ["recovery_guard_not_ready", "obstacle"])
async def test_retry_refused_flag_cannot_downgrade_a_fatal_guard(tmp_path, reported_reason):
    robot = RecoveryRobot(["obstacle"])
    mission = await recovery_mission(tmp_path, robot)
    await mission.execute_tool("advance", {"distance_m": 0.1, "reason": "Possible doorway"})
    # Even if a future retry path mislabels this response, classification must use
    # its captured guard rather than the now-healthy observation.
    robot.guard_reason = None
    mission.movement_result(
        {
            "arguments": {"distance_m": 0.1},
            "progress": {"negligible": True},
            "result": {
                "completed": False,
                "reason": reported_reason,
                "retry_refused": True,
                "guard_reason": "depth_too_close",
                "stop": {"acknowledged": True},
            },
        },
        await robot.observe(),
    )
    assert mission.result["status"] == "blocked"
    assert mission.recoverable_failures == 1


@pytest.mark.parametrize(
    "cause,acknowledged",
    [
        ("stale_health", True),
        ("stale_camera", True),
        ("disconnected", True),
        ("cancelled", True),
        ("obstacle", False),
    ],
)
async def test_fatal_motion_failures_do_not_enter_recovery(tmp_path, cause, acknowledged):
    robot = RecoveryRobot([cause], stop_acknowledged=acknowledged)
    mission = await recovery_mission(tmp_path, robot)
    await mission.execute_tool("advance", {"distance_m": 0.1, "reason": "Attempt route"})
    assert mission.result["status"] == "blocked"
    assert mission.recovery is None
    assert robot.calls.count("advance") == 1


async def test_action_budget_and_failed_stop_never_claim_arrival(tmp_path):
    session, robot = Session(), Robot()

    async def prompt(_):
        session.call("say", message="I will inspect the kitchen")

    session.on_prompt = prompt
    result, _, _ = await run(
        tmp_path, session, robot, goal="Kitchen", config=LiveConfig(max_actions=1)
    )
    assert result["status"] == "action_limit"
    robot.stop_ok = False
    result, _, _ = await run(tmp_path, Session(), robot, goal="Kitchen")
    assert result["status"] == "error"
    assert result["reason"] == "Final stop was not acknowledged"


async def test_motion_before_user_instruction_is_rejected(tmp_path):
    session = Session()
    session.call("advance", distance_m=0.1, reason="Invented mission")
    result, robot, _ = await run(tmp_path, session, audio=silent_audio())
    assert result["status"] == "error"
    assert robot.calls == ["stop", "stop"]


async def test_speech_tool_awaits_actual_playback(tmp_path):
    session = Session()
    spoken = []

    async def speak(message):
        spoken.append(message)

    async def prompt(_):
        session.call("say", message="I have reached the kitchen")

    async def response(_):
        assert spoken == ["I have reached the kitchen"]
        session.call("finish", status="goal_observed", reason="Inside by stove")

    session.on_prompt, session.on_response = prompt, response
    result, _, _ = await run(tmp_path, session, goal="Kitchen", speak=speak)
    assert result["status"] == "goal_observed"


async def test_next_serial_tool_can_arrive_before_response_send_resumes(tmp_path):
    session = Session()

    async def prompt(_):
        session.call("say", message="The current view includes the kitchen")

    async def response(response):
        if response["name"] == "say":
            session.call("finish", status="goal_observed", reason="Inside beside sink")
            # A delivered WebSocket write can still be waiting on local backpressure
            # while the peer has already sent the next valid blocking call.
            await asyncio.sleep(0)

    session.on_prompt, session.on_response = prompt, response
    result, robot, _ = await run(tmp_path, session, goal="Kitchen")
    assert result["status"] == "goal_observed"
    assert result["actions"] == 2
    assert robot.calls == ["stop", "stop"]


async def test_tool_after_terminal_response_is_never_executed(tmp_path):
    session = Session()

    async def prompt(_):
        session.call("finish", status="goal_observed", reason="Inside beside sink")

    async def response(_):
        session.call("advance", distance_m=0.1, reason="Late model call after arrival")
        await asyncio.sleep(0)

    session.on_prompt, session.on_response = prompt, response
    result, robot, _ = await run(tmp_path, session, goal="Kitchen")
    assert result["status"] == "goal_observed"
    assert result["actions"] == 1
    assert robot.calls == ["stop", "stop"]


async def test_start_does_not_replace_already_accepted_user_goal(tmp_path):
    session = Session()

    async def prompt(_):
        session.call("start_navigation", goal="A model paraphrase of the kitchen task")

    async def response(response):
        if response["name"] == "start_navigation":
            assert response["response"] == {
                "accepted": True,
                "goal": "Go to the kitchen",
                "already_active": True,
            }
            session.call("finish", status="blocked", reason="Door is closed")

    session.on_prompt, session.on_response = prompt, response
    result, _, _ = await run(tmp_path, session, goal="Go to the kitchen")
    assert result["goal"] == "Go to the kitchen"
    assert result["status"] == "blocked"


async def test_spoken_stop_interrupts_speech_playback(tmp_path):
    session = Session()
    speaking = asyncio.Event()

    async def speak(_):
        speaking.set()
        await asyncio.Event().wait()

    async def prompt(_):
        session.call("say", message="I am inspecting the next doorway")

    session.on_prompt = prompt
    task = asyncio.create_task(run(tmp_path, session, goal="Kitchen", speak=speak))
    await asyncio.wait_for(speaking.wait(), 1)
    session.push({"server_content": {"input_transcription": {"text": "Cancel the mission"}}})
    result, robot, _ = await asyncio.wait_for(task, 0.5)
    assert result["status"] == "cancelled"
    assert robot.calls == ["stop", "stop"]


@pytest.mark.parametrize(
    "tool,arguments",
    [
        ("advance", {"distance_m": True, "reason": "Bad numeric argument"}),
        ("advance", {"speed_m_s": 1, "reason": "Unknown parameter"}),
        ("enable_motors", {"reason": "Unexposed capability"}),
        ("finish", {"status": "success", "reason": "Invalid finish status"}),
    ],
)
async def test_invalid_model_tool_stops_before_motion(tmp_path, tool, arguments):
    session = Session()

    async def prompt(_):
        session.call(tool, **arguments)

    session.on_prompt = prompt
    result, robot, recorder = await run(tmp_path, session, goal="Kitchen")
    assert result["status"] == "error"
    assert robot.calls == ["stop", "stop"]
    assert '"event": "tool_failed"' in (recorder.path / "events.jsonl").read_text()


async def test_parallel_calls_are_not_executed(tmp_path):
    session = Session()
    session.push(
        {
            "tool_call": {
                "function_calls": [
                    {"id": "a", "name": "advance", "args": {"distance_m": 0.1, "reason": "One"}},
                    {"id": "b", "name": "advance", "args": {"distance_m": 0.1, "reason": "Two"}},
                ]
            }
        }
    )
    result, robot, _ = await run(tmp_path, session, goal="Kitchen")
    assert result["status"] == "error"
    assert robot.calls == ["stop", "stop"]


async def test_image_rate_limit_is_at_most_one_hertz(tmp_path):
    robot, session = Robot(), Session()
    mission = LiveMission(
        robot,
        robot,
        session,
        Recorder(tmp_path),
        audio=None,
        goal="Kitchen",
        config=LiveConfig(),
        speak=None,
        emit=lambda event: None,
    )
    await mission.image()
    first = time.monotonic()
    await mission.image()
    assert time.monotonic() - first >= 0.99
    assert len(session.inputs) == 2


@pytest.mark.parametrize("tool", ["look_at", "advance"])
async def test_tool_telemetry_matches_image_after_throttle(tmp_path, tool):
    class ChangingRobot(Robot):
        changed = False

        def snapshot(self):
            snapshot = super().snapshot()
            snapshot["camera"]["image"][:] = 255 if self.changed else 0
            return snapshot

        async def observe(self):
            return {
                **await super().observe(),
                "ready": not self.changed,
                "guard_reason": "obstacle" if self.changed else None,
                "depth_summary": {"view": "current" if self.changed else "previous"},
            }

    robot, session = ChangingRobot(), Session()
    mission = LiveMission(
        robot,
        robot,
        session,
        Recorder(tmp_path),
        audio=None,
        goal="Kitchen",
        config=LiveConfig(),
        speak=None,
        emit=lambda event: None,
    )
    mission.last_image = time.monotonic()

    async def sensors_change_during_throttle():
        await asyncio.sleep(0.05)
        robot.changed = True

    update = asyncio.create_task(sensors_change_during_throttle())
    arguments = {"x": 1, "y": 0, "z": 0} if tool == "look_at" else {"distance_m": 0.1}
    result = await mission.execute_tool(tool, {**arguments, "reason": "Inspect current scene"})
    await update
    jpeg = session.inputs[-1]["video"]["data"]
    assert np.asarray(Image.open(io.BytesIO(jpeg))).mean() > 250
    assert result["observation"]["ready"] is False
    assert result["observation"]["guard_reason"] == "obstacle"
    assert result["observation"]["depth"] == {"view": "current"}


class ArrivalReviewer:
    def __init__(self, verdicts):
        self.verdicts = iter(verdicts)
        self.calls = []

    async def review(self, goal, views):
        self.calls.append((goal, copy.deepcopy(views)))
        accepted = next(self.verdicts)
        return {
            "destination_visible": accepted,
            "inside_destination": accepted,
            "evidence": "Inside beside sink and stove"
            if accepted
            else "A flat wall fills the views",
            "uncertainty": "Visual assessment only",
        }


@pytest.fixture
def fast_images(monkeypatch):
    original = LiveMission.image

    async def send_image(mission):
        # Separate tests cover the real one-hertz throttle. These exercise review
        # state transitions while retaining actual JPEG encoding and recording.
        mission.last_image = -float("inf")
        return await original(mission)

    monkeypatch.setattr(LiveMission, "image", send_image)


async def test_arrival_review_rejects_wall_then_continues_original_goal(tmp_path, fast_images):
    session, reviewer = Session(), ArrivalReviewer([False])

    async def prompt(_):
        session.call("finish", status="goal_observed", reason="UNTRUSTED_EARLIER_ARRIVAL_CLAIM")

    async def response(response):
        if response["response"].get("continue_navigation"):
            assert response["response"]["goal"] == "Go to the kitchen"
            assert (
                response["response"]["arrival_review"]["evidence"] == "A flat wall fills the views"
            )
            session.call("advance", distance_m=0.1, reason="Continue toward visible clear floor")
        elif response["name"] == "advance":
            session.call("finish", status="blocked", reason="No further visible clear route")

    session.on_prompt, session.on_response = prompt, response
    result, robot, recorder = await run(
        tmp_path,
        session,
        goal="Go to the kitchen",
        arrival_reviewer=reviewer,
    )
    assert result["status"] == "blocked"
    assert result["goal"] == "Go to the kitchen"
    assert robot.calls == [
        "stop",
        "stop",
        "look_at",
        "look_at",
        "look_at",
        "look_at",
        "advance",
        "stop",
    ]
    goal, views = reviewer.calls[0]
    assert goal == "Go to the kitchen"
    assert [view["label"] for view in views] == ["front", "left45", "right45", "front_final"]
    assert all(set(view) == {"label", "jpeg"} for view in views)
    assert "UNTRUSTED_EARLIER_ARRIVAL_CLAIM" not in repr(reviewer.calls)
    assert "simulator_truth" not in repr(reviewer.calls)
    assert len(result["arrival_images"]) == 4
    for view, reference in zip(views, result["arrival_images"], strict=True):
        assert Path(reference["image_path"]).read_bytes() == view["jpeg"]
    assert '"event": "arrival_review_finished"' in (recorder.path / "events.jsonl").read_text()


async def test_accepted_arrival_remains_unverified_model_assessment(tmp_path, fast_images):
    session, reviewer = Session(), ArrivalReviewer([True])

    async def prompt(_):
        session.call("finish", status="goal_observed", reason="I think this is the kitchen")

    session.on_prompt = prompt
    result, robot, _ = await run(tmp_path, session, goal="Kitchen", arrival_reviewer=reviewer)
    assert result["status"] == "goal_observed"
    assert result["goal_verified"] is False
    assert result["reason"] == "Inside beside sink and stove"
    assert robot.calls.count("look_at") == 4
    assert "advance" not in robot.calls


@pytest.mark.parametrize(
    "reason", ["unhealthy", "state_stale", "depth_stale", "depth_too_close", "disconnected"]
)
async def test_arrival_cannot_override_guard_that_changes_during_review(
    tmp_path, fast_images, reason
):
    class ChangingHealth(Robot):
        guard_reason = None

        async def observe(self):
            observation = await super().observe()
            current = self.guard_reason
            # Simulate a transient fault that a subsequent observation would miss.
            self.guard_reason = None
            return {**observation, "ready": current is None, "guard_reason": current}

    robot, session = ChangingHealth(), Session()

    class Review(ArrivalReviewer):
        async def review(self, goal, views):
            verdict = await super().review(goal, views)
            robot.guard_reason = reason
            if reason == "disconnected":
                robot.connected = False
            return verdict

    async def prompt(_):
        session.call("finish", status="goal_observed", reason="Claim before review latency")

    session.on_prompt = prompt
    result, robot, recorder = await run(
        tmp_path, session, robot, goal="Kitchen", arrival_reviewer=Review([True])
    )
    assert result["status"] == "error"
    assert result["arrival_guard"]["guard_reason"] == reason
    assert result["arrival_review"]["inside_destination"] is True
    events = [
        json.loads(line) for line in (recorder.path / "events.jsonl").read_text().splitlines()
    ]
    assert not any(e["event"] == "arrival_review_finished" and e["accepted"] for e in events)
    assert robot.calls[-1] == "stop"


@pytest.mark.parametrize("failure", ["frozen_camera", "unsettled_motion"])
async def test_arrival_requires_fresh_settled_observation_after_review(
    tmp_path, fast_images, failure
):
    class ChangingSensors(Robot):
        review_finished_at = None

        def snapshot(self):
            snapshot = super().snapshot()
            if self.review_finished_at is not None:
                if failure == "frozen_camera":
                    snapshot["camera"]["received_at"] = self.review_finished_at
                else:
                    snapshot["state"]["data"]["move"]["applied"] = [0.1, 0, 0]
            return snapshot

    robot, session = ChangingSensors(), Session()

    class Review(ArrivalReviewer):
        async def review(self, goal, views):
            verdict = await super().review(goal, views)
            robot.review_finished_at = time.monotonic()
            return verdict

    async def prompt(_):
        session.call("finish", status="goal_observed", reason="Claim before sensors changed")

    session.on_prompt = prompt
    result, _, recorder = await run(
        tmp_path,
        session,
        robot,
        goal="Kitchen",
        arrival_reviewer=Review([True]),
        config=LiveConfig(observation_timeout_s=0.1),
    )
    assert result["status"] == "error"
    events = [
        json.loads(line) for line in (recorder.path / "events.jsonl").read_text().splitlines()
    ]
    assert not any(e["event"] == "arrival_review_finished" and e["accepted"] for e in events)


async def test_arrival_permits_ordinary_obstacle_while_safely_stopped(tmp_path, fast_images):
    class ObstacleRobot(Robot):
        async def observe(self):
            return {**await super().observe(), "ready": False, "guard_reason": "obstacle"}

    session = Session()

    async def prompt(_):
        session.call("finish", status="goal_observed", reason="Inside with furniture ahead")

    session.on_prompt = prompt
    result, _, _ = await run(
        tmp_path, session, ObstacleRobot(), goal="Kitchen", arrival_reviewer=ArrivalReviewer([True])
    )
    assert result["status"] == "goal_observed"
    assert result["arrival_guard"]["guard_reason"] == "obstacle"
    assert result["arrival_guard"]["motion"]["applied"] == [0, 0, 0]


async def test_three_rejected_arrival_claims_end_blocked(tmp_path, fast_images):
    session, reviewer = Session(), ArrivalReviewer([False, False, False])

    async def claim(_):
        session.call("finish", status="goal_observed", reason="Still claiming arrival")

    async def response(response):
        if response["response"].get("continue_navigation"):
            await claim(None)

    session.on_prompt, session.on_response = claim, response
    result, robot, _ = await run(tmp_path, session, goal="Kitchen", arrival_reviewer=reviewer)
    assert result["status"] == "blocked"
    assert len(reviewer.calls) == 3
    assert robot.calls.count("look_at") == 12
    assert result["actions"] == 3
    assert "advance" not in robot.calls


async def test_arrival_scan_failure_stops_without_review(tmp_path, fast_images):
    class BadGaze(Robot):
        async def look_at(self, x, y, z):
            self.calls.append("look_at")
            return {"completed": False, "reason": "gaze_timeout"}

    session, reviewer = Session(), ArrivalReviewer([True])

    async def prompt(_):
        session.call("finish", status="goal_observed", reason="Claim before failed scan")

    session.on_prompt = prompt
    result, robot, _ = await run(
        tmp_path,
        session,
        BadGaze(),
        goal="Kitchen",
        arrival_reviewer=reviewer,
    )
    assert result["status"] == "error"
    assert not reviewer.calls
    assert robot.calls[-1] == "stop"


async def test_spoken_cancel_interrupts_pending_arrival_review(tmp_path, fast_images):
    entered = asyncio.Event()

    class WaitingReview:
        async def review(self, goal, views):
            entered.set()
            await asyncio.Event().wait()

    session = Session()

    async def prompt(_):
        session.call("finish", status="goal_observed", reason="Claim before review")

    session.on_prompt = prompt
    task = asyncio.create_task(
        run(tmp_path, session, goal="Kitchen", arrival_reviewer=WaitingReview())
    )
    await asyncio.wait_for(entered.wait(), 1)
    session.push({"server_content": {"input_transcription": {"text": "Stop"}}})
    result, robot, _ = await asyncio.wait_for(task, 0.5)
    assert result["status"] == "cancelled"
    assert robot.calls[-1] == "stop"


class VisualPlanner:
    model = "visual-fixture"

    def __init__(self, decisions):
        self.decisions = iter(decisions)
        self.contexts = []
        self.images = []
        self.views = []

    async def decide(self, context, jpeg, *, views=None):
        self.contexts.append(copy.deepcopy(context))
        self.images.append(jpeg)
        self.views.append(copy.deepcopy(views))
        return next(self.decisions)


async def test_visual_planner_receives_current_course_before_and_after_walking(
    tmp_path, fast_images
):
    class CourseRobot(Robot):
        async def observe(self):
            return {
                **await super().observe(),
                "course": {
                    "active": True,
                    "target_yaw_deg": 90.0,
                    "error_deg": 12.0 if self.distance == 0 else 3.0,
                    "reset_reason": None,
                },
            }

    planner = VisualPlanner(
        [
            {"name": "advance", "args": {"distance_m": 0.1, "reason": "Clear corrective arc"}},
            {"name": "finish", "args": {"status": "blocked", "reason": "No onward route"}},
        ]
    )
    result, _, _ = await run(
        tmp_path, Session(), CourseRobot(), goal="Kitchen", navigation_planner=planner
    )
    assert result["status"] == "blocked"
    assert [context["course"]["error_deg"] for context in planner.contexts] == [12.0, 3.0]
    assert all(context["course"]["target_yaw_deg"] == 90.0 for context in planner.contexts)


def camera_snapshot(yaw_deg=0, pitch_deg=0, *, body_yaw=0, position=(0, 0, 0.12)):
    snapshot = Robot().snapshot()
    yaw, tilt = math.radians(yaw_deg) / 2, (math.pi / 2 - math.radians(pitch_deg)) / 2
    # Camera optical +Z first rotates toward trunk +X, then about trunk +Z.
    quaternion = [
        math.cos(yaw) * math.cos(tilt),
        -math.sin(yaw) * math.sin(tilt),
        math.cos(yaw) * math.sin(tilt),
        math.sin(yaw) * math.cos(tilt),
    ]
    snapshot["state"]["data"]["frames"] = {"camera": {"quat": quaternion}}
    snapshot["state"]["data"]["odom"] = {"position": list(position), "yaw": body_yaw}
    return snapshot


def test_retained_scans_use_measured_optical_direction_and_replace_nearby_heading(tmp_path):
    cache = StationaryViews(Recorder(tmp_path))
    for yaw in (0, 45, -45, 2):
        snapshot = camera_snapshot(yaw, -12)
        # Requested gaze is deliberately contradictory; measured frame is authoritative.
        snapshot["state"]["data"]["head"] = [0, 0, 0, 0]
        current, image_path = cache.retain(snapshot, bytes([int(yaw + 50)]))
    assert current["label"] == "front"
    assert current["yaw_deg"] == pytest.approx(2)
    assert current["pitch_deg"] == pytest.approx(-12)
    assert Path(image_path).read_bytes() == bytes([52])
    prior, references = cache.prior(snapshot, current)
    assert len(cache.views) == 3
    assert [view["camera"]["label"] for view in prior] == ["left", "right"]
    assert [view["camera"]["yaw_deg"] for view in prior] == pytest.approx([45, -45])
    assert all(0 <= view["camera"]["age_s"] < 1 for view in prior)
    assert all(set(view) == {"camera", "jpeg"} for view in prior)
    assert all(
        Path(ref["image_path"]).read_bytes() == view["jpeg"] for view, ref in zip(prior, references)
    )
    assert "simulator_truth" not in json.dumps(references)


@pytest.mark.parametrize("change", ["translated", "turned", "aged", "missing_odometry"])
def test_old_scans_are_discarded_after_pose_change_or_expiry(tmp_path, change):
    cache = StationaryViews(Recorder(tmp_path))
    cache.retain(camera_snapshot(45), b"left")
    current_snapshot = camera_snapshot()
    if change == "translated":
        current_snapshot["state"]["data"]["odom"]["position"][0] = 0.026
    elif change == "turned":
        current_snapshot["state"]["data"]["odom"]["yaw"] = math.radians(5.1)
    elif change == "aged":
        cache.views[0]["camera"]["received_at"] -= 31
    else:
        del current_snapshot["state"]["data"]["odom"]
    cache.prune(current_snapshot)
    assert not cache.views


def test_nearby_pose_retains_scans_with_signed_wrapped_heading_delta(tmp_path):
    cache = StationaryViews(Recorder(tmp_path))
    cache.retain(camera_snapshot(45, body_yaw=math.radians(179)), b"left")
    snapshot = camera_snapshot(body_yaw=math.radians(-179), position=(0.02, 0, 0.12))
    current, _ = cache.retain(snapshot, b"front")
    previous, _ = cache.prior(snapshot, current)
    assert len(previous) == 1
    assert previous[0]["camera"]["body_position_delta_m"] == pytest.approx(0.02)
    assert previous[0]["camera"]["body_heading_delta_deg"] == pytest.approx(2)


@pytest.mark.parametrize("failure", ["missing_frame", "bad_quaternion", "sensor_skew"])
def test_unknown_or_unmatched_camera_orientation_clears_retained_scans(tmp_path, failure):
    cache = StationaryViews(Recorder(tmp_path))
    cache.retain(camera_snapshot(45), b"left")
    snapshot = camera_snapshot()
    if failure == "missing_frame":
        del snapshot["state"]["data"]["frames"]["camera"]
    elif failure == "bad_quaternion":
        snapshot["state"]["data"]["frames"]["camera"]["quat"] = [0, 0, 0, 0]
    else:
        snapshot["state"]["received_at"] -= 0.2
    camera, _ = cache.retain(snapshot, b"unknown")
    assert camera is None
    assert not cache.views


async def test_standard_planner_compares_scans_at_same_pose_and_drops_them_after_motion(
    tmp_path, fast_images
):
    class GazeRobot(Robot):
        gaze_deg = 0

        def snapshot(self):
            snapshot = camera_snapshot(self.gaze_deg, -8, position=(self.distance, 0, 0.12))
            snapshot["camera"]["image"][:] = {0: 50, 45: 150, -45: 250}[self.gaze_deg]
            return snapshot

        async def observe(self):
            return {
                **await super().observe(),
                "ready": self.gaze_deg == 0,
                "guard_reason": None if self.gaze_deg == 0 else "head_not_forward",
                "depth_summary": {"sensor_yaw_deg": self.gaze_deg},
            }

        async def look_at(self, x, y, z):
            self.gaze_deg = round(math.degrees(math.atan2(y, x)))
            return await super().look_at(x, y, z)

    planner = VisualPlanner(
        [
            {"name": "look_at", "args": {"x": 1, "y": 1, "z": 0, "reason": "Inspect left"}},
            {"name": "look_at", "args": {"x": 1, "y": -1, "z": 0, "reason": "Inspect right"}},
            {"name": "look_at", "args": {"x": 1, "y": 0, "z": 0, "reason": "Recenter"}},
            {"name": "advance", "args": {"distance_m": 0.1, "reason": "Visible clear route"}},
            {"name": "finish", "args": {"status": "blocked", "reason": "No onward route"}},
        ]
    )
    result, robot, recorder = await run(
        tmp_path, Session(), GazeRobot(), goal="Kitchen", navigation_planner=planner
    )
    assert result["status"] == "blocked"
    assert robot.distance == pytest.approx(0.1)
    assert [len(views) for views in planner.views] == [0, 1, 2, 2, 0]
    assert [context["camera"]["label"] for context in planner.contexts] == [
        "front",
        "left",
        "right",
        "front",
        "front",
    ]
    assert [context["ready"] for context in planner.contexts] == [True, False, False, True, True]
    assert planner.contexts[3]["depth"] == {"sensor_yaw_deg": 0}
    assert [view["camera"]["label"] for view in planner.views[3]] == ["left", "right"]
    assert all(
        set(view["camera"]) & {"ready", "depth", "simulator_truth"} == set()
        for views in planner.views
        for view in views
    )
    events = [
        json.loads(line) for line in (recorder.path / "events.jsonl").read_text().splitlines()
    ]
    selected = [event for event in events if event["event"] == "visual_views_selected"]
    assert [event["context"] for event in selected] == planner.contexts
    for event, image, views in zip(selected, planner.images, planner.views, strict=True):
        assert Path(event["current"]["image_path"]).read_bytes() == image
        assert all(
            Path(ref["image_path"]).read_bytes() == view["jpeg"]
            for ref, view in zip(event["previous"], views, strict=True)
        )
    assert "simulator_truth" not in json.dumps(planner.contexts)


async def test_autonomous_visual_steps_finish_without_voice_calls_or_replies(tmp_path, fast_images):
    session = Session()
    planner = VisualPlanner(
        [
            {"name": "look_at", "args": {"x": 1, "y": 0, "z": 0, "reason": "Recenter first"}},
            {"name": "advance", "args": {"distance_m": 0.1, "reason": "Clear floor ahead"}},
            {"name": "finish", "args": {"status": "goal_observed", "reason": "Inside beside sink"}},
        ]
    )
    result, robot, recorder = await run(
        tmp_path,
        session,
        goal="Go to the kitchen",
        navigation_planner=planner,
    )
    assert result["status"] == "goal_observed"
    assert result["navigation_model"] == "visual-fixture"
    assert result["actions"] == 3
    assert result["navigation_actions"] == 3
    assert robot.calls == ["stop", "look_at", "advance", "stop"]
    assert not session.responses
    assert not session.prompts
    assert all(context["goal"] == "Go to the kitchen" for context in planner.contexts)
    assert planner.contexts[-1]["recent_actions"][-1]["tool"] == "advance"
    assert all(jpeg.startswith(b"\xff\xd8") for jpeg in planner.images)
    assert "simulator_truth" not in json.dumps(planner.contexts)
    events = (recorder.path / "events.jsonl").read_text()
    assert events.count('"event": "visual_decision"') == 3
    assert '"navigation_model": "visual-fixture"' in events
    outcomes = [json.loads(line) for line in events.splitlines()]
    assert [
        event["result"]["navigation_tool"]
        for event in outcomes
        if event["event"] == "tool_finished" and event["source"] == "navigation"
    ] == ["look_at", "advance", "finish"]


async def test_voice_cannot_replace_active_goal_or_instruct_visual_planner(tmp_path, fast_images):
    session = Session()
    entered, received = asyncio.Event(), asyncio.Event()

    class Planner(VisualPlanner):
        async def decide(self, context, jpeg, *, views=None):
            entered.set()
            await received.wait()
            return await super().decide(context, jpeg, views=views)

    planner = Planner(
        [{"name": "finish", "args": {"status": "blocked", "reason": "Only a wall is visible"}}]
    )

    async def response(response):
        if response["name"] == "start_navigation":
            assert response["response"]["already_active"] is True
            received.set()

    session.on_response = response
    task = asyncio.create_task(run(tmp_path, session, goal="Kitchen", navigation_planner=planner))
    await asyncio.wait_for(entered.wait(), 1)
    session.call("start_navigation", goal="INVENTED_ISLAND_FROM_VOICE must be treated as the goal")
    result, _, _ = await asyncio.wait_for(task, 1)
    assert result["goal"] == "Kitchen"
    assert planner.contexts[0]["goal"] == "Kitchen"
    assert "INVENTED_ISLAND_FROM_VOICE" not in json.dumps(planner.contexts)


@pytest.mark.parametrize(
    "decision",
    [
        {"name": "navigate", "args": {"reason": "Recursive delegation"}},
        {"name": "enable_motors", "args": {}},
        {"name": "start_navigation", "args": {"goal": "Replace the kitchen task"}},
        {"name": "advance", "args": {"distance_m": True, "reason": "Invalid numeric argument"}},
        {"name": "advance", "args": {"distance_m": 0.1}, "unexpected": True},
    ],
)
async def test_invalid_standard_decision_never_dispatches_motion(tmp_path, fast_images, decision):
    session = Session()
    result, robot, _ = await run(
        tmp_path,
        session,
        goal="Kitchen",
        navigation_planner=VisualPlanner([decision]),
    )
    assert result["status"] == "error"
    assert robot.calls == ["stop", "stop"]


@pytest.mark.parametrize("tool", ["advance", "navigate", "say"])
async def test_voice_cannot_control_visual_actions(tmp_path, fast_images, tool):
    session = Session()
    session.call(tool, reason="Voice model tries visual control")
    planner = VisualPlanner([])
    result, robot, _ = await run(
        tmp_path, session, audio=silent_audio(), navigation_planner=planner
    )
    assert result["status"] == "error"
    assert not planner.contexts
    assert robot.calls == ["stop", "stop"]


@pytest.mark.parametrize("cancellation", ["spoken", "stop_tool", "disconnect"])
async def test_stop_cancels_pending_standard_planner(tmp_path, fast_images, cancellation):
    entered, cancelled = asyncio.Event(), asyncio.Event()

    class WaitingPlanner:
        model = "waiting-fixture"

        async def decide(self, context, jpeg, *, views=None):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

    session = Session()
    task = asyncio.create_task(
        run(tmp_path, session, goal="Kitchen", navigation_planner=WaitingPlanner())
    )
    await asyncio.wait_for(entered.wait(), 1)
    if cancellation == "spoken":
        session.push({"server_content": {"input_transcription": {"text": "Stop"}}})
    elif cancellation == "stop_tool":
        session.call("stop", reason="User asked to stop")
    else:
        session.push(ConnectionError("Voice session lost"))
    result, robot, _ = await asyncio.wait_for(task, 0.5)
    assert result["status"] == ("error" if cancellation == "disconnect" else "cancelled")
    assert cancelled.is_set()
    assert robot.calls == ["stop", "stop"]


async def test_autonomous_rejected_arrival_continues_with_review_context(tmp_path, fast_images):
    session, reviewer = Session(), ArrivalReviewer([False])
    planner = VisualPlanner(
        [
            {"name": "finish", "args": {"status": "goal_observed", "reason": "Premature claim"}},
            {"name": "advance", "args": {"distance_m": 0.1, "reason": "Continue on visible floor"}},
            {"name": "finish", "args": {"status": "blocked", "reason": "No further safe route"}},
        ]
    )
    result, robot, recorder = await run(
        tmp_path,
        session,
        goal="Kitchen",
        navigation_planner=planner,
        arrival_reviewer=reviewer,
    )
    assert result["status"] == "blocked"
    assert result["goal"] == "Kitchen"
    assert robot.calls.count("advance") == 1
    assert planner.contexts[1]["arrival_claims"] == 1
    assert planner.contexts[1]["arrival_review"]["inside_destination"] is False
    assert planner.contexts[1]["progress_budget"]["observations_without_progress"] == 1
    assert planner.contexts[2]["progress_budget"]["observations_without_progress"] == 0
    outcomes = [
        json.loads(line)
        for line in (recorder.path / "events.jsonl").read_text().splitlines()
        if '"event": "tool_finished"' in line
    ]
    assert outcomes[0]["result"]["continue_navigation"] is True


async def test_spoken_goal_starts_navigation_while_voice_tool_response_is_pending(
    tmp_path, fast_images
):
    session = Session()
    planner = VisualPlanner(
        [
            {"name": "advance", "args": {"distance_m": 0.1, "reason": "Clear floor"}},
            {"name": "finish", "args": {"status": "blocked", "reason": "No onward route"}},
        ]
    )

    async def audio(_):
        session.call("start_navigation", goal="Go to the kitchen")

    async def response(_):
        await asyncio.Event().wait()

    async def pcm_fixture():
        # Energy opens a manual audio turn; this is a protocol fixture, not ASR proof.
        yield np.full(1600, 2000, dtype="<i2").tobytes()

    session.on_audio, session.on_response = audio, response
    result, robot, _ = await asyncio.wait_for(
        run(tmp_path, session, audio=pcm_fixture(), navigation_planner=planner),
        1,
    )
    assert result["status"] == "blocked"
    assert result["goal"] == "Go to the kitchen"
    assert result["actions"] == 3  # One voice acceptance and two visual decisions.
    assert result["navigation_actions"] == 2
    assert robot.calls == ["stop", "advance", "stop"]


async def test_standard_audio_brackets_two_utterances_and_closes_on_eof(tmp_path):
    robot, session = Robot(), Session()
    voiced = np.full(1600, 2000, dtype="<i2").tobytes()

    async def audio():
        for chunk in [bytes(3200)] * 3 + [voiced] + [bytes(3200)] * 8 + [voiced]:
            yield chunk

    mission = LiveMission(
        robot,
        robot,
        session,
        Recorder(tmp_path),
        audio=audio(),
        goal=None,
        config=LiveConfig(),
        speak=None,
        emit=lambda event: None,
        navigation_planner=VisualPlanner([]),
    )
    await mission.audio_input()
    boundaries = [message for message in session.inputs if "audio" not in message]
    assert boundaries == [
        {"activity_start": {}},
        {"activity_end": {}},
        {"activity_start": {}},
        {"activity_end": {}},
    ]
    first = next(message["audio"]["data"] for message in session.inputs if "audio" in message)
    assert first == bytes(6400) + voiced
    assert not any("audio_stream_end" in message for message in session.inputs)


async def test_standard_silence_does_not_create_user_activity(tmp_path):
    robot, session = Robot(), Session()
    mission = LiveMission(
        robot,
        robot,
        session,
        Recorder(tmp_path),
        audio=silent_audio(),
        goal=None,
        config=LiveConfig(),
        speak=None,
        emit=lambda event: None,
        navigation_planner=VisualPlanner([]),
    )
    await mission.audio_input()
    assert session.inputs == []


def test_standard_uses_manual_activity_while_streaming_keeps_automatic_vad():
    from google.genai import types

    standard = types.LiveConnectConfig(**connect_config("standard"))
    assert standard.realtime_input_config.automatic_activity_detection.disabled is True
    assert "realtime_input_config" not in connect_config("streaming")


async def test_standard_motion_is_serial_while_voice_can_confirm_existing_goal(
    tmp_path, fast_images
):
    robot, session = Robot(), Session()
    robot.advance_release = asyncio.Event()
    responded = asyncio.Event()
    planner = VisualPlanner(
        [
            {"name": "advance", "args": {"distance_m": 0.1, "reason": "First clear step"}},
            {"name": "advance", "args": {"distance_m": 0.1, "reason": "Next clear step"}},
            {"name": "finish", "args": {"status": "blocked", "reason": "No onward route"}},
        ]
    )

    async def response(_):
        responded.set()

    session.on_response = response
    task = asyncio.create_task(
        run(tmp_path, session, robot, goal="Kitchen", navigation_planner=planner)
    )
    await asyncio.wait_for(robot.advance_entered.wait(), 1)
    session.call("start_navigation", goal="Kitchen again")
    await asyncio.wait_for(responded.wait(), 1)
    assert robot.calls == ["stop", "advance"]
    assert len(planner.contexts) == 1
    robot.advance_release.set()
    result, _, recorder = await asyncio.wait_for(task, 1)
    assert result["status"] == "blocked"
    assert result["goal"] == "Kitchen"
    assert result["actions"] == 4
    assert result["navigation_actions"] == 3
    assert robot.calls == ["stop", "advance", "advance", "stop"]
    requested = [
        json.loads(line)["step"]
        for line in (recorder.path / "events.jsonl").read_text().splitlines()
        if '"event": "tool_requested"' in line
    ]
    assert requested == [1, 2, 3, 4]


@pytest.mark.parametrize("cancellation", ["spoken", "stop_tool", "interrupted"])
async def test_voice_stop_interrupts_autonomous_body_action(tmp_path, fast_images, cancellation):
    robot, session = Robot(), Session()
    robot.advance_release = asyncio.Event()
    planner = VisualPlanner(
        [{"name": "advance", "args": {"distance_m": 0.1, "reason": "Clear step"}}]
    )
    task = asyncio.create_task(
        run(tmp_path, session, robot, goal="Kitchen", navigation_planner=planner)
    )
    await asyncio.wait_for(robot.advance_entered.wait(), 1)
    if cancellation == "spoken":
        session.push({"server_content": {"input_transcription": {"text": "Stop"}}})
    elif cancellation == "stop_tool":
        session.call("stop", reason="User asked to stop")
    else:
        session.push({"server_content": {"interrupted": True}})
    result, _, _ = await asyncio.wait_for(task, 0.5)
    assert result["status"] == "cancelled"
    assert robot.calls == ["stop", "advance", "stop"]
    assert robot.distance == 0
    assert len(planner.contexts) == 1


async def test_voice_silence_after_goal_cannot_time_out_autonomous_navigation(
    tmp_path, fast_images
):
    class SlowStep(Robot):
        async def advance(self, distance_m, heading_deg=0):
            await asyncio.sleep(0.15)
            return await super().advance(distance_m, heading_deg)

    session = Session()
    session.call("start_navigation", goal="Kitchen")
    planner = VisualPlanner(
        [
            {"name": "advance", "args": {"distance_m": 0.1, "reason": "Clear floor"}},
            {"name": "finish", "args": {"status": "blocked", "reason": "No onward route"}},
        ]
    )
    result, robot, _ = await run(
        tmp_path,
        session,
        SlowStep(),
        audio=silent_audio(),
        navigation_planner=planner,
        config=LiveConfig(model_timeout_s=0.05),
    )
    assert result["status"] == "blocked"
    assert robot.distance == 0.1
    assert not session.prompts


async def test_voice_model_timeout_still_applies_before_goal_acceptance(tmp_path, fast_images):
    session = Session()
    session.push({"server_content": {"input_transcription": {"text": "Go to the kitchen"}}})
    planner = VisualPlanner([])
    result, robot, _ = await run(
        tmp_path,
        session,
        audio=silent_audio(),
        navigation_planner=planner,
        config=LiveConfig(model_timeout_s=0.05),
    )
    assert result["status"] == "model_timeout"
    assert not planner.contexts
    assert robot.calls == ["stop", "stop"]


async def test_autonomous_planner_timeout_stops_mission(tmp_path, fast_images):
    class SlowPlanner:
        async def decide(self, context, jpeg, *, views=None):
            await asyncio.Event().wait()

    result, robot, _ = await run(
        tmp_path,
        Session(),
        goal="Kitchen",
        navigation_planner=SlowPlanner(),
        config=LiveConfig(model_timeout_s=0.05),
    )
    assert result["status"] == "error"
    assert robot.calls == ["stop", "stop"]


async def test_autonomous_action_budget_bounds_visual_steps(tmp_path, fast_images):
    planner = VisualPlanner(
        [{"name": "advance", "args": {"distance_m": 0.1, "reason": "Clear floor"}}] * 3
    )
    result, robot, _ = await run(
        tmp_path,
        Session(),
        goal="Kitchen",
        navigation_planner=planner,
        config=LiveConfig(max_actions=2),
    )
    assert result["status"] == "action_limit"
    assert result["actions"] == result["navigation_actions"] == 2
    assert robot.calls == ["stop", "advance", "advance", "stop"]
    assert len(planner.contexts) == 2


@pytest.mark.parametrize("tool", ["look_at", "remember_place", "observe"])
async def test_observation_budget_stops_repeated_successful_scans(tmp_path, fast_images, tool):
    arguments = {
        "look_at": {"x": 1, "y": 0, "z": 0, "reason": "Inspect clear floor"},
        "remember_place": {"name": "hall", "observation": "A wall", "explored": True},
        "observe": {"reason": "Reassess the view"},
    }
    planner = VisualPlanner(
        [{"name": tool, "args": arguments[tool]}] * 12
        + [{"name": "advance", "args": {"distance_m": 0.1, "reason": "Must not dispatch"}}]
    )
    result, robot, _ = await run(tmp_path, Session(), goal="Kitchen", navigation_planner=planner)
    assert result["status"] == "blocked"
    assert result["reason"].startswith("Observation budget exhausted")
    assert result["progress_budget"] == {
        "observations_without_progress": 12,
        "limit": 12,
        "remaining": 0,
    }
    assert result["navigation_actions"] == len(planner.contexts) == 12
    assert [context["progress_budget"]["remaining"] for context in planner.contexts] == list(
        range(12, 0, -1)
    )
    assert "advance" not in robot.calls
    assert robot.calls[-1] == "stop"
    assert result["stop"]["completed"] is True


@pytest.mark.parametrize("completed, displacement", [(True, 0.001), (False, 0.1)])
async def test_uncompleted_or_negligible_advance_cannot_reset_observation_budget(
    tmp_path, fast_images, completed, displacement
):
    class NoProgressRobot(Robot):
        async def advance(self, distance_m, heading_deg=0):
            self.calls.append("advance")
            self.distance += displacement
            return {
                "completed": completed,
                "reason": "distance_reached" if completed else "obstacle",
                "stop": {"acknowledged": True, "physical_settling_verified": True},
            }

    observation = {"name": "observe", "args": {"reason": "Inspect floor"}}
    planner = VisualPlanner(
        [observation] * 11
        + [{"name": "advance", "args": {"distance_m": 0.1, "reason": "Visible floor"}}]
        + [observation]
    )
    result, robot, _ = await run(
        tmp_path, Session(), NoProgressRobot(), goal="Kitchen", navigation_planner=planner
    )
    assert result["status"] == "blocked"
    assert result["reason"].startswith("Observation budget exhausted")
    assert result["navigation_actions"] == 13
    assert planner.contexts[-1]["progress_budget"]["observations_without_progress"] == 11
    assert result["progress_budget"]["observations_without_progress"] == 12
    assert robot.calls == ["stop", "advance", "stop"]


async def test_measured_completed_advance_resets_scan_budget_and_preserves_arrival(
    tmp_path, fast_images
):
    observation = {"name": "observe", "args": {"reason": "Inspect floor"}}
    planner = VisualPlanner(
        [observation] * 11
        + [{"name": "advance", "args": {"distance_m": 0.1, "reason": "Visible floor"}}]
        + [observation] * 11
        + [{"name": "finish", "args": {"status": "goal_observed", "reason": "Inside kitchen"}}]
    )
    result, robot, _ = await run(
        tmp_path,
        Session(),
        goal="Kitchen",
        navigation_planner=planner,
        arrival_reviewer=ArrivalReviewer([True]),
    )
    assert result["status"] == "goal_observed"
    assert result["navigation_actions"] == 24
    assert planner.contexts[11]["progress_budget"]["observations_without_progress"] == 11
    assert planner.contexts[12]["progress_budget"]["observations_without_progress"] == 0
    assert result["progress_budget"]["observations_without_progress"] == 11
    assert robot.distance == 0.1
    assert robot.calls[-1] == "stop"


async def test_budget_exhaustion_still_requires_acknowledged_final_stop(tmp_path, fast_images):
    class FailedStopRobot(Robot):
        async def look_at(self, x, y, z):
            outcome = await super().look_at(x, y, z)
            if self.calls.count("look_at") == 12:
                self.stop_ok = False
            return outcome

    planner = VisualPlanner(
        [{"name": "look_at", "args": {"x": 1, "y": 0, "z": 0, "reason": "Inspect"}}] * 12
    )
    result, _, _ = await run(
        tmp_path, Session(), FailedStopRobot(), goal="Kitchen", navigation_planner=planner
    )
    assert result["status"] == "error"
    assert result["reason"] == "Final stop was not acknowledged"
    assert result["progress_budget"]["remaining"] == 0


async def test_voice_stop_wins_while_last_scan_is_pending(tmp_path, fast_images):
    robot, session = Robot(), Session()
    entered = asyncio.Event()

    class LastScanPlanner(VisualPlanner):
        async def decide(self, context, jpeg, *, views=None):
            if context["progress_budget"]["remaining"] == 1:
                entered.set()
                await asyncio.Event().wait()
            return await super().decide(context, jpeg, views=views)

    planner = LastScanPlanner([{"name": "observe", "args": {"reason": "Inspect"}}] * 12)
    task = asyncio.create_task(
        run(tmp_path, session, robot, goal="Kitchen", navigation_planner=planner)
    )
    await asyncio.wait_for(entered.wait(), 1)
    session.push({"server_content": {"input_transcription": {"text": "Stop"}}})
    result, _, _ = await asyncio.wait_for(task, 0.5)
    assert result["status"] == "cancelled"
    assert result["progress_budget"]["observations_without_progress"] == 11
    assert robot.calls == ["stop", "stop"]


async def test_delayed_voice_reply_cannot_cancel_last_allowed_visual_step(tmp_path, fast_images):
    robot, session = Robot(), Session()
    robot.advance_release = asyncio.Event()
    session.call("start_navigation", goal="Kitchen")
    planner = VisualPlanner(
        [{"name": "advance", "args": {"distance_m": 0.1, "reason": "One allowed clear step"}}]
    )
    reply_sent = asyncio.Event()

    async def response(_):
        await robot.advance_entered.wait()
        reply_sent.set()

    session.on_response = response
    task = asyncio.create_task(
        run(
            tmp_path,
            session,
            robot,
            audio=silent_audio(),
            navigation_planner=planner,
            config=LiveConfig(max_actions=2),
        )
    )
    await asyncio.wait_for(reply_sent.wait(), 1)
    await asyncio.sleep(0)
    assert not task.done()
    robot.advance_release.set()
    result, _, _ = await asyncio.wait_for(task, 1)
    assert result["status"] == "action_limit"
    assert result["actions"] == 2
    assert result["navigation_actions"] == 1
    assert robot.distance == 0.1


def test_standard_mode_exposes_only_voice_goal_and_stop_tools():
    from google.genai import types

    config = connect_config("standard")
    types.LiveConnectConfig(**config)
    tools = config["tools"][0]["function_declarations"]
    assert {tool["name"] for tool in tools} == {"start_navigation", "stop"}
    assert all(tool["behavior"] == "BLOCKING" for tool in tools)
    assert {tool["name"] for tool in visual_declarations()} == {
        "observe",
        "look_at",
        "advance",
        "advance_to_floor",
        "remember_place",
        "finish",
    }
    assert parser().parse_args([]).visual_planner == "standard"
    assert parser().parse_args(["--visual-planner", "streaming"]).visual_planner == "streaming"
    assert parser().parse_args([]).visual_model == "gemini-robotics-er-2-preview"
    assert (
        parser().parse_args(["--visual-model", "gemini-3.8-flash"]).visual_model
        == "gemini-3.8-flash"
    )


async def test_streaming_rejects_unapplied_visual_model_before_connecting():
    from duck_nav.live import run as run_cli

    args = parser().parse_args(
        ["--visual-planner", "streaming", "--visual-model", "gemini-3.8-flash"]
    )
    with pytest.raises(ValueError, match="applies only"):
        await run_cli(args)


async def test_run_records_actual_visual_generation_config(tmp_path, fast_images):
    planner = VisualPlanner(
        [{"name": "finish", "args": {"status": "blocked", "reason": "Fixture ends"}}]
    )
    planner.generation_config = {
        "thinkingConfig": {"thinkingLevel": "MEDIUM"},
        "maxOutputTokens": 8192,
    }
    result, _, recorder = await run(tmp_path, Session(), goal="Kitchen", navigation_planner=planner)
    assert result["navigation_generation_config"] == planner.generation_config
    events = [
        json.loads(line) for line in (recorder.path / "events.jsonl").read_text().splitlines()
    ]
    started = next(event for event in events if event["event"] == "live_session_started")
    assert started["navigation_generation_config"] == planner.generation_config


def test_live_schema_and_extended_house_budget():
    from google.genai import types

    schema = types.LiveConnectConfig(**connect_config())
    assert str(schema.response_modalities[0]) in {"TEXT", "Modality.TEXT"}
    assert all(t["behavior"] == "BLOCKING" for t in declarations())
    assert {d["name"] for d in declarations()} >= {"advance", "start_navigation", "finish"}
    assert "move_for" not in {d["name"] for d in declarations()}
    assert LiveConfig().max_actions >= 600
    assert LiveConfig().max_seconds >= 1800


@pytest.mark.parametrize(
    "text", ["Stop!", "Please stop moving", "Duck stop", "Cancel the mission", "halt"]
)
def test_spoken_stop_phrases(text):
    assert is_spoken_stop(text)


@pytest.mark.parametrize(
    "text", ["Go to the bus stop", "Don't stop", "I see a stop sign", "Stop by the kitchen"]
)
def test_destination_words_do_not_cancel(text):
    assert not is_spoken_stop(text)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"heartbeat_s": 0.1},
        {"max_actions": 1.1},
        {"max_seconds": float("nan")},
        {"model_timeout_s": True},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises((ValueError, TypeError)):
        LiveConfig(**kwargs)
