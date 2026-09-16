"""Exercise actual receive-turn lifetimes and concurrent interruption of movement."""

import asyncio
import copy
import json
import time
from types import SimpleNamespace

import numpy as np
import pytest

from duck_nav.cli import Recorder
from duck_nav.live import (
    LiveConfig,
    LiveMission,
    connect_config,
    declarations,
    is_spoken_stop,
    run_live,
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
