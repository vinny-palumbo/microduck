"""The tool entry point never turns malformed input into robot commands."""

import asyncio
import json

import numpy as np
import pytest

from duck_nav.cli import Recorder, dispatch, parser, serve


class Robot:
    def __init__(self):
        self.calls = []

    async def move_for(self, speed_m_s, duration_s):
        self.calls.append((speed_m_s, duration_s))
        return {"completed": True}

    async def stop(self):
        self.calls.append("stop")
        return {"completed": True}


@pytest.mark.asyncio
async def test_bad_tool_cannot_reach_transport():
    robot = Robot()
    for request in (
        {"tool": "robot.enable"},
        {"tool": "__dict__"},
        [],
        {"tool": "stop", "extra": True},
        {"tool": "stop", "arguments": []},
    ):
        with pytest.raises((ValueError, TypeError)):
            await dispatch(robot, request)
    assert robot.calls == []


@pytest.mark.asyncio
async def test_extra_parameters_do_not_get_ignored():
    robot = Robot()
    with pytest.raises(TypeError):
        await dispatch(
            robot,
            {
                "tool": "move_for",
                "arguments": {"speed_m_s": 0.05, "duration_s": 0.5, "unbounded": True},
            },
        )
    assert robot.calls == []


@pytest.mark.asyncio
async def test_stop_does_not_require_observation():
    robot = Robot()
    assert await dispatch(robot, {"tool": "stop"}) == {"completed": True}
    assert robot.calls == ["stop"]


def test_recording_keeps_raw_sensors_and_image_separate(tmp_path):
    recorder = Recorder(tmp_path)
    snapshot = {
        "connected": True,
        "camera": {
            "sequence": 3,
            "received_at": 1.0,
            "image": np.zeros((5, 7, 3), dtype=np.uint8),
            "metadata": {},
        },
        "state": {"sequence": 2, "received_at": 1.0, "data": {"odom": {"position": [0, 0, 0.12]}}},
        "depth": None,
    }
    record = recorder.capture(snapshot)
    recorder.write({"observation": record})
    saved = json.loads((recorder.path / "events.jsonl").read_text())
    assert "image" not in saved["observation"]["camera"]
    assert (recorder.path / "frame-0001.jpg").is_file()
    assert saved["observation"]["state"]["data"]["odom"]["position"] == [0, 0, 0.12]
    assert "image" in snapshot["camera"]


def test_default_endpoint_is_simulator():
    args = parser().parse_args(["observe"])
    assert args.host == "127.0.0.1"
    assert args.port == 8443


class BlockingRobot:
    def __init__(self):
        self.started = asyncio.Event()
        self.stopped = asyncio.Event()
        self.cleaned_up = False

    async def move_for(self, speed_m_s, duration_s):
        self.started.set()
        try:
            await self.stopped.wait()
            return {"completed": False, "reason": "stopped"}
        finally:
            self.cleaned_up = True

    async def stop(self):
        self.stopped.set()
        return {"completed": True}


class Transport:
    def snapshot(self):
        return {"connected": True}


@pytest.mark.asyncio
async def test_server_accepts_stop_during_action(tmp_path):
    robot, replies = BlockingRobot(), []

    async def lines():
        yield '{"id":1,"tool":"move_for","arguments":{"speed_m_s":0.05,"duration_s":2}}'
        await robot.started.wait()
        yield '{"id":2,"tool":"stop"}'
        await asyncio.sleep(0.01)

    await asyncio.wait_for(
        serve(robot, Transport(), Recorder(tmp_path), lines(), replies.append), 0.5
    )
    assert robot.cleaned_up
    by_id = {reply["id"]: reply for reply in replies}
    assert by_id[1]["result"]["reason"] == "stopped"
    assert by_id[2]["result"]["completed"]


@pytest.mark.asyncio
async def test_input_eof_cancels_running_action(tmp_path):
    robot, replies = BlockingRobot(), []

    async def lines():
        yield '{"id":1,"tool":"move_for","arguments":{"speed_m_s":0.05,"duration_s":2}}'
        await robot.started.wait()

    await asyncio.wait_for(
        serve(robot, Transport(), Recorder(tmp_path), lines(), replies.append), 0.5
    )
    assert robot.cleaned_up
    assert replies[0]["error"] == "cancelled: input closed"
