"""Transport checks: replayed samples and failed sensors must never enable movement."""

import asyncio
import json
import time
import unittest
from fractions import Fraction
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

import numpy as np
from av import AudioFrame

from duck_nav.transport import WebRtcRobot, _load_module


class Frame:
    time_base = Fraction(1, 90000)

    def __init__(self, pts):
        self.pts = pts

    def to_ndarray(self, format):
        assert format == "rgb24"
        return np.arange(18, dtype=np.uint8).reshape(2, 3, 3)


class SampleTests(unittest.TestCase):
    def setUp(self):
        self.robot = WebRtcRobot()

    def test_state_coast_does_not_refresh_sensor_age(self):
        # robotd can publish repeatedly while sensor reads fail; t_ns, not arrival, is truth.
        with patch("duck_nav.transport.time.monotonic", return_value=10):
            self.robot._record_notification("robot.state", {"t_ns": 100, "t": 1})
        with patch("duck_nav.transport.time.monotonic", return_value=20):
            self.robot._record_notification("robot.state", {"t_ns": 100, "t": 2})
        self.assertEqual(self.robot.snapshot()["state"]["received_at"], 10)
        self.assertEqual(self.robot.snapshot()["state"]["sequence"], 1)

    def test_depth_replay_and_restart_do_not_refresh_sensor_age(self):
        with patch("duck_nav.transport.time.monotonic", return_value=10):
            self.robot._record_notification("tof.frame", {"t_ns": 100, "seq": 5, "at_us": 20})
        with patch("duck_nav.transport.time.monotonic", return_value=20):
            for data in (
                {"t_ns": 100, "seq": 5, "at_us": 20},
                {"t_ns": 101, "seq": 5, "at_us": 21},
                {"t_ns": 99, "seq": 6, "at_us": 21},
                {"t_ns": 101, "seq": 0, "at_us": 0},
            ):
                self.robot._record_notification("tof.frame", data)
        self.assertEqual(self.robot.snapshot()["depth"]["received_at"], 10)
        self.assertEqual(self.robot.snapshot()["depth"]["sequence"], 1)

    def test_unstamped_and_nonfinite_notifications_are_not_observations(self):
        self.robot._record_notification("robot.state", {"t": float("nan")})
        self.robot._record_notification("tof.frame", {"distance_mm": [1000]})
        self.assertIsNone(self.robot.snapshot()["state"])
        self.assertIsNone(self.robot.snapshot()["depth"])

    def test_camera_replay_does_not_refresh_and_rotation_can_arrive_later(self):
        with patch("duck_nav.transport.time.monotonic", return_value=10):
            self.robot._record_camera(Frame(0))
        with patch("duck_nav.transport.time.monotonic", return_value=20):
            self.robot._record_camera(Frame(0))
        self.robot._record_notification("media.video", {"rotate": 90})
        sample = self.robot.snapshot()["camera"]
        self.assertEqual(sample["received_at"], 10)
        self.assertEqual(sample["sequence"], 1)
        self.assertEqual(sample["image"].shape, (3, 2, 3))
        np.testing.assert_array_equal(sample["image"], np.rot90(Frame(0).to_ndarray("rgb24"), -1))

    def test_snapshots_cannot_modify_stored_observations(self):
        self.robot._record_camera(Frame(0))
        self.robot._record_notification("robot.state", {"t_ns": 100, "odom": {"yaw": 2}})
        snapshot = self.robot.snapshot()
        snapshot["camera"]["image"][:] = 255
        snapshot["state"]["data"]["odom"]["yaw"] = 999
        self.assertEqual(self.robot.snapshot()["state"]["data"]["odom"]["yaw"], 2)
        self.assertEqual(self.robot.snapshot()["camera"]["image"][0, 0, 0], 0)


class FakeMediaStreamError(Exception):
    pass


class FakeChannel:
    readyState = "open"

    def __init__(self, consumer):
        self.consumer = consumer

    def send(self, payload):
        call = json.loads(payload)
        self.consumer.calls.append(call)
        if "id" not in call or call["method"] == "silent":
            return
        result = self.consumer.responses.get(call["method"], {"accepted": True})
        self.consumer._rpc.on_message(json.dumps({"jsonrpc": "2.0", "id": call["id"], **result}))


class FakeLanConsumer:
    responses: ClassVar[dict] = {
        "robot.subscribe": {"result": {"accepted": True}},
        "tof.stream": {"result": {"accepted": True, "sensor": "sim"}},
        "media.video": {"result": {"rotate": 90}},
        "robot.stop": {"result": {"accepted": True}},
    }

    def __init__(self, host, rpc, port):
        self._rpc = rpc
        self._cmd_channel = FakeChannel(self)
        self._cmd_channel_open = True
        self._loop = None
        self.error = None
        self.calls = []
        self.stopped = False

    async def start(self):
        self._loop = asyncio.get_running_loop()
        self._rpc.bound_to(self.send_command)

    async def stop(self):
        self.stopped = True

    def status(self):
        return {"connected": not self.stopped}

    async def _pump(self):
        return


class ConnectionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        root = Path(__file__).resolve().parents[3]
        Rpc = _load_module("_duck_nav_test_control", root / "spaces/shared/control.py").Rpc
        self.deps = patch(
            "duck_nav.transport._load_dependencies",
            return_value=(
                Rpc,
                FakeLanConsumer,
                FakeMediaStreamError,
            ),
        )
        self.deps.start()
        self.robot = WebRtcRobot(rpc_timeout=0.03)

    async def asyncTearDown(self):
        await self.robot.close()
        self.deps.stop()

    async def test_connect_subscribes_correctly_without_requiring_sensor_frames(self):
        await self.robot.connect()
        self.assertEqual(
            [(call["method"], call["params"]) for call in self.robot._consumer.calls],
            [
                ("robot.subscribe", {"hz": 20}),
                ("tof.stream", {}),
                ("media.video", {}),
            ],
        )
        self.assertTrue(self.robot.snapshot()["connected"])
        self.assertIsNone(self.robot.snapshot()["camera"])

    async def test_sensor_refusal_leaves_stop_available(self):
        responses = dict(FakeLanConsumer.responses)
        responses["tof.stream"] = {"result": {"accepted": True, "unavailable": "no sensor"}}
        responses["media.video"] = {"error": {"code": -1, "message": "no camera"}}
        with patch.object(FakeLanConsumer, "responses", responses):
            await self.robot.connect()
        self.assertTrue((await self.robot.request("robot.stop"))["accepted"])
        self.assertEqual(set(self.robot.snapshot()["errors"]), {"tof.stream", "media.video"})

    async def test_reconnect_discards_old_samples_and_late_callbacks(self):
        await self.robot.connect()
        old_rpc = self.robot._rpc
        notification = json.dumps({"method": "robot.state", "params": {"t_ns": 100}})
        old_rpc.on_message(notification)
        await self.robot.connect()
        old_rpc.on_message(notification)
        old_rpc.abandon("late close from the old session")
        self.assertTrue(self.robot.snapshot()["connected"])
        self.assertIsNone(self.robot.snapshot()["state"])
        self.robot._rpc.on_message(notification)
        self.assertEqual(self.robot.snapshot()["state"]["sequence"], 1)

    async def test_connection_loss_rejects_notifications_immediately(self):
        await self.robot.connect()
        self.robot._rpc.abandon("link gone")
        self.assertFalse(self.robot.snapshot()["connected"])
        with self.assertRaisesRegex(ConnectionError, "link gone"):
            self.robot.notify("robot.move", {"vx": 0.1})

    async def test_signalling_end_marks_connection_lost(self):
        await self.robot.connect()
        await self.robot._consumer._pump()
        self.assertFalse(self.robot.snapshot()["connected"])

    async def test_send_failure_fails_closed(self):
        await self.robot.connect()
        with (
            patch.object(
                self.robot._consumer._cmd_channel, "send", side_effect=RuntimeError("closed")
            ),
            self.assertRaises(ConnectionError),
        ):
            self.robot.notify("robot.stop")
        self.assertFalse(self.robot.snapshot()["connected"])

    async def test_request_has_bounded_timeout_and_cleans_pending_rpc(self):
        await self.robot.connect()
        with self.assertRaisesRegex(Exception, "no answer"):
            await asyncio.wait_for(self.robot.request("silent"), 0.5)
        self.assertEqual(self.robot._rpc._pending, {})

    async def test_audio_keeps_recent_pcm_and_bounds_latency(self):
        await self.robot.connect()
        pcm = bytes(range(256)) * 100
        self.robot._record_audio(pcm)
        self.assertEqual(self.robot._audio_queue.qsize(), 25)
        stream = self.robot.audio_chunks()
        chunk = await anext(stream)
        self.assertEqual(len(chunk), 640)
        self.assertEqual(chunk, pcm[15 * 640 : 16 * 640])
        await stream.aclose()

    async def test_audio_listener_wakes_on_loss_and_drops_buffered_speech(self):
        await self.robot.connect()
        stream = self.robot.audio_chunks()
        pending = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        self.robot._record_audio(b"\x00\x00" * 320)
        self.robot._rpc.abandon("link gone")
        with self.assertRaisesRegex(ConnectionError, "link gone"):
            await pending

    async def test_audio_drops_chunks_older_than_half_second(self):
        await self.robot.connect()
        self.robot._audio_queue.put_nowait((time.monotonic() - 1, b"old"))
        self.robot._record_audio(b"new")
        stream = self.robot.audio_chunks()
        self.assertEqual(await anext(stream), b"new")
        await stream.aclose()

    async def test_robot_stereo_audio_is_resampled_to_model_pcm(self):
        await self.robot.connect()
        release = asyncio.Event()

        class Microphone:
            sent = False

            async def recv(self):
                if self.sent:
                    await release.wait()
                self.sent = True
                frame = AudioFrame.from_ndarray(
                    np.full((1, 9600), 1000, dtype=np.int16), format="s16", layout="stereo"
                )
                frame.sample_rate = 48000
                frame.pts = 0
                frame.time_base = Fraction(1, 48000)
                return frame

        task = asyncio.create_task(self.robot._consumer._consume_audio(Microphone()))
        stream = self.robot.audio_chunks()
        try:
            pcm = await asyncio.wait_for(anext(stream), 1)
            samples = np.frombuffer(pcm, dtype=np.int16)
            self.assertEqual(len(samples), 320)
            self.assertTrue(np.all(samples > 900))
        finally:
            task.cancel()
            await task
            await stream.aclose()


if __name__ == "__main__":
    unittest.main()
