"""A timestamped navigation client over the repository's existing WebRTC transport."""

from __future__ import annotations

import asyncio
import copy
import importlib.util
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from av import AudioResampler

logger = logging.getLogger(__name__)


def _load_module(name: str, path: Path) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load the repository transport from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _load_dependencies() -> tuple[type, type, type]:
    # Both existing clients are source modules, not packages. Keep their names private and
    # limit lan.py's legacy `from control import ...` alias to the duration of its import.
    root = Path(__file__).resolve().parents[3]
    control = _load_module("_duck_nav_shared_control", root / "spaces/shared/control.py")
    previous = sys.modules.get("control")
    sys.modules["control"] = control
    try:
        lan = _load_module("_duck_nav_shared_lan", root / "spaces/policy-shop/lan.py")
    finally:
        if previous is None:
            sys.modules.pop("control", None)
        else:
            sys.modules["control"] = previous
    return control.Rpc, lan.LanConsumer, lan.MediaStreamError


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


class WebRtcRobot:
    """One LAN session; all async methods and notifications belong to its event loop.

    This deliberately does not reconnect during an action. A new session must subscribe again
    and obtain new observations before its caller can decide whether movement is safe.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8443, *, rpc_timeout: float = 1.0):
        self.host, self.port = host, port
        self.rpc_timeout = rpc_timeout
        self._rpc: Any = None
        self._consumer: Any = None
        self._ready = False
        self._lost_reason: str | None = None
        self._generation = 0
        self._audio_tasks: set[asyncio.Task] = set()
        self._clear_samples()

    def _clear_samples(self) -> None:
        # Wake listeners on the old session; buffered speech must never cross reconnects.
        if hasattr(self, "_audio_queue"):
            self._end_audio()
        self._audio_queue: asyncio.Queue = asyncio.Queue(maxsize=25)
        self._samples: dict[str, dict[str, Any] | None] = {
            "camera": None,
            "state": None,
            "depth": None,
        }
        self._markers: dict[str, dict[str, Any]] = {}
        self._video_metadata: dict[str, Any] = {}
        self._subscription_errors: dict[str, str] = {}

    def _advance(self, stream: str, markers: dict[str, Any]) -> bool:
        if not markers:
            return False
        previous = self._markers.get(stream)
        # A daemon restart or replay is not permission to refresh the safety clock. The caller
        # must establish a new session; sequence/timestamp regressions stay stale until then.
        if previous is not None and (
            markers.keys() != previous.keys()
            or any(value <= previous[key] for key, value in markers.items())
        ):
            return False
        self._markers[stream] = markers
        return True

    def _record_notification(self, method: str, data: Any) -> None:
        if not isinstance(data, dict):
            return
        if method == "media.video":
            self._video_metadata = copy.deepcopy(data)
            return
        stream = {"robot.state": "state", "tof.frame": "depth"}.get(method)
        if stream is None:
            return
        markers: dict[str, Any] = {}
        stamp = data.get("t_ns")
        if _number(stamp) and stamp > 0:
            markers["t_ns"] = stamp
        if stream == "state" and not markers:
            stamp = data.get("t")
            if _number(stamp) and stamp >= 0:
                markers["t"] = stamp
        elif stream == "depth":
            for key in ("seq", "at_us"):
                stamp = data.get(key)
                if _number(stamp) and stamp >= 0:
                    markers[key] = stamp
        if not self._advance(stream, markers):
            return
        previous = self._samples[stream]
        self._samples[stream] = {
            "sequence": previous["sequence"] + 1 if previous else 1,
            "received_at": time.monotonic(),
            "data": copy.deepcopy(data),
        }

    def _record_camera(self, frame: Any) -> None:
        # PyAV PTS is the source RTP presentation clock. Receiving the same decoded frame
        # again must not make a stalled camera look current.
        if frame.pts is None or frame.time_base is None:
            return
        if not self._advance("camera", {"pts_seconds": frame.pts * frame.time_base}):
            return
        picture = frame.to_ndarray(format="rgb24")
        previous = self._samples["camera"]
        self._samples["camera"] = {
            "sequence": previous["sequence"] + 1 if previous else 1,
            "received_at": time.monotonic(),
            "image": picture,
            "pts": frame.pts,
            "time_base": str(frame.time_base),
        }

    def _lost(self, reason: str, generation: int) -> None:
        if generation == self._generation:
            self._ready = False
            self._lost_reason = reason
            self._end_audio()

    def _end_audio(self) -> None:
        while not self._audio_queue.empty():
            self._audio_queue.get_nowait()
        self._audio_queue.put_nowait(None)

    def _record_audio(self, pcm: bytes) -> None:
        # 20 ms chunks bound the queue to half a second even for long source frames.
        for start in range(0, len(pcm), 640):
            if self._audio_queue.full():
                self._audio_queue.get_nowait()
            self._audio_queue.put_nowait((time.monotonic(), pcm[start : start + 640]))

    async def audio_chunks(self):
        """Robot microphone as 16 kHz mono PCM; fail if no microphone is offered."""
        queue, generation = self._audio_queue, self._generation
        while generation == self._generation:
            try:
                item = await asyncio.wait_for(queue.get(), 10)
            except TimeoutError:
                raise ConnectionError(
                    "No robot microphone audio; use --audio mic or --audio-wav in simulation"
                ) from None
            if item is None:
                raise ConnectionError(self._lost_reason or "robot audio session closed")
            received_at, pcm = item
            if time.monotonic() - received_at <= 0.5:
                yield pcm

    def _make_session(self) -> tuple[Any, Any]:
        Rpc, LanConsumer, MediaStreamError = _load_dependencies()
        owner, generation = self, self._generation

        class ObservedRpc(Rpc):
            def on_message(self, raw: Any) -> None:
                super().on_message(raw)
                try:
                    message = json.loads(raw)
                except (ValueError, TypeError):
                    return
                if (
                    isinstance(message, dict)
                    and message.get("id") is None
                    and generation == owner._generation
                ):
                    owner._record_notification(message.get("method"), message.get("params"))

            def abandon(self, why: str) -> None:
                super().abandon(why)
                owner._lost(why, generation)

        class ObservedConsumer(LanConsumer):
            async def _build_pc(self) -> None:
                await super()._build_pc()
                pc = self._pc

                @pc.on("track")
                def audio_track(track: Any) -> None:
                    if track.kind == "audio":
                        task = asyncio.create_task(self._consume_audio(track))
                        owner._audio_tasks.add(task)
                        task.add_done_callback(owner._audio_tasks.discard)

                @pc.on("connectionstatechange")
                def connection_changed() -> None:
                    if pc.connectionState in {"closed", "failed", "disconnected"}:
                        self._rpc.abandon(f"peer connection {pc.connectionState}")

            async def _pump(self) -> None:
                try:
                    await super()._pump()
                finally:
                    self._rpc.abandon(self.error or "the signalling session ended")

            async def _handle(self, message: dict[str, Any]) -> None:
                await super()._handle(message)
                if self.error:
                    self._rpc.abandon(self.error)

            def send_command(self, envelope: dict[str, Any]) -> bool:
                channel, loop = self._cmd_channel, self._loop
                if (
                    channel is None
                    or loop is None
                    or not self._cmd_channel_open
                    or channel.readyState != "open"
                    or owner._lost_reason
                ):
                    return False
                payload = json.dumps(envelope, allow_nan=False)

                def send() -> bool:
                    try:
                        channel.send(payload)
                        return True
                    except Exception as exc:  # noqa: BLE001 - any failed send loses this session
                        self._rpc.abandon(f"control send failed: {exc}")
                        return False

                try:
                    current = asyncio.get_running_loop()
                except RuntimeError:
                    current = None
                if current is loop:
                    return send()
                try:
                    loop.call_soon_threadsafe(send)
                except RuntimeError:
                    return False
                return True

            async def _consume_video(self, track: Any) -> None:
                try:
                    while generation == owner._generation:
                        frame = await track.recv()
                        if generation == owner._generation:
                            owner._record_camera(frame)
                except (MediaStreamError, asyncio.CancelledError):
                    return
                except Exception:
                    logger.exception("camera stream ended")

            async def _consume_audio(self, track: Any) -> None:
                resampler = AudioResampler(format="s16", layout="mono", rate=16000)
                try:
                    while generation == owner._generation:
                        frame = await track.recv()
                        if generation != owner._generation:
                            return
                        for converted in resampler.resample(frame):
                            owner._record_audio(converted.to_ndarray().tobytes())
                except (MediaStreamError, asyncio.CancelledError):
                    pass
                except Exception:
                    logger.exception("microphone stream ended")
                finally:
                    if generation == owner._generation:
                        owner._end_audio()

        rpc = ObservedRpc(timeout=self.rpc_timeout)
        return rpc, ObservedConsumer(self.host, rpc, self.port)

    def _control_open(self) -> bool:
        return bool(
            self._lost_reason is None
            and self._rpc is not None
            and self._rpc.is_open()
            and self._consumer is not None
            and self._consumer.status().get("connected")
        )

    async def connect(self, timeout: float = 15.0) -> None:
        await self.close()
        self._lost_reason = None
        self._rpc, self._consumer = self._make_session()

        async def open_session() -> None:
            await self._consumer.start()
            while not self._control_open():
                if self._lost_reason or self._consumer.error:
                    raise ConnectionError(self._lost_reason or self._consumer.error)
                await asyncio.sleep(0.02)
            # Missing sensors must fail the movement guard, while leaving robot.stop usable.
            # Requests still have a short deadline: a silent daemon cannot stall setup forever.
            for method, params in (("robot.subscribe", {"hz": 20}), ("tof.stream", None)):
                try:
                    result = await self.request(method, params)
                    if result.get("accepted") is not True or result.get("unavailable"):
                        raise ConnectionError(f"subscription unavailable: {result}")
                except Exception as exc:  # noqa: BLE001 - sensor failures must leave stop available
                    self._subscription_errors[method] = str(exc)
            try:
                self._record_notification("media.video", await self.request("media.video"))
            except Exception as exc:  # noqa: BLE001 - missing metadata must leave stop available
                self._subscription_errors["media.video"] = str(exc)
            if not self._control_open():
                raise ConnectionError(self._lost_reason or "control channel closed during setup")
            self._ready = True

        try:
            await asyncio.wait_for(open_session(), timeout)
        except BaseException:
            await self.close()
            raise

    async def close(self) -> None:
        self._generation += 1
        self._ready = False
        tasks = list(self._audio_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        consumer, rpc = self._consumer, self._rpc
        self._consumer = self._rpc = None
        self._clear_samples()
        if rpc is not None:
            rpc.abandon("navigation client closed")
        if consumer is not None:
            await consumer.stop()

    async def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self._control_open():
            raise ConnectionError(self._lost_reason or "no connected control channel")
        result = await asyncio.to_thread(self._rpc.call, method, params, self.rpc_timeout)
        if not isinstance(result, dict):
            raise TypeError(f"{method} returned a non-object response")
        return result

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        if not self._ready or not self._control_open():
            raise ConnectionError(self._lost_reason or "no connected control channel")
        if not self._consumer.send_command(
            {
                "jsonrpc": "2.0",
                "method": method,
                "params": params or {},
            }
        ):
            raise ConnectionError("control channel did not accept the notification")

    def snapshot(self) -> dict[str, Any]:
        snapshot: dict[str, Any] = {
            "connected": self._ready and self._control_open(),
            "state": copy.deepcopy(self._samples["state"]),
            "depth": copy.deepcopy(self._samples["depth"]),
            "camera": None,
            "errors": dict(self._subscription_errors),
        }
        camera = self._samples["camera"]
        if camera is not None:
            metadata = copy.deepcopy(self._video_metadata)
            degrees = int(metadata.get("rotate") or 0) % 360
            if degrees % 90:
                raise ValueError(f"unsupported camera rotation: {degrees}")
            # Match mediad's console and policy-shop: clockwise mount correction.
            picture = np.rot90(camera["image"], k=-(degrees // 90)).copy()
            metadata.update({"pts": camera["pts"], "time_base": camera["time_base"]})
            snapshot["camera"] = {
                "sequence": camera["sequence"],
                "received_at": camera["received_at"],
                "image": picture,
                "metadata": metadata,
            }
        return snapshot
