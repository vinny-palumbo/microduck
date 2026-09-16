"""Cancellable 16 kHz PCM sources for the persistent robot conversation."""

from __future__ import annotations

import asyncio
import shutil
import wave
from collections.abc import AsyncIterator
from pathlib import Path

import av
import numpy as np

RATE = 16000
CHUNK_SAMPLES = 1600


async def wav_chunks(path: Path, *, realtime: bool = True) -> AsyncIterator[bytes]:
    """Play an actual PCM WAV into the microphone path, including VAD trailing silence.

    Pacing is necessary: sending an entire recording in one WebSocket message does
    not exercise the same activity detector as live speech.
    """
    with wave.open(str(path), "rb") as source:
        if source.getsampwidth() != 2 or source.getnchannels() not in (1, 2):
            raise ValueError("WAV input must be uncompressed 16-bit mono or stereo PCM")
        channels, rate = source.getnchannels(), source.getframerate()
        resampler = av.AudioResampler(format="s16", layout="mono", rate=RATE)
        buffered = bytearray()
        deadline = asyncio.get_running_loop().time()

        async def paced(chunk):
            nonlocal deadline
            if realtime:
                await asyncio.sleep(max(0, deadline - asyncio.get_running_loop().time()))
                deadline += len(chunk) / (2 * RATE)
            return chunk

        while raw := source.readframes(max(1, rate // 10)):
            frame = av.AudioFrame.from_ndarray(
                np.frombuffer(raw, dtype="<i2").reshape(1, -1),
                format="s16",
                layout="mono" if channels == 1 else "stereo",
            )
            frame.sample_rate = rate
            for output in resampler.resample(frame):
                buffered.extend(output.to_ndarray().astype("<i2", copy=False).tobytes())
            while len(buffered) >= CHUNK_SAMPLES * 2:
                chunk = bytes(buffered[: CHUNK_SAMPLES * 2])
                del buffered[: CHUNK_SAMPLES * 2]
                yield await paced(chunk)
        for output in resampler.resample(None):
            buffered.extend(output.to_ndarray().astype("<i2", copy=False).tobytes())
        if buffered:
            yield await paced(bytes(buffered))
        for _ in range(8):
            yield await paced(bytes(CHUNK_SAMPLES * 2))


async def _terminate(process: asyncio.subprocess.Process) -> None:
    if process.returncode is None:
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), 2)
        except TimeoutError:
            process.kill()
            await process.wait()


async def microphone_chunks(device: str = "default") -> AsyncIterator[bytes]:
    """Capture the Linux/WSLg PulseAudio default microphone.

    A subprocess keeps shutdown cancellable even when an audio device stalls.
    WSLg exposes the Windows microphone through its PulseAudio server.
    """
    executable = shutil.which("parec") or shutil.which("parecord")
    if executable:
        command = [
            executable,
            "--raw",
            "--format=s16le",
            "--rate=16000",
            "--channels=1",
            "--latency-msec=20",
        ]
        if device != "default":
            command.append(f"--device={device}")
    elif executable := shutil.which("ffmpeg"):
        command = [
            executable,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "pulse",
            "-i",
            device,
            "-ac",
            "1",
            "-ar",
            str(RATE),
            "-f",
            "s16le",
            "pipe:1",
        ]
    else:
        raise RuntimeError("Local microphone requires parec or ffmpeg and a PulseAudio input")
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        while True:
            try:
                chunk = await process.stdout.readexactly(CHUNK_SAMPLES * 2)
            except asyncio.IncompleteReadError as error:
                raise ConnectionError(
                    "Microphone ended; check the PulseAudio input device"
                ) from error
            yield chunk
    finally:
        await _terminate(process)


async def speak_local(message: str) -> None:
    """Speak text on the bridge host; never interpret generated text as shell code."""
    executable = shutil.which("espeak-ng") or shutil.which("espeak")
    if executable is None:
        raise RuntimeError("Local speech requires espeak-ng or espeak")
    process = await asyncio.create_subprocess_exec(
        executable,
        "--stdin",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        await asyncio.wait_for(process.communicate(message.encode("utf-8")), 30)
        if process.returncode:
            raise RuntimeError("Local speech playback failed")
    finally:
        await _terminate(process)
