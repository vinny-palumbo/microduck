"""Verify streaming audio framing, resampling and subprocess cleanup."""

import asyncio
import wave

import numpy as np
import pytest

from duck_nav.audio import (
    CHUNK_SAMPLES,
    RATE,
    PcmActivity,
    microphone_chunks,
    speak_local,
    wav_chunks,
)


def test_manual_activity_retains_prefix_and_closes_each_short_utterance():
    segmenter = PcmActivity()
    quiet = np.full(1600, 20, dtype="<i2").tobytes()
    voiced = np.full(1600, 2000, dtype="<i2").tobytes()
    for _ in range(20):
        assert segmenter.push(quiet) == []
    first = segmenter.push(voiced)
    assert first[0] == {"activity_start": {}}
    assert first[1]["audio"]["data"] == quiet * 2 + voiced
    assert segmenter.push(voiced) == [PcmActivity.audio(voiced)]
    for _ in range(4):
        assert segmenter.push(quiet) == [PcmActivity.audio(quiet)]
    assert segmenter.push(quiet) == [PcmActivity.audio(quiet), {"activity_end": {}}]
    assert segmenter.push(quiet) == []
    assert segmenter.push(voiced)[0] == {"activity_start": {}}
    assert segmenter.finish() == [{"activity_end": {}}]
    assert segmenter.finish() == []


def test_activity_silence_uses_sample_duration_not_chunk_count():
    segmenter = PcmActivity()
    segmenter.push(np.full(160, 2000, dtype="<i2").tobytes())
    messages = []
    for _ in range(49):
        messages.extend(segmenter.push(bytes(320)))
    assert not any("activity_end" in message for message in messages)
    assert segmenter.push(bytes(320))[-1] == {"activity_end": {}}


def test_low_level_noise_never_opens_a_turn_and_prefix_is_bounded():
    segmenter = PcmActivity()
    noise = np.tile(np.array([-299, 299], dtype="<i2"), 800).tobytes()
    for _ in range(200):
        assert segmenter.push(noise) == []
    assert len(segmenter.prefix) == segmenter.prefix_samples * 2
    assert segmenter.finish() == []


def test_continuous_energy_cannot_leave_a_turn_open_indefinitely():
    segmenter = PcmActivity()
    chunk = np.full(1600, 3000, dtype="<i2").tobytes()
    messages = []
    for _ in range(150):
        messages.extend(segmenter.push(chunk))
    assert messages[-1] == {"activity_end": {}}
    assert not segmenter.active


@pytest.mark.parametrize("bad", [None, b"", b"x", "pcm"])
def test_activity_rejects_invalid_pcm(bad):
    with pytest.raises(ValueError):
        PcmActivity().push(bad)


@pytest.mark.parametrize("rate,channels", [(16000, 1), (48000, 2), (22050, 1)])
async def test_wav_resampling_produces_pcm16_mono_and_trailing_silence(tmp_path, rate, channels):
    path = tmp_path / "instruction.wav"
    duration = 0.2
    tone = (np.sin(np.arange(round(rate * duration)) * 2 * np.pi * 440 / rate) * 10000).astype(
        "<i2"
    )
    signal = np.repeat(tone[:, None], channels, axis=1)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(signal.tobytes())
    chunks = [chunk async for chunk in wav_chunks(path, realtime=False)]
    samples = np.frombuffer(b"".join(chunks), dtype="<i2")
    assert abs(len(samples) - RATE * (duration + 0.8)) <= 2
    assert max(abs(samples)) > 5000
    assert not np.any(samples[-8 * CHUNK_SAMPLES :])
    assert all(len(chunk) % 2 == 0 for chunk in chunks)


async def test_wav_rejects_8bit_audio(tmp_path):
    path = tmp_path / "bad.wav"
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(1)
        output.setframerate(16000)
        output.writeframes(b"\0" * 160)
    with pytest.raises(ValueError, match="16-bit"):
        _ = [chunk async for chunk in wav_chunks(path, realtime=False)]


class Process:
    def __init__(self):
        self.stdout = asyncio.StreamReader()
        self.stdout.feed_data(bytes(3200))
        self.returncode = None
        self.terminated = False
        self.input = None

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    async def wait(self):
        return self.returncode

    async def communicate(self, input):
        self.input = input
        self.returncode = 0
        return b"", b""


async def test_microphone_prefers_parec_and_closes_on_cancellation(monkeypatch):
    process, command = Process(), []

    async def create(*args, **kwargs):
        command.extend(args)
        return process

    monkeypatch.setattr("duck_nav.audio.shutil.which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr("duck_nav.audio.asyncio.create_subprocess_exec", create)
    source = microphone_chunks("test-source")
    assert len(await anext(source)) == 3200
    task = asyncio.create_task(anext(source))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert process.terminated
    assert command[0] == "/usr/bin/parec"
    assert "--device=test-source" in command


async def test_unexpected_microphone_eof_is_failure(monkeypatch):
    process = Process()
    process.stdout.feed_eof()

    async def create(*args, **kwargs):
        return process

    monkeypatch.setattr("duck_nav.audio.shutil.which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr("duck_nav.audio.asyncio.create_subprocess_exec", create)
    source = microphone_chunks()
    await anext(source)
    with pytest.raises(ConnectionError, match="Microphone ended"):
        await anext(source)
    assert process.terminated


async def test_tts_passes_model_text_as_stdin(monkeypatch):
    process, command = Process(), []

    async def create(*args, **kwargs):
        command.extend(args)
        return process

    monkeypatch.setattr("duck_nav.audio.shutil.which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr("duck_nav.audio.asyncio.create_subprocess_exec", create)
    text = "Kitchen; $(touch /tmp/must-not-run)"
    await speak_local(text)
    assert process.input == text.encode()
    assert text not in command
