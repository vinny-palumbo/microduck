"""Validate a prerecorded spoken STOP during a model-selected simulator body action.

Run from apps/navigation with the local simulator already running. Both WAVs are
explicit audio fixtures, not a claim that a person spoke during this run. The goal
comes only from the goal WAV; this harness never selects a route or movement tool.
The loopback body protocol is required before WebRTC is opened. Its read-only
samples are verification evidence and never enter a model request or controller.
Exit 0 means all recorded checks passed; exit 2 includes inconclusive runs.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import time
import wave
from pathlib import Path

from calibrate_turns import body_read

from duck_nav.arrival import GeminiArrivalReviewer
from duck_nav.audio import CHUNK_SAMPLES, RATE, wav_chunks
from duck_nav.cli import Recorder
from duck_nav.credentials import load_gemini_key
from duck_nav.live import (
    DEFAULT_MODEL,
    LiveConfig,
    connect_config,
    is_spoken_stop,
    run_live,
    visual_declarations,
)
from duck_nav.navigation import GaitNavigator
from duck_nav.planning import GeminiVisualPlanner
from duck_nav.transport import WebRtcRobot


def vector(value):
    if (
        isinstance(value, list)
        and len(value) == 3
        and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
            for v in value
        )
    ):
        return list(value)
    return None


def motion_state(robot, transport):
    snapshot = transport.snapshot()
    state = snapshot.get("state") or {}
    move = state.get("data", {}).get("move", {})
    requested, applied = vector(move.get("requested")), vector(move.get("applied"))
    fresh = (
        snapshot.get("connected") is True
        and time.monotonic() - state.get("received_at", -math.inf) <= robot.config.state_max_age_s
    )
    active = bool(robot._active)
    commanded = (
        fresh
        and active
        and requested is not None
        and applied is not None
        and any(abs(v) > 0.001 for v in requested)
        and any(abs(v) > 0.001 for v in applied)
    )
    return {
        "fresh": fresh,
        "active": active,
        "requested": requested,
        "applied": applied,
        "commanded": commanded,
    }


def score(evidence, result, final_motion):
    trigger = evidence.get("trigger")
    audio = evidence.get("stop_audio")
    terminal = evidence.get("terminal_at")
    transcript = evidence.get("stop_transcript_at")
    advance = evidence.get("advance_at")
    points = evidence.get("active_trunk_positions", [])
    travel = max((math.dist(points[0], point) for point in points), default=0)
    stop = result.get("stop", {})
    checks = {
        "local_simulator_handshake": evidence.get("simulator_confirmed") is True,
        "model_selected_body_action": advance is not None
        and trigger is not None
        and advance <= trigger["monotonic_s"],
        "active_nonzero_command_trigger": trigger is not None and trigger["commanded"] is True,
        "stop_audio_began_during_motion": audio is not None
        and trigger is not None
        and audio["commanded"] is True
        and trigger["monotonic_s"] <= audio["monotonic_s"],
        "physical_motion_observed": len(points) >= 2 and travel >= 0.001,
        "stop_transcribed_before_terminal": transcript is not None
        and terminal is not None
        and audio is not None
        and audio["monotonic_s"] <= transcript <= terminal,
        "trigger_before_terminal": trigger is not None
        and terminal is not None
        and trigger["monotonic_s"] < terminal,
        "mission_cancelled": result.get("status") == "cancelled",
        "stop_acknowledged": stop.get("completed") is True
        and stop.get("stop", {}).get("acknowledged") is True,
        "final_command_zero": final_motion.get("fresh") is True
        and final_motion.get("requested") == [0, 0, 0]
        and final_motion.get("applied") is not None
        and all(abs(v) < 0.001 for v in final_motion["applied"]),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "active_trunk_displacement_m": travel,
        "physical_settling_verified": False,
        "note": "Verifies spoken cancellation and zero commands; does not certify settled pose.",
    }


async def fixture_audio(args, robot, transport, recorder, evidence):
    recorder.write({"event": "fixture_goal_audio_started", "source": "prerecorded WAV"})
    async for chunk in wav_chunks(args.goal_wav):
        yield chunk
    deadline = time.monotonic() + min(100.0, args.timeout)
    while time.monotonic() < deadline:
        motion = motion_state(robot, transport)
        if motion["commanded"]:
            evidence["trigger"] = {"monotonic_s": time.monotonic(), **motion}
            recorder.write({"event": "voice_stop_motion_trigger", **evidence["trigger"]})
            break
        yield bytes(CHUNK_SAMPLES * 2)
        await asyncio.sleep(CHUNK_SAMPLES / RATE)
    else:
        raise TimeoutError("No active body command before the fixture trigger deadline")
    first = True
    async for chunk in wav_chunks(args.stop_wav):
        if first:
            first = False
            evidence["stop_audio"] = {
                "monotonic_s": time.monotonic(),
                **motion_state(robot, transport),
            }
            recorder.write({"event": "fixture_stop_audio_started", **evidence["stop_audio"]})
        yield chunk
    # Continue a live microphone-like stream until cancellation, including VAD silence.
    while True:
        yield bytes(CHUNK_SAMPLES * 2)
        await asyncio.sleep(CHUNK_SAMPLES / RATE)


async def sample_body(reader, writer, robot, transport, recorder, evidence):
    while True:
        body = await body_read(reader, writer)
        position = vector(body["trunk"])
        if position is None:
            raise ValueError("Invalid simulator body position")
        motion = motion_state(robot, transport)
        if motion["commanded"] and evidence.get("advance_at") is not None:
            evidence["active_trunk_positions"].append(position)
        recorder.write(
            {
                "event": "voice_stop_body_verification",
                "monotonic_s": time.monotonic(),
                "sim_time": body["sim_time"],
                "trunk": position,
                "motion": motion,
            }
        )
        await asyncio.sleep(0.05)


def validate_args(args):
    if not math.isfinite(args.timeout) or not 0 < args.timeout <= 180:
        raise ValueError("timeout must be greater than zero and at most 180 seconds")
    for path in (args.goal_wav, args.stop_wav):
        with wave.open(str(path), "rb") as source:
            if source.getsampwidth() != 2 or source.getnchannels() not in (1, 2):
                raise ValueError("Fixtures must be 16-bit mono or stereo PCM WAVs")
            if source.getnframes() == 0 or source.getcomptype() != "NONE":
                raise ValueError("Fixtures must contain uncompressed PCM audio")


async def run(args):
    validate_args(args)
    from google import genai

    recorder = Recorder(args.runs_dir)
    print(f"Recording to {recorder.path.resolve()}", flush=True)
    evidence = {"active_trunk_positions": [], "simulator_confirmed": False}
    result, final_motion = {"status": "error"}, {}
    transport = WebRtcRobot("127.0.0.1", 8443)
    robot = GaitNavigator(transport)
    client = writer = sampler = mission = None
    heard = ""

    def emit(event):
        nonlocal heard
        now = time.monotonic()
        if event.get("event") == "visual_decision" and event["decision"]["name"] == "advance":
            evidence.setdefault("advance_at", now)
        if event.get("event") == "voice_transcript" and evidence.get("stop_audio"):
            fragment = event.get("text", "")
            heard += fragment
            if is_spoken_stop(fragment) or is_spoken_stop(heard):
                evidence.setdefault("stop_transcript_at", now)
                recorder.write(
                    {
                        "event": "voice_stop_transcript_verified",
                        "monotonic_s": now,
                        "motion": motion_state(robot, transport),
                    }
                )
        if event.get("event") == "live_session_finished":
            evidence["terminal_at"] = now

    recorder.write(
        {
            "event": "voice_stop_validation_started",
            "audio_source": "prerecorded fixtures, not live user speech",
            "goal_wav_sha256": hashlib.sha256(args.goal_wav.read_bytes()).hexdigest(),
            "stop_wav_sha256": hashlib.sha256(args.stop_wav.read_bytes()).hexdigest(),
            "timeout_s": args.timeout,
        }
    )
    try:
        async with asyncio.timeout(args.timeout):
            reader, writer = await asyncio.wait_for(asyncio.open_connection("127.0.0.1", 7801), 3)
            await body_read(reader, writer)
            evidence["simulator_confirmed"] = True
            key = await asyncio.to_thread(load_gemini_key)
            if not key:
                raise ValueError("No stored Gemini key or environment credential")
            await transport.connect()
            await robot.initialize()
            client = genai.Client(api_key=key)
            async with client.aio.live.connect(
                model=DEFAULT_MODEL, config=connect_config("standard")
            ) as session:
                sampler = asyncio.create_task(
                    sample_body(reader, writer, robot, transport, recorder, evidence)
                )
                mission = asyncio.create_task(
                    run_live(
                        robot,
                        transport,
                        session,
                        recorder,
                        audio=fixture_audio(args, robot, transport, recorder, evidence),
                        goal=None,
                        config=LiveConfig(max_seconds=args.timeout),
                        speak=None,
                        emit=emit,
                        arrival_reviewer=GeminiArrivalReviewer(key),
                        navigation_planner=GeminiVisualPlanner(key, visual_declarations()),
                    )
                )
                try:
                    done, _ = await asyncio.wait(
                        {mission, sampler}, return_when=asyncio.FIRST_COMPLETED
                    )
                    if sampler in done:
                        await sampler
                    result = await mission
                finally:
                    # Stop before provider-session cleanup if verification fails
                    # or the outer deadline interrupts a moving mission.
                    if not mission.done():
                        mission.cancel()
                        await robot.stop()
                        await asyncio.gather(mission, return_exceptions=True)
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                final_motion = motion_state(robot, transport)
                if (
                    final_motion["fresh"]
                    and final_motion["requested"] == [0, 0, 0]
                    and final_motion["applied"] is not None
                    and all(abs(v) < 0.001 for v in final_motion["applied"])
                ):
                    break
                await asyncio.sleep(0.025)
    except Exception as error:  # noqa: BLE001 - diagnostic errors must stop and never expose keys
        recorder.write({"event": "voice_stop_validation_error", "error_type": type(error).__name__})
        result = {"status": "error", "error_type": type(error).__name__}
    finally:
        for task in (mission, sampler):
            if task is not None and not task.done():
                task.cancel()
        try:
            await robot.close()
        finally:
            await asyncio.gather(
                *(task for task in (mission, sampler) if task), return_exceptions=True
            )
            await transport.close()
            if client is not None:
                await client.aio.aclose()
                client.close()
            if writer is not None:
                writer.close()
                await writer.wait_closed()
    report = {
        **score(evidence, result, final_motion),
        "evidence": evidence,
        "mission_result": result,
        "final_motion": final_motion,
    }
    recorder.write({"event": "voice_stop_validation_finished", **report})
    (recorder.path / "voice-stop-validation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"passed": report["passed"], "checks": report["checks"]}), flush=True)
    return 0 if report["passed"] else 2


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--goal-wav", type=Path, required=True)
    result.add_argument("--stop-wav", type=Path, required=True)
    result.add_argument("--timeout", type=float, default=120)
    result.add_argument(
        "--runs-dir", type=Path, default=Path(__file__).resolve().parents[1] / "runs"
    )
    return result


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(run(parser().parse_args())))
    except KeyboardInterrupt:
        raise SystemExit(2) from None
    except Exception as error:  # noqa: BLE001 - credentials/provider exceptions must stay redacted
        print(f"Voice stop validation failed: {type(error).__name__}")
        raise SystemExit(2) from None
