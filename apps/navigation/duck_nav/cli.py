"""CLI and JSON-lines tools; model credentials are deliberately unnecessary."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import stat
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from PIL import Image

from .core import GuardedRobot
from .transport import WebRtcRobot

TOOLS = [
    {
        "name": "observe",
        "description": "Read current camera, depth, pose and motion readiness.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "move_for",
        "description": "Move forward at up to 0.10 m/s for at most 2 seconds.",
        "parameters": {
            "type": "object",
            "properties": {
                "speed_m_s": {"type": "number", "exclusiveMinimum": 0, "maximum": 0.1},
                "duration_s": {"type": "number", "exclusiveMinimum": 0, "maximum": 2},
            },
            "required": ["speed_m_s", "duration_s"],
            "additionalProperties": False,
        },
    },
    {
        "name": "turn_by",
        "description": "Turn up to 30 degrees using odometry; positive is left.",
        "parameters": {
            "type": "object",
            "properties": {
                "angle_deg": {"type": "number", "minimum": -30, "maximum": 30, "not": {"const": 0}}
            },
            "required": ["angle_deg"],
            "additionalProperties": False,
        },
    },
    {
        "name": "look_at",
        "description": "Look toward a trunk-frame point in metres (x forward, y left, z up).",
        "parameters": {
            "type": "object",
            "properties": {axis: {"type": "number"} for axis in ("x", "y", "z")},
            "required": ["x", "y", "z"],
            "additionalProperties": False,
        },
    },
    {
        "name": "stop",
        "description": "Cancel movement and request standing still.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
]


class Recorder:
    """One inspectable run directory; no unbounded in-memory frame queue."""

    def __init__(self, root: Path):
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.path = root / f"{stamp}-{uuid4().hex[:8]}"
        self.path.mkdir(parents=True)
        self.count = 0

    def write(self, event: dict) -> None:
        with (self.path / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"at": time.time(), **event}, allow_nan=False) + "\n")

    def capture(self, snapshot: dict) -> dict:
        now = time.monotonic()
        record = {"connected": snapshot.get("connected", False)}
        for name in ("camera", "state", "depth"):
            sample = snapshot.get(name)
            if sample is None:
                record[name] = None
                continue
            clean = {key: value for key, value in sample.items() if key != "image"}
            clean["age_s"] = now - sample["received_at"]
            if name == "camera" and sample.get("image") is not None:
                self.count += 1
                path = self.path / f"frame-{self.count:04d}.jpg"
                Image.fromarray(sample["image"]).save(path, quality=90)
                clean["image_path"] = str(path.resolve())
            record[name] = clean
        return record


async def dispatch(robot: GuardedRobot, request: dict) -> dict:
    if not isinstance(request, dict) or set(request) - {"id", "tool", "arguments"}:
        raise ValueError("Expected {id?, tool, arguments?}")
    name = request.get("tool")
    if name not in {tool["name"] for tool in TOOLS}:
        raise ValueError(f"Unknown tool: {name!r}")
    arguments = request.get("arguments", {})
    if not isinstance(arguments, dict):
        raise TypeError("arguments must be an object")
    return await getattr(robot, name)(**arguments)


async def stdin_lines():
    # An executor thread blocked in readline cannot be cancelled at Ctrl-C. Pipes and
    # terminals can instead be consumed by the event loop; regular files never block.
    if stat.S_ISREG(os.fstat(sys.stdin.fileno()).st_mode):
        for line in sys.stdin:
            yield line
        return
    reader = asyncio.StreamReader()
    pipe, _ = await asyncio.get_running_loop().connect_read_pipe(
        lambda: asyncio.StreamReaderProtocol(reader), sys.stdin
    )
    try:
        while line := await reader.readline():
            yield line.decode("utf-8")
    finally:
        pipe.close()


def reject_constant(value: str):
    raise ValueError(f"Non-finite JSON number: {value}")


async def serve(robot, transport, recorder, lines, emit):
    active = None

    async def respond(request):
        try:
            result = await execute(robot, transport, recorder, request)
        except asyncio.CancelledError:
            result = {"id": request.get("id"), "error": "cancelled: input closed"}
            recorder.write({"event": "cancelled", "request": request})
        except (ValueError, TypeError, RuntimeError, OSError) as error:
            result = {
                "id": request.get("id") if isinstance(request, dict) else None,
                "error": f"{type(error).__name__}: {error}",
            }
            recorder.write({"request": request, "result": result})
        emit(result)

    try:
        async for line in lines:
            try:
                request = json.loads(line, parse_constant=reject_constant)
            except ValueError as error:
                emit({"error": str(error)})
                continue
            if isinstance(request, dict) and request.get("tool") == "stop":
                await respond(request)
            elif active is not None and not active.done():
                emit(
                    {
                        "id": request.get("id") if isinstance(request, dict) else None,
                        "error": "An action is running; send stop or wait for its result",
                    }
                )
            else:
                active = asyncio.create_task(respond(request))
                # Let the action register before accepting a following stop/EOF.
                await asyncio.sleep(0)
    finally:
        if active is not None:
            if not active.done():
                active.cancel()
            await active


async def run(args: argparse.Namespace) -> int:
    if args.command == "tools":
        print(json.dumps(TOOLS, indent=2))
        return 0
    recorder = Recorder(Path(args.runs_dir))
    print(f"Recording to {recorder.path.resolve()}", file=sys.stderr)
    transport = WebRtcRobot(host=args.host, port=args.port)
    robot = GuardedRobot(transport)
    try:
        await transport.connect()
        # A stop remains available even if sensors are absent or the robot is unhealthy.
        if args.command != "stop":
            await robot.initialize()
        if args.command == "serve":
            print('Ready for JSON lines: {"tool":"observe"}', file=sys.stderr)
            await serve(
                robot,
                transport,
                recorder,
                stdin_lines(),
                lambda result: print(json.dumps(result, allow_nan=False), flush=True),
            )
            return 0
        request = {"tool": args.command, "arguments": {}}
        if args.command == "move_for":
            request["arguments"] = {"speed_m_s": args.speed, "duration_s": args.seconds}
        elif args.command == "turn_by":
            request["arguments"] = {"angle_deg": args.degrees}
        elif args.command == "look_at":
            request["arguments"] = {"x": args.x, "y": args.y, "z": args.z}
        result = await execute(robot, transport, recorder, request)
        print(json.dumps(result, indent=2, allow_nan=False))
        outcome = result["result"]
        return 2 if outcome.get("completed") is False else 0
    finally:
        try:
            await robot.close()
        finally:
            await transport.close()


async def execute(
    robot: GuardedRobot, transport: WebRtcRobot, recorder: Recorder, request: dict
) -> dict:
    # Persist the intent before executing so interruption still leaves an audit trail.
    recorder.write({"event": "requested", "request": request})
    outcome = await dispatch(robot, request)
    response = {
        "id": request.get("id"),
        "result": outcome,
        "observation": recorder.capture(transport.snapshot()),
    }
    recorder.write({"event": "finished", "request": request, "response": response})
    return response


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--host", default="127.0.0.1")
    result.add_argument("--port", type=int, default=8443)
    result.add_argument("--runs-dir", default="runs")
    commands = result.add_subparsers(dest="command", required=True)
    for name in ("tools", "observe", "stop", "serve"):
        commands.add_parser(name)
    move = commands.add_parser("move_for")
    move.add_argument("--speed", type=float, default=0.05)
    move.add_argument("--seconds", type=float, default=0.5)
    turn = commands.add_parser("turn_by")
    turn.add_argument("degrees", type=float)
    look = commands.add_parser("look_at")
    for axis in ("x", "y", "z"):
        look.add_argument(axis, type=float)
    return result


def main() -> None:
    try:
        code = asyncio.run(run(parser().parse_args()))
    except KeyboardInterrupt:
        code = 130
    except (ValueError, TypeError, RuntimeError, OSError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        code = 1
    raise SystemExit(code)


if __name__ == "__main__":
    main()
