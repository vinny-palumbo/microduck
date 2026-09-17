"""Score one to five consecutive walking arcs using independent simulator truth.

A pass checks settled progress and agreement with odometry, not general path
accuracy. Independent truth is recorded only; it never enters the executor.
"""

import argparse
import asyncio
import json
import math
import time
from pathlib import Path

from calibrate_turns import body_read, yaw

from duck_nav.cli import Recorder
from duck_nav.navigation import GaitNavigator, angle_delta
from duck_nav.transport import WebRtcRobot


def independent_change(before, after):
    return {
        "distance_m": math.dist(before["trunk"][:2], after["trunk"][:2]),
        "heading_deg": math.degrees(
            angle_delta(yaw(after["imu"]["quat"]), yaw(before["imu"]["quat"]))
        ),
        "before_position": before["trunk"],
        "after_position": after["trunk"],
    }


async def run(args):
    steps_requested = getattr(args, "steps", 1)
    if type(steps_requested) is not int or not 1 <= steps_requested <= 5:
        raise ValueError("steps must be an integer from 1 to 5")
    recorder = Recorder(Path("runs"))
    print(recorder.path, flush=True)
    # Requiring the local body protocol keeps this calibration off a physical robot.
    reader, writer = await asyncio.open_connection("127.0.0.1", 7801)
    await body_read(reader, writer)
    transport = WebRtcRobot()
    robot = GaitNavigator(transport)
    try:
        await transport.connect()
        await robot.initialize()
        if not (await robot.stop())["completed"]:
            raise RuntimeError("initial stop was not acknowledged")
        gaze = await robot.look_at(1, 0, 0)
        recorder.write({"event": "gaze", "result": gaze})
        if not gaze["completed"]:
            raise RuntimeError(f"camera recenter failed: {gaze['reason']}")
        await asyncio.sleep(1)
        steps = []
        initial = None
        for index in range(steps_requested):
            before = await body_read(reader, writer)
            if initial is None:
                initial = before
            recorder.capture(transport.snapshot())
            started_wall, started = time.time(), time.monotonic()
            # advance performs its own guarded stop and settling. A public
            # stop here would erase the course retained between these arcs.
            result = await robot.advance(args.distance, args.heading)
            ended_wall, elapsed = time.time(), time.monotonic() - started
            after = await body_read(reader, writer)
            recorder.capture(transport.snapshot())
            observation = await robot.observe()
            truth = independent_change(before, after)
            passed = (
                result["completed"]
                and result["stop"]["acknowledged"]
                and result["stop"]["physical_settling_verified"]
                and observation["ready"]
                and truth["distance_m"] >= 0.03
                and abs(truth["distance_m"] - result["distance_m"]) < 0.02
                and abs(truth["heading_deg"] - result["heading_deg"]) < 2
            )
            step = {
                "step": index + 1,
                "requested": {"distance_m": args.distance, "heading_deg": args.heading},
                "wall_interval": {
                    "start": started_wall,
                    "end": ended_wall,
                    "elapsed_s": elapsed,
                },
                "result": result,
                "truth": truth,
                "course": observation.get("course"),
                "guard_reason": observation["guard_reason"],
                "passed": passed,
            }
            steps.append(step)
            recorder.write(
                {"event": "calibration_step", **step, "body_before": before, "body_after": after}
            )
            if not passed:
                break
        passed = len(steps) == steps_requested and all(step["passed"] for step in steps)
        report = {
            "result": result,
            "truth": independent_change(initial, after),
            "passed": passed,
            "passed_scope": "Settled progress and odometry agreement; not general path accuracy.",
            "steps_requested": steps_requested,
            "steps_executed": len(steps),
            "steps": steps,
        }
        recorder.write({"event": "calibration", **report})
        (recorder.path / "calibration.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
        return 0 if passed else 2
    finally:
        await robot.close()
        await transport.close()
        writer.close()
        await writer.wait_closed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distance", type=float, default=0.1)
    parser.add_argument("--heading", type=float, default=0)
    parser.add_argument("--steps", type=int, choices=range(1, 6), default=1)
    raise SystemExit(asyncio.run(run(parser.parse_args())))
