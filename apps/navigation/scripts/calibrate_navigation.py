"""Score one walking arc in the real daemon/MuJoCo loop using independent truth."""

import argparse
import asyncio
import json
import math
from pathlib import Path

from calibrate_turns import body_read, yaw

from duck_nav.cli import Recorder
from duck_nav.navigation import GaitNavigator, angle_delta
from duck_nav.transport import WebRtcRobot


async def run(args):
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
        await robot.stop()
        gaze = await robot.look_at(1, 0, 0)
        recorder.write({"event": "gaze", "result": gaze})
        if not gaze["completed"]:
            raise RuntimeError(f"camera recenter failed: {gaze['reason']}")
        await asyncio.sleep(1)
        before = await body_read(reader, writer)
        recorder.capture(transport.snapshot())
        result = await robot.advance(args.distance, args.heading)
        after = await body_read(reader, writer)
        recorder.capture(transport.snapshot())
        truth = {
            "distance_m": math.dist(before["trunk"][:2], after["trunk"][:2]),
            "heading_deg": math.degrees(
                angle_delta(yaw(after["imu"]["quat"]), yaw(before["imu"]["quat"]))
            ),
            "before_position": before["trunk"],
            "after_position": after["trunk"],
        }
        passed = (
            result["completed"]
            and result["stop"]["physical_settling_verified"]
            and truth["distance_m"] >= 0.03
            and abs(truth["distance_m"] - result["distance_m"]) < 0.02
            and abs(truth["heading_deg"] - result["heading_deg"]) < 2
        )
        report = {"result": result, "truth": truth, "passed": passed}
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
    raise SystemExit(asyncio.run(run(parser.parse_args())))
