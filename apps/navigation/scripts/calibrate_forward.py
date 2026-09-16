"""Simulator-only forward response probe; not a model tool or distance guarantee.

Three sequential 0.30 m/s probes, each capped at 1.5 seconds, 50 mm odometry or
10 degrees heading change, with ordinary depth/health guards. Stops on any guard
abort or settled heading excursion above 10 degrees. Uses MuJoCo truth only to
score results; this script must not be pointed at hardware.
"""

import asyncio
import json
import math
from pathlib import Path

from calibrate_turns import body_read, yaw

from duck_nav.cli import Recorder
from duck_nav.core import GuardedRobot
from duck_nav.transport import WebRtcRobot


async def main():
    t = WebRtcRobot()
    r = GuardedRobot(t)
    record = Recorder(Path("runs"))
    writer = None
    print(str(record.path), flush=True)
    results = []
    passed = False
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", 7801)
        await body_read(reader, writer)
        await t.connect()
        await r.initialize()
        await r.stop()
        await asyncio.sleep(1)
        for speed in [0.3] * 3:
            look = await r.look_at(1, 0, 0)
            record.write({"look": look, "snapshot": record.capture(t.snapshot())})
            if not look["completed"]:
                print(json.dumps(look), flush=True)
                break
            await asyncio.sleep(1)
            before = await body_read(reader, writer)
            snap = t.snapshot()
            origin = snap["state"]["data"]["odom"]
            record.write({"before": record.capture(snap)})
            action = asyncio.create_task(r._motion("simulator_forward_probe", speed, 0, 1.5))
            cutoff = None
            try:
                while not action.done():
                    odom = t.snapshot()["state"]["data"]["odom"]
                    distance = math.dist(origin["position"][:2], odom["position"][:2])
                    angle = abs(
                        math.atan2(
                            math.sin(odom["yaw"] - origin["yaw"]),
                            math.cos(odom["yaw"] - origin["yaw"]),
                        )
                    )
                    if distance >= 0.05 or angle >= math.radians(10):
                        cutoff = "distance" if distance >= 0.05 else "heading"
                        await r.stop()
                        break
                    await asyncio.sleep(0.025)
                outcome = await action
            finally:
                if not action.done():
                    action.cancel()
                    try:
                        await action
                    except asyncio.CancelledError:
                        pass
            await asyncio.sleep(2)
            after = await body_read(reader, writer)
            snap = t.snapshot()
            result = {
                "speed": speed,
                "duration": 1.5,
                "outcome": outcome,
                "cutoff": cutoff,
                "truth_displacement_m": math.dist(before["trunk"][:2], after["trunk"][:2]),
                "truth_yaw_deg": math.degrees(
                    math.atan2(
                        math.sin(yaw(after["imu"]["quat"]) - yaw(before["imu"]["quat"])),
                        math.cos(yaw(after["imu"]["quat"]) - yaw(before["imu"]["quat"])),
                    )
                ),
                "final_move": snap["state"]["data"]["move"],
            }
            record.write({"result": result, "after": record.capture(snap)})
            results.append(result)
            (record.path / "forward.json").write_text(json.dumps(results, indent=2))
            print(json.dumps(result), flush=True)
            if (not outcome["completed"] and cutoff != "distance") or abs(
                result["truth_yaw_deg"]
            ) > 10:
                break
        passed = len(results) == 3 and all(
            item["cutoff"] == "distance"
            and abs(item["truth_displacement_m"] - 0.05) <= 0.02
            and abs(item["truth_yaw_deg"]) <= 10
            and item["outcome"]["stop"]["acknowledged"]
            and all(v == 0 for v in item["final_move"]["requested"])
            and all(abs(v) < 0.001 for v in item["final_move"]["applied"])
            for item in results
        )
        (record.path / "forward-summary.json").write_text(
            json.dumps({"passed": passed, "results": results}, indent=2)
        )
    finally:
        try:
            await r.close()
        finally:
            await t.close()
        if writer:
            writer.close()
            await writer.wait_closed()
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
