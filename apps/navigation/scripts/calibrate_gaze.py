"""Measure bounded gaze feedback through WebRTC; no body commands.

Run against the local level-floor simulator with no other controller.
Exit 1 means at least one target failed; recordings retain every result.
"""

import asyncio
import json
from pathlib import Path

from duck_nav.cli import Recorder
from duck_nav.core import GuardedRobot
from duck_nav.transport import WebRtcRobot


async def main():
    transport = WebRtcRobot()
    robot = GuardedRobot(transport)
    record = Recorder(Path("runs"))
    results = []
    try:
        await transport.connect()
        await robot.initialize()
        for _ in range(2):
            for point in ((1, 0, 0), (1, 0.25, 0), (1, -0.25, 0), (1, 0, 0.15), (1, 0, 0)):
                outcome = await robot.look_at(*point)
                result = {"target": point, "outcome": outcome}
                results.append(result)
                record.write({**result, "snapshot": record.capture(transport.snapshot())})
                print(json.dumps(result), flush=True)
                if not outcome.get("stop", {}).get("acknowledged", False):
                    break
            else:
                continue
            break
    finally:
        try:
            await robot.close()
        finally:
            await transport.close()
    passed = len(results) == 10 and all(r["outcome"]["completed"] for r in results)
    summary = {"passed": passed, "results": results}
    (record.path / "gaze.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(record.path, flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
