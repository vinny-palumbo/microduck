"""Live simulator check: observes, moves 0.08 m/s for 0.5 s, then cancels a move.

Run only with the level-floor simulator at 127.0.0.1:8443 and no other controller.
Offline tests are `uv run pytest`; this script deliberately moves the simulated robot.
"""

import asyncio
import json
import math
import time
from pathlib import Path

from duck_nav.cli import Recorder, execute
from duck_nav.core import GuardedRobot
from duck_nav.transport import WebRtcRobot


async def wait_stopped(transport):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        snapshot = transport.snapshot()
        state = (snapshot.get("state") or {}).get("data", {})
        applied = state.get("move", {}).get("applied")
        requested = state.get("move", {}).get("requested")
        # The daemon slews applied velocity exponentially; an exact floating-point
        # zero would reject a stopped robot. Still demand an exactly zero intent.
        if (
            applied is not None
            and all(abs(value) < 0.001 for value in applied)
            and requested is not None
            and all(value == 0 for value in requested)
        ):
            return applied
        await asyncio.sleep(0.05)
    raise AssertionError("applied velocity did not return to zero")


async def main():
    record = Recorder(Path("runs"))
    transport = WebRtcRobot()
    robot = GuardedRobot(transport)
    evidence = {"run_directory": str(record.path.resolve())}
    try:
        await transport.connect()
        initial = await robot.initialize()
        assert initial["ready"], initial["guard_reason"]
        await execute(robot, transport, record, {"tool": "observe"})
        initial_camera = transport.snapshot()["camera"]["sequence"]
        movement = await execute(
            robot,
            transport,
            record,
            {"tool": "move_for", "arguments": {"speed_m_s": 0.08, "duration_s": 0.5}},
        )
        outcome = movement["result"]
        assert outcome["completed"], outcome
        evidence["bounded_move"] = outcome
        before, after = outcome["before_odom"]["position"], outcome["after_odom"]["position"]
        evidence["odometry_displacement_m"] = math.dist(before[:2], after[:2])
        assert evidence["odometry_displacement_m"] > 0.001, "no observed displacement"
        evidence["applied_after_move"] = await wait_stopped(transport)
        assert transport.snapshot()["camera"]["sequence"] > initial_camera

        # Stop races a live action; it must cancel rather than permit later pulses.
        pending = asyncio.create_task(robot.move_for(0.05, 1))
        await asyncio.sleep(0.15)
        evidence["explicit_stop"] = await robot.stop()
        evidence["cancelled_move"] = await pending
        assert evidence["cancelled_move"]["completed"] is False
        assert evidence["cancelled_move"]["reason"] == "stopped"
        await asyncio.sleep(0.2)
        evidence["applied_after_cancel"] = await wait_stopped(transport)
        evidence["final_health"] = await transport.request("robot.health")
        assert evidence["final_health"]["healthy"]
        await execute(robot, transport, record, {"tool": "observe"})
        evidence["passed"] = True
        record.write({"event": "live_smoke", "evidence": evidence})
        (record.path / "smoke.json").write_text(json.dumps(evidence, indent=2) + "\n")
        print(json.dumps(evidence, indent=2))
    except Exception as error:
        evidence.update({"passed": False, "error": f"{type(error).__name__}: {error}"})
        record.write({"event": "live_smoke_failed", "evidence": evidence})
        (record.path / "smoke.json").write_text(json.dumps(evidence, indent=2) + "\n")
        raise
    finally:
        try:
            await robot.close()
        finally:
            await transport.close()


if __name__ == "__main__":
    asyncio.run(main())
