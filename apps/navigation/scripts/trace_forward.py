"""Trace one simulator-only forward probe; restart the simulator before each run.

Compare --posture neutral and recentered with the same policy/scene. Motion uses
WebRTC guards, 0.30 m/s, a 1.5 s cap, 50 mm odometry and 10 degree yaw cutoffs.
MuJoCo truth is recorded for evaluation only, never used for control.
"""

import argparse
import asyncio
import hashlib
import itertools
import json
import math
import statistics
import time
from dataclasses import asdict
from pathlib import Path

from calibrate_turns import body_read, unwrap, yaw

from duck_nav.cli import Recorder
from duck_nav.core import GuardedRobot
from duck_nav.transport import WebRtcRobot


def summarize(samples, start, end):
    angles = unwrap([s["truth_yaw"] for s in samples])
    pre = [i for i, s in enumerate(samples) if start - 0.3 <= s["t"] < start]
    post = [i for i, s in enumerate(samples) if s["t"] >= samples[-1]["t"] - 0.3]
    if len(pre) < 2 or len(post) < 2:
        raise ValueError("insufficient baseline or settled samples")
    reference = statistics.mean(angles[i] for i in pre)

    def at(t):
        return min(range(len(samples)), key=lambda i: abs(samples[i]["t"] - t))

    early, stopped = at(min(start + 0.5, end)), at(end)
    settled = statistics.mean(angles[i] for i in post)
    return {
        "startup_0_5s_deg": math.degrees(angles[early] - reference),
        "walking_after_0_5s_deg": math.degrees(angles[stopped] - angles[early]),
        "post_action_deg": math.degrees(settled - angles[stopped]),
        "total_deg": math.degrees(settled - reference),
        "heading_at_action_end_deg": math.degrees(angles[stopped] - reference),
        "peak_abs_deg": max(abs(math.degrees(a - reference)) for a in angles),
        "settled_span_deg": math.degrees(
            max(angles[i] for i in post) - min(angles[i] for i in post)
        ),
        "distance_m": math.dist(samples[pre[-1]]["trunk"][:2], samples[post[-1]]["trunk"][:2]),
        "action_s": end - start,
        "max_gap_s": max(b["t"] - a["t"] for a, b in itertools.pairwise(samples)),
        "requested_yaw_max": max(abs(s["requested"][2]) for s in samples),
        "applied_yaw_max": max(abs(s["applied"][2]) for s in samples),
        "final_requested": samples[-1]["requested"],
        "final_applied": samples[-1]["applied"],
    }


async def main(posture):
    record = Recorder(Path("runs"))
    print(record.path, flush=True)
    transport = WebRtcRobot()
    robot = GuardedRobot(transport)
    writer = sampler = action = None
    samples = []
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", 7801)
        await body_read(reader, writer)
        await transport.connect()
        await robot.initialize()
        if not (await robot.stop())["completed"]:
            raise RuntimeError("initial stop failed")
        if posture == "neutral":
            transport.notify(
                "robot.head", {"neck_pitch": 0, "head_pitch": 0, "head_yaw": 0, "head_roll": 0}
            )
        else:
            look = await robot.look_at(1, 0, 0)
            record.write({"look": look})
            if not look["completed"]:
                raise RuntimeError("recenter failed")
        await asyncio.sleep(1)
        policy = Path.home() / ".cache/duck-sim/policies/current/velstand.onnx"
        record.write(
            {
                "posture": posture,
                "policy_sha256": hashlib.sha256(policy.read_bytes()).hexdigest(),
                "guard": asdict(robot.config),
                "before": record.capture(transport.snapshot()),
            }
        )

        async def sample():
            while True:
                truth = await body_read(reader, writer)
                snapshot = transport.snapshot()
                state = snapshot["state"]["data"]
                row = {
                    "t": time.monotonic(),
                    "sim_time": truth["sim_time"],
                    "truth_yaw": yaw(truth["imu"]["quat"]),
                    "trunk": truth["trunk"],
                    "odom_yaw": state["odom"]["yaw"],
                    "odom_position": state["odom"]["position"],
                    "requested": state["move"]["requested"],
                    "applied": state["move"]["applied"],
                    "head": state["head"],
                    "joints": state["joints"],
                    "gyro": state["imu"]["gyro"],
                }
                samples.append(row)
                record.write({"sample": row})
                await asyncio.sleep(0.02)

        sampler = asyncio.create_task(sample())
        await asyncio.sleep(0.6)
        if sampler.done():
            await sampler
        origin = transport.snapshot()["state"]["data"]["odom"]
        start = time.monotonic()
        action = asyncio.create_task(robot._motion("forward_trace", 0.3, 0, 1.5))
        cutoff = None
        while not action.done():
            if sampler.done():
                await sampler
            odom = transport.snapshot()["state"]["data"]["odom"]
            rotation = math.atan2(
                math.sin(odom["yaw"] - origin["yaw"]), math.cos(odom["yaw"] - origin["yaw"])
            )
            if (
                abs(rotation) >= math.radians(10)
                or math.dist(odom["position"][:2], origin["position"][:2]) >= 0.05
            ):
                cutoff = "heading" if abs(rotation) >= math.radians(10) else "distance"
                await robot.stop()
                break
            await asyncio.sleep(0.025)
        outcome = await action
        end = time.monotonic()
        await asyncio.sleep(2)
        if sampler.done():
            await sampler
        sampler.cancel()
        try:
            await sampler
        except asyncio.CancelledError:
            pass
        result = {
            "posture": posture,
            "cutoff": cutoff,
            "outcome": outcome,
            "start": start,
            "end": end,
            "summary": summarize(samples, start, end),
        }
        record.write({"result": result, "after": record.capture(transport.snapshot())})
        (record.path / "trace.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        return 0
    finally:
        for pending in (action, sampler):
            if pending is not None and not pending.done():
                pending.cancel()
                try:
                    await pending
                except asyncio.CancelledError:
                    pass
        try:
            await robot.close()
        finally:
            await transport.close()
            if writer:
                writer.close()
                await writer.wait_closed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--posture", choices=("neutral", "recentered"), required=True)
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(args.posture)))
