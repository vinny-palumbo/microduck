"""Measure guarded turns against MuJoCo ground truth (local simulator only).

Ground truth is recorded for evaluation; it is never passed to the controller.
Start the flat-apartment simulator, then run from apps/navigation with uv run.
"""

import argparse
import asyncio
import hashlib
import itertools
import json
import math
import statistics
import sys
import time
import tomllib
from dataclasses import asdict
from pathlib import Path

from duck_nav.cli import Recorder
from duck_nav.core import GuardedRobot
from duck_nav.transport import WebRtcRobot


def yaw(quat):
    w, x, y, z = quat
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def unwrap(values):
    out = [values[0]]
    for before, after in itertools.pairwise(values):
        out.append(out[-1] + math.atan2(math.sin(after - before), math.cos(after - before)))
    return out


def summarize(samples, outcome, angle, action_start, action_end):
    truth = unwrap([sample["truth_yaw"] for sample in samples])
    odom = unwrap([sample["odom_yaw"] for sample in samples])
    pre = [i for i, s in enumerate(samples) if action_start - 0.3 <= s["t"] < action_start]
    post = [i for i, s in enumerate(samples) if s["t"] > samples[-1]["t"] - 0.3]
    if not pre or len(post) < 2:
        raise ValueError("insufficient baseline or settled samples")
    reference = statistics.mean(truth[i] for i in pre)
    settled = statistics.mean(truth[i] for i in post) - reference
    odom_settled = statistics.mean(odom[i] for i in post) - statistics.mean(odom[i] for i in pre)
    error = math.degrees(settled) - angle if angle is not None else None
    tolerance = min(2.0, abs(angle) / 4) if angle is not None else None
    post_span = math.degrees(max(truth[i] for i in post) - min(truth[i] for i in post))
    zero_command = (
        samples[-1]["requested_yaw_rate"] == 0 and abs(samples[-1]["applied_yaw_rate"]) < 0.001
    )
    # A transient crossing of the target must not count as an accurate turn.
    passed = (
        (
            outcome["completed"]
            and outcome["stop"]["acknowledged"]
            and abs(error) <= tolerance
            and post_span <= 0.5
            and zero_command
            and abs(math.degrees(odom_settled - settled)) <= 0.5
        )
        if angle is not None
        else None
    )
    requested_integral = sum(
        a["requested_yaw_rate"] * (b["t"] - a["t"]) for a, b in itertools.pairwise(samples)
    )
    applied_integral = sum(
        a["applied_yaw_rate"] * (b["t"] - a["t"]) for a, b in itertools.pairwise(samples)
    )
    return {
        "angle_deg": angle,
        "completed": outcome["completed"],
        "reason": outcome["reason"],
        "action_s": action_end - action_start,
        "truth_settled_deg": math.degrees(settled),
        "odom_settled_deg": math.degrees(odom_settled),
        "error_deg": error,
        "tolerance_deg": tolerance,
        "passed": passed,
        "settled_window_span_deg": post_span,
        "zero_command": zero_command,
        "requested_integral_deg": math.degrees(requested_integral),
        "applied_integral_deg": math.degrees(applied_integral),
        "peak_excursion_deg": max(abs(math.degrees(v - reference)) for v in truth),
        "position_drift_m": math.dist(samples[pre[-1]]["trunk"][:2], samples[-1]["trunk"][:2]),
        "post_requested_yaw_rate": samples[-1]["requested_yaw_rate"],
        "post_applied_yaw_rate": samples[-1]["applied_yaw_rate"],
        "max_sample_gap_s": max(b["t"] - a["t"] for a, b in itertools.pairwise(samples)),
        "real_time_factor": (samples[-1]["sim_time"] - samples[0]["sim_time"])
        / (samples[-1]["t"] - samples[0]["t"]),
    }


async def body_read(reader, writer):
    writer.write(b'{"op":"read"}\n')
    await writer.drain()
    result = json.loads(await asyncio.wait_for(reader.readline(), 0.5))
    # Require simulator-specific fields before any movement is attempted.
    for field in ("sim_time", "trunk", "imu"):
        if field not in result:
            raise ValueError(f"body endpoint is not the expected simulator: missing {field}")
    return result


async def probe_rate(robot, rate, duration):
    """Simulator-only response probe, retaining sensor/health/deadman guards.

    Stop at 25 degrees of measured excursion or two seconds, whichever is
    first. This is a diagnostic, not an exposed navigation action.
    """
    start = robot.transport.snapshot()["state"]["data"]["odom"]["yaw"]
    action = asyncio.create_task(robot._motion("rate_probe", 0, rate, duration))
    try:
        while not action.done():
            current = robot.transport.snapshot()["state"]["data"]["odom"]["yaw"]
            delta = math.atan2(math.sin(current - start), math.cos(current - start))
            if abs(delta) >= math.radians(25):
                await robot.stop()
                break
            await asyncio.sleep(0.025)
        return await action
    finally:
        if not action.done():
            action.cancel()
            try:
                await action
            except asyncio.CancelledError:
                pass


async def run(args):
    record = Recorder(Path("runs"))
    print(f"Recording to {record.path}", flush=True)
    transport, reader, writer = WebRtcRobot(), None, None
    robot = GuardedRobot(transport)
    results = []
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", args.body_port)
        await body_read(reader, writer)
        await transport.connect()
        await robot.initialize()
        if not (await robot.stop())["completed"]:
            raise RuntimeError("initial stop was not acknowledged")
        await asyncio.sleep(1)
        config_path = Path(args.state_dir).expanduser() / "robotd-duck-a.toml"
        daemon_config = tomllib.loads(config_path.read_text())
        policy_path = Path(daemon_config["policy"]["walk"])
        initial_state = transport.snapshot()["state"]["data"]
        git = await asyncio.create_subprocess_exec(
            "git", "rev-parse", "HEAD", stdout=asyncio.subprocess.PIPE
        )
        git_stdout, _ = await git.communicate()
        if git.returncode:
            raise RuntimeError("cannot record repository revision")
        meta = {
            "label": args.label,
            "guard_config": asdict(robot.config),
            "policy": str(policy_path),
            "policy_sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
            "git_head": git_stdout.decode().strip(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "angles": args.angles,
            "repeats": args.repeats,
            "probe_rates": args.probe_rates,
            "probe_duration_s": args.probe_duration,
            "initial_head_command": initial_state["head"],
            "initial_joints": initial_state["joints"],
            "initial_truth": await body_read(reader, writer),
            "body_port": args.body_port,
            "daemon_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "preroll_s": 0.6,
            "postroll_s": 2.0,
            "average_window_s": 0.3,
        }
        trials_expected = args.repeats * len(args.probe_rates or args.angles)
        record.write({"metadata": meta})
        for repetition in range(args.repeats):
            for value in args.probe_rates or args.angles:
                angle = None if args.probe_rates else value
                samples = []
                stop_sampling = asyncio.Event()

                async def sample(samples=samples, stop_sampling=stop_sampling):
                    while not stop_sampling.is_set():
                        truth = await body_read(reader, writer)
                        stamp = time.monotonic()
                        observed = transport.snapshot()
                        state = observed["state"]["data"]
                        samples.append(
                            {
                                "t": stamp,
                                "sim_time": truth["sim_time"],
                                "truth_yaw": yaw(truth["imu"]["quat"]),
                                "truth_gyro_z": truth["imu"]["gyro"][2],
                                "trunk": truth["trunk"],
                                "odom_yaw": state["odom"]["yaw"],
                                "state_t_ns": state["t_ns"],
                                "requested_yaw_rate": state["move"]["requested"][2],
                                "applied_yaw_rate": state["move"]["applied"][2],
                                "imu_gyro_z": state["imu"]["gyro"][2],
                            }
                        )
                        await asyncio.sleep(0.04)

                record.write({"before_trial": record.capture(transport.snapshot())})
                # If truth recording fails, TaskGroup cancels the active action;
                # GuardedRobot then requests stop in its cancellation handler.
                async with asyncio.TaskGroup() as group:
                    group.create_task(sample())
                    try:
                        await asyncio.sleep(0.6)
                        action_start = time.monotonic()
                        outcome = (
                            await probe_rate(robot, value, args.probe_duration)
                            if args.probe_rates
                            else await robot.turn_by(angle)
                        )
                        action_end = time.monotonic()
                        await asyncio.sleep(2)
                    finally:
                        stop_sampling.set()
                summary = summarize(samples, outcome, angle, action_start, action_end)
                summary["repetition"] = repetition + 1
                if args.probe_rates:
                    summary["probe_rate_rad_s"] = value
                results.append(summary)
                record.write(
                    {
                        "trial": summary,
                        "outcome": outcome,
                        "samples": samples,
                        "action_start": action_start,
                        "action_end": action_end,
                        "after_trial": record.capture(transport.snapshot()),
                    }
                )
                complete = len(results) == trials_expected
                benchmark_passed = (
                    complete and all(r["passed"] for r in results) if not args.probe_rates else None
                )
                (record.path / "turns.json").write_text(
                    json.dumps(
                        {
                            "metadata": meta,
                            "trials": results,
                            "benchmark_passed": benchmark_passed,
                            "complete": complete,
                            "trials_expected": trials_expected,
                        },
                        indent=2,
                    )
                )
                print(json.dumps(summary), flush=True)
                expected = {"angle_reached", "turn_timeout", "turn_no_progress"}
                if args.probe_rates:
                    expected |= {"duration_elapsed", "stopped"}
                if outcome["reason"] not in expected or not outcome["stop"]["acknowledged"]:
                    print(
                        f"Aborting remaining trials: {outcome['reason']}; see {record.path / 'turns.json'}",
                        file=sys.stderr,
                        flush=True,
                    )
                    return 2
        print(f"Results: {record.path / 'turns.json'}", flush=True)
        return 1 if benchmark_passed is False else 0
    finally:
        try:
            await robot.close()
        finally:
            await transport.close()
            if writer:
                writer.close()
                await writer.wait_closed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--angles", type=float, nargs="+", default=[5, -5, 15, -15, 30, -30])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--label", default="baseline")
    parser.add_argument(
        "--probe-rates", type=float, nargs="+", help="simulator-only rates in rad/s"
    )
    parser.add_argument("--probe-duration", type=float, default=2.0)
    parser.add_argument("--body-port", type=int, default=7801)
    parser.add_argument("--state-dir", default="~/.cache/duck-sim")
    args = parser.parse_args()
    if args.repeats < 1 or any(not 0 < abs(a) <= 30 for a in args.angles):
        parser.error("use positive repeats and nonzero angles between -30 and 30")
    if not 0 < args.probe_duration <= 2 or any(
        not 0 < abs(rate) <= 1 for rate in args.probe_rates or []
    ):
        parser.error("probes require rates up to 1 rad/s and durations up to two seconds")
    raise SystemExit(asyncio.run(run(args)))
