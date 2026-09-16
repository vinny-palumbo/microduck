"""Offline policy response measurements, never connected to a running robot.

Run with microduck_rl/.venv/bin/python. The deployment rehearsal builds observations;
this script adds robotd's default command/target filtering and measures complete
move-and-settle responses. MuJoCo qpos is evaluation truth, not an odometry claim.
Each trial starts a new physical simulation from HOME. No poses are changed after
initialization, and joint targets always come from the unmodified deployed policy.
This screens candidate primitives; it does not verify camera guards, real-time
daemon timing, odometry accuracy, hardware safety, or arrival at a destination.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import sys
from pathlib import Path

import numpy as np


def heading(quaternion):
    w, x, y, z = quaternion
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def requested_command(t, speed, yaw_rate, duration, stop_profile):
    if t < 0:
        return np.zeros(3)
    if t < duration:
        return np.array([speed, 0.0, yaw_rate])
    stopping = t - duration
    if stop_profile == "ramp" and stopping < 0.4:
        return np.array([speed, 0.0, yaw_rate]) * (1 - stopping / 0.4)
    if stop_profile == "counter_yaw" and stopping < 0.2:
        return np.array([0.0, 0.0, -yaw_rate])
    return np.zeros(3)


def steering(mode, desired, turned, left_limit=0.5, right_limit=0.3):
    """Compare feedback laws; desired/turned are radians, limits are rad/s."""
    error = math.atan2(math.sin(desired - turned), math.cos(desired - turned))
    if mode == "proportional":
        return max(-left_limit, min(left_limit, 2 * error))
    if desired == 0:
        return 0.0
    sign = math.copysign(1, desired)
    limit = left_limit if sign > 0 else right_limit
    if mode == "fixed":
        return sign * limit
    if mode == "clipped":
        return sign * max(0.0, min(limit, 2 * error * sign))
    raise ValueError(f"unknown steering mode: {mode}")


def load_rehearsal(rl_path):
    path = rl_path / "scripts/infer_policy.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("navigation_gait_rehearsal", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = module.ort.InferenceSession

    def single_thread_session(path):
        options = module.ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        return original(path, sess_options=options)

    module.ort.InferenceSession = single_thread_session
    return module


def contact_names(model, data, robot_bodies):
    """Keep floor support distinct from obstacle/body strikes and self contacts."""
    unusual = set()
    for contact in data.contact:
        if not any(
            int(model.geom_bodyid[int(g)]) in robot_bodies for g in (contact.geom1, contact.geom2)
        ):
            continue
        names = [
            model.geom(int(g)).name or model.body(model.geom_bodyid[int(g)]).name
            for g in (contact.geom1, contact.geom2)
        ]
        support = any(n == "floor" or n.startswith("floor_") for n in names)
        foot = any(n in {"left_foot_collision", "right_foot_collision"} for n in names)
        if not (support and foot):
            unusual.add(tuple(sorted(names)))
    return unusual


def descendants(model, root):
    bodies = {root}
    for body in range(root + 1, model.nbody):
        if int(model.body_parentid[body]) in bodies:
            bodies.add(body)
    return bodies


def probe(
    ip,
    args,
    speed,
    yaw_rate,
    duration,
    stop_profile,
    repeat,
    steering_mode="rate",
    desired_heading=0.0,
    distance_cutoff=None,
):
    import mujoco

    with contextlib.redirect_stdout(io.StringIO()):
        bam = None
        if args.bam:
            model, data, bam, _ = ip.load_mujoco_with_bam(
                str(args.scene), ip.load_bam_model(200, 7.4, None), 0.005, None, 6.0
            )
        else:
            model = mujoco.MjModel.from_xml_path(str(args.scene))
            model.opt.timestep = 0.005
            data = mujoco.MjData(model)
        controller = ip.PolicyInference(
            model,
            data,
            walking_onnx_path=str(args.policy),
            action_scale=args.action_scale,
            bam_ctrl=bam,
            new_cmd_obs=True,
            use_projected_gravity=True,
        )
    root = int(model.joint("trunk_base_freejoint").qposadr[0])
    robot_bodies = descendants(model, int(model.body("trunk_base").id))
    spawn_yaw = math.radians(args.spawn[2])
    data.qpos[root : root + 7] = [
        args.spawn[0],
        args.spawn[1],
        0.125,
        math.cos(spawn_yaw / 2),
        0,
        0,
        math.sin(spawn_yaw / 2),
    ]
    data.qpos[controller.joint_qpos_indices] = controller.default_pose
    if bam is not None:
        bam.reset(data.qpos)
    controller.set_position_targets(controller.default_pose)
    mujoco.mj_forward(model, data)

    # Vary preparation duration across repetitions to expose phase sensitivity.
    warmup_ticks = 100 + repeat * 7
    warmup = warmup_ticks / 50
    ticks = warmup_ticks + round((duration + args.settle) * 50)
    smooth = np.zeros(3)
    previous = None
    traces = []
    unusual_contacts = set()
    obstacle_contacts = set()
    first_obstacle_contact_s = None
    first_contact_s = None
    stop_reason = None
    stop_at = None
    reference_xy = None
    reference_yaw = None
    head = np.array(args.head)
    last_feedback_tick = -1
    feedback_yaw = 0.0
    for tick in range(ticks):
        elapsed = (tick - warmup_ticks) / 50
        if reference_xy is None and elapsed >= 0:
            reference_xy = data.qpos[root : root + 2].copy()
            reference_yaw = heading(data.qpos[root + 3 : root + 7])
        command = requested_command(elapsed, speed, yaw_rate, duration, stop_profile)
        if steering_mode != "rate" and 0 <= elapsed < duration:
            feedback_tick = math.floor((elapsed + 1e-9) / 0.05)
            if feedback_tick != last_feedback_tick:
                turned = heading(data.qpos[root + 3 : root + 7]) - reference_yaw
                feedback_yaw = steering(steering_mode, math.radians(desired_heading), turned)
                last_feedback_tick = feedback_tick
            command[2] = feedback_yaw
        if stop_reason:
            command = np.zeros(3)
        smooth += args.command_alpha * (command - smooth)
        controller.vel_cmd[:] = smooth
        controller.head_offset[:] = head
        controller._update_command()
        action = controller.infer()
        targets = controller.default_pose + args.action_scale * action
        if not args.no_target_filter and previous is not None:
            alpha = np.array([0.7] * 5 + [0.5] * 4 + [0.7] * 5)
            targets = alpha * targets + (1 - alpha) * previous
        controller.set_position_targets(targets)
        previous = targets.copy()
        for _ in range(4):
            if bam is not None:
                bam.update()
            mujoco.mj_step(model, data)
            if elapsed >= 0:
                contacts = contact_names(model, data, robot_bodies)
                unusual_contacts.update(contacts)
                obstacles = {
                    pair
                    for pair in contacts
                    if not any(name == "floor" or name.startswith("floor_") for name in pair)
                }
                obstacle_contacts.update(obstacles)
                if obstacles and first_obstacle_contact_s is None:
                    first_obstacle_contact_s = elapsed
                if contacts and first_contact_s is None:
                    first_contact_s = elapsed
        xyz = data.qpos[root : root + 3].copy()
        q = data.qpos[root + 3 : root + 7]
        yaw = heading(q)
        tilt = math.degrees(math.acos(np.clip(1 - 2 * (q[1] ** 2 + q[2] ** 2), -1, 1)))
        if elapsed >= 0:
            excursion = abs(
                math.atan2(math.sin(yaw - reference_yaw), math.cos(yaw - reference_yaw))
            )
            if stop_reason is None:
                if tilt > 35 or xyz[2] < 0.075:
                    stop_reason = "unsafe_pose"
                elif np.linalg.norm(xyz[:2] - reference_xy) > args.max_distance:
                    stop_reason = "distance_cutoff"
                elif (
                    distance_cutoff is not None
                    and np.linalg.norm(xyz[:2] - reference_xy) >= distance_cutoff
                ):
                    stop_reason = "target_distance_cutoff"
                elif excursion > math.radians(args.max_turn):
                    stop_reason = "heading_cutoff"
                if stop_reason:
                    stop_at = elapsed
        traces.append([elapsed, *xyz, yaw, tilt, *command, *smooth])

    samples = np.array(traces)
    unwrapped = np.unwrap(samples[:, 4])
    before = (samples[:, 0] >= -0.3) & (samples[:, 0] < 0)
    after = samples[:, 0] >= samples[-1, 0] - 0.3
    baseline = samples[before, 1:3].mean(axis=0)
    initial_yaw = unwrapped[before].mean()
    final_position = samples[after, 1:3].mean(axis=0)
    final_yaw = unwrapped[after].mean()
    action_end = int(np.searchsorted(samples[:, 0], stop_at if stop_at is not None else duration))
    delta = final_position - baseline
    rotate = np.array(
        [
            [math.cos(initial_yaw), math.sin(initial_yaw)],
            [-math.sin(initial_yaw), math.cos(initial_yaw)],
        ]
    )
    forward, lateral = rotate @ delta
    result = {
        "command": [speed, 0.0, yaw_rate],
        "steering": steering_mode,
        "desired_heading_deg": desired_heading,
        "distance_cutoff_m": distance_cutoff,
        "feedback_source": "perfect_simulator_yaw_and_position_for_controller_screening_only"
        if steering_mode != "rate"
        else "none",
        "spawn": args.spawn,
        "duration_s": duration,
        "stop_profile": stop_profile,
        "repeat": repeat,
        "warmup_s": warmup,
        "head": args.head,
        "stop_reason": stop_reason or "duration_elapsed",
        "stop_at_s": stop_at if stop_at is not None else duration,
        "displacement_m": float(np.linalg.norm(delta)),
        "forward_m": float(forward),
        "lateral_m": float(lateral),
        "active_displacement_m": float(np.linalg.norm(samples[action_end, 1:3] - baseline)),
        "after_stop_displacement_m": float(
            np.linalg.norm(final_position - samples[action_end, 1:3])
        ),
        "active_yaw_deg": float(np.degrees(unwrapped[action_end] - initial_yaw)),
        "settled_yaw_deg": float(np.degrees(final_yaw - initial_yaw)),
        "after_stop_yaw_deg": float(np.degrees(final_yaw - unwrapped[action_end])),
        "settled_yaw_range_deg": float(np.degrees(np.ptp(unwrapped[after]))),
        "settled_position_range_m": float(np.linalg.norm(np.ptp(samples[after, 1:3], axis=0))),
        "max_tilt_deg": float(samples[:, 5].max()),
        "min_height_m": float(samples[:, 3].min()),
        "final_applied_command": smooth.tolist(),
        "nonfoot_contacts": sorted(unusual_contacts),
        "obstacle_or_self_contacts": sorted(obstacle_contacts),
        "first_obstacle_contact_s": first_obstacle_contact_s,
        "collision_free_steering_trial": not obstacle_contacts,
        "first_nonfoot_contact_s": first_contact_s,
        "truth_source": "independent_MuJoCo_freejoint_qpos_not_robot_odometry",
    }
    return result, traces


def main():
    root = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rl", type=Path, default=root / "microduck_rl")
    parser.add_argument("--scene", type=Path)
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path.home() / ".cache/duck-sim/policies/current/velstand.onnx",
    )
    parser.add_argument("--speeds", nargs="+", type=float, default=[0.3, 0.4])
    parser.add_argument("--yaw-rates", nargs="+", type=float, default=[0, 0.5, -0.5, 1, -1])
    parser.add_argument("--durations", nargs="+", type=float, default=[2])
    parser.add_argument(
        "--stop-profiles", nargs="+", choices=["zero", "ramp", "counter_yaw"], default=["zero"]
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--head", nargs=4, type=float, default=[0, 0, 0, 0])
    parser.add_argument(
        "--spawn", nargs=3, type=float, default=[0, 0, 0], metavar=("X", "Y", "YAW_DEG")
    )
    parser.add_argument(
        "--controllers",
        nargs="+",
        choices=["rate", "proportional", "clipped", "fixed"],
        default=["rate"],
    )
    parser.add_argument("--headings", nargs="+", type=float, default=[0])
    parser.add_argument("--distance-cutoffs", nargs="+", type=float, default=[None])
    parser.add_argument("--settle", type=float, default=3)
    parser.add_argument("--command-alpha", type=float, default=0.2)
    parser.add_argument("--action-scale", type=float, default=0.9)
    parser.add_argument("--no-target-filter", action="store_true")
    parser.add_argument("--bam", action="store_true")
    parser.add_argument("--max-distance", type=float, default=0.75)
    parser.add_argument("--max-turn", type=float, default=80)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    numeric = [
        *args.speeds,
        *args.yaw_rates,
        *args.durations,
        *args.head,
        *args.spawn,
        *args.headings,
        *(value for value in args.distance_cutoffs if value is not None),
        args.settle,
        args.command_alpha,
        args.action_scale,
        args.max_distance,
        args.max_turn,
    ]
    if not all(math.isfinite(value) for value in numeric):
        parser.error("probe parameters must be finite")
    if any(not 0 <= v <= 0.4 for v in args.speeds) or any(abs(w) > 1 for w in args.yaw_rates):
        parser.error("probe commands must remain inside trained forward/yaw ranges")
    if not 0 < args.command_alpha <= 1 or not 1 <= args.repeats <= 10:
        parser.error("command-alpha must be (0,1]; repeats must be 1–10")
    if any(not 0 < d <= 10 for d in args.durations) or not 2 <= args.settle <= 10:
        parser.error("duration must be (0,10] and settle must be 2–10 seconds")
    if (
        not 0 < args.action_scale <= 1
        or not 0 < args.max_distance <= 1
        or not 0 < args.max_turn <= 90
    ):
        parser.error("action scale must be (0,1], distance cutoff (0,1]m, turn cutoff (0,90]deg")
    if any(abs(value) > limit for value, limit in zip(args.head, [1.1, 1.1, 1.4, 0.31])):
        parser.error("head offsets exceed the training command ranges")
    if any(abs(value) > 30 for value in args.headings):
        parser.error("feedback headings must remain within +/-30 degrees")
    if any(value is not None and not 0.03 <= value <= 0.2 for value in args.distance_cutoffs):
        parser.error("feedback distance cutoffs must be 0.03–0.2 metres")
    args.scene = (
        args.scene or args.rl / "src/mjlab_microduck/robot/microduck/scene_allcollisions.xml"
    )
    ip = load_rehearsal(args.rl)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "policy": str(args.policy.resolve()),
        "policy_sha256": hashlib.sha256(args.policy.read_bytes()).hexdigest(),
        "scene": str(args.scene.resolve()),
        "scene_sha256": hashlib.sha256(args.scene.read_bytes()).hexdigest(),
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "scope": "offline dynamics screening; no live transport, odometry or room-navigation claim",
        "trace_columns": [
            "t",
            "x",
            "y",
            "z",
            "yaw",
            "tilt_deg",
            "requested_vx",
            "requested_vy",
            "requested_vyaw",
            "applied_vx",
            "applied_vy",
            "applied_vyaw",
        ],
        "trials": [],
    }
    for speed in args.speeds:
        for yaw_rate in args.yaw_rates:
            for duration in args.durations:
                for profile in args.stop_profiles:
                    for repeat in range(args.repeats):
                        for mode in args.controllers:
                            for desired in args.headings:
                                for cutoff in args.distance_cutoffs:
                                    result, trace = probe(
                                        ip,
                                        args,
                                        speed,
                                        yaw_rate,
                                        duration,
                                        profile,
                                        repeat,
                                        mode,
                                        desired,
                                        cutoff,
                                    )
                                    print(json.dumps(result, allow_nan=False), flush=True)
                                    report["trials"].append({**result, "trace": trace})
                                    args.output.write_text(
                                        json.dumps(report, indent=2, allow_nan=False)
                                    )


if __name__ == "__main__":
    main()
