"""The benchmark must score lasting rotation, not a transient body twist."""

import asyncio
import json
import math
import runpy
from pathlib import Path

import pytest

calibration = runpy.run_path(str(Path(__file__).parents[1] / "scripts/calibrate_turns.py"))
summarize = calibration["summarize"]
probe_rate = calibration["probe_rate"]


def samples_for(angle, initial=179):
    samples = []
    for tick in range(101):
        stamp = tick * 0.04
        delta = angle * min(1, max(0, stamp - 1))
        heading = math.radians(initial + delta)
        heading = math.atan2(math.sin(heading), math.cos(heading))
        samples.append(
            {
                "t": stamp,
                "sim_time": stamp,
                "truth_yaw": heading,
                "odom_yaw": heading,
                "requested_yaw_rate": 1.0 if 1 <= stamp < 2 else 0.0,
                "applied_yaw_rate": 1.0 if 1 <= stamp < 2 else 0.0,
                "trunk": [0, 0, 0.12],
            }
        )
    return samples


def outcome(completed=True, acknowledged=True):
    return {
        "completed": completed,
        "reason": "angle_reached",
        "stop": {"acknowledged": acknowledged},
    }


@pytest.mark.parametrize("angle,initial", [(5, 179), (-5, -179)])
def test_scores_both_directions_across_yaw_wrap(angle, initial):
    result = summarize(samples_for(angle, initial), outcome(), angle, 1, 2)
    assert result["passed"]
    assert result["truth_settled_deg"] == pytest.approx(angle)
    assert result["requested_integral_deg"] == pytest.approx(math.degrees(1))
    assert result["real_time_factor"] == pytest.approx(1)


def test_transient_target_crossing_is_not_a_success():
    samples = samples_for(0)
    # A controller might claim success while the planted feet merely twist.
    samples[45]["truth_yaw"] += math.radians(15)
    samples[45]["odom_yaw"] += math.radians(15)
    result = summarize(samples, outcome(), 15, 1, 2)
    assert not result["passed"]
    assert result["peak_excursion_deg"] == pytest.approx(15)
    assert result["truth_settled_deg"] == pytest.approx(0)


@pytest.mark.parametrize("failure", ["unfinished", "no_ack", "command", "drift", "odom"])
def test_heading_accuracy_alone_cannot_pass(failure):
    samples = samples_for(15)
    reply = outcome(completed=failure != "unfinished", acknowledged=failure != "no_ack")
    if failure == "command":
        samples[-1]["applied_yaw_rate"] = 0.02
    if failure == "drift":
        samples[-1]["truth_yaw"] += math.radians(0.6)
    if failure == "odom":
        for sample in samples[50:]:
            sample["odom_yaw"] += math.radians(3)
    assert not summarize(samples, reply, 15, 1, 2)["passed"]


def test_probe_has_no_angle_success_claim():
    result = summarize(samples_for(1), outcome(), None, 1, 2)
    assert result["passed"] is None
    assert result["error_deg"] is None


class ProbeRobot:
    def __init__(self):
        self.transport = self
        self.heading = 0
        self.cancel = asyncio.Event()
        self.finished = asyncio.Event()

    def snapshot(self):
        return {"state": {"data": {"odom": {"yaw": self.heading}}}}

    async def _motion(self, action, vx, rate, duration):
        try:
            while not self.cancel.is_set():
                self.heading += math.copysign(0.1, rate)
                await asyncio.sleep(0.005)
            return outcome(False)
        finally:
            self.finished.set()

    async def stop(self):
        self.cancel.set()


@pytest.mark.parametrize("rate", [1, -1])
async def test_probe_stops_on_measured_excursion_in_either_direction(rate):
    robot = ProbeRobot()
    await asyncio.wait_for(probe_rate(robot, rate, 2), 1)
    assert robot.cancel.is_set()
    assert robot.finished.is_set()
    assert abs(robot.heading) >= math.radians(25)


async def test_probe_cancellation_awaits_motion_cleanup():
    robot = ProbeRobot()
    task = asyncio.create_task(probe_rate(robot, 1, 2))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert robot.finished.is_set()


async def test_requires_simulator_fields_before_motion():
    reader = asyncio.StreamReader()
    reader.feed_data(json.dumps({"positions": []}).encode() + b"\n")

    class Writer:
        def write(self, data):
            assert json.loads(data) == {"op": "read"}

        async def drain(self):
            pass

    with pytest.raises(ValueError, match="not the expected simulator"):
        await calibration["body_read"](reader, Writer())
