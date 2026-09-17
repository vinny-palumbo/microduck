"""Calibration sequencing uses fake body/robot endpoints; no simulator runs."""

import json
import math
import runpy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest


@pytest.fixture
def calibration(monkeypatch, tmp_path):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    run = runpy.run_path(str(scripts / "calibrate_navigation.py"))["run"]
    scope = run.__globals__
    events = []

    class Recorder:
        def __init__(self, _root):
            self.path = tmp_path

        def write(self, event):
            events.append(event)

        def capture(self, _snapshot):
            return None

    transport = SimpleNamespace(
        connect=AsyncMock(), close=AsyncMock(), snapshot=Mock(return_value={})
    )
    robot = SimpleNamespace(
        initialize=AsyncMock(),
        stop=AsyncMock(return_value={"completed": True}),
        look_at=AsyncMock(return_value={"completed": True}),
        close=AsyncMock(),
    )
    failures = {}
    count = 0

    async def advance(distance, heading):
        nonlocal count
        count += 1
        result = {
            "completed": True,
            "reason": "distance_budget",
            "distance_m": 0.1,
            "heading_deg": 5,
            "stop": {"acknowledged": True, "physical_settling_verified": True},
            "effective_heading_deg": heading,
            "target_yaw_deg": 179 + 5 * count,
        }
        result.update(failures.get(count, {}))
        return result

    async def observe():
        return {
            "ready": failures.get("guard_step") != count,
            "guard_reason": "unhealthy" if failures.get("guard_step") == count else None,
            "course": {"active": True, "target_yaw_deg": 179 + 5 * count, "error_deg": 0},
        }

    robot.advance = AsyncMock(side_effect=advance)
    robot.observe = AsyncMock(side_effect=observe)

    async def body_read(_reader, _writer):
        heading = math.radians(179 + count * 5)
        return {
            "sim_time": count,
            "trunk": [count * 0.1, 0, 0.12],
            "imu": {"quat": [math.cos(heading / 2), 0, 0, math.sin(heading / 2)]},
        }

    writer = SimpleNamespace(close=Mock(), wait_closed=AsyncMock())
    open_connection = AsyncMock(return_value=(object(), writer))
    body = AsyncMock(side_effect=body_read)
    transport_factory = Mock(return_value=transport)
    monkeypatch.setitem(scope, "Recorder", Recorder)
    monkeypatch.setitem(scope, "WebRtcRobot", transport_factory)
    monkeypatch.setitem(scope, "GaitNavigator", Mock(return_value=robot))
    monkeypatch.setitem(scope, "body_read", body)
    monkeypatch.setattr(scope["asyncio"], "open_connection", open_connection)
    monkeypatch.setattr(scope["asyncio"], "sleep", AsyncMock())
    return SimpleNamespace(
        run=run,
        path=tmp_path / "calibration.json",
        events=events,
        robot=robot,
        body=body,
        failures=failures,
        transport_factory=transport_factory,
        open_connection=open_connection,
    )


async def test_three_arcs_keep_course_and_report_independent_total(calibration):
    args = SimpleNamespace(distance=0.1, heading=5, steps=3)
    assert await calibration.run(args) == 0
    calibration.robot.stop.assert_awaited_once()
    calibration.robot.look_at.assert_awaited_once_with(1, 0, 0)
    assert calibration.robot.advance.await_args_list == [((0.1, 5),)] * 3
    calibration.open_connection.assert_awaited_once_with("127.0.0.1", 7801)
    report = json.loads(calibration.path.read_text())
    assert report["passed"]
    assert report["steps_executed"] == 3
    assert report["truth"]["heading_deg"] == pytest.approx(15)
    assert report["truth"]["distance_m"] == pytest.approx(0.3)
    assert [step["course"]["target_yaw_deg"] for step in report["steps"]] == [184, 189, 194]
    assert all(
        step["wall_interval"]["end"] >= step["wall_interval"]["start"] for step in report["steps"]
    )
    recorded = [event for event in calibration.events if event["event"] == "calibration_step"]
    assert len(recorded) == 3
    assert recorded[0]["body_before"]["trunk"] == [0, 0, 0.12]
    assert recorded[-1]["body_after"]["trunk"][0] == pytest.approx(0.3)


@pytest.mark.parametrize("failure", ["guard", "action", "stop", "settling", "odometry"])
async def test_failed_second_step_prevents_third_arc(calibration, failure):
    if failure == "guard":
        calibration.failures["guard_step"] = 2
    elif failure == "action":
        calibration.failures[2] = {"completed": False, "reason": "obstacle"}
    elif failure in ("stop", "settling"):
        calibration.failures[2] = {
            "stop": {
                "acknowledged": failure != "stop",
                "physical_settling_verified": failure != "settling",
            }
        }
    else:
        calibration.failures[2] = {"heading_deg": 15}
    assert await calibration.run(SimpleNamespace(distance=0.1, heading=5, steps=3)) == 2
    assert calibration.robot.advance.await_count == 2
    calibration.robot.stop.assert_awaited_once()
    report = json.loads(calibration.path.read_text())
    assert not report["passed"]
    assert report["steps_executed"] == 2
    assert not report["steps"][-1]["passed"]


async def test_default_preserves_single_arc_fields(calibration):
    assert await calibration.run(SimpleNamespace(distance=0.1, heading=5)) == 0
    report = json.loads(calibration.path.read_text())
    assert report["steps_requested"] == report["steps_executed"] == 1
    assert report["result"] == report["steps"][0]["result"]
    assert report["truth"] == report["steps"][0]["truth"]
    assert report["passed"] is True


async def test_local_body_handshake_must_succeed_before_robot_connection(calibration):
    calibration.body.side_effect = ValueError("wrong body protocol")
    with pytest.raises(ValueError, match="wrong body protocol"):
        await calibration.run(SimpleNamespace(distance=0.1, heading=0, steps=3))
    calibration.transport_factory.assert_not_called()
    calibration.robot.advance.assert_not_awaited()


@pytest.mark.parametrize("steps", [0, 6, 1.5, True])
async def test_invalid_step_count_cannot_connect(calibration, steps):
    with pytest.raises(ValueError, match="steps must"):
        await calibration.run(SimpleNamespace(distance=0.1, heading=0, steps=steps))
    calibration.open_connection.assert_not_awaited()
