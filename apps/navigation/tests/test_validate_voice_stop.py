"""A spoken-stop proof cannot pass without motion, transcription, and acknowledged stop."""

import copy
import runpy
from pathlib import Path

import pytest


@pytest.fixture
def score(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    return runpy.run_path(str(scripts / "validate_voice_stop.py"))["score"]


def evidence():
    return {
        "simulator_confirmed": True,
        "advance_at": 1,
        "trigger": {"monotonic_s": 2, "commanded": True},
        "stop_audio": {"monotonic_s": 2.1, "commanded": True},
        "stop_transcript_at": 2.5,
        "terminal_at": 3,
        "active_trunk_positions": [[0, 0, 0.12], [0.003, 0, 0.12]],
    }


def result():
    return {"status": "cancelled", "stop": {"completed": True, "stop": {"acknowledged": True}}}


def stopped():
    return {"fresh": True, "requested": [0, 0, 0], "applied": [0, 0, 0]}


def test_complete_evidence_passes_without_claiming_settled_pose(score):
    report = score(evidence(), result(), stopped())
    assert report["passed"] is True
    assert report["active_trunk_displacement_m"] == pytest.approx(0.003)
    assert report["physical_settling_verified"] is False


@pytest.mark.parametrize(
    "field",
    [
        "simulator_confirmed",
        "advance_at",
        "trigger",
        "stop_audio",
        "stop_transcript_at",
        "terminal_at",
    ],
)
def test_missing_proof_is_inconclusive(score, field):
    proof = evidence()
    del proof[field]
    assert not score(proof, result(), stopped())["passed"]


@pytest.mark.parametrize(
    "change",
    [
        {"advance_at": 2.2},
        {"terminal_at": 1.9},
        {"stop_transcript_at": 3.1},
        {"stop_transcript_at": 2.0},
        {"stop_audio": {"monotonic_s": 1.5, "commanded": True}},
        {"stop_audio": {"monotonic_s": 2.1, "commanded": False}},
        {"trigger": {"monotonic_s": 2, "commanded": False}},
        {"active_trunk_positions": [[0, 0, 0.12]] * 4},
    ],
)
def test_ordering_and_actual_motion_are_required(score, change):
    proof = {**evidence(), **copy.deepcopy(change)}
    assert not score(proof, result(), stopped())["passed"]


@pytest.mark.parametrize(
    "outcome",
    [
        {"status": "timeout", "stop": result()["stop"]},
        {"status": "goal_observed", "stop": result()["stop"]},
        {"status": "cancelled", "stop": {"completed": True}},
        {"status": "cancelled", "stop": {"completed": False, "stop": {"acknowledged": True}}},
    ],
)
def test_only_acknowledged_cancellation_passes(score, outcome):
    assert not score(evidence(), outcome, stopped())["passed"]


@pytest.mark.parametrize(
    "change", [{"fresh": False}, {"requested": [0.1, 0, 0]}, {"applied": [0.01, 0, 0]}]
)
def test_final_commands_must_be_fresh_and_zero(score, change):
    assert not score(evidence(), result(), {**stopped(), **change})["passed"]
