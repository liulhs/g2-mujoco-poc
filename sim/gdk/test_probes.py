"""Tests for the read-only robot probe and the chassis convention probe (fake agibot_gdk)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import chassis_probe as cp
import robot_probe as probe
import run_poc_gdk as rp
from test_run_poc_gdk import FakeRobot, _install


def test_leg_needs_exactly_one_component() -> None:
    with pytest.raises(rp.Abort, match="exactly one"):
        cp.leg(0.1, 0.1, 0.0)
    with pytest.raises(rp.Abort, match="exactly one"):
        cp.leg(0.0, 0.0, 0.0)


def test_leg_limits() -> None:
    with pytest.raises(rp.Abort, match="0.3 m"):
        cp.leg(0.0, 0.5, 0.0)
    with pytest.raises(rp.Abort, match="30"):
        cp.leg(0.0, 0.0, 45.0)


def test_dry_run_sends_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "agibot_gdk", raising=False)
    assert cp.main(["--dy", "0.1"]) == 0   # would fail importing agibot_gdk if it tried to connect


def test_execute_sends_one_leg(monkeypatch: pytest.MonkeyPatch) -> None:
    robot = FakeRobot()
    _install(monkeypatch, robot, [rp.ESTOP_PHRASE, cp.GO_PHRASE])
    assert cp.main(["--dy", "0.1", "--execute"]) == 0
    assert robot.log == ["drive:+0.000,+0.100,+0.0"]


def test_wrong_phrase_sends_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    robot = FakeRobot()
    _install(monkeypatch, robot, [rp.ESTOP_PHRASE, "go"])
    assert cp.main(["--dx", "0.1", "--execute"]) == 2
    assert robot.log == []


def test_cancel_after_cancels_running_leg(monkeypatch: pytest.MonkeyPatch) -> None:
    robot = FakeRobot()
    _install(monkeypatch, robot, [rp.ESTOP_PHRASE, cp.GO_PHRASE], pnc_state=2)   # 2: still running
    monkeypatch.setattr(cp, "WATCH_AFTER_CANCEL", 0.05)
    assert cp.main(["--dx", "0.2", "--execute", "--cancel-after", "0"]) == 0
    assert robot.log == ["drive:+0.200,+0.000,+0.0", "cancel"]


def test_stale_task_never_taken_for_the_leg(monkeypatch: pytest.MonkeyPatch) -> None:
    robot = FakeRobot()
    _install(monkeypatch, robot, [rp.ESTOP_PHRASE, cp.GO_PHRASE], pnc_lag=3)
    monkeypatch.setattr(rp, "TASK_START_TIMEOUT", 5.0)
    assert cp.main(["--dyaw-deg", "20", "--execute"]) == 0
    assert robot.log[0].startswith("drive:") and "cancel" not in robot.log


def test_probe_survey_reads_without_commanding(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    robot = FakeRobot(end_model="o10_t2")
    _install(monkeypatch, robot, [])
    (tmp_path / "share" / "o10_t2").mkdir(parents=True)
    (tmp_path / "share" / "o10_t2" / "hand.urdf").write_text("<robot/>")
    (tmp_path / "share" / "unrelated.urdf").write_text("<robot/>")
    assert probe.main(["--out", str(tmp_path / "o"), "--search-root", str(tmp_path / "share")]) == 0
    doc = json.loads(next((tmp_path / "o").glob("robot_probe_*[0-9].json")).read_text())
    assert doc["gdk"]["whole_body_status"]["right_end_model"] == "o10_t2"
    assert doc["gdk"]["task_state"]["state"] == rp.TASK_CANCELED
    assert doc["model_files"] == [str(tmp_path / "share" / "o10_t2" / "hand.urdf")]
    assert robot.log == []   # nothing sent
    assert next((tmp_path / "o").glob("*_help.txt")).read_text().count("=====") >= 2
