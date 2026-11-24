"""Tests for the GDK runner, using a fake agibot_gdk (no robot needed)."""
from __future__ import annotations

import json
import math
import sys
import types
from pathlib import Path

import pytest

import run_poc_gdk as rp

PLANS = sorted((Path(__file__).resolve().parents[1] / "out").glob("bimanual_*.json"))


def _doc(tmp_path: Path, all_ok: bool = True, arm: str = "crs", measured: bool = True) -> Path:
    base = {n: 0.0 for n in rp.COMMANDED}
    wps = [
        dict(name="HOME", note="", gripper=0.0, joints=dict(base)),
        dict(name="A", note="", gripper=-0.785, joints={**base, "idx61_arm_r_joint1": 0.6}),
        dict(name="GRASP", note="", gripper=0.0, joints={**base, "idx61_arm_r_joint1": 0.6}),
        dict(name="B", note="", gripper=0.0, joints={**base, "idx61_arm_r_joint1": 0.6, "idx05_body_joint5": -0.3}),
    ]
    p = tmp_path / "w.json"
    p.write_text(json.dumps(dict(all_ok=all_ok, expected_arm=arm, layout=dict(measured=measured), waypoints=wps)))
    return p


SERVO_MODE = 5  # GDK 2.6.3 reports mode=5 (G2 servo) on the real robot


class FakeRobot:
    def __init__(self, arm_type: str | None = "crs", mode: int = SERVO_MODE, rc: int = 0,
                 gripper_works: bool = True, end_model: str | None = "omnipicker") -> None:
        self.q = {n: 0.0 for n in rp.COMMANDED}
        self.grip = 0.0
        self.log: list[str] = []
        self.arm_type, self.mode, self.rc, self.gripper_works = arm_type, mode, rc, gripper_works
        self.end_model = end_model

    def get_joint_states(self) -> dict:
        return {"states": [{"name": n, "motor_position": v} for n, v in self.q.items()]}

    def joint_control_request(self, req) -> int:
        assert req.life_time > 0 and len(req.joint_names) == len(req.joint_positions) == len(req.joint_velocities)
        assert max(req.joint_velocities) <= rp.MAX_SPEED + 1e-9
        if self.rc == 0:
            self.q.update(dict(zip(req.joint_names, req.joint_positions)))
        self.log.append(f"joints:{req.detail}")
        return self.rc

    def move_ee_pos(self, cmd) -> int:
        assert cmd.group in ("left_tool", "right_tool") and cmd.target_type == "omnipicker"
        if self.gripper_works:
            self.grip = cmd.states[0].position
        if cmd.group == "right_tool":
            self.log.append(f"grip:{cmd.states[0].position:+.3f}")
        return 0

    def get_end_state(self) -> dict:
        return {f"{side}_end_state": {"end_states": [{"position": self.grip}]} for side in ("left", "right")}

    def get_whole_body_status(self) -> dict:
        st = {"right_arm_error": 0, "right_arm_estop": False}
        if self.arm_type:
            st["right_arm_type"] = self.arm_type
        if self.end_model:
            st["left_end_model"] = st["right_end_model"] = self.end_model
        return st

    def get_motion_control_status(self):
        # GDK 2.6.3: the field is `mode`; there is no `control_mode`
        return types.SimpleNamespace(error_code=0, error_msg="", mode=self.mode)

    def set_control_mode(self, mode) -> int:
        self.log.append(f"mode:{mode}")  # the runner must never call this
        return 0


class FakePnc:
    """Starts with a stale cancelled task (state 7), as seen on the robot; a new task id
    appears `lag` polls after relative_move."""

    def __init__(self, robot: FakeRobot, final_state: int = 9, lag: int = 0) -> None:
        self.robot, self.final_state, self.lag = robot, final_state, lag
        self.id, self.state, self.pending = 1498285016, rp.TASK_CANCELED, None

    def relative_move(self, req) -> None:
        dyaw = 2 * math.atan2(req.target.orientation.z, req.target.orientation.w)
        self.robot.log.append(f"drive:{req.target.position.x:+.3f},{req.target.position.y:+.3f},"
                              f"{math.degrees(dyaw):+.1f}")
        self.pending = self.lag

    def get_task_state(self):
        if self.pending is not None:
            if self.pending == 0:
                self.id, self.state, self.pending = self.id + 1, self.final_state, None
            else:
                self.pending -= 1
        return types.SimpleNamespace(id=self.id, state=self.state, type=3)

    def cancel_task(self) -> None:
        self.robot.log.append("cancel")


def _navi_req():
    return types.SimpleNamespace(target=types.SimpleNamespace(position=types.SimpleNamespace(x=0.0, y=0.0),
                                                              orientation=types.SimpleNamespace(z=0.0, w=1.0)))


def _install(monkeypatch: pytest.MonkeyPatch, robot: FakeRobot, answers: list[str], pnc_state: int = 9,
             pnc_lag: int = 0) -> None:
    class Obj:
        pass
    gdk = types.SimpleNamespace(GDKRes=types.SimpleNamespace(kSuccess=0), gdk_init=lambda: 0,
                                gdk_release=lambda: None, Robot=lambda: robot,
                                Pnc=lambda: FakePnc(robot, pnc_state, pnc_lag), NaviReq=_navi_req,
                                JointControlReq=Obj, JointState=Obj, JointStates=Obj)
    monkeypatch.setitem(sys.modules, "agibot_gdk", gdk)
    monkeypatch.setattr(rp.time, "sleep", lambda s: None)
    it = iter(answers)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(it))


def _run(path: Path, *extra: str) -> int:
    return rp.main(["--waypoints", str(path), "--execute", "--speed", "0.3",
                    "--control-mode", str(SERVO_MODE), *extra])


def test_refuses_unvalidated_plan(tmp_path: Path) -> None:
    with pytest.raises(rp.Abort, match="all_ok"):
        rp.load_plan(_doc(tmp_path, all_ok=False))


def test_synced_velocities_arrive_together() -> None:
    q0 = {n: 0.0 for n in rp.COMMANDED}
    q1 = {**q0, "idx61_arm_r_joint1": 0.6, "idx05_body_joint5": -0.3}
    vel, dur = rp.synced_velocities(q0, q1, 0.3)
    assert dur == pytest.approx(2.0)
    assert vel[rp.COMMANDED.index("idx61_arm_r_joint1")] == pytest.approx(0.3)
    assert vel[rp.COMMANDED.index("idx05_body_joint5")] == pytest.approx(0.15)


def test_plan_moves_then_actuates_gripper(tmp_path: Path) -> None:
    plan = rp.load_plan(_doc(tmp_path))
    steps = rp.plan_steps(plan.moves, plan.moves[0].joints, 0.3)
    assert [(s.kind, s.name) for s in steps] == [
        ("gripper", "HOME"), ("joints", "A"), ("gripper", "A"), ("gripper", "GRASP"), ("joints", "B")]


def test_execute_happy_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    robot = FakeRobot()
    _install(monkeypatch, robot, [rp.ESTOP_PHRASE] + [""] * 10)
    assert _run(_doc(tmp_path)) == 0
    assert robot.log == ["grip:+0.000", "joints:g2_poc:A", "grip:-0.785", "grip:+0.000", "joints:g2_poc:B"]
    assert robot.q["idx05_body_joint5"] == pytest.approx(-0.3)


def test_refuses_other_control_mode_and_never_switches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    robot = FakeRobot(mode=2)
    _install(monkeypatch, robot, [rp.ESTOP_PHRASE] + [""] * 10)
    assert _run(_doc(tmp_path)) == 2
    assert robot.log == []


def test_needs_control_mode_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    robot = FakeRobot()
    _install(monkeypatch, robot, [rp.ESTOP_PHRASE] + [""] * 10)
    assert rp.main(["--waypoints", str(_doc(tmp_path)), "--execute"]) == 2
    assert robot.log == []


def test_refuses_when_mode_field_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    robot = FakeRobot()
    robot.get_motion_control_status = lambda: types.SimpleNamespace(error_code=0, error_msg="")
    _install(monkeypatch, robot, [rp.ESTOP_PHRASE] + [""] * 10)
    assert _run(_doc(tmp_path)) == 2
    assert robot.log == []


def test_refuses_dexterous_hands(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    robot = FakeRobot(end_model="o10_t2")
    _install(monkeypatch, robot, [rp.ESTOP_PHRASE] + [""] * 10)
    assert _run(_doc(tmp_path)) == 2
    assert robot.log == []


def test_needs_confirm_end_effector_when_not_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    robot = FakeRobot(end_model=None)
    _install(monkeypatch, robot, [rp.ESTOP_PHRASE] + [""] * 10)
    assert _run(_doc(tmp_path)) == 2
    _install(monkeypatch, robot, [rp.ESTOP_PHRASE] + [""] * 10)
    assert _run(_doc(tmp_path), "--confirm-end-effector", "omnipicker") == 0


def test_refuses_when_not_at_first_pose(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    robot = FakeRobot()
    robot.q["idx62_arm_r_joint2"] = 0.5
    _install(monkeypatch, robot, [rp.ESTOP_PHRASE])
    assert _run(_doc(tmp_path)) == 2
    assert not any(e.startswith("joints") for e in robot.log)


def test_approach_home_needs_typed_confirmation_and_is_slow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    robot = FakeRobot()
    robot.q["idx62_arm_r_joint2"] = 0.5
    _install(monkeypatch, robot, ["APPROACH", rp.ESTOP_PHRASE] + [""] * 10)
    plan = rp.load_plan(_doc(tmp_path))
    first = rp._joint_step("HOME", "", {**plan.moves[0].joints, "idx62_arm_r_joint2": 0.5},
                           plan.moves[0].joints, rp.APPROACH_SPEED)
    assert max(first.velocities) <= rp.APPROACH_SPEED + 1e-9
    assert _run(_doc(tmp_path), "--approach-home") == 0
    assert robot.log[0] == "joints:g2_poc:HOME"


def test_approach_home_refuses_large_gap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    robot = FakeRobot()
    robot.q["idx61_arm_r_joint1"] = -2.67  # e.g. the plan sits on the other shoulder branch
    _install(monkeypatch, robot, ["APPROACH", rp.ESTOP_PHRASE] + [""] * 10)
    assert _run(_doc(tmp_path), "--approach-home") == 2
    assert robot.log == []


def test_refuses_wrong_arm_type(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    robot = FakeRobot(arm_type="crsB")
    _install(monkeypatch, robot, [rp.ESTOP_PHRASE])
    assert _run(_doc(tmp_path, arm="crs")) == 2
    assert robot.log == []


def test_needs_confirm_arm_when_not_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    robot = FakeRobot(arm_type=None)
    _install(monkeypatch, robot, [rp.ESTOP_PHRASE] + [""] * 10)
    assert _run(_doc(tmp_path)) == 2
    _install(monkeypatch, robot, [rp.ESTOP_PHRASE] + [""] * 10)
    assert _run(_doc(tmp_path), "--confirm-arm", "crs") == 0


def test_refuses_assumed_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    robot = FakeRobot()
    _install(monkeypatch, robot, [rp.ESTOP_PHRASE] + [""] * 10)
    assert _run(_doc(tmp_path, measured=False)) == 2
    assert robot.log == []


def test_refuses_without_estop_phrase(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    robot = FakeRobot()
    _install(monkeypatch, robot, ["yes"])
    assert _run(_doc(tmp_path)) == 2
    assert robot.log == []


def test_stops_on_nonzero_return_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    robot = FakeRobot(rc=5)
    _install(monkeypatch, robot, [rp.ESTOP_PHRASE] + [""] * 10)
    assert _run(_doc(tmp_path)) == 2
    assert robot.log.count("joints:g2_poc:A") == 1 and "joints:g2_poc:B" not in robot.log


def test_stops_when_gripper_does_not_move(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    robot = FakeRobot(gripper_works=False)
    _install(monkeypatch, robot, [rp.ESTOP_PHRASE] + [""] * 10)
    t = iter(range(10_000))
    monkeypatch.setattr(rp.time, "time", lambda: next(t) * 0.5)
    assert _run(_doc(tmp_path)) == 2
    assert "joints:g2_poc:B" not in robot.log


def test_no_step_requires_prior_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    robot = FakeRobot()
    _install(monkeypatch, robot, [rp.ESTOP_PHRASE])
    assert _run(_doc(tmp_path), "--no-step") == 2


@pytest.mark.parametrize("plan", PLANS, ids=lambda p: p.name)
def test_generated_plans_are_well_formed(plan: Path) -> None:
    doc = json.loads(plan.read_text())
    if not doc.get("all_ok"):
        with pytest.raises(rp.Abort):
            rp.load_plan(plan)
        return
    p = rp.load_plan(plan)
    steps = rp.plan_steps(p.moves, p.moves[0].joints, 0.3)
    joint_steps = [s for s in steps if s.kind == "joints"] + [j for s in steps if s.kind == "combo" for j in s.sub]
    assert all(max(s.velocities) <= 0.3 + 1e-9 for s in joint_steps)
    assert not any(math.isnan(v) for s in steps for v in s.positions)
    assert rp.main(["--waypoints", str(plan)]) == 0


def _drive_doc(tmp_path: Path, rel: dict, arms_move: bool = False) -> Path:
    base = {n: 0.0 for n in rp.COMMANDED}
    after = {**base, "idx61_arm_r_joint1": 0.2} if arms_move else dict(base)
    wps = [dict(name="START", note="", kind="arm", gripper=0.0, joints=dict(base)),
           dict(name="LEG", note="", kind="drive", gripper=0.0, joints=after, relative_move=rel)]
    p = tmp_path / "d.json"
    p.write_text(json.dumps(dict(all_ok=True, expected_arm="crs", layout=dict(measured=True), waypoints=wps)))
    return p


def test_180_turn_is_split_in_same_direction(tmp_path: Path) -> None:
    plan = rp.load_plan(_drive_doc(tmp_path, dict(dx_m=0.0, dy_m=0.0, dyaw_rad=-math.pi)))
    steps = [s for s in rp.plan_steps(plan.moves, plan.moves[0].joints, 0.3) if s.kind == "chassis"]
    assert [round(math.degrees(s.rel[2])) for s in steps] == [-90, -90]


def test_mixed_translation_and_rotation_is_refused(tmp_path: Path) -> None:
    plan = rp.load_plan(_drive_doc(tmp_path, dict(dx_m=0.5, dy_m=0.0, dyaw_rad=0.3)))
    with pytest.raises(rp.Abort, match="mixes"):
        rp.plan_steps(plan.moves, plan.moves[0].joints, 0.3)


def test_arms_must_hold_during_drive(tmp_path: Path) -> None:
    plan = rp.load_plan(_drive_doc(tmp_path, dict(dx_m=0.5, dy_m=0.0, dyaw_rad=0.0), arms_move=True))
    with pytest.raises(rp.Abort, match="hold"):
        rp.plan_steps(plan.moves, plan.moves[0].joints, 0.3)


def test_chassis_leg_runs_after_relocalization_confirmed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    robot = FakeRobot()
    _install(monkeypatch, robot, [rp.ESTOP_PHRASE, "RELOCALIZED"] + [""] * 10)
    assert _run(_drive_doc(tmp_path, dict(dx_m=0.7, dy_m=0.1, dyaw_rad=0.0))) == 0
    assert robot.log == ["grip:+0.000", "drive:+0.700,+0.100,+0.0"]


def test_chassis_needs_relocalization_phrase(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    robot = FakeRobot()
    _install(monkeypatch, robot, [rp.ESTOP_PHRASE, "yes"])
    assert _run(_drive_doc(tmp_path, dict(dx_m=0.7, dy_m=0.0, dyaw_rad=0.0))) == 2
    assert robot.log == []


def test_chassis_failure_stops_the_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    robot = FakeRobot()
    _install(monkeypatch, robot, [rp.ESTOP_PHRASE, "RELOCALIZED"] + [""] * 10, pnc_state=8)
    assert _run(_drive_doc(tmp_path, dict(dx_m=0.7, dy_m=0.0, dyaw_rad=0.0))) == 2
    assert robot.log[-1] == "cancel"


def test_stale_task_state_is_ignored_until_new_task_appears(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    robot = FakeRobot()  # the Pnc starts on an old cancelled task (state 7)
    _install(monkeypatch, robot, [rp.ESTOP_PHRASE, "RELOCALIZED"] + [""] * 10, pnc_lag=3)
    assert _run(_drive_doc(tmp_path, dict(dx_m=0.7, dy_m=0.0, dyaw_rad=0.0))) == 0
    assert "cancel" not in robot.log


def test_cancels_when_no_new_task_appears(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    robot = FakeRobot()
    _install(monkeypatch, robot, [rp.ESTOP_PHRASE, "RELOCALIZED"] + [""] * 10, pnc_lag=10_000)
    t = iter(range(100_000))
    monkeypatch.setattr(rp.time, "time", lambda: next(t) * 0.1)
    assert _run(_drive_doc(tmp_path, dict(dx_m=0.7, dy_m=0.0, dyaw_rad=0.0))) == 2
    assert robot.log[-2:] == ["drive:+0.700,+0.000,+0.0", "cancel"]


def test_chassis_timeout_cancels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    robot = FakeRobot()
    _install(monkeypatch, robot, [rp.ESTOP_PHRASE, "RELOCALIZED"] + [""] * 10, pnc_state=2)  # still running
    t = iter(range(100_000))
    monkeypatch.setattr(rp.time, "time", lambda: next(t) * 1.0)
    assert _run(_drive_doc(tmp_path, dict(dx_m=0.7, dy_m=0.0, dyaw_rad=0.0))) == 2
    assert robot.log[-1] == "cancel"


def test_ctrl_c_during_a_leg_cancels_the_chassis(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    robot = FakeRobot()
    _install(monkeypatch, robot, [rp.ESTOP_PHRASE, "RELOCALIZED"] + [""] * 10, pnc_state=2)

    def interrupt(s: float) -> None:
        if s < 1.0:  # the chassis poll, not the DDS discovery wait in G2.__init__
            raise KeyboardInterrupt
    monkeypatch.setattr(rp.time, "sleep", interrupt)
    assert _run(_drive_doc(tmp_path, dict(dx_m=0.7, dy_m=0.0, dyaw_rad=0.0))) == 130
    assert robot.log[-1] == "cancel"


def _combo_doc(tmp_path: Path, rel: dict) -> Path:
    base = {n: 0.0 for n in rp.COMMANDED}
    wps = [dict(name="START", note="", kind="arm", gripper=0.0, joints=dict(base)),
           dict(name="LEG~via", note="", kind="drive", combo_group="LEG", gripper=0.0,
                joints={**base, "idx61_arm_r_joint1": 0.1}),
           dict(name="LEG", note="", kind="drive", combo_group="LEG", gripper=0.0,
                joints={**base, "idx61_arm_r_joint1": 0.2}, relative_move=rel)]
    p = tmp_path / "c.json"
    p.write_text(json.dumps(dict(all_ok=True, expected_arm="crs", layout=dict(measured=True), waypoints=wps)))
    return p


def test_arms_move_while_chassis_translates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    robot = FakeRobot()
    _install(monkeypatch, robot, [rp.ESTOP_PHRASE, "RELOCALIZED"] + [""] * 10)
    assert _run(_combo_doc(tmp_path, dict(dx_m=0.0, dy_m=-1.0, dyaw_rad=0.0))) == 0
    assert robot.log == ["grip:+0.000", "drive:+0.000,-1.000,+0.0", "joints:g2_poc:LEG~via", "joints:g2_poc:LEG"]


def test_arms_move_during_a_split_turn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    robot = FakeRobot()
    _install(monkeypatch, robot, [rp.ESTOP_PHRASE, "RELOCALIZED"] + [""] * 10)
    assert _run(_combo_doc(tmp_path, dict(dx_m=0.0, dy_m=0.0, dyaw_rad=math.pi))) == 0
    # second half of the turn starts as soon as the first half reports success, between arm poses
    assert robot.log == ["grip:+0.000", "drive:+0.000,+0.000,+90.0", "joints:g2_poc:LEG~via",
                         "drive:+0.000,+0.000,+90.0", "joints:g2_poc:LEG"]


def test_combo_turn_still_needs_pure_rotation(tmp_path: Path) -> None:
    plan = rp.load_plan(_combo_doc(tmp_path, dict(dx_m=0.3, dy_m=0.0, dyaw_rad=1.0)))
    with pytest.raises(rp.Abort, match="mixes"):
        rp.plan_steps(plan.moves, plan.moves[0].joints, 0.3)


def test_chassis_is_cancelled_when_an_arm_move_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    robot = FakeRobot(rc=5)
    _install(monkeypatch, robot, [rp.ESTOP_PHRASE, "RELOCALIZED"] + [""] * 10)
    assert _run(_combo_doc(tmp_path, dict(dx_m=0.5, dy_m=0.0, dyaw_rad=0.0))) == 2
    assert robot.log[-2:] == ["joints:g2_poc:LEG~via", "cancel"]


def test_until_stops_after_named_waypoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    robot = FakeRobot()
    _install(monkeypatch, robot, [rp.ESTOP_PHRASE] + [""] * 10)
    assert _run(_doc(tmp_path), "--accept-assumed-layout", "--until", "A") == 0
    assert robot.log == ["grip:+0.000", "joints:g2_poc:A", "grip:-0.785"]


def test_until_unknown_waypoint_refused(tmp_path: Path) -> None:
    assert rp.main(["--waypoints", str(_doc(tmp_path)), "--until", "NOPE"]) == 2
