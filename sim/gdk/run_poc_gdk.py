"""Replay the MuJoCo-validated bimanual cup-carry-pour plan on a real AgiBot G2 via GDK.

Arm/waist/head waypoints -> Robot.joint_control_request; both Omnipickers ->
Robot.move_ee_pos (left_tool / right_tool); chassis legs -> Pnc.relative_move
(turns over 90 deg are split so the robot cannot choose the other direction).
"Arms-during-drive" legs (waypoints sharing a combo_group) start the relative_move, send the
arm poses while the chassis translates or turns, then wait for the chassis; the planner checked
those legs for every combination of chassis and arm progress, so their relative timing does not
matter. A turn split into several relative_moves starts its next part as soon as the previous
one reports success (polled after each arm pose; no threads, GDK is never called concurrently).
If an arm move fails during such a leg, the chassis task is cancelled.

Default is a DRY RUN that only prints the plan. Real motion needs --execute plus the
safety confirmations below; every move waits for Enter unless --no-step is allowed.

    python run_poc_gdk.py --waypoints ../out/bimanual_plan.json                  # dry run
    python run_poc_gdk.py --waypoints ... --execute --speed 0.15                  # real robot
    python run_poc_gdk.py --waypoints ... --check-joints                          # sign check
    python run_poc_gdk.py --waypoints ... --execute --until START                 # staged: stop after START

Safety gates on --execute:
- the plan file must be marked all_ok by the simulator;
- the layout in the plan must be marked measured (or --accept-assumed-layout);
- the robot's arm type must match the plan (read from GDK, or --confirm-arm);
- both end effectors must be the plan's (Omnipicker; read from GDK, or --confirm-end-effector);
- the motion control mode read from GDK must equal --control-mode; the script never switches modes;
- the robot must already be within 0.05 rad of the plan's first pose (or --approach-home, only for
  joints within 0.5 rad: a bigger gap usually means a different IK branch, not a small offset);
- the operator types ESTOP-READY: a move cannot be cancelled from software once sent;
- for chassis legs: the operator confirms G02 Pad relocalization (RELOCALIZED) and that the
  robot was placed at the plan's START station; relative_move stops at obstacles, never avoids them.

Chassis legs: the task id is read before each relative_move and only the new task's state is
trusted (get_task_state can still show an earlier task); any abort or Ctrl-C while a leg may be
running calls Pnc.cancel_task first.

API used (GDK 2.6.x / 3.3.x Python): gdk_init/gdk_release, Robot.get_joint_states,
Robot.joint_control_request(JointControlReq), Robot.move_ee_pos(JointStates),
Robot.get_end_state, get_whole_body_status, get_motion_control_status (`mode`),
Pnc.relative_move/get_task_state/cancel_task.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOG = logging.getLogger("g2_poc")

BODY = [f"idx0{i}_body_joint{i}" for i in range(1, 6)]
ARM_R = [f"idx6{i}_arm_r_joint{i}" for i in range(1, 8)]
ARM_L = [f"idx2{i}_arm_l_joint{i}" for i in range(1, 8)]
HEAD = ["idx11_head_joint1", "idx12_head_joint2", "idx13_head_joint3"]
COMMANDED = BODY + HEAD + ARM_L + ARM_R

MAX_SPEED = 0.5            # rad/s, upper bound regardless of --speed (the plan is physics-replayed at 0.5)
APPROACH_SPEED = 0.1       # rad/s, for the optional unchecked move to the first pose
START_TOL = 0.05           # rad, robot must be this close to the first waypoint
MAX_APPROACH_DELTA = 0.5   # rad, largest per-joint gap --approach-home will close (it is not collision-checked)
REACHED_TOL = 0.02         # rad, post-move check
GRIPPER_TOL = 0.08         # rad
GRIPPER_TIMEOUT = 2.0      # s
CHASSIS_TIMEOUT = 60.0     # s per relative_move leg
TASK_START_TIMEOUT = 3.0   # s for get_task_state to show the new leg's task id
MAX_TURN = 1.5708 + 1e-6   # rad, longest single relative_move rotation
TASK_SUCCESS, TASK_FAIL, TASK_CANCELED = 9, 8, 7
ESTOP_PHRASE = "ESTOP-READY"


class Abort(RuntimeError):
    """Refuse to continue; nothing further is sent to the robot."""


@dataclass(frozen=True)
class Move:
    name: str
    note: str
    joints: dict[str, float]
    gripper: float
    kind: str = "arm"                                 # "arm" or "drive"
    rel: tuple[float, float, float] | None = None     # (dx m, dy m, dyaw rad) in the chassis frame
    group: str | None = None                          # drive leg during which the arms move


@dataclass(frozen=True)
class Step:
    kind: str                 # "joints", "gripper", "chassis" or "combo" (arms move while the chassis drives)
    name: str
    note: str
    names: tuple[str, ...] = ()
    positions: tuple[float, ...] = ()
    velocities: tuple[float, ...] = ()
    duration: float = 0.0
    gripper: float = 0.0
    rel: tuple[float, float, float] = (0.0, 0.0, 0.0)
    sub: tuple["Step", ...] = ()   # combo: joint steps sent while the chassis leg runs
    legs: tuple["Step", ...] = ()  # combo: the chassis relative_moves (a long turn is split)


@dataclass(frozen=True)
class Plan:
    moves: list[Move]
    expected_arm: str
    expected_end_effector: str
    layout_measured: bool
    source: Path


def load_plan(path: Path) -> Plan:
    doc = json.loads(path.read_text())
    if not doc.get("all_ok"):
        raise Abort(f"{path.name} is not marked all_ok by the simulator; refusing to run it")
    if doc.get("expected_arm") not in ("crs", "crsB"):
        raise Abort(f"{path.name} does not say which arm (crs/crsB) it was planned for")
    moves = []
    for w in doc["waypoints"]:
        missing = [n for n in COMMANDED if n not in w["joints"]]
        if missing:
            raise Abort(f"waypoint {w['name']} lacks joints {missing}")
        rel = None
        if w.get("kind") == "drive" and "relative_move" in w:
            r = w["relative_move"]
            rel = (float(r["dx_m"]), float(r["dy_m"]), float(r["dyaw_rad"]))
        elif w.get("kind") == "drive" and not w.get("combo_group"):
            raise Abort(f"drive waypoint {w['name']} has no relative_move")
        moves.append(Move(w["name"], w.get("note", ""), {n: float(w["joints"][n]) for n in COMMANDED},
                          float(w["gripper"]), w.get("kind", "arm"), rel, w.get("combo_group")))
    return Plan(moves, doc["expected_arm"], doc.get("expected_end_effector", "omnipicker"), bool(doc.get("layout", {}).get("measured")), path)


def synced_velocities(q0: dict[str, float], q1: dict[str, float], vmax: float) -> tuple[tuple[float, ...], float]:
    """Per-joint speeds so every joint arrives together; peak joint speed = vmax."""
    deltas = [abs(q1[n] - q0[n]) for n in COMMANDED]
    duration = max(max(deltas) / vmax, 0.5)
    vel = tuple(max(d / duration, 0.01) for d in deltas)
    return vel, duration


def _joint_step(name: str, note: str, q0: dict[str, float], q1: dict[str, float], vmax: float) -> Step:
    vel, dur = synced_velocities(q0, q1, vmax)
    return Step("joints", name, note, tuple(COMMANDED), tuple(q1[n] for n in COMMANDED), vel, dur)


def chassis_legs(name: str, note: str, rel: tuple[float, float, float]) -> list[Step]:
    """One relative_move per leg; long turns split into equal parts of at most 90 deg."""
    dx, dy, dyaw = rel
    if abs(dx) > 1e-4 or abs(dy) > 1e-4:
        if abs(dyaw) > 1e-3:
            raise Abort(f"{name}: plan leg mixes translation and rotation; the planner must split them")
        return [Step("chassis", name, note, rel=(dx, dy, 0.0))]
    n = max(1, math.ceil(abs(dyaw) / MAX_TURN))
    return [Step("chassis", f"{name}[{i + 1}/{n}]" if n > 1 else name, note, rel=(0.0, 0.0, dyaw / n))
            for i in range(n)]


def combo_step(group: list[Move], q_prev: dict[str, float], vmax: float) -> Step:
    """One chassis leg (translation or turn) with the arm path (its via-points) sent while it moves."""
    last = group[-1]
    if last.rel is None or any(m.rel is not None for m in group[:-1]):
        raise Abort(f"{last.name}: only the last waypoint of an arms-during-drive leg may carry the relative_move")
    sub, q = [], q_prev
    for m in group:
        if any(abs(m.joints[n] - q[n]) > 1e-4 for n in COMMANDED):
            sub.append(_joint_step(m.name, m.note, q, m.joints, vmax))
        q = m.joints
    legs = tuple(chassis_legs(last.name, last.note, last.rel))
    return Step("combo", last.name, last.note, rel=last.rel, sub=tuple(sub), legs=legs)


def plan_steps(moves: list[Move], start: dict[str, float], vmax: float) -> list[Step]:
    """Joint move to each waypoint, then any gripper change; drive legs become chassis steps."""
    steps, q_prev, g_prev = [], start, None
    i = 0
    while i < len(moves):
        mv = moves[i]
        i += 1
        if mv.group:
            group = [mv]
            while i < len(moves) and moves[i].group == mv.group:
                group.append(moves[i])
                i += 1
            steps.append(combo_step(group, q_prev, vmax))
            q_prev = group[-1].joints
            continue
        if mv.kind == "drive":
            if any(abs(mv.joints[n] - q_prev[n]) > 1e-4 for n in COMMANDED):
                raise Abort(f"{mv.name}: arms must hold their pose during a chassis leg")
            steps.extend(chassis_legs(mv.name, mv.note, mv.rel))
            continue
        if any(abs(mv.joints[n] - q_prev[n]) > 1e-4 for n in COMMANDED):
            steps.append(_joint_step(mv.name, mv.note, q_prev, mv.joints, vmax))
        if g_prev is None or abs(mv.gripper - g_prev) > 1e-3:
            steps.append(Step("gripper", mv.name, mv.note, gripper=mv.gripper))
            g_prev = mv.gripper
        q_prev = mv.joints
    return steps


def truncate_until(steps: list[Step], until: str | None) -> list[Step]:
    """Keep the steps up to and including the last one that reaches waypoint `until` (staged testing)."""
    if until is None:
        return steps
    hits = [i for i, s in enumerate(steps) if s.name.split("[")[0] == until]
    if not hits:
        names = list(dict.fromkeys(s.name.split("[")[0] for s in steps if "~via" not in s.name))
        raise Abort(f"--until {until}: no such step; choose one of {names}")
    return steps[:hits[-1] + 1]


def start_deviation(start: dict[str, float], first: Move) -> dict[str, float]:
    return {n: start[n] - first.joints[n] for n in COMMANDED if abs(start[n] - first.joints[n]) > START_TOL}


def _as_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    return {k: getattr(obj, k) for k in dir(obj) if not k.startswith("_") and not callable(getattr(obj, k))}


class G2:
    """Thin wrapper over agibot_gdk; each method maps to documented GDK calls."""

    def __init__(self, gdk: Any) -> None:
        self.gdk = gdk
        if gdk.gdk_init() != gdk.GDKRes.kSuccess:
            raise Abort("gdk_init failed: check cable, 10.42.1.x address and `source env.sh`")
        self.robot = gdk.Robot()
        self.pnc = gdk.Pnc() if hasattr(gdk, "Pnc") else None
        self.required_mode: int | None = None
        self.task_id: int | None = None   # id of the chassis task this script started last
        time.sleep(2.0)  # let DDS discovery finish

    def control_mode(self) -> int:
        """Current motion control mode: `mode` in GDK 2.6.x, `control_mode` in some other builds."""
        mc = _as_dict(self.robot.get_motion_control_status())
        for key in ("mode", "control_mode"):
            if mc.get(key) is not None:
                return int(mc[key])
        raise Abort(f"get_motion_control_status() has no mode field ({sorted(mc)}); cannot check the control mode")

    def mode_names(self) -> str:
        enum = getattr(self.gdk, "MotionControlMode", None)
        members = getattr(enum, "__members__", None)
        if not members:
            return "(this GDK does not expose MotionControlMode)"
        return ", ".join(f"{int(v)}={k}" for k, v in members.items())

    def require_control_mode(self, required: int | None) -> None:
        """Never switch modes: the operator states which mode joint_control_request is meant for."""
        mode = self.control_mode()
        LOG.info("motion control mode: %s; GDK modes: %s", mode, self.mode_names())
        if required is None:
            raise Abort(f"robot is in motion control mode {mode}. Confirm that joint_control_request is meant to "
                        f"be used in this mode, then pass --control-mode {mode}")
        if mode != required:
            raise Abort(f"robot is in motion control mode {mode}, not {required} (--control-mode); "
                        "this script does not switch modes")
        self.required_mode = required

    def close(self) -> None:
        try:
            self.gdk.gdk_release()
        except Exception as exc:  # gdk_release is known to abort noisily after success
            LOG.warning("gdk_release raised %s (known GDK behaviour)", exc)

    def arm_types(self) -> dict[str, Any]:
        wb = _as_dict(self.robot.get_whole_body_status())
        return {k: wb[k] for k in ("right_arm_type", "left_arm_type", "urdf_type") if k in wb}

    def end_models(self) -> dict[str, Any]:
        wb = _as_dict(self.robot.get_whole_body_status())
        return {k: wb[k] for k in ("left_end_model", "right_end_model") if k in wb}

    def joint_positions(self) -> dict[str, float]:
        js = self.robot.get_joint_states()
        return {s["name"]: float(s["motor_position"]) for s in js["states"]}

    def gripper_positions(self) -> dict[str, float]:
        st = self.robot.get_end_state()
        return {side: float(st[f"{side}_end_state"]["end_states"][0]["position"]) for side in ("left", "right")}

    def check_ok(self) -> None:
        wb = _as_dict(self.robot.get_whole_body_status())
        errs = {k: v for k, v in wb.items() if k.endswith("_error") and v}
        estop = {k: v for k, v in wb.items() if k.endswith("_estop") and v}
        mc = _as_dict(self.robot.get_motion_control_status())
        mode = self.control_mode()
        if errs or estop or mc.get("error_code") or mode != self.required_mode:
            raise Abort(f"robot not ready: errors={errs} estop={estop} mode={mode} "
                        f"(required {self.required_mode}) mc={mc.get('error_msg', '')}")

    def move_joints(self, step: Step) -> None:
        req = self.gdk.JointControlReq()
        req.joint_names = list(step.names)
        req.joint_positions = list(step.positions)
        req.joint_velocities = list(step.velocities)
        req.life_time = step.duration * 1.5 + 2.0
        req.detail = f"g2_poc:{step.name}"
        rc = self.robot.joint_control_request(req)  # documented to block until reached
        if rc not in (None, 0):
            raise Abort(f"{step.name}: joint_control_request returned {rc}")
        now = self.joint_positions()
        worst = max(abs(now[n] - p) for n, p in zip(step.names, step.positions))
        if worst > REACHED_TOL:
            raise Abort(f"{step.name}: joints off target by {worst:.3f} rad after the move")

    def set_grippers(self, value: float) -> None:
        """Both Omnipickers to the same value (cupping posture keeps them closed)."""
        for group in ("left_tool", "right_tool"):
            s = self.gdk.JointState()
            s.position = value
            cmd = self.gdk.JointStates()
            cmd.group = group
            cmd.target_type = "omnipicker"
            cmd.states = [s]
            cmd.nums = 1
            self.robot.move_ee_pos(cmd)  # returns 0 even if nothing moves -> poll below
        t0 = time.time()
        while time.time() - t0 < GRIPPER_TIMEOUT:
            if all(abs(v - value) < GRIPPER_TOL for v in self.gripper_positions().values()):
                return
            time.sleep(0.05)
        raise Abort(f"grippers did not reach {value:.3f} (now {self.gripper_positions()})")

    def _task(self) -> tuple[int | None, int]:
        """(task id, state) from Pnc.get_task_state; id is None if this GDK does not report one."""
        st = self.pnc.get_task_state()
        if isinstance(st, int):
            return None, st
        d = _as_dict(st)
        tid = d.get("id", d.get("task_id"))
        return (None if tid is None else int(tid)), int(d.get("state", d.get("task_state", -1)))

    def _leg_state(self, step: Step) -> int:
        tid, state = self._task()
        if tid != self.task_id:
            raise Abort(f"{step.name}: chassis task changed from {self.task_id} to {tid} while the leg ran")
        return state

    def relative_move(self, step: Step) -> None:
        try:
            self.start_relative_move(step)
            self.wait_chassis(step)
        except BaseException:
            self.cancel_chassis()
            raise

    def start_relative_move(self, step: Step) -> None:
        """Send one leg and wait until get_task_state reports its (new) task id."""
        if self.pnc is None:
            raise Abort("this GDK build has no Pnc client; cannot drive the chassis")
        old_id, old_state = self._task()
        if old_id is None:
            raise Abort("Pnc.get_task_state() reports no task id, so this leg's state cannot be told apart "
                        "from an earlier task's; refusing to drive")
        dx, dy, dyaw = step.rel
        req = self.gdk.NaviReq()
        req.target.position.x, req.target.position.y = dx, dy
        req.target.orientation.z, req.target.orientation.w = math.sin(dyaw / 2), math.cos(dyaw / 2)
        self.pnc.relative_move(req)  # no return value; raises on failure; returns while the leg runs
        t0 = time.time()
        while time.time() - t0 < TASK_START_TIMEOUT:
            tid, _ = self._task()
            if tid != old_id:
                self.task_id = tid
                return
            time.sleep(0.05)
        raise Abort(f"{step.name}: get_task_state still shows the earlier task {old_id} (state {old_state}) "
                    f"{TASK_START_TIMEOUT:.0f} s after relative_move")

    def cancel_chassis(self) -> None:
        if self.pnc is not None and hasattr(self.pnc, "cancel_task"):
            try:
                self.pnc.cancel_task()
            except Exception as exc:
                LOG.warning("cancel_task raised %s", exc)

    def combo(self, step: Step) -> None:
        """Start the chassis leg, send the arm path while it moves, then wait for the chassis.

        A split turn starts its next part after an arm pose once the current part has finished.
        """
        pending = list(step.legs)
        try:
            current = pending.pop(0)
            self.start_relative_move(current)
            for js in step.sub:
                self.move_joints(js)
                if pending and self._chassis_done(current):
                    current = pending.pop(0)
                    self.start_relative_move(current)
            self.wait_chassis(current)
            for leg in pending:
                self.start_relative_move(leg)
                self.wait_chassis(leg)
        except BaseException:
            self.cancel_chassis()
            raise

    def _chassis_done(self, step: Step) -> bool:
        state = self._leg_state(step)
        if state in (TASK_FAIL, TASK_CANCELED):
            raise Abort(f"{step.name}: relative_move ended with task state {state}")
        return state == TASK_SUCCESS

    def wait_chassis(self, step: Step) -> None:
        """Callers cancel the chassis task when this raises."""
        t0 = time.time()
        while time.time() - t0 < CHASSIS_TIMEOUT:
            if self._chassis_done(step):
                return
            time.sleep(0.1)
        raise Abort(f"{step.name}: relative_move did not finish within {CHASSIS_TIMEOUT:.0f} s")


def describe(step: Step) -> str:
    if step.kind == "gripper":
        state = "OPEN" if step.gripper < -0.4 else "CLOSE"
        return f"[grippers] both {state:5s} ({step.gripper:+.3f})  @ {step.name}: {step.note}"
    if step.kind == "combo":
        dx, dy, dyaw = step.rel
        return (f"[combo  ] {step.name:22s} relative_move dx={dx:+.3f} m dy={dy:+.3f} m "
                f"dyaw={math.degrees(dyaw):+6.1f} deg in {len(step.legs)} part(s), arms meanwhile "
                f"through {len(step.sub)} poses ({sum(s.duration for s in step.sub):.1f}s): {step.note}")
    if step.kind == "chassis":
        dx, dy, dyaw = step.rel
        return (f"[chassis] {step.name:22s} relative_move dx={dx:+.3f} m dy={dy:+.3f} m "
                f"dyaw={math.degrees(dyaw):+6.1f} deg: {step.note}")
    return (f"[joints ] -> {step.name:16s} {step.duration:5.1f}s, peak {max(step.velocities):.2f} rad/s: "
            f"{step.note}")


def verify_arm(g2: G2, plan: Plan, confirm_arm: str | None) -> None:
    types = g2.arm_types()
    LOG.info("robot reports arm/urdf type: %s", types or "(not exposed by this GDK)")
    if types:
        right = str(types.get("right_arm_type", types.get("urdf_type", "")))
        # crsB contains "crs", so compare the crsB marker explicitly
        is_b = "crsb" in right.lower()
        if (plan.expected_arm == "crsB") != is_b:
            raise Abort(f"plan is for {plan.expected_arm} but robot reports {types}")
    elif confirm_arm != plan.expected_arm:
        raise Abort(f"GDK does not report the arm type; pass --confirm-arm {plan.expected_arm} "
                    "after checking the arm (upper arm ~287 mm crs vs ~332 mm crsB)")


def verify_end_effector(g2: G2, plan: Plan, confirm: str | None) -> None:
    models = g2.end_models()
    LOG.info("robot reports end effectors: %s", models or "(not exposed by this GDK)")
    want = plan.expected_end_effector.lower()
    if models:
        wrong = {k: v for k, v in models.items() if want not in str(v).lower()}
        if wrong:
            raise Abort(f"plan was made for {plan.expected_end_effector} end effectors but robot reports {models}; "
                        "gripper commands and collision checks do not apply")
    elif confirm != plan.expected_end_effector:
        raise Abort(f"GDK does not report the end effectors; pass --confirm-end-effector "
                    f"{plan.expected_end_effector} after checking both hands")


def check_joints(g2: G2, delta: float = 0.05) -> None:
    """Jog each commanded joint toward the middle of its travel and back, to confirm sign."""
    base = g2.joint_positions()
    for n in COMMANDED:
        sign = -1.0 if base[n] > 0 else 1.0
        input(f"Jog {n} by {sign * delta:+.2f} rad (Enter to move, Ctrl-C to stop) ")
        for target in (base[n] + sign * delta, base[n]):
            q = {**{k: base[k] for k in COMMANDED}, n: target}
            g2.move_joints(_joint_step(n, "jog", g2.joint_positions(), q, 0.1))


def _require(answer: str, phrase: str) -> None:
    if answer.strip() != phrase:
        raise Abort(f"confirmation phrase not entered ({phrase}); nothing was moved")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--waypoints", type=Path, required=True)
    ap.add_argument("--execute", action="store_true", help="move the real robot")
    ap.add_argument("--speed", type=float, default=0.15, help="peak joint speed, rad/s (max 0.5)")
    ap.add_argument("--no-step", action="store_true", help="do not pause before each move")
    ap.add_argument("--i-have-run-this-before", action="store_true", help="required with --no-step")
    ap.add_argument("--check-joints", action="store_true")
    ap.add_argument("--confirm-arm", choices=["crs", "crsB"])
    ap.add_argument("--confirm-end-effector", choices=["omnipicker"])
    ap.add_argument("--control-mode", type=int,
                    help="motion control mode (get_motion_control_status().mode) the robot must already be in")
    ap.add_argument("--accept-assumed-layout", action="store_true")
    ap.add_argument("--until", metavar="WAYPOINT",
                    help="stop after the step that reaches this waypoint (e.g. START), for staged testing")
    ap.add_argument("--approach-home", action="store_true",
                    help="allow an UNCHECKED slow joint move from the current pose to the first waypoint")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        plan = load_plan(args.waypoints)
    except Abort as exc:
        LOG.error("ABORT: %s", exc)
        return 2
    vmax = min(max(args.speed, 0.02), MAX_SPEED)

    if not args.execute and not args.check_joints:
        try:
            steps = truncate_until(plan_steps(plan.moves, plan.moves[0].joints, vmax), args.until)
        except Abort as exc:
            LOG.error("ABORT: %s", exc)
            return 2
        LOG.info("DRY RUN: %s, planned for the %s arm, layout measured: %s", plan.source.name,
                 plan.expected_arm, plan.layout_measured)
        LOG.info("The robot must already be within %.2f rad of %s on every joint.", START_TOL, plan.moves[0].name)
        for s in steps:
            LOG.info("  %s", describe(s))
        LOG.info("arm-only motion ~%.0f s at %.2f rad/s, plus %d chassis legs (%d with the arms moving)",
                 sum(s.duration for s in steps), vmax, sum(1 for s in steps if s.kind in ("chassis", "combo")),
                 sum(1 for s in steps if s.kind == "combo"))
        return 0

    import agibot_gdk  # only available on the robot / GDK dev PC

    if args.no_step and not args.i_have_run_this_before:
        LOG.error("ABORT: --no-step needs --i-have-run-this-before (run step-by-step first)")
        return 2
    g2 = G2(agibot_gdk)
    try:
        g2.require_control_mode(args.control_mode)
        g2.check_ok()
        verify_arm(g2, plan, args.confirm_arm)
        verify_end_effector(g2, plan, args.confirm_end_effector)
        if args.check_joints:
            check_joints(g2)
            return 0
        if not plan.layout_measured and not args.accept_assumed_layout:
            raise Abort("plan was made with an ASSUMED layout (bin height / frame); measure, re-plan, "
                        "or pass --accept-assumed-layout")
        start = g2.joint_positions()
        dev = start_deviation(start, plan.moves[0])
        steps = plan_steps(plan.moves, {n: start[n] for n in COMMANDED}, vmax)
        if dev:
            LOG.warning("robot is not at %s: %s", plan.moves[0].name, {k: round(v, 3) for k, v in dev.items()})
            if not args.approach_home:
                raise Abort("bring the robot to the first pose first (or pass --approach-home)")
            far = {k: round(v, 3) for k, v in dev.items() if abs(v) > MAX_APPROACH_DELTA}
            if far:
                raise Abort(f"--approach-home only closes gaps up to {MAX_APPROACH_DELTA} rad; these joints are "
                            f"further off: {far}. A gap this large usually means the plan uses a different IK "
                            "branch than the robot; report it instead of moving there")
            first = _joint_step(plan.moves[0].name, "UNCHECKED approach to first pose", start,
                                plan.moves[0].joints, APPROACH_SPEED)
            steps = [first] + plan_steps(plan.moves, plan.moves[0].joints, vmax)
            _require(input("The first move is NOT collision-checked. Type APPROACH to allow it: "), "APPROACH")
        steps = truncate_until(steps, args.until)
        _require(input(f"A move cannot be cancelled from software once sent. Keep a hand on the hardware "
                       f"E-STOP and type {ESTOP_PHRASE}: "), ESTOP_PHRASE)
        if any(s.kind in ("chassis", "combo") for s in steps):
            _require(input("Chassis legs follow. Confirm the robot was relocalized on the G02 Pad and placed at "
                           "the plan's START station; type RELOCALIZED: "), "RELOCALIZED")
        for s in steps:
            LOG.info(describe(s))
            if not args.no_step:
                input("  Enter = send this step (E-STOP to stop it), Ctrl-C = abort before sending ")
            g2.check_ok()
            if s.kind == "gripper":
                g2.set_grippers(s.gripper)
            elif s.kind == "chassis":
                g2.relative_move(s)
            elif s.kind == "combo":
                g2.combo(s)
            else:
                g2.move_joints(s)
        LOG.info("done")
        return 0
    except KeyboardInterrupt:
        LOG.warning("aborted by operator: any step already sent ran to completion; no further steps sent")
        return 130
    except Abort as exc:
        LOG.error("ABORT: %s", exc)
        return 2
    finally:
        g2.close()


if __name__ == "__main__":
    sys.exit(main())
