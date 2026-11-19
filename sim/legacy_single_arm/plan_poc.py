"""Solve the 7-step POC waypoints, check the joint-space path, write waypoints.json."""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

import g2_model
import ik

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
IN = g2_model.IN

# ---- task geometry (inches -> metres) ---------------------------------------
BIN_RIM_Z = g2_model.SOURCE_BIN["h"] * IN                       # 30 in assumed
PICK_XY = tuple(float(v) * IN for v in os.environ.get("PICK_IN", "137,83").split(","))  # inside the P1 corner
HOP_C = np.array([g2_model.HOPPER["cx"], g2_model.HOPPER["cy"]]) * IN
P2 = np.array([62.0, 93.0]) * IN
_out = (P2 - HOP_C) / np.linalg.norm(P2 - HOP_C)                # hopper centre -> P2
DUMP_XY = HOP_C + _out * float(os.environ.get("DUMP_R_IN", "16")) * IN                                # 17 in from hopper centre (rim r=19 in, P2 at 17.3)
PRE_RIM_XY = HOP_C + _out * 25 * IN                             # just outside the rim
DOWN = (0.0, 0.0, -1.0)
_tilt = np.deg2rad(75)  # 60 deg puts the wrist into the 63 in frame; 75 deg clears it
DUMP_APPROACH = tuple(np.cos(_tilt) * np.array(DOWN) + np.sin(_tilt) * np.append(-_out, 0.0))

# Each step: (name, target or None, gripper value, note)
STEPS = [
    ("HOME", ik.Target((2.95, 2.18, 1.05), DOWN, 0.02, 10), ik.GRIP_CLOSED, "start / end pose"),
    ("ABOVE_BIN", ik.Target((*PICK_XY, BIN_RIM_Z + 0.15), DOWN, 0.01, 20), ik.GRIP_OPEN, "1 open gripper above P1"),
    ("IN_BIN", ik.Target((*PICK_XY, BIN_RIM_Z - float(os.environ.get("PICK_DEPTH", "0.0"))), DOWN, 0.015, 20), ik.GRIP_OPEN, "2 descend into the bin"),
    ("GRASP", None, ik.GRIP_CLOSED, "3 close gripper (fake grab)"),
    ("LIFT", ik.Target((*PICK_XY, BIN_RIM_Z + 0.20), DOWN, 0.01, 20), ik.GRIP_CLOSED, "4 lift out of the bin"),
    ("TRANSIT", ik.Target((2.55, 2.55, float(os.environ.get("TRANSIT_Z", "1.40"))), DOWN, 0.03, 15), ik.GRIP_CLOSED, "5a raise and turn"),
    ("PRE_RIM", ik.Target((*PRE_RIM_XY, float(os.environ.get("PRE_RIM_Z", "1.40"))), DUMP_APPROACH, 0.02, 10), ik.GRIP_CLOSED, "5b outside the rim, under the frame"),
    ("ABOVE_HOPPER", ik.Target((*DUMP_XY, float(os.environ.get("DUMP_Z", "1.35"))), DUMP_APPROACH, 0.03, 15), ik.GRIP_CLOSED, "5c over the hopper at P2"),
    ("RELEASE", None, ik.GRIP_OPEN, "6 open gripper (fake dump)"),
    ("RELEASE_DONE", None, ik.GRIP_CLOSED, "6b close gripper again before backing out (open fingers would reach the frame)"),
    ("RETREAT", "PRE_RIM", ik.GRIP_CLOSED, "7a back out over the rim (same pose as PRE_RIM)"),
    ("RETURN_TRANSIT", "TRANSIT", ik.GRIP_CLOSED, "7b turn back (same pose as TRANSIT)"),
    ("HOME_END", None, ik.GRIP_CLOSED, "7c home"),
]

# poses solved with the gripper OPEN (for clearance) even though they are reached closed
SOLVE_OPEN = {"ABOVE_HOPPER"}
LEFT_TUCK_TARGET = np.array([0.20, 0.32, 0.85])  # robot frame (x fwd, y left), from the chassis centre
HEAD = {n: 0.0 for n in ik.HEAD}
SAMPLES_PER_SEGMENT = 120
VMAX = 0.3  # rad/s, peak joint speed used for the independent-arrival profile


def tuck_left_arm(chk: ik.Checker) -> dict[str, float]:
    """Put the idle left arm low, beside and in front of the torso, clear of the body."""
    m = chk.model
    chk.set_q({})
    base_id = m.body("base_link").id
    R, base = chk.data.xmat[base_id].reshape(3, 3).copy(), chk.data.xpos[base_id].copy()
    tgt = base + R @ LEFT_TUCK_TARGET
    lo, hi = ik.bounds(m, ik.ARM_L)
    b = m.body("gripper_l_base_link").id
    left_pairs = [(a, c) for a, c in chk.self_pairs
                  if "L" in (ik._side(m.body(m.geom_bodyid[a]).name), ik._side(m.body(m.geom_bodyid[c]).name))
                  and "R" not in (ik._side(m.body(m.geom_bodyid[a]).name), ik._side(m.body(m.geom_bodyid[c]).name))]

    def res(x: np.ndarray) -> np.ndarray:
        chk.set_q(dict(zip(ik.ARM_L, x)))
        d, _ = chk.min_distance(left_pairs, 0.15)
        return np.concatenate([chk.data.xpos[b] - tgt, [30.0 * max(0.0, 0.06 - d)], 0.02 * x])

    best = None
    for x0 in ([0.0, -1.2, 0.0, -1.5, 0.0, 0.0, 0.0], [0.5, -1.0, 0.5, -1.8, 0.0, 0.3, 0.0],
               [-0.5, -1.4, -0.5, -1.2, 0.0, -0.3, 0.0]):
        sol = least_squares(res, np.clip(np.array(x0), lo + 1e-3, hi - 1e-3), bounds=(lo, hi), max_nfev=300)
        chk.set_q(dict(zip(ik.ARM_L, sol.x)))
        d, _ = chk.min_distance(left_pairs, 1.0)
        if best is None or d > best[0]:
            best = (d, sol.x)
    print(f"left-arm tuck: clearance to body {best[0]:.3f} m")
    return dict(zip(ik.ARM_L, best[1]))


def _profiles(qa: dict[str, float], qb: dict[str, float]):
    """Joint-space paths the real controller might take between two waypoints.

    sync: all joints arrive together (straight line in joint space, what the runner commands).
    independent: every joint moves at the same peak speed and stops on arrival (worst case if
    GDK does not synchronise arrival).
    """
    names = list(qa)
    a = np.array([qa[n] for n in names])
    b = np.array([qb[n] for n in names])
    for s in np.linspace(0.0, 1.0, SAMPLES_PER_SEGMENT):
        yield "sync", dict(zip(names, a + s * (b - a)))
    d = b - a
    t_end = float(np.max(np.abs(d))) / VMAX
    for t in np.linspace(0.0, t_end, SAMPLES_PER_SEGMENT):
        step = np.sign(d) * np.minimum(np.abs(d), VMAX * t)
        yield "indep", dict(zip(names, a + step))


def path_check(chk: ik.Checker, qa: dict[str, float], qb: dict[str, float]) -> dict:
    env_pairs = chk.env_pairs()
    worst_env, worst_self, worst_cpl, near = 1.0, 1.0, -1.0, ""
    m = chk.model
    for prof, q in _profiles(qa, qb):
        s = prof
        chk.set_q(q)
        d_env, (ga, gb) = chk.min_distance(env_pairs, 1.0)
        if d_env < worst_env:
            worst_env, near = d_env, f"{m.body(m.geom_bodyid[ga]).name}<->{m.geom(gb).name}[{s}]"
        worst_self = min(worst_self, chk.min_distance(chk.self_pairs, 1.0)[0])
        worst_cpl = max(worst_cpl, chk.coupled_violation())
    return dict(closest=near, min_env_clearance_m=round(worst_env, 4), min_self_clearance_m=round(worst_self, 4),
                max_coupled_violation=round(max(0.0, worst_cpl), 4))


def segment_ok(pc: dict) -> bool:
    return (pc["min_env_clearance_m"] >= ik.ENV_CLEARANCE
            and pc["min_self_clearance_m"] >= ik.SELF_CLEARANCE
            and pc["max_coupled_violation"] <= 1e-3)


def _via(chk: ik.Checker, a: dict, b: dict, depth: int) -> list[dict]:
    """Return via-points (excluding a and b) that make a -> b collision-free."""
    same_pose = all(abs(a["joints"][n] - b["joints"][n]) < 1e-6 for n in ik.IK_JOINTS)
    if same_pose or depth >= 4 or segment_ok(path_check(chk, a["joints"], b["joints"])):
        return []
    mid = {n: 0.5 * (a["joints"][n] + b["joints"][n]) for n in a["joints"]}
    fixed = {n: v for n, v in a["joints"].items() if n not in ik.IK_JOINTS}
    q_arm = ik.repair(chk, mid, fixed)
    q = {**fixed, **q_arm}
    chk.set_q(q)
    tcp, app = chk.tcp()
    rep = ik.evaluate(chk, q)
    w = dict(name=f"{a['name'].split('~')[0]}~via", note="auto via-point (collision-free path)",
             gripper=a["gripper"], ok=True, tcp_world_m=np.round(tcp, 4).tolist(),
             approach=np.round(app, 3).tolist(),
             check={k: round(v, 4) for k, v in asdict(rep).items()},
             joints={n: round(float(v), 5) for n, v in q.items()})
    return _via(chk, a, w, depth + 1) + [w] + _via(chk, w, b, depth + 1)


def refine_path(chk: ik.Checker, wps: list[dict]) -> list[dict]:
    out = [wps[0]]
    for a, b in zip(wps[:-1], wps[1:]):
        out.extend(_via(chk, a, b, 0))
        out.append(b)
    return out


def main() -> int:
    model = g2_model.build_model()
    chk = ik.Checker(model)
    left = tuck_left_arm(chk)
    fixed_base = {**HEAD, **left, ik.GRIP_L: ik.GRIP_CLOSED}

    seed = {n: 0.0 for n in ik.IK_JOINTS}
    seed.update({"idx62_arm_r_joint2": 1.0, "idx64_arm_r_joint4": -1.5})
    waypoints, all_ok = [], True
    home_q = None
    for name, target, grip, note in STEPS:
        fixed = {**fixed_base, ik.GRIP_R: grip}
        if name == "HOME_END":
            q_arm = home_q
        elif isinstance(target, str):
            src = next(w for w in waypoints if w["name"] == target)
            q_arm = {n: src["joints"][n] for n in ik.IK_JOINTS}
            target = None
        elif target is None:
            q_arm = {n: waypoints[-1]["joints"][n] for n in ik.IK_JOINTS}
        else:
            solve_fixed = {**fixed, ik.GRIP_R: ik.GRIP_OPEN} if name in SOLVE_OPEN else fixed
            q_arm = seed
            for _ in range(3):  # re-solve from the last answer if constraints are still violated
                q_arm = ik.solve(chk, target, q_arm, solve_fixed)
                if ik.evaluate(chk, {**solve_fixed, **q_arm}, target).ok(target):
                    break
        q = {**fixed, **q_arm}
        rep = ik.evaluate(chk, q, target)
        ok = rep.ok(target) if target else True
        all_ok &= ok
        if name == "HOME":
            home_q = q_arm
        seed = q_arm
        tcp, app = chk.tcp()
        waypoints.append(dict(name=name, note=note, gripper=grip, ok=ok,
                              tcp_world_m=np.round(tcp, 4).tolist(),
                              approach=np.round(app, 3).tolist(),
                              check={k: round(v, 4) for k, v in asdict(rep).items()},
                              joints={n: round(float(v), 5) for n, v in q.items()}))
        if target is not None and not ok:
            print("   fail detail:", rep)
        print(f"{name:15s} ok={ok} pos_err={rep.pos_err:.3f} app={rep.approach_err_deg:4.1f}deg "
              f"env={rep.env_clear:.3f} self={rep.self_clear:.3f} tilt={rep.tilt_deg:.1f} "
              f"shift={rep.shift:.3f} coupled={rep.coupled:.3f} tcp={np.round(tcp, 3)}")

    waypoints = refine_path(chk, waypoints)
    segments = []
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        pc = path_check(chk, a["joints"], b["joints"])
        seg_ok = segment_ok(pc)
        all_ok &= seg_ok
        segments.append(dict(frm=a["name"], to=b["name"], ok=seg_ok, **pc))
        print(f"  {a['name']:>22s} -> {b['name']:<22s} ok={seg_ok} {pc}")

    OUT.mkdir(exist_ok=True)
    doc = dict(
        robot=f"AgiBot G2, {g2_model.ARM} arm, Omnipicker",
        expected_arm=g2_model.ARM,
        model_source=dict(repo="github.com/AgibotTech/genie_sim_robot_model", commit=g2_model.OFFICIAL_COMMIT,
                          xacro=f"robots/genie/g2/g2_{g2_model.ARM}_omnipicker.urdf.xacro", collision_type="fine"),
        units=dict(joints="rad", gripper="omnipicker joint, -0.785 open .. 0 closed (same as GDK move_ee_pos)"),
        layout=dict(robot_base_in=g2_model.ROBOT_BASE_IN, robot_yaw_deg=g2_model.ROBOT_YAW_DEG,
                    measured=os.environ.get("LAYOUT_MEASURED", "0") == "1",
                    assumed=dict(bin_height_in=g2_model.SOURCE_BIN["h"], frame_top_in=g2_model.FRAME_TOP_IN,
                                 frame_beam_depth_in=g2_model.FRAME_SLAB_IN, frame_posts="not modelled")),
        path_profiles=["sync", "independent-arrival"],
        clearance_required_m=dict(env=ik.ENV_CLEARANCE, self=ik.SELF_CLEARANCE),
        all_ok=bool(all_ok), waypoints=waypoints, segments=segments)
    (OUT / os.environ.get("PLAN_OUT", f"waypoints_{g2_model.ARM}.json")).write_text(json.dumps(doc, indent=2))
    print("ALL_OK", all_ok)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
