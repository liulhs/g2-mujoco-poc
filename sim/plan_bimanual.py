"""Plan the bimanual cup-carry-pour POC with a driving chassis.

Sequence (like a person): from the robot's reset pose bring the hands to the chest, drive to the source bin, cup a handful with BOTH hands
(closed Omnipickers side by side), hold the cupping posture while the chassis backs off,
turns and drives to the hopper, raise both hands over the 50 in rim (under the frame),
pour with both wrists, then drive back.

Every chassis leg is a pure in-place rotation or a pure straight translation, so each
maps to one GDK pnc.relative_move. Checks on every waypoint and every leg:
>= 5 cm robot<->workcell (chassis included), >= 2 cm between non-adjacent robot parts
(both arms, both grippers, torso), AgiBot coupled-joint limits, torso lean <= 10 deg,
joint-limit margin. Arm legs are checked for synchronised and independent-arrival
joint timing. Output: out/bimanual_plan.json.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

import g2_model
import ik

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
IN = g2_model.IN
SAMPLES = 120
VMAX = 0.3              # rad/s peak joint speed (independent-arrival profile)
HAND_GAP = 0.11         # m between the two TCPs: closed grippers (~10 cm wide) side by side, ~1 cm apart
POSE_BUDGET_M = 0.03    # chassis stop error the station arm legs must tolerate (runner checks <= 2 cm)
POSE_BUDGET_DEG = 2.0
RIM_BIN = g2_model.SOURCE_BIN["h"] * IN
HOP = g2_model.HOPPER
HOP_C = np.array([HOP["cx"], HOP["cy"]]) * IN

# chassis poses: (x_in, y_in, yaw_deg). Every leg is a pure turn or a pure translation (the swerve
# base can translate in any direction, so sideways/diagonal moves replace turn-drive-turn).
START = (84.0, 60.0, 90.0)          # centre of the drawing's G2 placement area, facing +y
BIN_STATION = (float(os.environ.get("BIN_STATION_X", "112")), 60.0, 0.0)  # front edge ~5 in from the bin wall
BIN_BACKOFF = (103.0, 60.0, 0.0)
TURN_POINT = (103.0, 60.0, 180.0)   # turned around at the back-off point
HOP_STANDOFF = (103.0, 105.5, 180.0)  # in line with the hopper centre, facing it
HOP_DOCK = (float(os.environ.get("HOP_DOCK_X", "89")), 105.5, 180.0)      # front edge ~4 in from the hopper base
HOME_TURN = (84.0, 60.0, 180.0)     # back at the start spot, still facing west
GRID = 21                           # chassis x arm progress samples per axis for overlapped legs
STATIONS = {"bin": BIN_STATION, "hopper": HOP_DOCK}

GRIP = ik.GRIP_CLOSED               # both Omnipickers closed the whole time (cupping posture)
HEAD = {n: 0.0 for n in ik.HEAD}


@dataclass(frozen=True)
class ArmGoal:
    """Both TCPs; pos is the midpoint between the hands, in WORLD (world=True) or BASE frame."""
    name: str
    note: str
    pos: tuple[float, float, float]
    down_deg: float                 # approach tilt below horizontal, along the robot's forward axis
    world: bool = False
    tol: float = 0.04
    app_tol: float = 10.0


def base_q(pose: tuple[float, float, float], yaw_rad: float | None = None) -> dict[str, float]:
    return {"base_x": pose[0] * IN, "base_y": pose[1] * IN,
            "base_yaw": np.deg2rad(pose[2]) if yaw_rad is None else yaw_rad}


def hand_targets(goal: ArmGoal, bq: dict[str, float]) -> dict[str, ik.Target]:
    yaw = bq["base_yaw"]
    fwd, left = np.array([np.cos(yaw), np.sin(yaw), 0.0]), np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    if goal.world:
        mid = np.array(goal.pos)
    else:
        mid = np.array([bq["base_x"], bq["base_y"], 0.0]) + goal.pos[0] * fwd + goal.pos[1] * left
        mid[2] = goal.pos[2]
    th = np.deg2rad(goal.down_deg)
    up_w = np.array([0, 0, 1.0])
    app = tuple(np.cos(th) * fwd - np.sin(th) * up_w)
    # cupping "tray": both grippers same orientation, finger plane level (camera side up),
    # pitched with the approach; fingers open sideways (local y), hands side by side
    up = tuple(np.sin(th) * fwd + np.cos(th) * up_w)
    return {"tcp_l": ik.Target(tuple(mid + left * HAND_GAP / 2), app, goal.tol, goal.app_tol, up),
            "tcp_r": ik.Target(tuple(mid - left * HAND_GAP / 2), app, goal.tol, goal.app_tol, up)}


CARRY = ArmGoal("CARRY", "both hands cupped together, level, in front of the chest", (0.42, 0.0, 1.00), 0.0)
HAND_Z_OVER = float(os.environ.get("HAND_Z_OVER", "1.55"))   # hands height when crossing the rim
POUR_Z = float(os.environ.get("POUR_Z", "1.36"))
POUR_DEG = float(os.environ.get("POUR_DEG", "55"))
RAISE = ArmGoal("RAISE", "raise the cupped hands above the 50 in rim, below the frame", (0.42, 0.0, HAND_Z_OVER), 0.0)
PRE_SCOOP = ArmGoal("PRE_SCOOP", "1 both hands above the bin, tilted down",
                    ((139 - BIN_STATION[0]) * IN, 0.0, RIM_BIN + 0.12), 60.0)   # base frame at BIN_STATION
READY = ArmGoal("READY", "hands part way to the bin pose", (0.55, 0.0, 0.95), 30.0)
# ("drive", name, note, pose[, ArmGoal]): with an ArmGoal (base frame) the arms move DURING the chassis
# leg; the leg is checked for every combination of chassis and arm progress, and falls back to
# "arms first, then drive" if that check fails.
# ("reset", name, note, pose): the robot's own reset pose (ik.RESET_POSE, waist upright) at that chassis
# pose. The plan starts here so the reset -> START arm move is collision-checked (with via-points if
# needed) instead of being left to the runner's unchecked --approach-home.
SEQUENCE: list[tuple] = [
    ("reset", "RESET", "robot reset pose (both arms j2 = -1.571), as read on the robot", START),
    ("arm", ArmGoal("START", "start: hands cupped in front", CARRY.pos, 0.0), START),
    ("drive", "TURN_TO_BIN", "turn in place to face the source bin, starting to lift the hands", (START[0], START[1], 0.0),
     READY),
    ("drive", "DRIVE_TO_BIN", "drive forward to the source bin, hands moving above the bin", BIN_STATION, PRE_SCOOP),
    ("arm", ArmGoal("SCOOP_IN", "2 dip both hands into the bin", (140 * IN, 60 * IN, RIM_BIN - 0.05), 60.0, True), None),
    ("arm", ArmGoal("SCOOP_SWEEP", "3 scoop forward, rolling the hands up", (141 * IN, 60 * IN, RIM_BIN - 0.02), 30.0, True, 0.03, 10.0), None),
    ("arm", ArmGoal("SCOOP_LIFT", "4 lift the handful out level", (140 * IN, 60 * IN, RIM_BIN + 0.22), 0.0, True), None),
    ("drive", "BACK_OFF", "back away from the bin, holding the handful", BIN_BACKOFF),
    ("drive", "TURN_AROUND", "turn around in place, bringing the cupped hands to the chest", TURN_POINT, CARRY),
    ("drive", "SLIDE_TO_HOPPER", "slide sideways to the hopper, raising the cupped hands", HOP_STANDOFF, RAISE),
    ("drive", "DOCK", "move up to the hopper, reaching over the rim",
     HOP_DOCK, ArmGoal("OVER_HOPPER", "5 reach over the rim into the hopper", (63 * IN, 105.5 * IN, HAND_Z_OVER), 0.0, True)),
    ("arm", ArmGoal("POUR", "6 tip both hands down into the hopper: pour", (60 * IN, 105.5 * IN, POUR_Z), POUR_DEG, True, 0.03, 12.0), None),
    ("drive", "UNDOCK", "back off from the hopper, levelling the hands and pulling them back over the rim",
     HOP_STANDOFF, ArmGoal("RETRACT", "level the hands and pull them back over the rim", RAISE.pos, 0.0)),
    ("drive", "RETURN", "slide diagonally back to the start, lowering the hands",
     HOME_TURN, ArmGoal("HALF_LOWER", "lower the hands part way", (0.42, 0.0, 0.5 * (HAND_Z_OVER + CARRY.pos[2])), 0.0)),
    ("drive", "TURN_FINAL", "turn to the start heading, lowering the hands to the carry posture", START,
     ArmGoal("LOWER", "lower to the carry posture", CARRY.pos, 0.0)),
]


# ---------------- path checking ----------------------------------------------------------
def _arm_profiles(qa: dict[str, float], qb: dict[str, float]):
    names = list(qa)
    a, b = np.array([qa[n] for n in names]), np.array([qb[n] for n in names])
    for s in np.linspace(0.0, 1.0, SAMPLES):
        yield "sync", dict(zip(names, a + s * (b - a)))
    d = b - a
    arm = np.array([n not in ik.BASE for n in names])
    t_end = float(np.max(np.abs(d[arm]))) / VMAX if arm.any() else 0.0
    for t in np.linspace(0.0, t_end, SAMPLES):
        step = np.where(arm, np.sign(d) * np.minimum(np.abs(d), VMAX * t), 0.0)
        yield "indep", dict(zip(names, a + step))


def _drive_profile(qa: dict[str, float], qb: dict[str, float]):
    names = list(qa)
    a, b = np.array([qa[n] for n in names]), np.array([qb[n] for n in names])
    for s in np.linspace(0.0, 1.0, SAMPLES):
        yield "drive", dict(zip(names, a + s * (b - a)))


def _combo_profile(qa: dict[str, float], qb: dict[str, float]):
    """Arms and chassis move together but with unknown relative timing: every progress pair."""
    names = list(qa)
    a, b = np.array([qa[n] for n in names]), np.array([qb[n] for n in names])
    base = np.array([n in ik.BASE for n in names])
    for sa in np.linspace(0.0, 1.0, GRID):
        for sb in np.linspace(0.0, 1.0, GRID):
            yield "combo", dict(zip(names, a + np.where(base, sb, sa) * (b - a)))


def path_check(chk: ik.Checker, qa: dict[str, float], qb: dict[str, float], kind: str) -> dict:
    env_pairs, m = chk.env_pairs(), chk.model
    worst_env, worst_self, worst_hand, worst_cpl, near = 1.0, 1.0, 1.0, -1.0, ""
    worst_tilt, worst_shift = 0.0, 0.0
    gen = {"drive": _drive_profile, "combo": _combo_profile}.get(kind, _arm_profiles)(qa, qb)
    for prof, q in gen:
        chk.set_q(q)
        d_env, (ga, gb) = chk.min_distance(env_pairs, 1.0)
        if d_env < worst_env and ga >= 0:
            worst_env, near = d_env, f"{m.body(m.geom_bodyid[ga]).name}<->{m.geom(gb).name}[{prof}]"
        worst_self = min(worst_self, chk.min_distance(chk.self_pairs, 1.0)[0])
        worst_hand = min(worst_hand, chk.min_distance(chk.hand_pairs, 1.0)[0])
        worst_cpl = max(worst_cpl, chk.coupled_violation())
        tilt, shift = chk.torso_state()
        worst_tilt, worst_shift = max(worst_tilt, tilt), max(worst_shift, shift)
    return dict(closest=near, min_env_clearance_m=round(worst_env, 4),
                min_self_clearance_m=round(worst_self, 4), min_hand_gap_m=round(worst_hand, 4),
                max_coupled_violation=round(max(0.0, worst_cpl), 4),
                max_torso_tilt_deg=round(float(np.rad2deg(worst_tilt)), 2), max_waist_shift_m=round(worst_shift, 4))


def segment_ok(pc: dict) -> bool:
    return (pc["min_env_clearance_m"] >= ik.ENV_CLEARANCE and pc["min_self_clearance_m"] >= ik.SELF_CLEARANCE
            and pc["min_hand_gap_m"] >= ik.HAND_CLEARANCE
            and pc["max_coupled_violation"] <= 5e-3   # still >= 15 mrad inside AgiBot's limit (margin 20 mrad)
            and pc["max_torso_tilt_deg"] <= np.rad2deg(ik.MAX_TORSO_TILT) + 0.5
            and pc["max_waist_shift_m"] <= ik.MAX_WAIST_SHIFT + 0.015)


def _offset(q: dict[str, float], dx: float, dy: float, dyaw: float) -> dict[str, float]:
    """Shift the chassis by (dx, dy) in its own frame and turn it by dyaw."""
    c, s = np.cos(q["base_yaw"]), np.sin(q["base_yaw"])
    return {**q, "base_x": q["base_x"] + c * dx - s * dy, "base_y": q["base_y"] + s * dx + c * dy,
            "base_yaw": q["base_yaw"] + dyaw}


def pose_budget_check(chk: ik.Checker, qa: dict[str, float], qb: dict[str, float]) -> dict:
    """Worst workcell clearance of an arm leg when the chassis stopped up to the budget off its pose."""
    env_pairs, m = chk.env_pairs(), chk.model
    worst, where = 1.0, ""
    b, y = POSE_BUDGET_M, np.deg2rad(POSE_BUDGET_DEG)
    for dx in (-b, 0.0, b):
        for dy in (-b, 0.0, b):
            for dyaw in (-y, 0.0, y):
                if dx == dy == dyaw == 0.0:
                    continue
                qa2, qb2 = _offset(qa, dx, dy, dyaw), _offset(qb, dx, dy, dyaw)
                for prof, q in _arm_profiles(qa2, qb2):
                    if prof != "sync":
                        continue
                    chk.set_q(q)
                    d, (ga, gb) = chk.min_distance(env_pairs, 0.3)
                    if d < worst and ga >= 0:
                        worst, where = d, (f"{m.body(m.geom_bodyid[ga]).name}<->{m.geom(gb).name} "
                                           f"offset=({dx * 100:+.0f}cm,{dy * 100:+.0f}cm,{np.rad2deg(dyaw):+.0f}deg)")
    return dict(budget=f"+-{POSE_BUDGET_M * 100:.0f} cm, +-{POSE_BUDGET_DEG:.0f} deg",
                min_env_clearance_m=round(worst, 4), closest=where)


def _waypoint(chk: ik.Checker, name: str, note: str, kind: str, q: dict[str, float],
              rep: ik.Report, ok: bool, extra: dict | None = None) -> dict:
    chk.set_q(q)
    tl, al = chk.tcp("tcp_l")
    tr, ar = chk.tcp("tcp_r")
    return dict(name=name, note=note, kind=kind, ok=bool(ok), gripper=GRIP,
                tcp_l_world_m=np.round(tl, 4).tolist(), tcp_r_world_m=np.round(tr, 4).tolist(),
                approach_l=np.round(al, 3).tolist(), approach_r=np.round(ar, 3).tolist(),
                check={k: round(float(v), 4) for k, v in asdict(rep).items()},
                joints={n: round(float(v), 5) for n, v in q.items()}, **(extra or {}))


def _via(chk: ik.Checker, a: dict, b: dict, depth: int) -> list[dict]:
    if depth >= 4 or segment_ok(path_check(chk, a["joints"], b["joints"], "arm")):
        return []
    mid = {n: 0.5 * (a["joints"][n] + b["joints"][n]) for n in a["joints"]}
    fixed = {n: v for n, v in a["joints"].items() if n not in ik.BIMANUAL_JOINTS}
    q = {**fixed, **ik.repair_multi(chk, mid, fixed)}
    rep = ik.evaluate_multi(chk, q)
    w = _waypoint(chk, f"{b['name'].split('~')[0]}~via", "auto via-point (collision-free)", "arm", q,
                  rep, rep.ok())
    return _via(chk, a, w, depth + 1) + [w] + _via(chk, w, b, depth + 1)


def _combo_chain(chk: ik.Checker, prev_wp: dict, q_end: dict[str, float], rep_end: ik.Report,
                 name: str, note: str, goal_name: str) -> tuple[list[dict] | None, dict | None]:
    """Arms follow their (via-refined) path while the chassis drives the whole leg.

    Each arm sub-segment is checked against EVERY chassis position of the leg, so the result
    holds for any relative timing of arms and chassis on the real robot.
    Returns the group's waypoints (base joints at the arm-progress fraction) or None.
    """
    qa = prev_wp["joints"]
    base_a = {n: qa[n] for n in ik.BASE}
    base_b = {n: q_end[n] for n in ik.BASE}
    # arm path at a fixed chassis pose (arm self/hand clearance does not depend on the chassis)
    a_fix = {**prev_wp, "joints": dict(qa)}
    b_fix = {**prev_wp, "name": name, "joints": {**q_end, **base_a}}
    vias = _via(chk, a_fix, b_fix, 0)
    chain = [w["joints"] for w in vias] + [b_fix["joints"]]
    reps = [ik.Report(**w["check"]) for w in vias] + [rep_end]
    oks = [w["ok"] for w in vias] + [True]
    arm_names = [n for n in qa if n not in ik.BASE]
    travel = [max(abs(q1[n] - q0[n]) for n in arm_names) for q0, q1 in zip([qa] + chain[:-1], chain)]
    total = sum(travel) or 1.0
    frac, members, worst = 0.0, [], None
    for i, (q_arm, rep, ok, tr) in enumerate(zip(chain, reps, oks, travel)):
        q0 = {**(members[-1]["joints"] if members else qa), **base_a}
        pc = path_check(chk, q0, {**q_arm, **base_b}, "combo")
        if not (ok and segment_ok(pc)):
            return None, pc
        frac += tr / total
        q = {**q_arm, **{n: base_a[n] + frac * (base_b[n] - base_a[n]) for n in ik.BASE}}
        last = i == len(chain) - 1
        extra = dict(combo_group=name, leg_check=pc)
        if last:
            extra.update(relative_move=relative_move(qa, q_end), arms_during_drive=goal_name)
        members.append(_waypoint(chk, name if last else f"{name}~via",
                                 f"{note} [{goal_name} during the leg]" if last else "auto via-point (arms, during the leg)",
                                 "drive", q_end if last else q, rep, ok, extra))
        if worst is None or pc["min_env_clearance_m"] < worst["min_env_clearance_m"]:
            worst = pc
    return members, worst


def _turn_choices(prev: dict[str, float], pose: tuple[float, float, float]) -> list[dict[str, float]]:
    """End chassis poses for a leg: the short way round (both ways for a 180 deg turn)."""
    target_yaw = np.deg2rad(pose[2])
    out = []
    for k in (-1, 0, 1):
        yaw = target_yaw + 2 * np.pi * k
        if abs(yaw - prev["base_yaw"]) <= np.pi + 1e-6:
            out.append(base_q(pose, yaw))
    return out


def relative_move(qa: dict[str, float], qb: dict[str, float]) -> dict:
    """GDK relative_move target, in the chassis frame at the start of the leg."""
    yaw0 = qa["base_yaw"]
    dxw, dyw = qb["base_x"] - qa["base_x"], qb["base_y"] - qa["base_y"]
    c, s = np.cos(yaw0), np.sin(yaw0)
    return dict(dx_m=round(c * dxw + s * dyw, 4), dy_m=round(-s * dxw + c * dyw, 4),
                dyaw_rad=round(qb["base_yaw"] - yaw0, 4))


def main() -> int:
    chk = ik.Checker(g2_model.build_model())
    fixed_static = {**HEAD, ik.GRIP_L: GRIP, ik.GRIP_R: GRIP}
    # seed from the robot's own reset pose: START then sits near it (the IK is regularised toward the seed)
    seed = {n: 0.0 for n in ik.BIMANUAL_JOINTS}
    seed.update(ik.RESET_POSE)
    wps: list[dict] = []
    all_ok = True
    bq = base_q(START)
    for item in SEQUENCE:
        kind = item[0]
        if kind == "reset":
            _, name, note, pose = item
            bq = base_q(pose)
            q = {**fixed_static, **bq, **{n: 0.0 for n in ik.BODY}, **ik.RESET_POSE}
            rep = ik.evaluate_multi(chk, q)
            wps.append(_waypoint(chk, name, note, "arm", q, rep, rep.ok()))
        elif kind == "drive":
            _, name, note, pose = item[:4]
            goal = item[4] if len(item) > 4 else None
            target_yaw = np.deg2rad(pose[2])
            prev = wps[-1]["joints"]
            arms = {n: v for n, v in prev.items() if n not in ik.BASE}
            if goal is not None:
                # arms move during the leg: solve the base-frame goal at the leg's end pose
                bq_end = _turn_choices(prev, pose)[0]
                tg = hand_targets(goal, bq_end)
                fixed = {**fixed_static, **bq_end}
                q_arm = seed
                for _ in range(3):
                    q_arm = ik.solve_multi(chk, tg, q_arm, fixed)
                    if ik.evaluate_multi(chk, {**fixed, **q_arm}, tg).ok(next(iter(tg.values()))):
                        break
                q_end = {**fixed, **q_arm}
                rep_end = ik.evaluate_multi(chk, q_end, tg)
                arm_ok = rep_end.ok(next(iter(tg.values())))
                members, pc = None, None
                for bq_try in (_turn_choices(prev, pose) if arm_ok else []):
                    q_try = {**q_end, **bq_try}
                    members, pc = _combo_chain(chk, wps[-1], q_try, rep_end, name, note, goal.name)
                    if members:
                        q_end = q_try
                        break
                if members:
                    seed = {n: q_end[n] for n in ik.BIMANUAL_JOINTS}
                    bq = {n: q_end[n] for n in ik.BASE}
                    wps.extend(members)
                    all_ok &= all(m_["ok"] for m_ in members)
                    print(f"combo {name:16s} ok=True  {len(members)} arm moves during the leg, "
                          f"env={pc['min_env_clearance_m']:.3f} self={pc['min_self_clearance_m']:.3f} "
                          f"rel={members[-1]['relative_move']}", flush=True)
                    continue
                print(f"combo {name:16s} overlap rejected (arm_ok={arm_ok}, leg={pc}); arms first, then drive",
                      flush=True)
                # fallback: same arm goal at the current base pose (base-frame goal), then the plain leg
                tg = hand_targets(goal, bq)
                fixed = {**fixed_static, **bq}
                q_arm = ik.solve_multi(chk, tg, seed, fixed)
                q_a = {**fixed, **q_arm}
                rep_a = ik.evaluate_multi(chk, q_a, tg)
                wps.append(_waypoint(chk, goal.name, goal.note, "arm", q_a, rep_a, rep_a.ok(next(iter(tg.values())))))
                all_ok &= wps[-1]["ok"]
                seed = {n: q_a[n] for n in ik.BIMANUAL_JOINTS}
                prev = wps[-1]["joints"]
                arms = {n: v for n, v in prev.items() if n not in ik.BASE}
            # choose the shorter / clearer rotation direction
            best = None
            for k in (-1, 0, 1):
                yaw = target_yaw + 2 * np.pi * k
                if abs(yaw - prev["base_yaw"]) > np.pi + 1e-6:   # only the short way (both ways for 180 deg)
                    continue
                q = {**arms, **base_q(pose, yaw)}
                pc = path_check(chk, prev, q, "drive")
                if best is None or pc["min_env_clearance_m"] > best[1]["min_env_clearance_m"]:
                    best = (q, pc)
            q, pc = best
            bq = {n: q[n] for n in ik.BASE}
            rep = ik.evaluate_multi(chk, q)
            ok = segment_ok(pc)
            wps.append(_waypoint(chk, name, note, "drive", q, rep, ok,
                                 dict(relative_move=relative_move(prev, q), leg_check=pc)))
        else:
            if isinstance(item[1], str):     # copy of an earlier pose
                _, name, note, src = item
                q = dict(next(w for w in wps if w["name"] == src)["joints"])
                rep, ok = ik.evaluate_multi(chk, q), True
            else:
                goal, pose = item[1], item[2]
                if pose is not None:
                    bq = base_q(pose)
                tg = hand_targets(goal, bq)
                fixed = {**fixed_static, **bq}
                q_arm = seed
                for _ in range(3):
                    q_arm = ik.solve_multi(chk, tg, q_arm, fixed)
                    if ik.evaluate_multi(chk, {**fixed, **q_arm}, tg).ok(next(iter(tg.values()))):
                        break
                q = {**fixed, **q_arm}
                rep = ik.evaluate_multi(chk, q, tg)
                ok = rep.ok(next(iter(tg.values())))
                name, note = goal.name, goal.note
            seed = {n: q[n] for n in ik.BIMANUAL_JOINTS}
            wps.append(_waypoint(chk, name, note, "arm", q, rep, ok))
        all_ok &= wps[-1]["ok"]
        w, c = wps[-1], wps[-1]["check"]
        print(f"{w['kind']:5s} {w['name']:16s} ok={w['ok']} pos={c['pos_err']:.3f} app={c['approach_err_deg']:4.1f} "
              f"env={c['env_clear']:.3f} self={c['self_clear']:.3f} tilt={c['tilt_deg']:.1f} shift={c['shift']:.3f}"
              + (f"  leg_env={w['leg_check']['min_env_clearance_m']:.3f} rel={w['relative_move']}" if kind == "drive" else ""),
              flush=True)

    # arm legs: refine with via-points and check
    final = [wps[0]]
    for a, b in zip(wps[:-1], wps[1:]):
        if b["kind"] == "arm":
            final.extend(_via(chk, a, b, 0))
        final.append(b)
    for w in final:
        if not w["ok"]:
            all_ok = False
            print(f"  FAIL waypoint {w['name']}: {w['check']}", flush=True)
    station_q = {k: base_q(v) for k, v in STATIONS.items()}

    def at_station(q: dict[str, float]) -> str | None:
        for k, sq in station_q.items():
            if (abs(q["base_x"] - sq["base_x"]) < 1e-4 and abs(q["base_y"] - sq["base_y"]) < 1e-4
                    and abs(np.angle(np.exp(1j * (q["base_yaw"] - sq["base_yaw"])))) < 1e-4):
                return k
        return None

    segments = []
    for a, b in zip(final[:-1], final[1:]):
        pc = b["leg_check"] if b["kind"] == "drive" else path_check(chk, a["joints"], b["joints"], "arm")
        ok = segment_ok(pc)
        if b["kind"] == "arm" and at_station(a["joints"]):
            pb = pose_budget_check(chk, a["joints"], b["joints"])
            pc = {**pc, "pose_budget": pb}
            pc["pose_budget"]["meets_5cm"] = pb["min_env_clearance_m"] >= ik.ENV_CLEARANCE  # reported, not gating
        all_ok &= ok
        segments.append(dict(frm=a["name"], to=b["name"], kind=b["kind"], ok=ok, **pc))
        if not ok:
            print(f"  FAIL {a['name']} -> {b['name']} ({b['kind']}): {pc}", flush=True)

    off_branch = [w["name"] for w in final if not ik.on_robot_branch(w["joints"])]
    if off_branch:
        all_ok = False
        print(f"  FAIL shoulder branch differs from the robot's (j2 >= 0) at {off_branch}", flush=True)

    OUT.mkdir(exist_ok=True)
    doc = dict(
        task="bimanual cup-carry-pour with driving chassis", robot=f"AgiBot G2, {g2_model.ARM} arms, Omnipicker x2",
        expected_arm=g2_model.ARM, expected_end_effector="omnipicker",
        robot_reset_pose=ik.RESET_POSE,
        model_source=dict(repo="github.com/AgibotTech/genie_sim_robot_model", commit=g2_model.OFFICIAL_COMMIT,
                          xacro=f"robots/genie/g2/g2_{g2_model.ARM}_omnipicker.urdf.xacro", collision_type="fine"),
        units=dict(joints="rad", base_x_y="m (world, room origin)", base_yaw="rad",
                   gripper="omnipicker -0.785 open .. 0 closed"),
        layout=dict(measured=os.environ.get("LAYOUT_MEASURED", "0") == "1",
                    stations_in_deg=dict(start=START, bin=BIN_STATION, bin_backoff=BIN_BACKOFF,
                                         turn_point=TURN_POINT, hopper_standoff=HOP_STANDOFF, hopper_dock=HOP_DOCK,
                                         home_return=HOME_TURN),
                    overhead_frame_modelled=g2_model.FRAME_SLAB_IN > 0,
                    assumed=dict(bin_height_in=g2_model.SOURCE_BIN["h"], frame_top_in=g2_model.FRAME_TOP_IN,
                                 frame_beam_depth_in=g2_model.FRAME_SLAB_IN, frame_posts="not modelled")),
        clearance_required_m=dict(env=ik.ENV_CLEARANCE, self=ik.SELF_CLEARANCE, hands=ik.HAND_CLEARANCE),
        chassis_pose_budget=dict(m=POSE_BUDGET_M, deg=POSE_BUDGET_DEG,
                                 note="station arm legs keep >= env clearance with the chassis this far off; "
                                      "the runner must verify the stop pose to within 2 cm / 2 deg"),
        hand_posture=dict(gap_between_tcps_m=HAND_GAP, roll="both grippers same orientation, finger plane level "
                          "(camera side up), pitched with the approach", payload="none (demo only)"),
        all_ok=bool(all_ok), waypoints=final, segments=segments)
    out = OUT / os.environ.get("PLAN_OUT", "bimanual_plan.json")
    out.write_text(json.dumps(doc, indent=2))
    print("ALL_OK", all_ok, "->", out.name, f"({len(final)} waypoints)")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
