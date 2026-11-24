"""Constrained IK for the G2 right arm + waist, using the MuJoCo model itself.

Constraints (soft, via least squares residuals; checked hard afterwards):
- TCP position and approach-axis direction
- AgiBot coupled-joint limits (A q <= b - margin) from genie_sim_moveit_plugins
- torso upright and waist shift (keeps the centre of mass near the chassis)
- clearance to the workcell and to the robot's own body
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
import yaml
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parent

BODY = [f"idx0{i}_body_joint{i}" for i in range(1, 6)]
ARM_R = [f"idx6{i}_arm_r_joint{i}" for i in range(1, 8)]
ARM_L = [f"idx2{i}_arm_l_joint{i}" for i in range(1, 8)]
HEAD = ["idx11_head_joint1", "idx12_head_joint2", "idx13_head_joint3"]
GRIP_R = "idx71_gripper_r_inner_joint1"
GRIP_L = "idx31_gripper_l_inner_joint1"
GRIP_OPEN, GRIP_CLOSED = -0.785, 0.0
IK_JOINTS = BODY + ARM_R
BIMANUAL_JOINTS = BODY + ARM_L + ARM_R
BASE = ["base_x", "base_y", "base_yaw"]

TCP_SITE = "tcp_r"
COUPLED_MARGIN = 0.02      # rad, same idea as the MoveIt plugins' margin
ENV_CLEARANCE = 0.05       # m, required distance robot <-> workcell
SELF_CLEARANCE = 0.02      # m, required distance between non-neighbouring robot parts
HAND_CLEARANCE = 0.005     # m, the two cupped grippers sit ~1 cm apart but must never touch
MAX_WAIST_SHIFT = 0.13     # m, horizontal shift of body_link5 from its upright position
MAX_TORSO_TILT = np.deg2rad(10.0)
LIMIT_MARGIN = 0.03        # rad, stay off the hard joint limits (GDK soft limits sit there)


# Shoulder branch. The shoulder axes j1..j3 intersect, so (j1 - pi, -j2, j3 + pi) puts the hand in the
# same pose; the IK must pick the branch the real robot uses. On the robot's reset pose (read with
# GDK, 2026-11-23) both arms have j2 = -1.571: L = (1.571, -1.571, -1.571, -1.571, 0, 0, 0),
# R = (-1.571, -1.571, 1.571, -1.571, 0, 0, 0). Keeping j2 < 0 keeps the plan on that branch; on the
# other one the right arm sat ~3 rad away from the robot's pose in every joint j1..j3.
RESET_POSE = {
    **dict(zip(ARM_L, (1.571, -1.571, -1.571, -1.571, 0.0, 0.0, 0.0))),
    **dict(zip(ARM_R, (-1.571, -1.571, 1.571, -1.571, 0.0, 0.0, 0.0))),
}
SHOULDER_J2 = (ARM_L[1], ARM_R[1])
SHOULDER_J2_MAX = -0.05    # rad, also keeps the IK off the j2 = 0 shoulder singularity


def bounds(model: mujoco.MjModel, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    lo = np.array([model.jnt_range[model.joint(n).id][0] for n in names]) + LIMIT_MARGIN
    hi = np.array([model.jnt_range[model.joint(n).id][1] for n in names]) - LIMIT_MARGIN
    hi = np.array([min(h, SHOULDER_J2_MAX) if n in SHOULDER_J2 else h for n, h in zip(names, hi)])
    return lo, hi


def on_robot_branch(q: dict[str, float]) -> bool:
    return all(q[n] < 0.0 for n in SHOULDER_J2 if n in q)


def _side(body_name: str) -> str:
    if body_name.startswith(("arm_r", "gripper_r")):
        return "R"
    if body_name.startswith(("arm_l", "gripper_l")):
        return "L"
    return "B"  # body / head / chassis


@dataclass(frozen=True)
class Target:
    pos: tuple[float, float, float]          # world, metres
    approach: tuple[float, float, float]     # desired TCP local-z direction (world)
    pos_tol: float = 0.01
    approach_tol_deg: float = 5.0
    up: tuple[float, float, float] | None = None   # desired gripper local x (camera side), defines roll


@dataclass
class Checker:
    model: mujoco.MjModel
    data: mujoco.MjData = field(init=False)
    qadr: dict[str, int] = field(init=False)
    coupled: list[tuple[list[int], np.ndarray, np.ndarray]] = field(init=False)
    env_geoms: list[int] = field(init=False)
    robot_geoms: list[int] = field(init=False)
    self_pairs: list[tuple[int, int]] = field(init=False)
    hand_pairs: list[tuple[int, int]] = field(init=False)
    upright_waist_b: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        m = self.model
        self.data = mujoco.MjData(m)
        self.qadr = {m.joint(j).name: m.jnt_qposadr[j] for j in range(m.njnt)}
        cfg = yaml.safe_load((ROOT / "coupled_constraints.yaml").read_text())
        self.coupled = [
            ([self.qadr[n] for n in c["joint_names"]], np.array(c["A"]), np.array(c["b"]))
            for c in cfg["coupled_constraints"]
        ]
        self.env_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == 0 and m.geom(g).name != "floor"]
        self.robot_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] != 0 and m.geom_contype[g] != 0]
        pairs = self._self_pairs()
        name = lambda g: m.body(m.geom_bodyid[g]).name  # noqa: E731
        is_hand_pair = lambda a, b: {name(a)[:9], name(b)[:9]} == {"gripper_l", "gripper_r"}  # noqa: E731
        self.hand_pairs = [(a, b) for a, b in pairs if is_hand_pair(a, b)]
        self.self_pairs = [(a, b) for a, b in pairs if not is_hand_pair(a, b)]
        mujoco.mj_forward(m, self.data)
        self.upright_waist_b = self._in_base(self.data.xpos[m.body("body_link5").id])[:2].copy()

    def _in_base(self, p: np.ndarray) -> np.ndarray:
        b = self.model.body("base_link").id
        R = self.data.xmat[b].reshape(3, 3)
        return R.T @ (p - self.data.xpos[b])

    # -- geometry helpers ---------------------------------------------------
    def _chain(self, body: int) -> set[int]:
        out = set()
        while body != 0:
            out.add(body)
            body = self.model.body_parentid[body]
        return out

    def _self_pairs(self) -> list[tuple[int, int]]:
        """Both arms/grippers vs. torso/head/chassis, and right arm vs. left arm.

        Skipped: geoms on the same arm (official hulls of neighbouring links overlap by
        design) and each arm's first link vs. the torso link it is mounted on.
        """
        m = self.model
        name = lambda g: m.body(m.geom_bodyid[g]).name  # noqa: E731
        pairs = []
        for i, a in enumerate(self.robot_geoms):
            for b in self.robot_geoms[i + 1:]:
                na, nb = name(a), name(b)
                sa, sb = _side(na), _side(nb)
                if sa == sb:
                    continue
                arm, other = (na, nb) if sa != "B" else (nb, na)
                if arm.endswith(("_link1", "_link2")) and other in ("arm_base_link", "body_link5"):
                    continue  # shoulder mount: sits 2 cm from the torso shell by design
                ba, bb = m.geom_bodyid[a], m.geom_bodyid[b]
                if m.body_parentid[ba] == bb or m.body_parentid[bb] == ba:
                    continue
                pairs.append((a, b))
        return pairs

    def set_q(self, q: dict[str, float]) -> None:
        for n, v in q.items():
            self.data.qpos[self.qadr[n]] = v
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        # keep mimic gripper links consistent for collision checks
        self._apply_mimic()

    def _apply_mimic(self) -> None:
        m, d = self.model, self.data
        for e in range(m.neq):
            if m.eq_type[e] != mujoco.mjtEq.mjEQ_JOINT:
                continue
            j1, j2 = m.eq_obj1id[e], m.eq_obj2id[e]
            c = m.eq_data[e]
            x = d.qpos[m.jnt_qposadr[j2]]
            d.qpos[m.jnt_qposadr[j1]] = c[0] + c[1] * x + c[2] * x**2
        mujoco.mj_kinematics(m, d)

    def min_distance(self, pairs: list[tuple[int, int]], cutoff: float) -> tuple[float, tuple[int, int]]:
        best, arg = cutoff, (-1, -1)
        fromto = np.zeros(6)
        if not pairs:
            return best, arg
        pa = np.asarray(pairs)
        # broad phase: skip pairs whose bounding spheres are already farther apart than the cutoff
        gap = (np.linalg.norm(self.data.geom_xpos[pa[:, 0]] - self.data.geom_xpos[pa[:, 1]], axis=1)
               - self.model.geom_rbound[pa[:, 0]] - self.model.geom_rbound[pa[:, 1]])
        for a, b in pa[gap < cutoff]:
            dist = mujoco.mj_geomDistance(self.model, self.data, a, b, cutoff, fromto)
            if dist < best:
                best, arg = dist, (a, b)
        return best, arg

    def env_pairs(self) -> list[tuple[int, int]]:
        m = self.model
        # every robot collision geom (arms, grippers, torso, legs, chassis) vs the workcell
        return [(a, b) for a in self.robot_geoms for b in self.env_geoms]

    def tcp(self, site: str = TCP_SITE) -> tuple[np.ndarray, np.ndarray]:
        s = self.model.site(site).id
        return self.data.site_xpos[s].copy(), self.data.site_xmat[s].reshape(3, 3)[:, 2].copy()

    def tcp_up(self, site: str) -> np.ndarray:
        """Gripper local x axis (camera side) in world frame."""
        return self.data.site_xmat[self.model.site(site).id].reshape(3, 3)[:, 0].copy()

    def coupled_violation(self) -> float:
        worst = -np.inf
        for idx, A, b in self.coupled:
            q = self.data.qpos[idx]
            worst = max(worst, float(np.max(A @ q - (b - COUPLED_MARGIN))))
        return worst

    def torso_state(self) -> tuple[float, float]:
        b = self.model.body("body_link5").id
        z = self.data.xmat[b].reshape(3, 3)[:, 2]
        tilt = float(np.arccos(np.clip(z[2], -1, 1)))
        shift = float(np.linalg.norm(self._in_base(self.data.xpos[b])[:2] - self.upright_waist_b))
        return tilt, shift


def solve(chk: Checker, target: Target, seed: dict[str, float],
          fixed: dict[str, float], reg: float = 0.02) -> dict[str, float]:
    """Solve IK for IK_JOINTS; `fixed` joints (head, left arm, grippers) are held."""
    lo, hi = bounds(chk.model, IK_JOINTS)
    x0 = np.clip(np.array([seed[n] for n in IK_JOINTS]), lo + 1e-4, hi - 1e-4)
    env_pairs = chk.env_pairs()
    tpos, tapp = np.array(target.pos), np.array(target.approach) / np.linalg.norm(target.approach)

    def residual(x: np.ndarray) -> np.ndarray:
        chk.set_q({**fixed, **dict(zip(IK_JOINTS, x))})
        p, a = chk.tcp()
        res = [10.0 * (p - tpos), 2.0 * (a - tapp)]
        cv = []
        for idx, A, b in chk.coupled:
            cv.extend(np.maximum(0.0, A @ chk.data.qpos[idx] - (b - COUPLED_MARGIN)))
        res.append(20.0 * np.array(cv))
        tilt, shift = chk.torso_state()
        res.append(20.0 * np.array([max(0.0, tilt - MAX_TORSO_TILT), max(0.0, shift - MAX_WAIST_SHIFT)]))
        d_env, _ = chk.min_distance(env_pairs, ENV_CLEARANCE + 0.05)
        d_self, _ = chk.min_distance(chk.self_pairs, SELF_CLEARANCE + 0.05)
        res.append(30.0 * np.array([max(0.0, ENV_CLEARANCE + 0.01 - d_env),
                                    max(0.0, SELF_CLEARANCE + 0.01 - d_self)]))
        res.append(reg * (x - x0))
        return np.concatenate(res)

    sol = least_squares(residual, x0, bounds=(lo, hi), diff_step=1e-3, max_nfev=400)
    return dict(zip(IK_JOINTS, sol.x))


def repair(chk: Checker, q_mid: dict[str, float], fixed: dict[str, float]) -> dict[str, float]:
    """Nudge a via-point so it meets clearance/coupled/torso limits, staying near q_mid."""
    lo, hi = bounds(chk.model, IK_JOINTS)
    x0 = np.clip(np.array([q_mid[n] for n in IK_JOINTS]), lo + 1e-4, hi - 1e-4)
    env_pairs = chk.env_pairs()
    margin = 0.03  # aim above the requirement so the interpolated path keeps clearance

    def residual(x: np.ndarray) -> np.ndarray:
        chk.set_q({**fixed, **dict(zip(IK_JOINTS, x))})
        cv = []
        for idx, A, b in chk.coupled:
            cv.extend(np.maximum(0.0, A @ chk.data.qpos[idx] - (b - COUPLED_MARGIN)))
        tilt, shift = chk.torso_state()
        d_env, _ = chk.min_distance(env_pairs, ENV_CLEARANCE + margin + 0.05)
        d_self, _ = chk.min_distance(chk.self_pairs, SELF_CLEARANCE + margin + 0.05)
        return np.concatenate([
            20.0 * np.array(cv),
            20.0 * np.array([max(0.0, tilt - MAX_TORSO_TILT), max(0.0, shift - MAX_WAIST_SHIFT)]),
            30.0 * np.array([max(0.0, ENV_CLEARANCE + margin - d_env),
                             max(0.0, SELF_CLEARANCE + margin - d_self)]),
            0.05 * (x - x0),
        ])

    sol = least_squares(residual, x0, bounds=(lo, hi), diff_step=1e-3, max_nfev=300)
    return dict(zip(IK_JOINTS, sol.x))


@dataclass(frozen=True)
class Report:
    pos_err: float
    approach_err_deg: float
    coupled: float
    tilt_deg: float
    shift: float
    env_clear: float
    self_clear: float
    hand_clear: float = 1.0

    def ok(self, target: Target | None = None) -> bool:
        reach = target is None or (self.pos_err <= target.pos_tol and self.approach_err_deg <= target.approach_tol_deg)
        # torso limits are this project's own stability margins (not AgiBot limits); soft-solved, 1 cm / 0.5 deg slack
        return (reach and self.coupled <= 1e-3 and self.tilt_deg <= np.rad2deg(MAX_TORSO_TILT) + 0.5
                and self.shift <= MAX_WAIST_SHIFT + 0.01 and self.env_clear >= ENV_CLEARANCE
                and self.self_clear >= SELF_CLEARANCE and self.hand_clear >= HAND_CLEARANCE)


def evaluate(chk: Checker, q: dict[str, float], target: Target | None = None) -> Report:
    chk.set_q(q)
    p, a = chk.tcp()
    tilt, shift = chk.torso_state()
    d_env, _ = chk.min_distance(chk.env_pairs(), 1.0)
    d_self, _ = chk.min_distance(chk.self_pairs, 1.0)
    pos_err = float(np.linalg.norm(p - np.array(target.pos))) if target else 0.0
    app_err = 0.0
    if target:
        ta = np.array(target.approach) / np.linalg.norm(target.approach)
        app_err = float(np.rad2deg(np.arccos(np.clip(a @ ta, -1, 1))))
    return Report(pos_err, app_err, max(0.0, chk.coupled_violation()), float(np.rad2deg(tilt)),
                  shift, d_env, d_self)


def _residual_common(chk: Checker, env_pairs, margin: float) -> list[np.ndarray]:
    cv = []
    for idx, A, b in chk.coupled:
        cv.extend(np.maximum(0.0, A @ chk.data.qpos[idx] - (b - COUPLED_MARGIN)))
    tilt, shift = chk.torso_state()
    d_env, _ = chk.min_distance(env_pairs, ENV_CLEARANCE + margin + 0.05)
    d_self, _ = chk.min_distance(chk.self_pairs, SELF_CLEARANCE + margin + 0.05)
    d_hand, _ = chk.min_distance(chk.hand_pairs, HAND_CLEARANCE + 0.05)
    return [20.0 * np.array(cv),
            20.0 * np.array([max(0.0, tilt - (MAX_TORSO_TILT - np.deg2rad(0.3))),
                             max(0.0, shift - (MAX_WAIST_SHIFT - 0.003))]),
            30.0 * np.array([max(0.0, ENV_CLEARANCE + margin - d_env),
                             max(0.0, SELF_CLEARANCE + margin - d_self),
                             max(0.0, HAND_CLEARANCE + 0.002 - d_hand)])]


def solve_multi(chk: Checker, targets: dict[str, Target], seed: dict[str, float],
                fixed: dict[str, float], joints: list[str] = BIMANUAL_JOINTS,
                reg: float = 0.02, max_nfev: int = 500) -> dict[str, float]:
    """IK for several TCP sites at once (e.g. both hands), with all clearance constraints."""
    lo, hi = bounds(chk.model, joints)
    x0 = np.clip(np.array([seed[n] for n in joints]), lo + 1e-4, hi - 1e-4)
    env_pairs = chk.env_pairs()
    tgts = [(site, np.array(t.pos), np.array(t.approach) / np.linalg.norm(t.approach),
             None if t.up is None else np.array(t.up) / np.linalg.norm(t.up)) for site, t in targets.items()]

    def residual(x: np.ndarray) -> np.ndarray:
        chk.set_q({**fixed, **dict(zip(joints, x))})
        res = []
        for site, tp, ta, tu in tgts:
            p, a = chk.tcp(site)
            res += [10.0 * (p - tp), 2.0 * (a - ta)]
            if tu is not None:
                res.append(2.0 * (chk.tcp_up(site) - tu))
        res += _residual_common(chk, env_pairs, 0.01)
        res.append(reg * (x - x0))
        return np.concatenate(res)

    sol = least_squares(residual, x0, bounds=(lo, hi), diff_step=1e-3, max_nfev=max_nfev)
    return dict(zip(joints, sol.x))


def repair_multi(chk: Checker, q_mid: dict[str, float], fixed: dict[str, float],
                 joints: list[str] = BIMANUAL_JOINTS) -> dict[str, float]:
    """Nudge a via-point to meet clearance/coupled/torso limits, staying near q_mid."""
    lo, hi = bounds(chk.model, joints)
    x0 = np.clip(np.array([q_mid[n] for n in joints]), lo + 1e-4, hi - 1e-4)
    env_pairs = chk.env_pairs()

    def residual(x: np.ndarray) -> np.ndarray:
        chk.set_q({**fixed, **dict(zip(joints, x))})
        return np.concatenate(_residual_common(chk, env_pairs, 0.03) + [0.05 * (x - x0)])

    sol = least_squares(residual, x0, bounds=(lo, hi), diff_step=1e-3, max_nfev=300)
    return dict(zip(joints, sol.x))


def evaluate_multi(chk: Checker, q: dict[str, float], targets: dict[str, Target] | None = None) -> Report:
    """Worst-case report over all TCP targets (or just constraints when targets is None)."""
    chk.set_q(q)
    tilt, shift = chk.torso_state()
    d_env, _ = chk.min_distance(chk.env_pairs(), 1.0)
    d_self, _ = chk.min_distance(chk.self_pairs, 1.0)
    d_hand, _ = chk.min_distance(chk.hand_pairs, 1.0)
    pos_err = app_err = 0.0
    for site, t in (targets or {}).items():
        p, a = chk.tcp(site)
        ta = np.array(t.approach) / np.linalg.norm(t.approach)
        pos_err = max(pos_err, float(np.linalg.norm(p - np.array(t.pos))))
        app_err = max(app_err, float(np.rad2deg(np.arccos(np.clip(a @ ta, -1, 1)))))
        if t.up is not None:
            tu = np.array(t.up) / np.linalg.norm(t.up)
            app_err = max(app_err, float(np.rad2deg(np.arccos(np.clip(chk.tcp_up(site) @ tu, -1, 1)))))
    return Report(pos_err, app_err, max(0.0, chk.coupled_violation()), float(np.rad2deg(tilt)),
                  shift, d_env, d_self, d_hand)
