"""Physics replay of waypoints.json in MuJoCo.

Mimics how GDK will execute it: joint-space moves between waypoints, every joint
synchronised, peak joint speed capped (default 0.3 rad/s, the GDK examples' value),
gripper moved between waypoints. Logs robot<->workcell contacts, self contacts,
tracking error, and records a video.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

import g2_model
import ik
import render

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
MAX_JOINT_SPEED = float(os.environ.get("MAX_JOINT_SPEED", "0.3"))   # rad/s
BASE_LIN_SPEED = float(os.environ.get("BASE_LIN_SPEED", "0.25"))    # m/s, chassis translation
BASE_YAW_SPEED = float(os.environ.get("BASE_YAW_SPEED", "0.4"))     # rad/s, chassis rotation
GRIPPER_TIME = 1.0      # s per open/close
SETTLE_TIME = 0.4       # s hold at each waypoint
KP, KV = 30000.0, 2500.0  # waist/leg servo gains (stiff high-ratio actuators)
VIDEO_FPS = 30


def add_servos(spec: mujoco.MjSpec, joints: list[str]) -> None:
    for name in joints:
        a = spec.add_actuator()
        a.name = f"pos_{name}"
        a.target = name
        a.trntype = mujoco.mjtTrn.mjTRN_JOINT
        a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        if name in ("base_x", "base_y", "base_yaw"):
            kp, kv = 40000.0, 4000.0       # chassis servo (driven as a whole)
        elif "gripper" in name:
            kp, kv = 20.0, 1.0
        elif "body" in name:
            kp, kv = KP, KV
        elif name[-1] in "567" or "head" in name:   # wrist / head joints
            kp, kv = 1000.0, 60.0
        else:                                         # shoulder / elbow
            kp, kv = 4000.0, 300.0
        a.gainprm[0] = kp
        a.biasprm[:3] = [0.0, -kp, -kv]


def build() -> tuple[mujoco.MjModel, list[str]]:
    model = g2_model.build_model()
    spec = mujoco.MjSpec.from_string(Path(g2_model.BUILD / "g2_workcell.xml").read_text())
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    for b in spec.bodies:
        if b.name not in ("world",):
            b.gravcomp = 1.0  # the real controller compensates gravity
    for j in spec.joints:
        if j.name.startswith("idx"):
            j.armature = 0.05 if "gripper" in j.name else 0.2   # geared-motor rotor inertia
            j.damping = [0.05 if "gripper" in j.name else 1.0, 0, 0]
    joints = ik.BASE + ik.BODY + ik.ARM_R + ik.ARM_L + ik.HEAD + [ik.GRIP_R, ik.GRIP_L]
    add_servos(spec, joints)
    return spec.compile(), joints


def schedule(wps: list[dict], joints: list[str]) -> list[tuple[float, np.ndarray, str]]:
    """Piecewise timing: duration of each move = max joint travel / speed (min 1 s)."""
    keys, t = [], 0.0
    q_prev = np.array([wps[0]["joints"][n] for n in joints])
    keys.append((0.0, q_prev, wps[0]["name"]))
    arm_idx = [i for i, n in enumerate(joints) if "gripper" not in n and n not in ik.BASE]
    xy_idx = [joints.index("base_x"), joints.index("base_y")] if "base_x" in joints else []
    yaw_idx = joints.index("base_yaw") if "base_yaw" in joints else None
    for w in wps[1:]:
        q = np.array([w["joints"][n] for n in joints])
        travel = float(np.max(np.abs(q[arm_idx] - q_prev[arm_idx])))
        lin = float(np.linalg.norm(q[xy_idx] - q_prev[xy_idx])) if xy_idx else 0.0
        rot = abs(float(q[yaw_idx] - q_prev[yaw_idx])) if yaw_idx is not None else 0.0
        # 1.5: smoothstep peak-speed factor
        dur = max(1.0, 1.5 * max(travel / MAX_JOINT_SPEED, lin / BASE_LIN_SPEED, rot / BASE_YAW_SPEED))
        if travel < 1e-4 and lin < 1e-4 and rot < 1e-4:
            dur = GRIPPER_TIME
        t += dur
        keys.append((t, q, w["name"]))
        t += SETTLE_TIME
        keys.append((t, q, w["name"]))
        q_prev = q
    return keys


def target_at(keys: list, t: float) -> tuple[np.ndarray, str]:
    for (t0, q0, _), (t1, q1, n1) in zip(keys[:-1], keys[1:]):
        if t0 <= t <= t1:
            s = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            s = s * s * (3 - 2 * s)  # smoothstep: zero velocity at both ends
            return q0 + s * (q1 - q0), n1
    return keys[-1][1], keys[-1][2]


def main() -> int:
    plan = OUT / os.environ.get("PLAN_FILE", "bimanual_plan.json")
    tag = plan.stem.replace("waypoints_", "")
    doc = json.loads(plan.read_text())
    wps = doc["waypoints"]
    model, joints = build()
    chk = ik.Checker(model)            # same geoms; used for distance-based clearance on the live state
    env_pairs, self_pairs = chk.env_pairs(), chk.self_pairs
    min_env, min_env_at, min_self, min_self_at = 1.0, "", 1.0, ""
    fromto = np.zeros(6)
    data = mujoco.MjData(model)
    qadr = [model.jnt_qposadr[model.joint(n).id] for n in joints]
    data.qpos[qadr] = [wps[0]["joints"][n] for n in joints]
    data.ctrl[:] = data.qpos[qadr]
    mujoco.mj_forward(model, data)

    keys = schedule(wps, joints)
    t_end = keys[-1][0] + 1.0
    env_geoms = {g for g in range(model.ngeom) if model.geom_bodyid[g] == 0}
    robot_col = {g for g in range(model.ngeom) if model.geom_bodyid[g] != 0 and model.geom_contype[g] != 0}
    arm_idx = [i for i, n in enumerate(joints) if "gripper" not in n and n not in ik.BASE]
    base_idx = [i for i, n in enumerate(joints) if n in ik.BASE]
    max_base_err = 0.0

    contacts_env, contacts_self = {}, {}
    max_err, max_err_at = 0.0, ""
    frames, next_frame = [], 0.0
    renderer = mujoco.Renderer(model, height=540, width=960)
    cam = render.make_camera("overview")
    cam.lookat[:] = (2.4, 2.2, 0.8)
    cam.distance, cam.azimuth, cam.elevation = 5.2, -70.0, -35.0
    while data.time < t_end:
        q_ref, seg = target_at(keys, data.time)
        data.ctrl[:] = q_ref
        mujoco.mj_step(model, data)
        err = float(np.max(np.abs(data.qpos[qadr][arm_idx] - q_ref[arm_idx])))
        max_base_err = max(max_base_err, float(np.max(np.abs(data.qpos[qadr][base_idx] - q_ref[base_idx]))))
        if err > max_err:
            max_err, max_err_at = err, f"{seg}@{data.time:.1f}s"
        for c in data.contact[: data.ncon]:
            g1, g2 = int(c.geom1), int(c.geom2)
            if (g1 in env_geoms) != (g2 in env_geoms) and (g1 in robot_col or g2 in robot_col):
                rg = g2 if g1 in env_geoms else g1
                eg = g1 if g1 in env_geoms else g2
                if model.geom(eg).name == "floor":
                    continue
                k = f"{model.body(model.geom_bodyid[rg]).name}<->{model.geom(eg).name}"
                contacts_env.setdefault(k, f"{seg}@{data.time:.1f}s")
        if int(round(data.time / model.opt.timestep)) % 10 == 0:   # every 20 ms of sim time
            chk.data.qpos[:] = data.qpos
            mujoco.mj_kinematics(model, chk.data)
            d_env, (a, b) = chk.min_distance(env_pairs, 0.2)
            if d_env < min_env and a >= 0:
                min_env, min_env_at = d_env, f"{model.body(model.geom_bodyid[a]).name}<->{model.geom(b).name} {seg}@{data.time:.1f}s"
            d_self, (a, b) = chk.min_distance(self_pairs, 0.2)
            if d_self < min_self and a >= 0:
                min_self, min_self_at = d_self, f"{model.body(model.geom_bodyid[a]).name}<->{model.body(model.geom_bodyid[b]).name} {seg}@{data.time:.1f}s"
                if d_self < 0:
                    contacts_self.setdefault(min_self_at, "penetration")
        if data.time >= next_frame:
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render().copy())
            next_frame += 1.0 / VIDEO_FPS
    renderer.close()

    video = OUT / f"poc_replay_{tag}.mp4"
    imageio.mimsave(video, frames, fps=VIDEO_FPS, macro_block_size=8)
    result = dict(duration_s=round(keys[-1][0], 1), max_joint_speed_rad_s=MAX_JOINT_SPEED,
                  max_tracking_error_rad=round(max_err, 4), max_tracking_error_at=max_err_at,
                  max_chassis_tracking_error=round(max_base_err, 4),
                  chassis_speed=dict(linear_m_s=BASE_LIN_SPEED, yaw_rad_s=BASE_YAW_SPEED),
                  plan=plan.name, arm=g2_model.ARM, frame_beam_in=g2_model.FRAME_SLAB_IN,
                  robot_workcell_contacts=contacts_env, robot_self_contacts=contacts_self,
                  min_env_clearance_m=round(min_env, 4), min_env_at=min_env_at,
                  min_self_clearance_m=round(min_self, 4), min_self_at=min_self_at,
                  video=str(video.relative_to(ROOT)))
    (OUT / f"replay_report_{tag}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    ok = (not contacts_env and not contacts_self and max_err < 0.05
          and min_env >= ik.ENV_CLEARANCE - 0.005 and min_self >= ik.SELF_CLEARANCE - 0.005)
    print("REPLAY_OK", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
