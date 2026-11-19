"""Open the interactive MuJoCo viewer and play the bimanual cup-carry-pour POC in real time (loops).

macOS needs the mjpython launcher:
    .venv/bin/mjpython view_demo.py

Mouse: left-drag rotate, right-drag pan, scroll zoom, double-click to select a body.
The nearest robot<->workcell distance is drawn live (green/yellow/red) with the TCP trail.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("G2_ARM", "crs")
os.environ.setdefault("FRAME_SLAB_IN", "0")

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402
import numpy as np  # noqa: E402

import ik  # noqa: E402
import record_demo as rd  # noqa: E402
import replay  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
TRAIL = 300


def main() -> int:
    wps = json.loads((OUT / os.environ.get("PLAN_FILE", "bimanual_plan.json")).read_text())["waypoints"]
    model, joints = rd.build_visual_model()
    data = mujoco.MjData(model)
    chk = ik.Checker(model)
    env_pairs = chk.env_pairs()
    qadr = [model.jnt_qposadr[model.joint(n).id] for n in joints]
    keys = replay.schedule(wps, joints)
    t_cycle = keys[-1][0] + 2.0
    tcp_site, tcp_site_l = model.site("tcp_r").id, model.site("tcp_l").id

    def reset() -> None:
        mujoco.mj_resetData(model, data)
        data.qpos[qadr] = [wps[0]["joints"][n] for n in joints]
        data.ctrl[:] = data.qpos[qadr]
        mujoco.mj_forward(model, data)

    reset()
    trail: list[np.ndarray] = []
    trail_l: list[np.ndarray] = []
    with mujoco.viewer.launch_passive(model, data) as v:
        v.cam.lookat[:], v.cam.distance, v.cam.azimuth, v.cam.elevation = rd.CAM["overview"]
        v.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
        v.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        v.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        wall0 = time.time()
        last_step = ""
        while v.is_running():
            t = (time.time() - wall0) % t_cycle
            if t < data.time - 1e-6:  # cycle wrapped: restart from HOME
                reset()
                trail.clear()
                trail_l.clear()
            q_ref, seg = replay.target_at(keys, data.time)
            while data.time < t:
                data.ctrl[:] = q_ref
                mujoco.mj_step(model, data)
                q_ref, seg = replay.target_at(keys, data.time)
            if seg != last_step:
                num, zh, en = rd.STEP_TEXT.get(seg.split("~")[0], ("", seg, seg))
                print(f"[{data.time:5.1f}s] stage {num}/{rd.STAGES}  {zh}  ({en})", flush=True)
                last_step = seg
            chk.data.qpos[:] = data.qpos
            mujoco.mj_kinematics(model, chk.data)
            d, (a, b) = chk.min_distance(env_pairs, 0.5)
            trail.append(data.site_xpos[tcp_site].copy())
            trail_l.append(data.site_xpos[tcp_site_l].copy())
            del trail[:-TRAIL]
            del trail_l[:-TRAIL]
            with v.lock():
                scn = v.user_scn
                scn.ngeom = 0
                for i in range(1, len(trail), 2):
                    rd.add_line(scn, trail[i - 1], trail[i], 0.004, (1.0, 0.86, 0.24, 0.9))
                    rd.add_line(scn, trail_l[i - 1], trail_l[i], 0.004, (0.35, 0.90, 1.0, 0.9))
                if a >= 0:
                    ft = np.zeros(6)
                    mujoco.mj_geomDistance(model, chk.data, a, b, 0.5, ft)
                    col = rd.margin_colour(d)
                    rd.add_line(scn, ft[:3], ft[3:], 0.006, col)
                    rd.add_sphere(scn, ft[:3], 0.012, col)
                    rd.add_sphere(scn, ft[3:], 0.012, col)
            v.sync()
            time.sleep(0.01)
    return 0


if __name__ == "__main__":
    sys.exit(main())
