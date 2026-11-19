"""Sweep chassis placement: can one stationary pose reach both the bin pick and the hopper dump?"""
from __future__ import annotations

import itertools
import os
import sys

import numpy as np


def run(x: float, y: float, dump_r_in: float, tilt_deg: float) -> tuple:
    os.environ["G2_BASE_IN"] = f"{x},{y}"
    import importlib

    import g2_model
    import ik
    import plan_poc
    importlib.reload(g2_model)
    importlib.reload(plan_poc)
    chk = ik.Checker(g2_model.build_model())
    left = plan_poc.tuck_left_arm(chk)
    fixed = {**plan_poc.HEAD, **left, ik.GRIP_L: 0.0, ik.GRIP_R: ik.GRIP_OPEN}
    seed = {n: 0.0 for n in ik.IK_JOINTS}
    seed.update({"idx62_arm_r_joint2": 1.0, "idx64_arm_r_joint4": -1.5})
    pick = ik.Target((*plan_poc.PICK_XY, plan_poc.BIN_RIM_Z - 0.06), plan_poc.DOWN, 0.01, 20)
    q = ik.solve(chk, ik.Target((*plan_poc.PICK_XY, plan_poc.BIN_RIM_Z + 0.15), plan_poc.DOWN), seed, fixed)
    q = ik.solve(chk, pick, q, fixed)
    rp = ik.evaluate(chk, {**fixed, **q}, pick)
    out = plan_poc._out
    t = np.deg2rad(tilt_deg)
    app = tuple(np.cos(t) * np.array(plan_poc.DOWN) + np.sin(t) * np.append(-out, 0.0))
    dump = ik.Target((*(plan_poc.HOP_C + out * dump_r_in * g2_model.IN), 1.40), app, 0.01, 15)
    qd = seed
    for _ in range(2):
        qd = ik.solve(chk, dump, qd, fixed)
    rd = ik.evaluate(chk, {**fixed, **qd}, dump)
    return rp, rd, pick, dump


def main() -> int:
    rows = []
    for x, y, r, tilt in itertools.product([93, 95.5, 98], [89, 93, 97], [16, 17], [60, 75]):
        rp, rd, tp, td = run(x, y, r, tilt)
        ok = rp.ok(tp) and rd.ok(td)
        rows.append((ok, x, y, r, tilt, rp.pos_err, rd.pos_err, rd.approach_err_deg, rd.env_clear))
        print(f"x={x:5.1f} y={y:4.1f} r={r} tilt={tilt} ok={ok} pick_err={rp.pos_err:.3f} "
              f"(env {rp.env_clear:.3f}) dump_err={rd.pos_err:.3f} app={rd.approach_err_deg:.1f} "
              f"env={rd.env_clear:.3f} shift={rd.shift:.3f}", flush=True)
    good = [r for r in rows if r[0]]
    print("GOOD", good)
    return 0


if __name__ == "__main__":
    sys.exit(main())
