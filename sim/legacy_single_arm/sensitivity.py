"""Re-check a stored plan (no re-planning) against different bin heights / frame beam depths."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> int:
    import g2_model
    import ik
    import plan_poc
    doc = json.loads((ROOT / "out" / os.environ["PLAN_FILE"]).read_text())
    chk = ik.Checker(g2_model.build_model())
    wps = doc["waypoints"]
    worst_env, where = 1.0, ""
    for a, b in zip(wps[:-1], wps[1:]):
        pc = plan_poc.path_check(chk, a["joints"], b["joints"])
        if pc["min_env_clearance_m"] < worst_env:
            worst_env, where = pc["min_env_clearance_m"], f"{a['name']}->{b['name']} {pc['closest']}"
    print(f"bin={g2_model.BIN_HEIGHT_IN:.0f}in beam={g2_model.FRAME_SLAB_IN:.0f}in "
          f"min_env={worst_env:.4f} m at {where}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
