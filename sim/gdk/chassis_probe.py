"""One small chassis move to learn GDK's relative_move conventions, and optionally test cancel_task.

Answers, for the plan's chassis legs: which way +y goes (left or right), how far a leg really goes,
whether yaw > 0 turns left (counter-clockwise from above), how get_task_state reports a leg, and
whether cancel_task stops a running leg. The arms are not commanded.

Default is a DRY RUN. With --execute it needs a person at the robot with a hand on the E-STOP,
a clear area of at least 1 m around the robot, and two typed confirmations.

    python3 gdk/chassis_probe.py --dy 0.10                         # dry run
    python3 gdk/chassis_probe.py --dy 0.10 --execute               # 10 cm, then measure
    python3 gdk/chassis_probe.py --dx 0.30 --execute --cancel-after 1.0   # cancel mid-leg

Exactly one of --dx / --dy / --dyaw-deg is given. Limits: 0.30 m, 30 deg. Every abort or Ctrl-C
after relative_move calls Pnc.cancel_task.
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from typing import Any

import run_poc_gdk as rp

LOG = logging.getLogger("g2_chassis_probe")
MAX_TRANSLATION = 0.30   # m
MAX_TURN_DEG = 30.0
POLL = 0.1               # s between get_task_state reads
WATCH_AFTER_CANCEL = 3.0 # s of task states logged after cancel_task
GO_PHRASE = "GO"


def leg(dx: float, dy: float, dyaw_deg: float) -> tuple[float, float, float]:
    given = [v for v in (dx, dy, dyaw_deg) if v]
    if len(given) != 1:
        raise rp.Abort("give exactly one of --dx, --dy, --dyaw-deg (non-zero)")
    if abs(dx) > MAX_TRANSLATION or abs(dy) > MAX_TRANSLATION:
        raise rp.Abort(f"translation is limited to {MAX_TRANSLATION} m for this probe")
    if abs(dyaw_deg) > MAX_TURN_DEG:
        raise rp.Abort(f"rotation is limited to {MAX_TURN_DEG} deg for this probe")
    return dx, dy, math.radians(dyaw_deg)


def expectation(rel: tuple[float, float, float]) -> str:
    dx, dy, dyaw = rel
    if dx:
        return f"{'FORWARD' if dx > 0 else 'BACKWARD'} {abs(dx) * 100:.0f} cm"
    if dy:
        return f"sideways {abs(dy) * 100:.0f} cm, to the robot's {'LEFT' if dy > 0 else 'RIGHT'} if +y = left"
    return (f"turn {math.degrees(abs(dyaw)):.0f} deg {'LEFT (CCW from above)' if dyaw > 0 else 'RIGHT (CW)'} "
            "if +yaw = CCW")


def watch(g2: rp.G2, step: rp.Step, cancel_after: float | None) -> list[tuple[float, Any, int]]:
    """Log (t, id, state) until the leg ends, or cancel it after cancel_after seconds."""
    t0, log, cancelled_at = time.time(), [], None
    last = None
    while True:
        t = time.time() - t0
        tid, state = g2._task()
        if (tid, state) != last:
            LOG.info("  t=%5.2f s  task id=%s state=%s", t, tid, state)
            last = (tid, state)
        log.append((round(t, 2), tid, state))
        if cancelled_at is None and cancel_after is not None and t >= cancel_after:
            LOG.info("  t=%5.2f s  -> cancel_task()", t)
            g2.cancel_chassis()
            cancelled_at = t
        if cancelled_at is not None and t - cancelled_at >= WATCH_AFTER_CANCEL:
            return log
        if cancelled_at is None and state in (rp.TASK_SUCCESS, rp.TASK_FAIL, rp.TASK_CANCELED) and tid == g2.task_id:
            return log
        if t > rp.CHASSIS_TIMEOUT:
            raise rp.Abort(f"{step.name}: no end state within {rp.CHASSIS_TIMEOUT:.0f} s")
        time.sleep(POLL)


def run(g2: rp.G2, rel: tuple[float, float, float], cancel_after: float | None) -> None:
    g2.required_mode = g2.control_mode()   # chassis only: report the mode, do not require one
    g2.check_ok()
    tid, state = g2._task()
    LOG.info("before: task id=%s state=%s (an earlier task may still show here)", tid, state)
    rp._require(input(f"Hand on the E-STOP, >= 1 m clear around the robot. Type {rp.ESTOP_PHRASE}: "),
                rp.ESTOP_PHRASE)
    rp._require(input(f"Mark the floor at the chassis now. Expected: {expectation(rel)}. Type {GO_PHRASE}: "),
                GO_PHRASE)
    step = rp.Step("chassis", "PROBE", "chassis convention probe", rel=rel)
    try:
        g2.start_relative_move(step)
        log = watch(g2, step, cancel_after)
    except BaseException:
        g2.cancel_chassis()
        raise
    LOG.info("leg task id=%s, final state=%s, %d state reads", g2.task_id, log[-1][2], len(log))
    LOG.info("Now measure and report: direction moved, distance (cm) or angle (deg) from the floor mark%s.",
             ", and whether the chassis stopped right after cancel_task" if cancel_after is not None else "")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dx", type=float, default=0.0, help="m, forward > 0")
    ap.add_argument("--dy", type=float, default=0.0, help="m, the plan assumes left > 0")
    ap.add_argument("--dyaw-deg", type=float, default=0.0, help="deg, the plan assumes CCW (left) > 0")
    ap.add_argument("--cancel-after", type=float, help="call cancel_task this many seconds after the leg starts")
    ap.add_argument("--execute", action="store_true", help="move the real chassis")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        rel = leg(args.dx, args.dy, args.dyaw_deg)
    except rp.Abort as exc:
        LOG.error("ABORT: %s", exc)
        return 2
    LOG.info("relative_move target: x=%.3f m, y=%.3f m, yaw=%.1f deg -> expected %s%s", rel[0], rel[1],
             math.degrees(rel[2]), expectation(rel),
             f"; cancel_task after {args.cancel_after:.1f} s" if args.cancel_after is not None else "")
    if not args.execute:
        LOG.info("DRY RUN: nothing sent. Add --execute to move the chassis.")
        return 0

    import agibot_gdk  # only on the robot / GDK dev PC

    g2 = rp.G2(agibot_gdk)
    try:
        run(g2, rel, args.cancel_after)
        return 0
    except KeyboardInterrupt:
        LOG.warning("aborted by operator (cancel_task was called if a leg was running)")
        return 130
    except rp.Abort as exc:
        LOG.error("ABORT: %s", exc)
        return 2
    finally:
        g2.close()


if __name__ == "__main__":
    sys.exit(main())
