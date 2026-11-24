"""Read-only probe of a real G2: end effector (o10_t2 hands), control modes, GDK signatures, model files.

Sends NO motion command. It calls gdk_init, constructs Robot/Pnc, reads status getters and help()
texts, and searches the local filesystem for robot description files. Output: one JSON file plus a
text file with the help() texts, both named robot_probe_<time>.* in --out (default: current dir).

    python3 gdk/robot_probe.py --out ~/g2_poc/probe
    python3 gdk/robot_probe.py --no-gdk --search-root /opt     # filesystem search only (e.g. on the robot)
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

MODEL_EXT = (".urdf", ".xacro", ".srdf", ".mjcf", ".xml", ".stl", ".dae", ".obj", ".glb", ".yaml")
MODEL_NAME = re.compile(r"o10|hand|dexter|g2|omnipicker|end_effector|gripper", re.IGNORECASE)
SEARCH_ROOTS = ("/opt", "/home", "/root", "/usr/share", "/usr/local/share", "/data", "/agibot", "/workspace")
SKIP_DIRS = {"proc", "sys", "dev", "run", "tmp", "snap", "node_modules", ".git", "__pycache__", "site-packages"}
MAX_HITS = 400
HELP_TARGETS = (
    "Robot.joint_control_request", "Robot.move_ee_pos", "Robot.set_control_mode", "Robot.get_end_state",
    "Robot.get_motion_control_status", "JointControlReq", "JointStates", "JointState", "MotionControlMode",
    "Pnc.relative_move", "Pnc.get_task_state", "Pnc.cancel_task", "Pnc.pause_task", "Pnc.move_chassis",
    "NaviReq", "PNCTaskState",
)
INTERESTING = re.compile(r"hand|end|ee|finger|urdf|model|pose|odom|locali|chassis|mode", re.IGNORECASE)


def plain(obj: Any, depth: int = 0) -> Any:
    """GDK return values (pybind structs, dicts, lists) as JSON-able data."""
    if depth > 6:
        return repr(obj)
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): plain(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [plain(v, depth + 1) for v in obj]
    members = getattr(type(obj), "__members__", None)
    if members is not None and hasattr(obj, "name"):
        return f"{obj.name}={int(obj)}" if hasattr(obj, "__int__") else str(obj.name)
    fields = {k: getattr(obj, k) for k in dir(obj) if not k.startswith("_")}
    data = {k: v for k, v in fields.items() if not callable(v)}
    return {k: plain(v, depth + 1) for k, v in data.items()} if data else repr(obj)


def _call(results: dict[str, Any], key: str, fn: Any) -> None:
    try:
        results[key] = plain(fn())
    except Exception as exc:  # record and continue: this is a survey, not a gate
        results[key] = f"ERROR {type(exc).__name__}: {exc}"


def _resolve(gdk: Any, dotted: str) -> Any:
    obj = gdk
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def help_texts(gdk: Any) -> str:
    out = []
    for name in HELP_TARGETS:
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                help(_resolve(gdk, name))
        except AttributeError:
            buf.write("(not in this GDK)\n")
        out.append(f"===== {name} =====\n{buf.getvalue()}")
    return "\n".join(out)


def gdk_survey(gdk: Any) -> dict[str, Any]:
    res: dict[str, Any] = {"gdk_file": getattr(gdk, "__file__", None),
                           "gdk_names": sorted(n for n in dir(gdk) if not n.startswith("_"))}
    for cls in ("Robot", "Pnc"):
        names = sorted(n for n in dir(getattr(gdk, cls, object)) if not n.startswith("_"))
        res[f"{cls}_methods"] = names
        res[f"{cls}_methods_of_interest"] = [n for n in names if INTERESTING.search(n)]
    enum = getattr(gdk, "MotionControlMode", None)
    res["MotionControlMode"] = ({k: int(v) for k, v in enum.__members__.items()}
                                if getattr(enum, "__members__", None) else "not exposed")
    if gdk.gdk_init() != gdk.GDKRes.kSuccess:
        res["gdk_init"] = "FAILED"
        return res
    try:
        robot = gdk.Robot()
        pnc = gdk.Pnc() if hasattr(gdk, "Pnc") else None
        time.sleep(2.0)  # DDS discovery
        _call(res, "whole_body_status", robot.get_whole_body_status)
        _call(res, "motion_control_status", robot.get_motion_control_status)
        _call(res, "end_state", robot.get_end_state)
        _call(res, "joint_states", robot.get_joint_states)
        if pnc is not None:
            _call(res, "task_state", pnc.get_task_state)
    finally:
        with contextlib.suppress(Exception):
            gdk.gdk_release()
    return res


def find_model_files(roots: list[str]) -> list[str]:
    hits: list[str] = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root, onerror=lambda e: None):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            path_match = MODEL_NAME.search(dirpath)
            for f in filenames:
                if f.lower().endswith(MODEL_EXT) and (path_match or MODEL_NAME.search(f)):
                    hits.append(os.path.join(dirpath, f))
                    if len(hits) >= MAX_HITS:
                        return hits
    return hits


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("."))
    ap.add_argument("--no-gdk", action="store_true", help="only search for model files")
    ap.add_argument("--search-root", action="append", help="directories to search (default: common ones + $HOME)")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    roots = args.search_root or [*SEARCH_ROOTS, str(Path.home())]
    res: dict[str, Any] = {"time": stamp, "python": sys.version, "search_roots": roots}
    if not args.no_gdk:
        import agibot_gdk  # only on the robot / GDK dev PC

        res["gdk"] = gdk_survey(agibot_gdk)
        (args.out / f"robot_probe_{stamp}_help.txt").write_text(help_texts(agibot_gdk))
    res["model_files"] = find_model_files(roots)
    path = args.out / f"robot_probe_{stamp}.json"
    path.write_text(json.dumps(res, indent=2, ensure_ascii=False, default=repr))
    print(f"wrote {path} ({len(res['model_files'])} model files found)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
