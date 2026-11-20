# AgiBot G2 bimanual cup-carry-pour POC (MuJoCo → GDK)

The official AgiBot G2 model (crs arms, two Omnipicker grippers) is simulated in MuJoCo on a mobile base. It does the following:

1. Drives to a source bin.
2. Cups a "handful" with both hands. The closed grippers are held side by side, so this is a demo posture with no real material.
3. Backs off, turns around while bringing the hands to the chest, and slides sideways to the hopper while raising them (the swerve base translates in any direction).
4. Pours with both hands.
5. Returns to the start.

The validated plan is then replayed on the real robot with AgiBot GDK. There is no camera, no ROS and no learning.

Demo video: see the GitHub Releases page.

## Layout

All layout numbers come from drawing G1-LAY-001: a 192 × 160 in room with the origin at the lower-left corner.

The following values are **assumptions** until they are measured on site:
- the bin height (30 in);
- no overhead frame above the hopper (`FRAME_SLAB_IN=0`);
- the robot's start pose.

## Results (`sim/out/bimanual_plan.json`)

| Check | Result |
|---|---|
| Plan | 43 waypoints, all checks pass (`all_ok`) |
| Clearance limits | ≥5 cm to the workcell, ≥2 cm self-clearance, AgiBot coupled-joint limits, torso tilt ≤10° |
| Physics replay | **42.8 s** cycle, 0 contacts, min 5.8 cm to the workcell, min 2.3 cm self-clearance, max tracking error 0.037 rad |
| Speeds (sim) | joints 0.5 rad/s; chassis 0.5 m/s, 1.0 rad/s, ramps 0.5 m/s², 1 rad/s² |
| Chassis | Every leg is a pure turn in place or a pure translation (`Pnc.relative_move`) |
| Arms during chassis motion | 8 of 9 chassis legs (turns included) move the arms at the same time; checked for every combination of chassis and arm progress, so relative timing does not matter. Only the scoop and the pour run with the chassis stopped |

The speeds are environment variables of `replay.py`, `view_demo.py` and `record_demo.py`:
`MAX_JOINT_SPEED`, `BODY_JOINT_SPEED`, `BASE_LIN_SPEED`, `BASE_YAW_SPEED`, `BASE_LIN_ACCEL`, `BASE_YAW_ACCEL`.
At 0.6 rad/s and above the simulated arms lag their path by more than 0.05 rad (fail).
The servo gains in the simulation are estimates, not AgiBot's controller, so these timings are not a real-robot measurement.

With the 62 in frame (`FRAME_SLAB_IN=1`), the bimanual pour **fails** the 5 cm clearance limit.

## Run

```bash
cd sim
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python plan_bimanual.py          # first run clones the official model; writes out/bimanual_plan.json
.venv/bin/python replay.py                 # physics replay + report
.venv/bin/mjpython view_demo.py            # interactive viewer (macOS needs mjpython)
.venv/bin/python record_demo.py            # 1080p60 video
DEMO_LANG=en .venv/bin/python record_demo.py   # same video, English-only overlay
.venv/bin/python -m pytest -q gdk          # GDK runner tests (fake robot)
```

## Real robot (`sim/gdk/run_poc_gdk.py`)

The runner does a dry run by default. Real motion needs `--execute` and passing the following safety gates:
- the plan is marked `all_ok`;
- the arm type matches the plan;
- the robot is at the start pose;
- the operator types the `ESTOP-READY` confirmation;
- before the chassis moves, the operator confirms G02 Pad relocalization (`RELOCALIZED`);
- every move waits for Enter.

```bash
python gdk/run_poc_gdk.py --waypoints out/bimanual_plan.json                       # dry run
python gdk/run_poc_gdk.py --waypoints out/bimanual_plan.json --execute --speed 0.15 --accept-assumed-layout
```

The following are **not done yet**:
- chassis-leg hardening: cancel a leg by task id, Ctrl-C stops the chassis, pose check after each leg;
- measuring how far `relative_move` stops from an obstacle.

`relative_move` stops at obstacles but does not avoid them.

## Layout of the repo

- `sim/g2_model.py`: builds the official model and the workcell.
- `sim/ik.py`: IK, clearance checks and coupled-joint checks.
- `sim/plan_bimanual.py`: the planner.
- `sim/replay.py`: physics replay.
- `sim/record_demo.py`, `sim/view_demo.py`: video recording and the interactive viewer.
- `sim/gdk/`: the real-robot runner and its tests.
- `sim/legacy_single_arm/`: the earlier single-arm plan, now superseded.
- `docs/G2_POC_plan.md`: the plan notes, in Chinese.
- `docs/HANDOFF.md`: handoff notes for the scripts (Chinese).

See `THIRD_PARTY_NOTICES.md` for the official model, `coupled_constraints.yaml` and GDK licensing.
