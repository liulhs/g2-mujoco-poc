# (Superseded) single-arm G2 scoop & dump POC

> Superseded by the bimanual mobile plan described in the top-level README. Scripts here live in `sim/legacy_single_arm/`; outputs are not committed.
Fixed-waypoint fake pick (source bin) and fake dump (hopper, under the frame) for an
AgiBot G2, validated in MuJoCo and replayed on the real robot with AgiBot GDK.
No camera, no ROS, no learning.

## Robot model (official)

- Source: [AgibotTech/genie_sim_robot_model](https://github.com/AgibotTech/genie_sim_robot_model)
  (Apache-2.0), commit `fb6de335f6a293605204a17608cf0b22b1fef6de`, cloned into `official/`.
- Expanded from `robots/genie/g2/g2_<arm>_omnipicker.urdf.xacro` with the official
  `collision_type="fine"` option. `G2_ARM=crs` (default) or `G2_ARM=crsB`.
- Only changes for MuJoCo:
  - wheels locked;
  - gripper loop joints removed (mimic joints kept);
  - DAE visuals converted to OBJ, display only;
  - robot raised so its wheels rest on the floor (+40.8 mm);
  - TCP 0.131 m past `gripper_r_base_link`.
- The older `G2/genie_robot_description` copy in this repo is **not** used. It differs from the official files, including a 1.3 cm gripper offset.

## Run

```bash
cd sim && python3 -m venv .venv && .venv/bin/pip install mujoco numpy scipy pyyaml trimesh pycollada imageio imageio-ffmpeg xacro pytest
G2_ARM=crs FRAME_SLAB_IN=1 .venv/bin/python plan_poc.py      # -> out/waypoints_crs.json (rename/PLAN_OUT as needed)
G2_ARM=crs FRAME_SLAB_IN=1 PLAN_FILE=waypoints_crs_slab1.json .venv/bin/python replay.py
cd gdk && ../.venv/bin/python -m pytest -q
../.venv/bin/python run_poc_gdk.py --waypoints ../out/waypoints_crs_slab1.json   # dry run
```

Layout inputs (environment variables, inches unless noted):

| Variable | Meaning |
|---|---|
| `G2_BASE_IN` | Robot base position, default `101.5,84` |
| `G2_YAW_DEG` | Robot yaw in degrees, default `90` |
| `BIN_HEIGHT_IN` | Source-bin height, default `30` |
| `FRAME_SLAB_IN` | Frame beam depth below 63 in |
| `LAYOUT_MEASURED=1` | Set this once the layout is measured |

## Current results (all with ≥5 cm to the workcell, ≥2 cm self-clearance, AgiBot coupled-joint limits)

Checks run on every plan:
- every waypoint;
- every segment, under both the synchronised and the independent-arrival joint timing;
- a physics replay at 0.3 rad/s.

| Arm | Frame beam | Result |
|---|---|---|
| crs | 1 in (bottom at 62 in) | **pass**: plan + physics replay (0 contacts, min 5.7 cm env, 4.2 cm self, 0.03 rad tracking) |
| crs | 3 in (bottom at 60 in) | fail: 4.5 cm near the frame |
| crsB | 1 in | fail: 4.0 cm at the hopper wall, dump 6 cm short |
| crsB | 3 in | fail: 3.3 cm |

The plan is valid only for the bin height it was made with. With +2 in of bin height, clearance drops to 3 cm.

## Before the real robot

1. Arm type: **confirmed crs** (2026-11-20). The runner still checks GDK's `right_arm_type`; if GDK does not report it, pass `--confirm-arm crs`.
2. Measure:
   - the bin height and wall;
   - the frame: beam underside height and depth, and the post positions;
   - the robot's actual parking pose.

   Set the variables above with `LAYOUT_MEASURED=1` and re-plan.
3. Check that GDK `joint_control_request` arrives synchronised. Test with one multi-joint move away from the frame.
4. Measure shoulder height above the floor and compare with the model (`arm_r_link2` z = 1.382 m).
5. Run `run_poc_gdk.py --check-joints`, then the plan step by step at 0.15 rad/s, with a hand on the E-STOP.
