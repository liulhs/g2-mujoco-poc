# Third-party notices

- **AgiBot G2 robot model** — [AgibotTech/genie_sim_robot_model](https://github.com/AgibotTech/genie_sim_robot_model),
  Apache-2.0. Not included in this repository: `sim/g2_model.py` clones it at commit
  `fb6de335f6a293605204a17608cf0b22b1fef6de` into `sim/official/` on first run.
- **`sim/coupled_constraints.yaml`** — AgiBot coupled-joint limits copied unchanged from
  [AgibotTech/genie_sim](https://github.com/AgibotTech/genie_sim) (`source/geniesim_ros/src/ros_ws/src/genie_sim_moveit_plugins/config/coupled_constraints.yaml`),
  Mozilla Public License 2.0 (https://mozilla.org/MPL/2.0/). This file stays under MPL-2.0.
- **AgiBot GDK** (`agibot_gdk`) is AgiBot's SDK and is not included; `sim/gdk/run_poc_gdk.py` imports it on the robot.
