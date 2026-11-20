# 交接说明：G2 双手捧-运-倒 POC

仓库：https://github.com/liulhs/g2-mujoco-poc
演示视频（1080p60，约 52 秒；中英双语版和纯英文版）：https://github.com/liulhs/g2-mujoco-poc/releases/tag/v0.1

## 1. 这是什么

AgiBot G2 的一个演示动作，按固定点位回放：
1. 原地转向料桶，开过去，边开边把双手伸到桶口上方。
2. 两只 Omnipicker 夹爪合拢、并排，像捧东西一样从桶里“捧”一把。只是做出姿势，不装真东西。
3. 双手端着，底盘后退、原地掉头，再**横着平移**到料斗前，边移边把手抬高。G2 是舵轮底盘，可以往任意方向平移。
4. 靠近料斗，双手伸过料斗沿，一起往前翻手腕“倒”。
5. 边后退边放平、收回双手，再斜着平移回起点，边移边放低双手，最后转回起始朝向。

一整轮在仿真里约 **49 秒**。

不用相机，不用 ROS，不做学习。

流程分两步：
1. **仿真里规划并验证（MuJoCo）**，得到一个动作文件 `sim/out/bimanual_plan.json`。
2. **真机脚本回放这个文件（AgiBot GDK）**：`sim/gdk/run_poc_gdk.py`。

真机脚本自己不做任何规划，只按文件逐步执行。**改动作 = 改仿真规划，重新生成 json，再交给真机脚本。**

## 2. 模型和现场

- **机器人模型**：AgiBot 官方 `genie_sim_robot_model`（Apache-2.0），固定在 commit `fb6de33…`。第一次运行会自动 clone 到 `sim/official/`。
  - 手臂类型 **crs**，已和真机确认。
  - 夹爪 Omnipicker：关节值 `0` = 合拢，`-0.785` = 张开。
- **现场布局**：按图纸 G1-LAY-001 建模，房间 192 × 160 in，原点在左下角。

| 物体 | 位置 / 尺寸 |
|---|---|
| 料桶 | 中心 (132, 40) in，48×48 in，高 30 in（**假设**） |
| 料斗 | 中心 (50, 105.5) in，桶沿高 50 in，内半径 19 in，底座 40×35 in、高 34 in |
| 底盘停靠站点 (x, y, 朝向°) | 起点 (84, 60, 90) → 料桶前 (112, 60, 0) → 后退 (103, 60, 0) → 掉头 (103, 60, 180) → 横移到 (103, 105.5, 180) → 料斗前 (89, 105.5, 180) → 退回 (103, 105.5, 180) → 斜移到 (84, 60, 180) → 转回起点朝向 |

- **还是假设、上真机前要实测的**：
  - 料桶高度和桶壁；
  - **料斗正上方没有框架**（`FRAME_SLAB_IN=0`）。如果框架真的在约 62 in 高，双手倒料达不到 5 cm 余量，规划会失败。
  - 机器人起点位置。

## 3. 仿真端文件（`sim/`）

| 文件 | 作用 |
|---|---|
| `g2_model.py` | 把官方 xacro 展开成 URDF，再导入 MuJoCo，加上底盘平面关节（`base_x/base_y/base_yaw`）和工位几何。工位尺寸可用环境变量调整：`BIN_HEIGHT_IN`、`FRAME_SLAB_IN` 等 |
| `ik.py` | 双臂 IK 和检查器，检查项见下 |
| `plan_bimanual.py` | **规划主程序**。站点、手的高度（`HAND_Z_OVER` 1.55 m）、倒料高度和角度（`POUR_Z` 1.36 m、`POUR_DEG` 55°）都在这里。输出 `out/bimanual_plan.json` |
| `replay.py` | 带物理的回放，统计接触、最小间距和跟踪误差 |
| `view_demo.py` | 交互窗口，macOS 用 `.venv/bin/mjpython view_demo.py` |
| `record_demo.py` | 录 1080p 视频 |
| `coupled_constraints.yaml` | AgiBot 官方的腰部耦合关节限制（MPL-2.0） |

`ik.py` 的检查项：
- 离工位 ≥ 5 cm，手臂和身体之间 ≥ 2 cm，两手之间 ≥ 0.5 cm；
- 关节限位留 0.03 rad 余量；
- AgiBot 腰部耦合约束；
- 躯干倾斜 ≤ 10°，腰部平移 ≤ 0.13 m。

规划器检查每个点位，也检查点之间的每段路径（两种关节时序都查），不通过会自动插中间点。底盘每段只允许“原地转”或“纯平移”（前后、横向、斜向都可以，但不能边转边移）。

**边开边动手**：有 4 段底盘平移时手臂同时在动（开到料桶、横移到料斗、离开料斗、回起点）。这几段按“底盘走到任意位置 × 手臂走到任意位置”的所有组合都做了碰撞检查，所以真机上底盘和手臂谁快谁慢都没关系。检查不过的段（比如离开料桶时收手会碰桶壁）自动退回成“先动手、再开车”。

当前结果：
- 54 个点位，`all_ok = true`；
- 物理回放 **49.3 s**，零接触，离工位最近 5.8 cm，自身最近 2.6 cm，最大跟踪误差 0.037 rad。

仿真速度（`replay.py` 的环境变量，`view_demo.py` / `record_demo.py` 也读）：

| 变量 | 默认 | 含义 |
|---|---|---|
| `MAX_JOINT_SPEED` | 0.5 rad/s | 手臂、头关节峰值速度 |
| `BODY_JOINT_SPEED` | 0.5 rad/s | 腰、腿关节峰值速度 |
| `BASE_LIN_SPEED` / `BASE_LIN_ACCEL` | 0.5 m/s / 0.5 m/s² | 底盘平移速度和加速度 |
| `BASE_YAW_SPEED` / `BASE_YAW_ACCEL` | 1.0 rad/s / 1.0 rad/s² | 底盘转向速度和角加速度 |

- 关节 0.6 rad/s 以上，仿真里手臂跟不上路线，误差超过 0.05 rad，不通过。
- 不限底盘加速度时，急刹会让腿部关节 `idx01_body_joint1` 前后晃 0.13 rad，所以加了加速度上限。
- 这些伺服参数是估的，不是 AgiBot 控制器的，所以**这只是仿真里的时间，不是真机实测**。

## 4. 动作文件 `bimanual_plan.json`

- **顶层字段**：
  - `expected_arm`：`crs`；
  - `all_ok`；
  - `layout`：含 `measured`，目前是 `false`；
  - `waypoints`；
  - `segments`。
- **每个 waypoint**：
  - `name`、`note`；
  - `kind`：`arm`（手臂动作）或 `drive`（底盘动作）；
  - `joints`：27 个关节角，包括腰、头、左右臂各 7 个、两个夹爪；
  - `gripper`；
  - 底盘段带 `relative_move`，内容是 `dx_m / dy_m / dyaw_rad`，在机器人自身坐标系下。

## 5. 真机脚本 `sim/gdk/run_poc_gdk.py`

### 它怎么调 GDK

| 动作 | GDK 调用 |
|---|---|
| 腰、头、双臂（22 个关节） | `Robot.joint_control_request(JointControlReq)`。按 `--speed` 算出同步速度，默认 0.15 rad/s，上限 0.5（仿真验证过的速度）。发完会回读关节，误差超过 0.02 rad 就中止 |
| 两只夹爪 | `Robot.move_ee_pos`，group 为 `left_tool` / `right_tool`，`target_type="omnipicker"`。这个调用即使没动也会返回 0，所以脚本会轮询 `get_end_state` 确认到位，超时 2 s |
| 底盘 | `Pnc.relative_move(NaviReq)`，然后轮询 `get_task_state`：9 = 成功，8 = 失败，7 = 取消。每段超时 60 s。超过 90° 的转弯拆成两段，免得底盘往反方向转 |
| 边开边动手 | 先发 `relative_move`（它会立刻返回），再依次发这段的手臂姿态，最后等底盘完成。手臂任何一步失败，就调 `cancel_task` 停底盘 |
| 每一步之前 | `get_whole_body_status` / `get_motion_control_status` 检查：没有报错、没有急停、`control_mode == 1`（位置模式） |

### 跑之前的检查门（加 `--execute` 时）

1. 动作文件必须 `all_ok`。
2. 布局还没实测时，必须加 `--accept-assumed-layout`。
3. 手臂类型要和文件一致：脚本从 GDK 读取；读不到就用 `--confirm-arm crs`。
4. 机器人每个关节必须已在起点姿态 0.05 rad 以内。不在的话可以加 `--approach-home`，它会以 0.1 rad/s 慢慢移过去，但这一步**没有碰撞检查**，要输入 `APPROACH` 确认。
5. 输入 `ESTOP-READY`：命令一旦发出，软件就停不下来，要手放在急停上。
6. 有底盘段时，要先在 **G02 Pad 上重定位**，把机器人摆到起点，再输入 `RELOCALIZED`。
7. 默认每一步前按回车才发送。`--no-step` 必须配合 `--i-have-run-this-before` 才能用。

### 命令

都在 `sim/` 下运行；真机或 GDK 开发机上需要先 `source env.sh`。

1. 空跑，只打印，不需要 GDK：
   ```bash
   python gdk/run_poc_gdk.py --waypoints out/bimanual_plan.json
   ```
2. 检查关节方向：逐个关节往行程中间点动 0.05 rad 再回来，每个关节都要按回车。
   ```bash
   python gdk/run_poc_gdk.py --waypoints out/bimanual_plan.json --check-joints --confirm-arm crs
   ```
3. 真机运行：
   ```bash
   python gdk/run_poc_gdk.py --waypoints out/bimanual_plan.json --execute --speed 0.15 --accept-assumed-layout
   ```

测试用的是假的 GDK，不需要真机：`python -m pytest -q gdk`，共 22 个测试。

## 6. 已知缺口（上真机前要补）

- **底盘安全加固还没做**：
  - 底盘段运行中按 Ctrl-C **停不了底盘**，只能拍急停；
  - 取消任务没有传 task id；
  - 可能读到上一个任务的旧状态；
  - 每段开完后没有检查定位误差。
- **停车精度**：误差要控制在 2 cm、2° 以内。按 ±3 cm、±2° 算，料桶那边的余量最低降到约 2.4 cm。
- **`relative_move` 遇到障碍只会停，不会绕**：停车距离还没实测；料桶、料斗前的停靠点离障碍只有 4–5 in。
- **没有建模负载**：手里是空的。
- **没在真机上验证过**：
  - `joint_control_request` 是否真的多关节同步到达；
  - 肩高和模型是否一致：模型里 `arm_r_link2` 高 1.382 m；
  - `relative_move` 执行期间能不能同时发 `joint_control_request`（边开边动手依赖这一点）；
  - 横向、斜向的 `relative_move` 在真机上走得准不准。
- **真机一整轮会比 49 秒长**：
  - 底盘真实的速度和加速度由机器人自己的导航配置决定，脚本控制不了；
  - `joint_control_request` 每发一个姿态都要等它到位，所以手臂在中间点会短暂停一下，而仿真里是连续走过去的；
  - 脚本默认 0.15 rad/s，要接近仿真时间需要逐步提到 `--speed 0.5`。

## 7. 怎么改动作

1. 改 `plan_bimanual.py` 里的站点、高度、角度，或者用环境变量改工位尺寸（实测后加 `LAYOUT_MEASURED=1`）。
2. 运行 `.venv/bin/python plan_bimanual.py`，确认输出 `ALL_OK True`。
3. 运行 `.venv/bin/python replay.py`，确认 `REPLAY_OK True`。
4. 运行 `.venv/bin/mjpython view_demo.py` 看一遍。
5. 把新的 `out/bimanual_plan.json` 拷到真机，先空跑，再逐步执行。
