# 交接给机器人现场 agent：G2 双手捧-运-倒 POC 真机测试（v0.2）

> 读者：在 AgiBot G2 现场、通过 GDK 控制机器人的 coding agent。
> 目标：把仿真里跑通的流程，在真机上**安全地**逐步跑通。一期只要求流程能从头到尾跑通，不拿工具，也不装料。
> 这里所有东西都只在 MuJoCo 仿真里验证过，**还没有在真机上跑过一次**。

---

## 0. 最重要的一点：仿真给的数字可能是错的，由你来修正和适配

仿真端是照着官方模型和 GDK 文档做的，没有接触过这台机器人。所以下面这些**都可能不对**：
- 关节角度：正负方向、零位
- 底盘约定：y 正方向往哪边、yaw 正方向往哪边、坐标系
- GDK 接口的用法：控制模式、返回值、阻塞与否、任务状态码
- 夹爪指令的数值
- 场地尺寸和站位
- 脚本里的容差、超时、速度

第 1 轮只读检查已经证实了几处（右臂 IK 分支、`mode` 字段、旧任务状态），这些已经修了，但还会有别的。

**你被授权直接修正这些问题，不需要先报告、等仿真端改。** 包括改 `run_poc_gdk.py`、写新脚本、调参数，必要时改 `bimanual_plan.json` 里的数字。目标是让流程在这台机器人上跑通。

修的时候按第 2 节的规矩来。下面这条硬规则**不变**：

---

## 1. 硬规则（安全，不能改）

1. **任何会让机器人动的命令，都要现场的人类操作员确认后再发。** 包括带 `--execute`、`--check-joints` 的命令，`chassis_probe.py --execute`，以及你自己写的任何 GDK 运动调用。发之前说清楚三件事：发哪条命令、机器人会怎么动、动多远。操作员的手要一直放在**硬件急停**上。
2. 脚本里的交互确认（`ESTOP-READY`、`RELOCALIZED`、`APPROACH`、`GO`、每一步的回车）必须由人亲手输入。**不要用 `yes |`、`expect`、heredoc 之类的办法替人输入**，也不要加 `--no-step`。你写的新脚本，凡是会动机器人，也要有同样的人工确认。
3. 指令一旦发出，软件里就撤不回来。要停车，以拍急停为准。
4. 第一次做任何新动作，都用低速、小幅度，在空旷处做：关节 ≤ 0.15 rad/s，底盘一次 ≤ 30 cm 或 ≤ 30°。确认方向和幅度都对了，再放大。
5. 凡是动作和预期明显不一致（方向反了、多转、撞到或差点撞到东西），**立刻停下**，先查清原因再继续。不要换个参数直接重试。

---

## 2. 修正和适配的规矩

1. **先测，再改。** 每一处修改都要有依据：读到的数值、`help()` 的输出，或者人在现场看到、量到的结果。不要凭猜改。
2. **每次改动都记进 `CHANGES_ON_ROBOT.md`**，放在解压目录的根目录。每条写清：
   - 改了什么（文件、参数、旧值 → 新值）
   - 为什么改（依据）
   - 怎么验证的
   - 这个改动会不会让机器人走到仿真没检查过的位置（见下一条）

   这份记录要回传给仿真端，用来修模型和重新规划。
3. **分清两类改动：**
   - **约定类**：符号、单位、零位偏置、接口用法、状态码、超时、容差。改完以后机器人仍然走仿真规划好的那条路径，只是把指令翻译对了。可以直接改。
   - **几何类**：改变手或身体实际到达的位置。例如改 json 里的关节角、改底盘段的距离、因为场地不一样而挪位置。仿真的碰撞检查**不再覆盖**改过的部分。可以改，但要满足三点：
     - 幅度尽量小。单个关节偏离仿真值超过约 0.2 rad，或者手的位置偏离超过约 5 cm，先在空旷处、低速、人盯着单独试。
     - 改过的路点在 `CHANGES_ON_ROBOT.md` 里标为"未经碰撞检查"。
     - 场地尺寸实测后和第 4 节差得多时，把实测数据回传，让仿真端重新规划，比在真机上一点点挪更安全。
4. **改 json 时保留原文件**，另存一份（比如 `out/bimanual_plan.robot.json`），不要覆盖原来的。
5. 脚本自带的安全检查可以按实际调整，比如 `START_TOL`、`REACHED_TOL`、超时，但**不能删掉**：急停确认、`all_ok` 检查、到位回读、底盘中止时调 `cancel_task`。

---

## 3. 已知没验证过的假设（逐项核对）

| # | 假设 | 怎么核对 | 错了怎么办 |
|---|---|---|---|
| A1 | 末端是 **Omnipicker**（`0` = 合拢，`-0.785` = 张开），指令是 `move_ee_pos`，`target_type="omnipicker"`，夹爪关节是 `gripper_*_inner_joint1`，`get_end_state` 的 `end_states[0]` 就是它 | 抓夹装上后跑 `robot_probe.py`，看 `end_state` 里的关节名和数值；人工张合一次 | 改 `set_grippers` / `gripper_positions` |
| A2 | `joint_control_request` 在当前模式（2.6.3 读到 `mode=5`，伺服）下可以用，会阻塞到位，速度单位是 rad/s | `robot_probe.py` 导出的 `help()`；`--check-joints` 时观察 | 改 `--control-mode` 或 `move_joints` |
| A3 | 每个关节的**正负方向和零位**和模型一致（只用 reset 姿态核对过一次：两只手左右对称） | `--check-joints`，逐个关节动 0.05 rad，人看方向 | 方向反了：在发指令前给该关节取反；零位差：加固定偏置。两种都是约定类改动 |
| A4 | 真机 reset 姿态 = 计划第一个点 `RESET`：左臂 (1.571, −1.571, −1.571, −1.571, 0, 0, 0)，右臂 (−1.571, −1.571, 1.571, −1.571, 0, 0, 0)，腰部全 0 | 脚本启动时会比较，差超过 0.05 rad 就拒绝 | 小差距可以放宽 `START_TOL`，或者用 `--approach-home` |
| A5 | 底盘 `relative_move`：x 向前，**y 向左为正**，**yaw 逆时针（向左转）为正**，相对机器人当前位姿 | `chassis_probe.py`（见第 5 节第 2 步） | 在 `start_relative_move` 里统一换算 |
| A6 | 侧向和斜向的目标，底盘会**直接平移、车身不转** | 同上，用 `--dy` 测，看车身转没转 | 如果车身会转：要么改用 `move_chassis` 等别的接口，要么把横移拆成"转-走-转"（几何类改动） |
| A7 | `relative_move` 发出后立刻返回；`get_task_state` 状态码 9 = 成功、8 = 失败、7 = 取消；每发一段都会出现新的任务 `id` | `chassis_probe.py` 会打印状态时间线 | 改 `TASK_*` 常量和等待逻辑 |
| A8 | `cancel_task()` 能马上停住正在跑的底盘 | `chassis_probe.py --cancel-after 1` | 停不住的话，底盘的每段都要人手放在急停上 |
| A9 | `relative_move` 执行期间可以同时发 `joint_control_request`（边开边动手；9 段底盘里有 8 段依赖这一点） | 第 5 节第 6 步 | 不行的话改成"先动手、再开车"。计划里的手臂姿态不变，只是顺序变了，这样同样安全 |
| A10 | 场地尺寸（见第 4 节） | 现场量 | 回传实测值，让仿真端重新规划 |
| A11 | 腰部 5 个关节都能按计划动，计划里腰部前倾不超过 10° | `--check-joints` | 约定类：同 A3 |

---

## 4. 计划假设的场地（单位：英寸，原点在房间左下角，图纸 G1-LAY-001）

| 物体 | 假设 |
|---|---|
| 料桶 | 48 × 48，**高 30（未定）**，一角在 (132, 40) |
| 料斗 | 圆心 (50, 105.5)，内半径 19，**桶口高 50**，底座 40 × 35 × 34 |
| 顶上框架 | 顶部高 63（未量），不确定料斗上方有没有横梁 |
| 机器人起点 | (84, 60)，面朝 +y |
| 料桶停靠点 | (112, 60)，面朝 +x，车身前沿离桶壁约 5 in |
| 料斗停靠点 | (89, 105.5)，面朝 −x，车身前沿离料斗底座约 4 in |

现场还在搭。搭好以前，**先在空地上把手臂和底盘的约定都核对清楚**（第 5 节第 1–6 步）。这部分不依赖场地。

---

## 5. 推进顺序（第 2 轮）

工具都在 release 包的 `gdk/` 目录下，都只用 Python 标准库加 `agibot_gdk`。

### 第 1 步：只读探测（不动机器人，可以马上做）

```bash
python3 gdk/robot_probe.py --out probe
```

它会导出这些内容到 `probe/robot_probe_<时间>.json`，以及一份 `_help.txt`：
- 末端型号和每个末端关节
- 当前控制模式和 `MotionControlMode` 的全部取值
- 当前任务状态
- 所有关节的名字和角度
- `joint_control_request`、`move_ee_pos`、`relative_move` 等接口的 `help()`
- 本机上和机器人模型有关的文件（`*.urdf`、`*.xacro`、`*.stl` 等）

GDK 跑在开发机的 Docker 里，所以如果能 SSH 到机器人主控，再在主控上只搜文件：

```bash
python3 robot_probe.py --no-gdk --out probe_robot
```

主控上没有 Python 的话，用 `find / -name "*.urdf" 2>/dev/null` 代替。

### 第 2 步：底盘约定（会动，在空地上，人确认）

手臂保持现在的姿态。每条都先不加 `--execute` 看一眼会发什么，再加上真正执行：

```bash
python3 gdk/chassis_probe.py --dy 0.10 --execute        # 往哪边移？车身转了没？实际走了多远？
python3 gdk/chassis_probe.py --dy -0.10 --execute       # 应该回到原处
python3 gdk/chassis_probe.py --dx 0.10 --execute        # 前进
python3 gdk/chassis_probe.py --dyaw-deg 20 --execute    # 向左转还是向右转？
python3 gdk/chassis_probe.py --dx 0.30 --execute --cancel-after 1.0   # cancel_task 能不能停住
```

每次执行前在地上画好标记，执行后量出移动的距离和角度，记下来。结果对应第 3 节的 A5–A8。如果约定和假设不一样，在 `start_relative_move` 里统一换算，并记进 `CHANGES_ON_ROBOT.md`。

### 第 3 步：装上抓夹后，核对末端（A1）

再跑一次 `robot_probe.py`，确认 `left_end_model` / `right_end_model` 都是 `omnipicker`，并看夹爪关节的名字和数值。人工让夹爪张开、合拢各一次，确认 `0` 和 `-0.785` 分别对应哪个状态。

### 第 4 步：关节方向（会动，人确认）

```bash
python3 gdk/run_poc_gdk.py --waypoints out/bimanual_plan.json --check-joints --confirm-arm crs --control-mode <N>
```

逐个关节往行程中间方向动 0.05 rad 再回来，人看方向是否和预期一致。`<N>` 填第 1 步读到的模式，前提是你已经确认 `joint_control_request` 应该在这个模式下用。

### 第 5 步：分段跑手臂（会动，人确认）

机器人先回到 reset 姿态。只跑到 START，底盘不动：

```bash
python3 gdk/run_poc_gdk.py --waypoints out/bimanual_plan.json --execute --speed 0.15 --accept-assumed-layout --control-mode <N> --until START
```

`--until <路点名>` 跑到这个路点就停。路点名可以用 dry run 列出来。回到 reset 的办法：沿同一条路径倒着走（`START` → `START~via` → `RESET`），这条路径仿真里检查过。可以写个小脚本来做。

### 第 6 步：边开边动手（会动，空地上）

```bash
... --until TURN_TO_BIN     # 原地转 90°，同时抬手
... --until DRIVE_TO_BIN    # 前进 0.71 m，同时把手伸出去
```

在空地上做，确认手臂和底盘是不是真的同时在动（A9）。

### 第 7 步：场地搭好以后

1. 量场地，和第 4 节对比，把实测值回传。
2. 在 G02 Pad 上重定位，把机器人摆到起点站位和朝向。
3. 低速完整跑一遍：

```bash
python3 gdk/run_poc_gdk.py --waypoints out/bimanual_plan.json --execute --speed 0.15 --accept-assumed-layout --control-mode <N>
```

跑通以后，再逐步把速度提到 0.25、0.35、0.5。

---

## 6. 注意事项

- **料桶、料斗前余量很小。** 停靠点离桶壁只有 4–5 in；底盘停车误差要在 2 cm、2° 以内。按 ±3 cm、±2° 算，料桶那边的余量会降到约 2.4 cm。脚本每段开完后**不检查**实际停在哪里，误差会一段段累积，需要人观察。
- **中间点会一顿一顿。** `joint_control_request` 每发一个姿态都等它到位，所以经过中间点会短暂停一下，而仿真里是连续走过去的。这是预期行为，不是故障。
- **180° 掉头会拆成两个 90°**，防止底盘选另一个方向转。
- **时间会比仿真长。** 仿真时间是 0.5 rad/s 下的结果；真机的底盘速度由机器人自己的导航配置决定，脚本默认关节速度 0.15 rad/s。先把流程跑通，不要追时间。
- **手里不放东西。** 计划没有建模负载，一期只做空手的"捧"的动作。
- 模型里的肩高（`arm_r_link2` 离地 1.382 m）还没和真机核对过。

---

## 7. 下载

Release：https://github.com/liulhs/g2-mujoco-poc/releases/tag/v0.2，附件 `g2_poc_v0.2.zip`。

```bash
mkdir -p ~/g2_poc && cd ~/g2_poc
curl -fL -o v0.2.zip https://github.com/liulhs/g2-mujoco-poc/releases/download/v0.2/g2_poc_v0.2.zip
python3 -m zipfile -e v0.2.zip .
cd g2_poc_v0.2 && python3 gdk/run_poc_gdk.py --waypoints out/bimanual_plan.json   # dry run
```

解压后：

```
g2_poc_v0.2/
  README.txt
  AGENT_HANDOFF_G2_GDK.md          本文档
  gdk/run_poc_gdk.py               执行脚本（支持 --until 分段跑）
  gdk/robot_probe.py               只读探测
  gdk/chassis_probe.py             底盘约定测试
  out/bimanual_plan.json           轨迹：从 RESET 开始，54 个路点，9 段底盘
  out/replay_report_bimanual_plan.json   MuJoCo 物理回放报告
```

dry run 最后一行应该是 `plus 9 chassis legs (8 with the arms moving)`。

**v0.1 的三个版本都作废了。** 那三份计划里右臂处在另一组 IK 解上，和真机差约 3 rad，不要再用。

---

## 8. 回报内容

做完一步或者卡住时，回报这些：

1. `robot_probe.py` 导出的 json 和 `_help.txt`，以及找到的模型文件。
2. 第 2 步底盘测试的结果：每条命令的实际方向、距离或角度、车身有没有转、状态时间线、`cancel_task` 能不能停住。
3. `--check-joints` 时哪些关节方向反了。
4. `CHANGES_ON_ROBOT.md` 全文。
5. 每次中止的完整报错和当时读到的关节值。
6. 场地实测尺寸（场地搭好以后）。
7. 有没有任何地方离料桶、料斗或机器人自身太近（目测或拍照）。

---

## 9. 改动作的另一条路：回到仿真重新规划

几何类改动比较大，或者场地实测和第 4 节差得多时，回到仿真重新规划更安全。仿真里会重新做碰撞检查。

- 仓库：https://github.com/liulhs/g2-mujoco-poc（说明见 `docs/HANDOFF.md`）
- 在开发机上（不在机器人上）：
  1. 改 `sim/plan_bimanual.py`，或用环境变量改布局；实测过的话加 `LAYOUT_MEASURED=1`。
  2. 跑 `python plan_bimanual.py`，确认输出 `ALL_OK True`。
  3. 跑 `python replay.py`，确认输出 `REPLAY_OK True`。
  4. 把新的 `out/bimanual_plan.json` 拷到机器人上，先 dry run。
- 也可以只把实测数据和 `CHANGES_ON_ROBOT.md` 回传，由仿真端来重新规划。
