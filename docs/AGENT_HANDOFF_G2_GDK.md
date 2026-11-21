# 交接给机器人现场 agent：G2 双手捧-运-倒 POC 真机测试

> 读者：通过网线 SSH 连在 AgiBot G2 上的 coding agent。
> 目标：把仿真里验证过的三个版本轨迹，用 GDK 在真机上**安全地**逐步跑通。
> 这里所有东西都只在 MuJoCo 仿真里验证过，**还没有在真机上跑过一次**。

---

## 0. 给 agent 的硬规则（先读）

1. **会让机器人动的命令（带 `--execute`、`--check-joints`，或你自己写的任何 GDK 运动调用），必须由现场的人类操作员确认后再执行。** 发之前先说清楚：要发哪条命令、机器人会怎么动、动多远。操作员必须一直把手放在**硬件急停**上。
2. 脚本里有交互确认，需要人类亲手输入：`ESTOP-READY`、`RELOCALIZED`、`APPROACH`，每一步还要按回车。**不要用 `yes |`、`expect`、heredoc 之类的办法替人输入。** 也不要加 `--no-step`。
3. 指令一旦发出，软件里就**撤不回来**：底盘段运行中按 Ctrl-C 停不了底盘，只能拍急停。
4. 不要改 `bimanual_plan.json` 里的数字。轨迹是仿真里做过碰撞检查的，手改就等于没检查过。要改动作，得回到仿真重新规划（见第 6 节）。
5. 遇到和本文不一致的现象（方向反了、多转了一圈、某个调用不存在），**先停下来报告**，不要自己"修"了再接着跑。

---

## 1. 连接环境

文档里记录的网络配置（来自 `docs/G2_POC_plan.md`，以现场为准）：

| 项 | 值 |
|---|---|
| 机器人 | `10.42.1.101`（网线接调试口） |
| 开发机 | `10.42.1.102` |
| GDK 安装 | `curl -sSL http://10.42.1.101:8849/install.sh \| bash` |
| GDK 环境 | **每个终端**都要先 `source ~/.cache/agibot/app/env.sh` |
| GDK 版本 | 在 `http://10.42.1.101:8849` 查 |

先做下面几项检查，并把输出记下来：

```bash
source ~/.cache/agibot/app/env.sh
python3 --version
python3 -c "import agibot_gdk as g; print(g.__file__); print([n for n in dir(g) if not n.startswith('_')])"
python3 -c "import agibot_gdk as g; help(g.Pnc.relative_move); help(g.NaviReq)"
python3 -c "import agibot_gdk as g; print(hasattr(g.Robot,'set_control_mode'), hasattr(g.Pnc,'cancel_task'), hasattr(g.Pnc,'get_task_state'))"
```

- 脚本只用 Python 标准库加 `agibot_gdk`，不需要 numpy、MuJoCo 或任何 pip 包。
- GDK 2.1、2.6、3.3 的接口不完全一样。`set_control_mode`、`Pnc`、`cancel_task` 在你的版本里如果不存在，**先报告**。
- 如果脚本跑在开发机而不是机器人本机上：开发机防火墙要放行来自 `10.42.1.101` 的入站连接。否则读状态正常，但发指令会卡住。

---

## 2. 下载 GitHub release

Release 页面：https://github.com/liulhs/g2-mujoco-poc/releases/tag/v0.1

| 附件 | 用途 |
|---|---|
| `g2_gdk_plans_138s_49s_43s.zip`（67 KB） | **要用的就是这个**：三个版本的轨迹和执行脚本 |
| `g2_demo_v1_original_138s.mp4` / `v2_compressed_49s` / `v3_simultaneous_43s` | 三个版本的仿真视频，跑真机前先看一遍，知道机器人应该怎么动 |
| `g2_bimanual_demo_1080p*.mp4` | 43 s 版本的高画质视频 |

机器人能上网时：

```bash
mkdir -p ~/g2_poc && cd ~/g2_poc
curl -fL -o plans.zip https://github.com/liulhs/g2-mujoco-poc/releases/download/v0.1/g2_gdk_plans_138s_49s_43s.zip
python3 -m zipfile -e plans.zip .
ls gdk_versions
```

机器人不能上网时，在能上网的机器上下载，再 `scp plans.zip <user>@10.42.1.101:~/g2_poc/`。

- 解压用 `python3 -m zipfile -e`，不要用 `unzip`：文件夹名是中文，有些 `unzip` 会解成乱码。
- 解压后的结构：

```
gdk_versions/
  README.txt
  1_最初版_138s/      gdk/run_poc_gdk.py   out/bimanual_plan.json   out/replay_report_bimanual_plan.json
  2_压缩版_49s/       （同上）
  3_同步动作版_43s/   （同上）
```

三个文件夹里的 `run_poc_gdk.py` 是**同一份**最新脚本，三条轨迹都能用它执行（已在本地 dry run 过）。

---

## 3. 三个版本的区别

| 版本 | 仿真时长 | 底盘段 | 底盘怎么动 | 手臂和底盘同时动 |
|---|---|---|---|---|
| **1_最初版_138s** | 138 s | 13 段 | **只有原地转向和前进/后退**，没有横移 | 0 段：每段都是先停车再动手 |
| **2_压缩版_49s** | 49 s | 9 段（180° 掉头拆成 2 段，所以脚本里是 10 段） | 有**横移**（向右 1.16 m）和**斜移**（前 0.48 m + 左 1.16 m） | 4 段（都是平移段） |
| **3_同步动作版_43s** | 43 s | 9 段 | 同版本 2 | 8 段，**转弯时也在动手臂** |

每段底盘的 `relative_move` 都是相对机器人**当前**位姿的（单位 m、rad）。约定：x 向前，**y 向左为正**，yaw 逆时针为正（ROS REP-103）：

| 段 | v1 | v2 | v3 |
|---|---|---|---|
| 转向料桶 | yaw −90° | yaw −90° | yaw −90°（边转边抬手） |
| 开到料桶 | x +0.711 | x +0.711（边开边伸手） | 同 v2 |
| 后退 | x −0.229 | x −0.229 | x −0.229 |
| 去料斗 | 左转 90° → 前进 1.156 → 左转 90° | 掉头 180° → **横移 y −1.156** | 掉头 −180°（边转边收手）→ **横移 y −1.156** |
| 靠近料斗 | x +0.356 | x +0.356 | x +0.356（边靠边伸手） |
| 退出料斗 | x −0.356 | x −0.356（边退边收手） | 同 v2 |
| 回起点 | 左转 90° → 前进 1.156 → 右转 90° → 前进 0.483 | **斜移 x +0.483, y +1.156** | 同 v2（边走边放低手） |
| 转回起点朝向 | yaw −90° | yaw −90° | yaw −90°（边转边放手） |

**测试顺序必须是 v1 → v2 → v3。** v1 不依赖任何还没验证的底盘行为，v2 和 v3 依赖（见第 5 节）。

---

## 4. 怎么测

都在某个版本的文件夹里运行，例如 `cd ~/g2_poc/gdk_versions/1_最初版_138s`，并且已经 `source env.sh`。

### 4.1 Dry run（不动机器人，agent 可以自己跑）

```bash
python3 gdk/run_poc_gdk.py --waypoints out/bimanual_plan.json
```

三个版本都跑一遍，确认最后一行分别是：
- v1：`plus 13 chassis legs (0 with the arms moving)`
- v2：`plus 10 chassis legs (4 with the arms moving)`
- v3：`plus 9 chassis legs (8 with the arms moving)`

### 4.2 现场准备（人做）

1. **实测布局**。下面这些还是假设，和现场不符就**不要跑**，先回仿真重新规划：
   - 料桶高 30 in；
   - 料斗正上方没有框架（如果在约 62 in 高处有框架，双手倒料会碰）；
   - 起点站位 (84, 60) in，朝向 90°。坐标按图纸 G1-LAY-001，原点在房间左下角。
2. 底盘停靠点离料桶、料斗只有 4–5 in。第一次跑时，在料桶、料斗前留出余量，或者先在空地上跑。
3. 在 **G02 Pad 上重定位**，把机器人摆到起点站位和起点朝向。
4. 双臂摆到起点姿态（双手合拢在身前）。每个关节都要在 0.05 rad 以内，否则脚本拒绝执行。

### 4.3 关节方向检查（会动机器人，人来确认）

逐个关节往行程中间点动 0.05 rad 再回来，用来确认符号方向和模型一致：

```bash
python3 gdk/run_poc_gdk.py --waypoints out/bimanual_plan.json --check-joints --confirm-arm crs
```

真机手臂类型是 **crs**，夹爪是 **Omnipicker**：`0` = 合拢，`-0.785` = 张开。脚本读不到手臂类型时，用 `--confirm-arm crs`。

### 4.4 版本 1（会动机器人）

```bash
python3 gdk/run_poc_gdk.py --waypoints out/bimanual_plan.json --execute --speed 0.15 --accept-assumed-layout
```

- 默认每步前要按回车。每步之前确认机器人状态和上一步结束时一致。
- 机器人不在起点姿态时，可以加 `--approach-home`。它会以 0.1 rad/s 慢慢移过去，但这一步**没有碰撞检查**，要人输入 `APPROACH`。
- `--speed` 是关节峰值速度（rad/s），脚本内部上限 0.5。第一次用 0.15，全流程正常后再逐步提到 0.25、0.35、0.5。
- 记录每一段：底盘实际停在哪里（和计划差几 cm、几度）、有没有顿挫、总时长。

### 4.5 版本 2、3 之前：先单独测底盘平移（见 5.1）

### 4.6 版本 2、3（会动机器人）

命令和 4.4 一样，换文件夹即可。先 v2 再 v3。

---

## 5. 注意事项和 gotcha

### 5.1 底盘平移（translate）还没验证，这是最大的风险

v2 和 v3 依赖 `Pnc.relative_move` 能**原地横移、斜移，同时不转车身**（G2 是舵轮底盘）。这一点在真机上**从来没验证过**：

1. **y 的正负号。** 脚本假设 `NaviReq.target.position.y > 0` 是向机器人**左边**移。如果 GDK 的约定相反，`SLIDE_TO_HOPPER`（y −1.156）会往反方向横移 1.16 m。
2. **是不是真的平移。** 有些导航实现收到侧向目标时，会"先转向目标 → 开过去 → 再转回来"。这样车身会转动，手臂扫过的区域和仿真完全不一样，可能撞到料桶或料斗。
3. **斜移是不是一条直线。** `RETURN` 是 x、y 同时有值。如果底盘走折线或弧线，路径和仿真不一致。
4. **`relative_move` 是否立刻返回。** 脚本假设它发出后马上返回，任务在后台跑，再轮询 `get_task_state`（9 = 成功，8 = 失败，7 = 取消）。如果它会一直阻塞到底盘停下，"边开边动手"就会退化成"先开车再动手"：不危险，但 v2、v3 就测不出效果，也要报告。

**先在空地上，手臂在起点姿态，单独测一次小幅横移。** 建议让 agent 写一个小脚本，和 runner 用同样的 API（runner 里的 `G2.start_relative_move`），依次测：

- `dx=0, dy=+0.10, dyaw=0`：观察往哪边移、车身转没转；
- `dx=0, dy=-0.10, dyaw=0`：应该移回原处；
- `dx=+0.10, dy=+0.10, dyaw=0`：观察是不是斜着走直线。

`NaviReq` 的构造照抄 runner：`req.target.position.x/y = dx/dy`，`req.target.orientation.z/w = sin(dyaw/2)/cos(dyaw/2)`，然后 `pnc.relative_move(req)`，再轮询 `pnc.get_task_state()`。同样要人确认后再发。

**判定：**
- 向左、不转车身、走直线：可以跑 v2、v3。
- 方向反了：**不要**自己改脚本里的符号，报告回来，由我们在仿真端统一改并重新出 json。
- 车身会转，或者走折线：v2、v3 不能用，只跑 v1，并报告实际行为。

### 5.2 边开车边动手臂（v2、v3）

- 没验证过：`relative_move` 执行期间（包括原地转向时）能不能同时发 `joint_control_request`。v3 的 9 段底盘里有 8 段依赖这一点。
- 脚本不开线程：先发 `relative_move`，然后依次发这段的手臂姿态，最后等底盘完成。手臂任何一步失败，就调 `cancel_task` 停底盘。
- 规划时把"底盘走到任意位置 × 手臂走到任意位置"的组合（21×21）都做了碰撞检查，所以谁快谁慢不影响安全。**前提是底盘真的按直线平移或原地转。**
- 180° 掉头会拆成两个 90° 的 `relative_move`，避免底盘选另一个方向转。第二段在发完当前手臂姿态后才发，所以中间可能停一下。

### 5.3 底盘安全还没加固

- 底盘段运行中，Ctrl-C **停不了底盘**，只能拍急停。
- `cancel_task` 没有传 task id。
- `get_task_state` 可能读到**上一个任务的旧状态**（例如上一段的 9 = 成功），导致脚本以为这一段已经完成，提前进入下一步。第一次跑时，每段观察脚本是不是在底盘真正停下后才继续。
- 每段开完后，脚本**不检查**实际停在哪里。误差会一段段累积，需要人观察。
- 停车精度要控制在 2 cm、2° 以内。按 ±3 cm、±2° 算，料桶那边的余量会降到约 2.4 cm。
- `relative_move` 遇到障碍只会停，不会绕。停车距离还没实测。

### 5.4 手臂和夹爪

- `joint_control_request` 每发一个姿态都要等它到位，所以中间点会短暂停一下，而仿真里是连续走过去的。v2、v3 离开料斗那段有十几个中间点，真机上会**明显一顿一顿**。这是预期行为，不是故障。
- 每步发完后，脚本回读关节，误差超过 0.02 rad 就中止。如果总是在同一个关节中止，先报告读到的数值。
- `move_ee_pos`（夹爪）即使没动也会返回 0，所以脚本会轮询 `get_end_state` 确认到位，超时 2 s。
- 脚本要求 `control_mode == 1`（位置模式），不是的话会尝试 `set_control_mode(1)`。
- 没有建模负载：手里是空的，只是"捧"的姿势，不要真的放东西。
- 模型里的肩高（`arm_r_link2` 高 1.382 m）还没和真机核对过。

### 5.5 时间

- 138 s、49 s、43 s 都是**仿真**在 0.5 rad/s 下的时间。
- 真机会更慢：底盘的速度和加速度由机器人自己的导航配置决定，脚本控制不了；手臂在中间点会停顿；脚本默认 0.15 rad/s。
- 不要用加速度或速度相关的参数去"追"仿真时间。先把流程安全跑通。

---

## 6. 需要改动作时

真机脚本自己不做任何规划，只按 json 逐步执行。**改动作 = 回到仿真重新规划。**

- 仓库：https://github.com/liulhs/g2-mujoco-poc
- 详细说明：仓库里的 `docs/HANDOFF.md`
- 流程（在开发机上，不在机器人上）：
  1. 改 `sim/plan_bimanual.py`，或用环境变量改布局（实测后加 `LAYOUT_MEASURED=1`）；
  2. `python plan_bimanual.py`，确认 `ALL_OK True`；
  3. `python replay.py`，确认 `REPLAY_OK True`；
  4. 把新的 `out/bimanual_plan.json` 拷到机器人，先 dry run，再逐步执行。

---

## 7. 跑完要回报的内容

每个版本都请回报：

1. GDK 版本，以及第 1 节各项检查的输出。
2. 5.1 横移测试的结果：方向、车身有没有转、斜移是不是直线。
3. 每段底盘实际停的位置和计划差多少（cm、度）。
4. 有没有中止，中止时的完整报错和读到的关节值。
5. "边开边动手"是否真的同时进行（v2、v3）。
6. 每个版本的总时长和用的 `--speed`。
7. 有没有任何地方离料桶、料斗、身体太近（目测或拍照）。
