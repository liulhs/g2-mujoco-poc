"""Record a 1080p60 demo of the validated POC with debug overlays.

Physics replay identical to replay.py (same servos, timing, plan), rendered with:
- in-scene debug geometry: nearest robot<->workcell distance (coloured by margin),
  nearest self-distance, TCP trail, TCP frame, waypoint markers, contact points/forces;
- HUD: current step (7-step script), time, gripper state, live clearances, tracking error,
  torso tilt, contact count;
- labels on the workcell and a top-down mini-map (layout inches) showing where everything is.

    G2_ARM=crs .venv/bin/python record_demo.py           # bimanual cup-carry-pour, chassis driving
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("G2_ARM", "crs")
os.environ.setdefault("FRAME_SLAB_IN", "0")

import imageio.v2 as imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import g2_model  # noqa: E402
import ik  # noqa: E402
import replay  # noqa: E402

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
W, H, FPS = 1920, 1080, 60
INTRO_S, OUTRO_S = 5.0, 2.5
TRAIL_LEN = 240
IN = g2_model.IN

FONT_CJK = "/System/Library/Fonts/Hiragino Sans GB.ttc"
FONT_MONO = "/System/Library/Fonts/Menlo.ttc"

STEP_TEXT = {  # name: (stage, zh, en)   6 stages
    "START": ("0", "起点：双手合拢在胸前", "Start: hands cupped together"),
    "TURN_TO_BIN": ("1", "原地转向料桶", "Turn in place toward the bin"),
    "DRIVE_TO_BIN": ("1", "开到料桶前", "Drive to the source bin"),
    "PRE_SCOOP": ("2", "双手移到桶口上方", "Both hands above the bin"),
    "SCOOP_IN": ("2", "双手伸进桶口", "Dip both hands in"),
    "SCOOP_SWEEP": ("2", "向前捧起", "Scoop forward"),
    "SCOOP_LIFT": ("2", "捧着抬起", "Lift the handful"),
    "CARRY": ("3", "双手端平，准备移动", "Hold level, ready to move"),
    "BACK_OFF": ("3", "端着后退", "Back off, holding"),
    "TURN_LEFT": ("3", "端着原地左转", "Turn left in place, holding"),
    "DRIVE_NORTH": ("3", "端着向前开", "Drive forward, holding"),
    "TURN_TO_HOPPER": ("3", "端着转向料斗", "Turn to face the hopper, holding"),
    "DRIVE_TO_HOPPER": ("3", "端着开到料斗", "Drive to the hopper, holding"),
    "RAISE": ("4", "双手抬过料斗沿", "Raise the hands above the rim"),
    "DOCK": ("4", "靠近料斗", "Move up to the hopper"),
    "OVER_HOPPER": ("4", "双手伸到料斗上方", "Reach over the hopper"),
    "POUR": ("5", "双手翻腕倒出", "Tip both hands: pour"),
    "UNPOUR": ("5", "双手放平", "Level the hands"),
    "RETRACT": ("6", "收回双手", "Pull the hands back"),
    "UNDOCK": ("6", "离开料斗", "Back away from the hopper"),
    "LOWER": ("6", "放低到端持姿势", "Lower to the carry posture"),
    "TURN_HOME": ("6", "原地转向", "Turn in place"),
    "DRIVE_SOUTH": ("6", "向前开", "Drive forward"),
    "TURN_WEST": ("6", "原地转向起点", "Turn toward the start"),
    "DRIVE_HOME": ("6", "开回起点", "Drive back to the start"),
    "TURN_FINAL": ("6", "转回起始朝向", "Turn to the start heading"),
}
STAGES = 6

CAM = {  # lookat, distance, azimuth, elevation ("follow" lookat tracks the chassis)
    "overview": ((2.45, 2.10, 0.70), 5.6, -90.0, -42.0),
    "follow": ((0.0, 0.0, 0.85), 3.4, -60.0, -26.0),
    "bin": ((3.15, 1.62, 0.85), 2.3, -120.0, -20.0),
    "hopper": ((1.80, 2.68, 1.25), 2.5, -125.0, -14.0),
}
CAM_FOR = {"0": "overview", "1": "follow", "2": "bin", "3": "follow", "4": "hopper", "5": "hopper", "6": "follow"}


def build_visual_model() -> tuple[mujoco.MjModel, list[str]]:
    g2_model.build_model()
    spec = mujoco.MjSpec.from_string(Path(g2_model.BUILD / "g2_workcell.xml").read_text())
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    for b in spec.bodies:
        if b.name != "world":
            b.gravcomp = 1.0
    for j in spec.joints:
        if j.name.startswith("idx"):
            j.armature = 0.05 if "gripper" in j.name else 0.2
            j.damping = [0.05 if "gripper" in j.name else 1.0, 0, 0]
    joints = ik.BASE + ik.BODY + ik.ARM_R + ik.ARM_L + ik.HEAD + [ik.GRIP_R, ik.GRIP_L]
    replay.add_servos(spec, joints)
    sky = spec.add_texture()
    sky.name, sky.type, sky.builtin = "sky", mujoco.mjtTexture.mjTEXTURE_SKYBOX, mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
    sky.rgb1, sky.rgb2, sky.width, sky.height = [0.42, 0.50, 0.60], [0.08, 0.10, 0.13], 512, 3072
    grid = spec.add_texture()
    grid.name, grid.type, grid.builtin = "grid", mujoco.mjtTexture.mjTEXTURE_2D, mujoco.mjtBuiltin.mjBUILTIN_CHECKER
    grid.rgb1, grid.rgb2, grid.width, grid.height = [0.80, 0.82, 0.85], [0.72, 0.74, 0.77], 512, 512
    grid.mark, grid.markrgb = mujoco.mjtMark.mjMARK_EDGE, [0.55, 0.58, 0.62]
    mat = spec.add_material()
    mat.name = "floor_grid"
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    mat.texrepeat = [1 / 0.6096, 1 / 0.6096]  # one checker pair = 2 ft -> each square is 1 ft
    mat.texuniform = True
    mat.reflectance = 0.05
    floor = spec.geom("floor")
    floor.material = "floor_grid"
    floor.rgba = [1, 1, 1, 1]
    model = spec.compile()
    model.vis.global_.offwidth, model.vis.global_.offheight = W, H
    model.vis.quality.offsamples = 8
    model.vis.quality.shadowsize = 8192
    model.vis.headlight.ambient[:] = [0.35, 0.35, 0.35]
    model.vis.headlight.diffuse[:] = [0.45, 0.45, 0.45]
    model.vis.map.znear = 0.005
    return model, joints


# ---------- in-scene debug geometry ----------------------------------------------
def _add_geom(scn: mujoco.MjvScene) -> mujoco.MjvGeom | None:
    if scn.ngeom >= scn.maxgeom:
        return None
    g = scn.geoms[scn.ngeom]
    scn.ngeom += 1
    return g


def add_sphere(scn, pos, r, rgba) -> None:
    g = _add_geom(scn)
    if g is not None:
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([r, 0, 0]), np.asarray(pos, float),
                            np.eye(3).flatten(), np.asarray(rgba, np.float32))


def add_line(scn, a, b, width, rgba) -> None:
    g = _add_geom(scn)
    if g is not None:
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9),
                            np.asarray(rgba, np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, np.asarray(a, float), np.asarray(b, float))


def margin_colour(d: float) -> tuple[float, float, float, float]:
    if d >= ik.ENV_CLEARANCE + 0.03:
        return (0.10, 0.85, 0.35, 1.0)
    if d >= ik.ENV_CLEARANCE:
        return (0.98, 0.80, 0.10, 1.0)
    return (0.95, 0.15, 0.15, 1.0)


# ---------- projection for labels ------------------------------------------------
def project(scn: mujoco.MjvScene, p: np.ndarray) -> tuple[float, float] | None:
    cam = scn.camera[0]
    pos, fwd, up = np.array(cam.pos), np.array(cam.forward), np.array(cam.up)
    right = np.cross(fwd, up)
    v = np.asarray(p) - pos
    z = v @ fwd
    if z <= 1e-3:
        return None
    x, y = (v @ right) / z * cam.frustum_near, (v @ up) / z * cam.frustum_near
    half_w = cam.frustum_top * W / H  # frustum is defined by height; width follows the aspect ratio
    u = (x + half_w) / (2 * half_w) * W
    w_ = (cam.frustum_top - y) / (cam.frustum_top - cam.frustum_bottom) * H
    return float(u), float(w_)


@dataclass
class Fonts:
    big: ImageFont.FreeTypeFont
    mid: ImageFont.FreeTypeFont
    small: ImageFont.FreeTypeFont
    mono: ImageFont.FreeTypeFont
    label: ImageFont.FreeTypeFont


def load_fonts() -> Fonts:
    return Fonts(ImageFont.truetype(FONT_CJK, 40), ImageFont.truetype(FONT_CJK, 28),
                 ImageFont.truetype(FONT_CJK, 22), ImageFont.truetype(FONT_MONO, 21),
                 ImageFont.truetype(FONT_CJK, 24))


def panel(draw: ImageDraw.ImageDraw, box, alpha: int = 170) -> None:
    draw.rounded_rectangle(box, radius=14, fill=(12, 16, 22, alpha), outline=(255, 255, 255, 40), width=1)


HUD_BOXES = [(24, 24, 1060, 210), (24, 1080 - 330, 820, 1080 - 24), (844, 1080 - 140, 1500, 1080 - 24),
             (1460, 0, 1920, 420)]


def _overlap(a, b) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def tag(draw, xy, text, font, placed: list, colour=(255, 255, 255, 255), dot=(255, 200, 40, 255)) -> None:
    x, y = xy
    if not (0 <= x <= W and 0 <= y <= H) or any(x0 <= x <= x1 and y0 <= y <= y1 for x0, y0, x1, y1 in HUD_BOXES):
        return
    tw = draw.textlength(text, font=font)
    for dy in (-40, 16, -80, 56, -120, 96):
        bx, by = x + 14, y + dy
        box = (bx - 8, by - 4, bx + tw + 8, by + 30)
        if box[2] > W - 4:
            bx, box = x - 14 - tw, (x - 22 - tw, by - 4, x - 6, by + 30)
        if not any(_overlap(box, p) for p in placed + HUD_BOXES):
            break
    else:
        return
    placed.append(box)
    draw.ellipse([x - 6, y - 6, x + 6, y + 6], fill=dot, outline=(0, 0, 0, 255))
    draw.line([x, y, box[0], (box[1] + box[3]) / 2], fill=(255, 255, 255, 150), width=2)
    draw.rounded_rectangle(box, radius=8, fill=(12, 16, 22, 200))
    draw.text((bx, by), text, font=font, fill=colour)


# ---------- top-down mini map (layout inches) ------------------------------------
MAP_X0, MAP_Y0, MAP_S = W - 40 - 192 * 2.0, 40, 2.0  # 2 px per inch -> 384 x 320


def m2px(x_in: float, y_in: float) -> tuple[float, float]:
    return MAP_X0 + x_in * MAP_S, MAP_Y0 + (160 - y_in) * MAP_S


def draw_minimap(draw, fonts: Fonts, base_xy_yaw, tcp_trail_in, tcp_in) -> None:
    x0, y0 = m2px(0, 160)
    x1, y1 = m2px(192, 0)
    panel(draw, [x0 - 14, y0 - 44, x1 + 14, y1 + 40])
    draw.text((x0, y0 - 38), "俯视图 Top view (in)", font=fonts.small, fill=(255, 255, 255, 230))
    draw.rectangle([x0, y0, x1, y1], outline=(200, 200, 200, 200), width=2)
    hb = g2_model.HOPPER_BASE
    draw.rectangle([*m2px(hb["x0"], hb["y0"] + hb["d"]), *m2px(hb["x0"] + hb["w"], hb["y0"])],
                   fill=(120, 130, 140, 200))
    hp = g2_model.HOPPER
    cx, cy = m2px(hp["cx"], hp["cy"])
    r = (hp["inner_r"] + hp["wall"]) * MAP_S
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(80, 140, 230, 255), width=3)
    draw.rectangle([*m2px(hb["x0"], hb["y0"] + hb["d"]), *m2px(hb["x0"] + hb["w"], hb["y0"])],
                   outline=(240, 150, 60, 255), width=2)  # frame footprint
    sb = g2_model.SOURCE_BIN
    draw.rectangle([*m2px(sb["x0"], sb["y0"] + sb["d"]), *m2px(sb["x0"] + sb["w"], sb["y0"])],
                   outline=(80, 140, 230, 255), width=3)
    draw.text(m2px(sb["x0"] + 4, sb["y0"] + 30), "料桶", font=fonts.small, fill=(150, 190, 255, 255))
    draw.text(m2px(hb["x0"] + 6, hb["y0"] - 4), "料斗", font=fonts.small, fill=(150, 190, 255, 255))
    # robot footprint (760 L x 640 W mm), oriented by yaw
    bx, by, yaw = base_xy_yaw
    L, Wd = 760 / 25.4 / 2, 640 / 25.4 / 2
    c, s = np.cos(np.deg2rad(yaw)), np.sin(np.deg2rad(yaw))
    corners = [(bx + c * dx - s * dy, by + s * dx + c * dy) for dx, dy in ((L, Wd), (L, -Wd), (-L, -Wd), (-L, Wd))]
    draw.polygon([m2px(*p) for p in corners], fill=(220, 220, 225, 220), outline=(255, 255, 255, 255))
    fx, fy = m2px(bx + c * L * 0.8, by + s * L * 0.8)
    draw.ellipse([fx - 4, fy - 4, fx + 4, fy + 4], fill=(40, 40, 40, 255))
    draw.text(m2px(bx - 16, by - 18), "G2", font=fonts.small, fill=(255, 255, 255, 255))
    if len(tcp_trail_in) > 1:
        draw.line([m2px(*p) for p in tcp_trail_in], fill=(120, 230, 255, 200), width=3)  # chassis path
    tx, ty = m2px(*tcp_in)
    draw.ellipse([tx - 6, ty - 6, tx + 6, ty + 6], fill=(255, 220, 60, 255), outline=(0, 0, 0, 255))
    draw.text((x0, y1 + 8), f"G2 base ({bx:.1f}, {by:.1f}) in, yaw {yaw:.0f}°", font=fonts.small,
              fill=(230, 230, 230, 255))


@dataclass
class Debug:
    t: float
    step: str
    gripper: float
    env_d: float
    env_pair: str
    env_min_run: float
    self_d: float
    self_pair: str
    track_err: float
    tilt_deg: float
    contacts: int
    tcp_z: float
    base: tuple[float, float, float] = (0.0, 0.0, 0.0)


def draw_hud(img: Image.Image, fonts: Fonts, dbg: Debug, labels: list[tuple], base_xy_yaw, trail_in,
             tcp_in, intro: float | None) -> Image.Image:
    over = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    placed: list = []
    for xy, text, colour in labels:
        if xy is not None:
            tag(d, xy, text, fonts.label, placed, dot=colour)
    # top-left: title + step
    panel(d, [24, 24, 1060, 210])
    d.text((44, 34), "G2 Scoop & Dump POC · MuJoCo", font=fonts.big, fill=(255, 255, 255, 255))
    d.text((44, 88), "官方模型 AgibotTech/genie_sim_robot_model @fb6de33 · crs 手臂 + Omnipicker",
           font=fonts.small, fill=(190, 200, 210, 255))
    base = dbg.step.split("~")[0]
    num, zh, en = STEP_TEXT.get(base, ("", base, base))
    via = "（避障中间点）" if "~" in dbg.step else ""
    d.text((44, 124), f"阶段 {num}/{STAGES}  {zh}{via}", font=fonts.mid, fill=(255, 220, 90, 255))
    d.text((44, 164), en, font=fonts.small, fill=(220, 220, 220, 255))
    # bottom-left: debug panel
    y0 = H - 330
    panel(d, [24, y0, 820, H - 24])
    d.text((44, y0 + 12), "DEBUG", font=fonts.mid, fill=(120, 220, 255, 255))
    env_col = tuple(int(255 * c) for c in margin_colour(dbg.env_d)[:3]) + (255,)
    rows = [
        (f"t = {dbg.t:6.2f} s    joint speed cap 0.30 rad/s", (230, 230, 230, 255)),
        (f"grippers L+R {'OPEN -0.785' if dbg.gripper < -0.4 else 'CLOSED 0.000'}   chassis ({dbg.base[0]:5.1f},{dbg.base[1]:5.1f}) in {dbg.base[2]:4.0f}°", (230, 230, 230, 255)),
        (f"env clearance {dbg.env_d * 100:6.1f} cm  (run min "
         f"{'  -- ' if dbg.env_min_run >= 1.0 else f'{dbg.env_min_run * 100:4.1f}'} cm, need >=5)", env_col),
        (f"  nearest: {dbg.env_pair}", (190, 190, 190, 255)),
        (f"self clearance {dbg.self_d * 100:5.1f} cm", (200, 170, 255, 255)),
        (f"  nearest: {dbg.self_pair}", (190, 190, 190, 255)),
        (f"tracking err {dbg.track_err * 1000:5.1f} mrad   torso tilt {dbg.tilt_deg:4.1f}°", (230, 230, 230, 255)),
        (f"contacts robot-workcell: {dbg.contacts}    hands z {dbg.tcp_z:5.3f} m", (230, 230, 230, 255)),
    ]
    for i, (text, colour) in enumerate(rows):
        d.text((44, y0 + 54 + i * 32), text, font=fonts.mono, fill=colour)
    # legend
    lx, ly = 860, H - 128
    panel(d, [lx - 16, ly - 12, lx + 640, H - 24])
    for i, (c, t) in enumerate((((25, 215, 90), "绿/黄/红：机器人到工位最近距离（≥8 / ≥5 / <5 cm）"),
                                ((170, 120, 255), "紫：手臂到自身最近距离"),
                                ((255, 220, 60), "黄 / 青线：右手 / 左手夹爪中心轨迹"))):
        d.rectangle([lx, ly + 6 + i * 30, lx + 26, ly + 16 + i * 30], fill=c + (255,))
        d.text((lx + 38, ly + i * 30), t, font=fonts.small, fill=(230, 230, 230, 255))
    draw_minimap(d, fonts, base_xy_yaw, trail_in, tcp_in)
    if intro is not None:
        a = int(255 * min(1.0, intro))
        panel(d, [24, 226, 1060, 330], alpha=int(200 * min(1.0, intro)))
        d.text((44, 236), "开到料桶 → 双手捧起 → 端着开到料斗 → 双手倒出 → 回起点", font=fonts.mid,
               fill=(255, 255, 255, a))
        d.text((44, 282), "Layout: drawing G1-LAY-001 · bin 30 in (assumed) · no overhead frame at the hopper (assumed)",
               font=fonts.small, fill=(220, 220, 220, a))
    return Image.alpha_composite(img.convert("RGBA"), over).convert("RGB")


def blend_cam(cur: mujoco.MjvCamera, preset, k: float) -> None:
    lookat, dist, az, el = preset
    cur.lookat[:] = cur.lookat + k * (np.array(lookat) - cur.lookat)
    cur.distance += k * (dist - cur.distance)
    daz = (az - cur.azimuth + 180) % 360 - 180
    cur.azimuth += k * daz
    cur.elevation += k * (el - cur.elevation)


def main() -> int:
    plan = OUT / os.environ.get("PLAN_FILE", "bimanual_plan.json")
    doc = json.loads(plan.read_text())
    wps = doc["waypoints"]
    model, joints = build_visual_model()
    data = mujoco.MjData(model)
    chk = ik.Checker(model)
    env_pairs, self_pairs = chk.env_pairs(), chk.self_pairs
    qadr = [model.jnt_qposadr[model.joint(n).id] for n in joints]
    data.qpos[qadr] = [wps[0]["joints"][n] for n in joints]
    data.ctrl[:] = data.qpos[qadr]
    mujoco.mj_forward(model, data)
    keys = replay.schedule(wps, joints)
    t_motion = keys[-1][0] + 0.5
    arm_idx = [i for i, n in enumerate(joints) if "gripper" not in n and n not in ik.BASE]
    base_adr = [model.jnt_qposadr[model.joint(n).id] for n in ik.BASE]
    env_geoms = {g for g in range(model.ngeom) if model.geom_bodyid[g] == 0}
    tcp_site = model.site("tcp_r").id
    tcp_site_l = model.site("tcp_l").id
    grip_adr = model.jnt_qposadr[model.joint(ik.GRIP_R).id]

    fonts = load_fonts()
    renderer = mujoco.Renderer(model, height=H, width=W, max_geom=20000)
    opt = mujoco.MjvOption()
    opt.frame = mujoco.mjtFrame.mjFRAME_SITE
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    model.vis.scale.framelength = 0.6
    model.vis.scale.framewidth = 0.08
    cam = mujoco.MjvCamera()
    lookat, dist, az, el = CAM["overview"]
    cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = lookat, dist, az - 70, el

    marks = [(w["name"], (np.array(w["tcp_l_world_m"]) + np.array(w["tcp_r_world_m"])) / 2) for w in wps
             if "~" not in w["name"] and w["name"] in ("SCOOP_IN", "SCOOP_LIFT", "OVER_HOPPER", "POUR")]
    hb, hp, sb = g2_model.HOPPER_BASE, g2_model.HOPPER, g2_model.SOURCE_BIN
    static_labels = [
        (np.array([(sb["x0"] + sb["w"] / 2) * IN, (sb["y0"] + sb["d"] / 2) * IN, sb["h"] * IN]),
         f"料桶 Source bin · 高 {sb['h']:.0f} in（假设）", (80, 140, 230, 255)),
        (np.array([hp["cx"] * IN, (hp["cy"] + hp["inner_r"]) * IN, hp["rim"] * IN]),
         f"料斗 Hopper · 沿高 {hp['rim']} in", (80, 140, 230, 255)),
    ]
    if g2_model.FRAME_SLAB_IN > 0:
        static_labels.append((np.array([(hb["x0"] + 4) * IN, (hb["y0"] + hb["d"] - 4) * IN,
                                        (g2_model.FRAME_TOP_IN - g2_model.FRAME_SLAB_IN) * IN]),
                              f"框架 Frame · 梁底 {g2_model.FRAME_TOP_IN - g2_model.FRAME_SLAB_IN:.0f} in（假设）",
                              (240, 150, 60, 255)))
    base_trail: list[tuple[float, float]] = []

    out_path = OUT / "g2_bimanual_demo_1080p.mp4"
    writer = imageio.get_writer(out_path, fps=FPS, codec="libx264", quality=None, macro_block_size=1,
                                ffmpeg_params=["-crf", "15", "-preset", "slow", "-pix_fmt", "yuv420p",
                                               "-profile:v", "high", "-movflags", "+faststart"])
    trail: list[np.ndarray] = []
    trail_l: list[np.ndarray] = []
    env_min_run = 1.0
    total_t = INTRO_S + t_motion + OUTRO_S
    n_frames = int(total_t * FPS)
    frame_dt = 1.0 / FPS
    for f in range(n_frames):
        t_video = f * frame_dt
        t_sim = min(max(0.0, t_video - INTRO_S), t_motion)
        q_ref, seg = replay.target_at(keys, t_sim)
        while data.time < t_sim:
            data.ctrl[:] = q_ref
            mujoco.mj_step(model, data)
            q_ref, seg = replay.target_at(keys, data.time)
        if t_video < INTRO_S:
            seg = "START"
        # debug measurements on the live state
        chk.data.qpos[:] = data.qpos
        mujoco.mj_kinematics(model, chk.data)
        fromto_env = np.zeros(6)
        d_env, (ea, eb) = chk.min_distance(env_pairs, 0.5)
        if ea >= 0:
            mujoco.mj_geomDistance(model, chk.data, ea, eb, 0.5, fromto_env)
        d_self, (sa, sbb) = chk.min_distance(self_pairs, 0.5)
        fromto_self = np.zeros(6)
        if sa >= 0:
            mujoco.mj_geomDistance(model, chk.data, sa, sbb, 0.5, fromto_self)
        if t_video >= INTRO_S:
            env_min_run = min(env_min_run, d_env)
        contacts = sum(1 for c in data.contact[: data.ncon]
                       if (int(c.geom1) in env_geoms) != (int(c.geom2) in env_geoms)
                       and model.geom(int(c.geom1 if int(c.geom1) in env_geoms else c.geom2)).name != "floor")
        tcp = data.site_xpos[tcp_site].copy()
        tcp_l = data.site_xpos[tcp_site_l].copy()
        trail.append(tcp)
        trail_l.append(tcp_l)
        trail, trail_l = trail[-TRAIL_LEN:], trail_l[-TRAIL_LEN:]
        bx, by, byaw = (float(v) for v in data.qpos[base_adr])
        base_trail.append((bx / IN, by / IN))
        b5 = model.body("body_link5").id
        tilt = float(np.degrees(np.arccos(np.clip(data.xmat[b5].reshape(3, 3)[2, 2], -1, 1))))

        # camera
        if t_video < INTRO_S:
            s = t_video / INTRO_S
            cam.lookat[:] = CAM["overview"][0]
            cam.distance = CAM["overview"][1] + 0.4 * (1 - s)
            cam.azimuth = CAM["overview"][2] - 70 * (1 - s) ** 2
            cam.elevation = CAM["overview"][3] - 8 * (1 - s)
        else:
            stage = STEP_TEXT.get(seg.split("~")[0], ("0",))[0]
            key = CAM_FOR.get(stage, "overview")
            preset = CAM[key]
            if key == "follow":
                preset = ((bx, by, 0.85), *preset[1:])
            blend_cam(cam, preset, 0.03)

        renderer.update_scene(data, camera=cam, scene_option=opt)
        scn = renderer.scene
        for i in range(1, len(trail), 2):
            add_line(scn, trail[i - 1], trail[i], 0.004, (1.0, 0.86, 0.24, 0.9))
            add_line(scn, trail_l[i - 1], trail_l[i], 0.004, (0.35, 0.90, 1.0, 0.9))
        for _, p in marks:
            add_sphere(scn, p, 0.018, (1.0, 0.45, 0.15, 0.55))
        if ea >= 0 and d_env < 0.5:
            col = margin_colour(d_env)
            add_line(scn, fromto_env[:3], fromto_env[3:], 0.006, col)
            add_sphere(scn, fromto_env[:3], 0.012, col)
            add_sphere(scn, fromto_env[3:], 0.012, col)
        if sa >= 0 and d_self < 0.3:
            add_line(scn, fromto_self[:3], fromto_self[3:], 0.004, (0.67, 0.47, 1.0, 0.9))
        img = Image.fromarray(renderer.render())

        labels = [(project(scn, tcp), f"右手 z={tcp[2]:.2f} m", (255, 220, 60, 255)),
                  (project(scn, tcp_l), "左手", (90, 230, 255, 255))]
        for name, p in marks:
            if name == seg.split("~")[0]:
                labels.append((project(scn, p), "目标点 " + STEP_TEXT[name][1].split("，")[0], (255, 120, 40, 255)))
        labels += [(project(scn, p), t, c) for p, t, c in static_labels]
        dbg = Debug(t=max(0.0, t_video - INTRO_S), step=seg, gripper=float(data.qpos[grip_adr]),
                    env_d=d_env, env_pair=(f"{model.body(model.geom_bodyid[ea]).name} ↔ {model.geom(eb).name}"
                                          if ea >= 0 else "-"),
                    env_min_run=env_min_run, self_d=d_self,
                    self_pair=(f"{model.body(model.geom_bodyid[sa]).name}↔{model.body(model.geom_bodyid[sbb]).name}"
                               if sa >= 0 else ""),
                    track_err=float(np.max(np.abs(data.qpos[qadr][arm_idx] - q_ref[arm_idx]))),
                    tilt_deg=tilt, contacts=contacts, tcp_z=float((tcp[2] + tcp_l[2]) / 2),
                    base=(bx / IN, by / IN, float(np.degrees(byaw))))
        trail_in = base_trail[::6]
        intro = (1.0 - max(0.0, t_video - (INTRO_S - 1.0))) if t_video < INTRO_S else None
        frame = draw_hud(img, fonts, dbg, labels, (bx / IN, by / IN, float(np.degrees(byaw))), trail_in,
                         (((tcp[0] + tcp_l[0]) / 2) / IN, ((tcp[1] + tcp_l[1]) / 2) / IN), intro)
        writer.append_data(np.asarray(frame))
        if f % (FPS * 5) == 0:
            print(f"frame {f}/{n_frames}  t={t_video:5.1f}s  step={seg}  env={d_env:.3f}", flush=True)
            if os.environ.get("PREVIEW_AT") is None:
                pass
        if os.environ.get("PREVIEW_AT") and f in {int(float(x) * FPS) for x in os.environ["PREVIEW_AT"].split(",")}:
            frame.save(OUT / f"demo_preview_{f:05d}.png")
        if os.environ.get("MAX_FRAMES") and f >= int(os.environ["MAX_FRAMES"]):
            break
    writer.close()
    renderer.close()
    print("wrote", out_path, f"{n_frames} frames, run min env clearance {env_min_run:.4f} m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
