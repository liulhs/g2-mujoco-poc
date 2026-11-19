"""Build the G2 + workcell MuJoCo model from the official URDF.

All layout numbers come from drawing G1-LAY-001 (inches, origin = lower-left
inside corner of the room, x right, y up). They are converted to metres here.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
# Official AgiBot description: github.com/AgibotTech/genie_sim_robot_model (Apache-2.0),
# pinned commit below; URDF expanded from robots/genie/g2/g2_crs_omnipicker.urdf.xacro
# with the official collision_type="fine" option (convex STL hulls).
OFFICIAL_REPO = ROOT / "official" / "genie_sim_robot_model"
OFFICIAL_COMMIT = "fb6de335f6a293605204a17608cf0b22b1fef6de"
BUILD = ROOT / "build"
ARM = __import__("os").environ.get("G2_ARM", "crs")          # "crs" or "crsB": must match the real robot
if ARM not in ("crs", "crsB"):
    raise ValueError(f"G2_ARM must be crs or crsB, got {ARM!r}")
URDF_SRC = BUILD / "official" / f"g2_{ARM}_omnipicker_fine.urdf"
# Official TCP note: the URDF no longer has a gripper centre link and asks downstream
# code to offset from gripper_<side>_base_link; Genie Sim configs place the grasp
# centre ~0.13 m past the end-effector mount, i.e. ~0.131 m from gripper_r_base_link.
TCP_OFFSET_FROM_GRIPPER_BASE = 0.131

IN = 0.0254  # metres per inch

# ---- layout (inches) --------------------------------------------------------
ROBOT_BASE_IN = tuple(float(v) for v in __import__("os").environ.get("G2_BASE_IN", "101.5,84").split(","))  # chassis centre
ROBOT_YAW_DEG = float(__import__("os").environ.get("G2_YAW_DEG", "90"))  # 0: chassis +x faces the bin; 90: faces +y

HOPPER_BASE = dict(x0=30, y0=88, w=40, d=35, h=34)   # stainless base
HOPPER = dict(cx=50, cy=105.5, inner_r=19, wall=0.5, rim=50, depth=13)
FRAME_TOP_IN = 63.0                # frame-top height, still to be measured
# Frame over the hopper: geometry unmeasured. Default 0 = nothing overhead on the robot's
# approach side (the bimanual pour needs that). FRAME_SLAB_IN=1 models a solid ceiling
# at 62-63 in over the hopper base, under which the bimanual pour does NOT keep 5 cm.
FRAME_SLAB_IN = float(__import__("os").environ.get("FRAME_SLAB_IN", "0"))
BIN_HEIGHT_IN = float(__import__("os").environ.get("BIN_HEIGHT_IN", "30"))
SOURCE_BIN = dict(x0=132, y0=40, w=48, d=48, h=BIN_HEIGHT_IN, wall=0.5)  # height TBD

# joints that stay fixed in the POC (chassis does not move)
LOCKED_PATTERNS = ("wheel", "steer")


def ensure_official_urdf() -> Path:
    """Expand the official xacro (collision_type="fine") for the selected arm, once."""
    if URDF_SRC.exists():
        return URDF_SRC
    import shutil
    import subprocess
    if not (OFFICIAL_REPO / "robots").exists():  # fresh checkout: fetch the pinned official model
        OFFICIAL_REPO.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "https://github.com/AgibotTech/genie_sim_robot_model.git",
                        str(OFFICIAL_REPO)], check=True)
        subprocess.run(["git", "-C", str(OFFICIAL_REPO), "checkout", "--quiet", OFFICIAL_COMMIT], check=True)
    src = BUILD / "xacro_src"
    if src.exists():
        shutil.rmtree(src)
    g2 = OFFICIAL_REPO / "robots" / "genie" / "g2"
    shutil.copytree(g2 / "xacro", src / "xacro")
    for f in g2.glob("*.urdf.xacro"):
        shutil.copy(f, src / f.name)
    for f in src.rglob("*.xacro"):
        f.write_text(f.read_text().replace("$(find genie_sim_robot_model)/robots/genie/g2", str(src)))
    top = src / f"g2_{ARM}_omnipicker.urdf.xacro"
    fine = src / f"g2_{ARM}_omnipicker_fine.urdf.xacro"
    fine.write_text(top.read_text().replace('collision_type="coarse"', 'collision_type="fine"'))
    URDF_SRC.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(ROOT / ".venv" / "bin" / "xacro"), str(fine), "-o", str(URDF_SRC)], check=True)
    return URDF_SRC


def _prepared_urdf() -> Path:
    """Resolve mesh paths, drop loop joints (gripper 4-bar closures), lock the chassis."""
    BUILD.mkdir(exist_ok=True)
    text = ensure_official_urdf().read_text()
    text = text.replace("package://genie_sim_robot_model/", f"{OFFICIAL_REPO}/")
    tree = ET.ElementTree(ET.fromstring(text))
    root = tree.getroot()
    for loop in root.findall("loop_joint"):
        root.remove(loop)
    for joint in root.findall("joint"):
        name = joint.get("name", "")
        if any(p in name for p in LOCKED_PATTERNS) and joint.get("type") in ("revolute", "continuous"):
            joint.set("type", "fixed")
            for tag in ("limit", "axis", "dynamics", "mimic"):
                el = joint.find(tag)
                if el is not None:
                    joint.remove(el)
    mj = ET.SubElement(root, "mujoco")
    ET.SubElement(mj, "compiler", dict(discardvisual="true", fusestatic="false",
                                       strippath="false", balanceinertia="true"))
    out = BUILD / f"g2_{ARM}_prepared.urdf"
    tree.write(out)
    return out


def _obj_for(dae: Path) -> Path:
    """Convert an official .dae visual mesh to .obj (geometry unchanged) for MuJoCo."""
    import trimesh

    out = BUILD / "visual_obj" / dae.relative_to(OFFICIAL_REPO).with_suffix(".obj")
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        mesh = trimesh.load(dae, force="mesh")
        mesh.export(out)
    return out


def _rpy_quat(rpy: list[float]) -> list[float]:
    from scipy.spatial.transform import Rotation

    x, y, z, w = Rotation.from_euler("xyz", rpy).as_quat()
    return [w, x, y, z]


def _add_official_visuals(spec: mujoco.MjSpec) -> None:
    """Attach the official visual meshes (non-colliding) to each link body."""
    root = ET.fromstring(URDF_SRC.read_text().replace("package://genie_sim_robot_model/", f"{OFFICIAL_REPO}/"))
    for link in root.findall("link"):
        try:
            body = spec.body(link.get("name"))
        except Exception:
            body = None
        if body is None:
            continue
        for i, vis in enumerate(link.findall("visual")):
            mesh_el = vis.find("geometry/mesh")
            if mesh_el is None:
                continue
            src = Path(mesh_el.get("filename"))
            path = _obj_for(src) if src.suffix.lower() == ".dae" else src
            mname = f"vis_{link.get('name')}_{i}"
            mesh = spec.add_mesh()
            mesh.name = mname
            mesh.file = str(path)
            scale = mesh_el.get("scale")
            if scale:
                mesh.scale = [float(v) for v in scale.split()]
            g = body.add_geom()
            g.type = mujoco.mjtGeom.mjGEOM_MESH
            g.meshname = mname
            o = vis.find("origin")
            if o is not None:
                g.pos = [float(v) for v in o.get("xyz", "0 0 0").split()]
                g.quat = _rpy_quat([float(v) for v in o.get("rpy", "0 0 0").split()])
            g.contype = 0
            g.conaffinity = 0
            g.group = 2
            g.density = 0
            g.mass = 0
            g.rgba = [0.9, 0.9, 0.92, 1] if "arm" in link.get("name") or "body" in link.get("name") else [0.75, 0.75, 0.78, 1]


def _box(body, name, center, half, rgba, group=1):
    g = body.add_geom()
    g.name = name
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size = half
    g.pos = center
    g.rgba = rgba
    g.group = group
    g.contype = 2
    g.conaffinity = 1
    return g


def _add_workcell(spec: mujoco.MjSpec) -> None:
    wb = spec.worldbody
    floor = wb.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [6, 6, 0.1]
    floor.pos = [192 * IN / 2, 160 * IN / 2, 0]
    floor.rgba = [0.82, 0.84, 0.86, 1]
    floor.contype = 0      # visual only: the chassis is driven kinematically, wheels do not touch
    floor.conaffinity = 0

    steel = [0.62, 0.66, 0.70, 1]
    hb = HOPPER_BASE
    _box(wb, "hopper_base",
         [(hb["x0"] + hb["w"] / 2) * IN, (hb["y0"] + hb["d"] / 2) * IN, hb["h"] / 2 * IN],
         [hb["w"] / 2 * IN, hb["d"] / 2 * IN, hb["h"] / 2 * IN], steel)

    hp = HOPPER
    bottom_in = hp["rim"] - hp["depth"]
    # pedestal between the base top and the hopper bottom (treated as solid)
    _box(wb, "hopper_pedestal",
         [hp["cx"] * IN, hp["cy"] * IN, (hb["h"] + bottom_in) / 2 * IN],
         [10 * IN, 10 * IN, (bottom_in - hb["h"]) / 2 * IN], steel)
    floor_g = wb.add_geom()
    floor_g.name = "hopper_floor"
    floor_g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    floor_g.size = [(hp["inner_r"] + hp["wall"]) * IN, 0.25 * IN, 0]
    floor_g.pos = [hp["cx"] * IN, hp["cy"] * IN, (bottom_in + 0.25) * IN]
    floor_g.rgba = [0.25, 0.45, 0.75, 1]
    floor_g.contype = 2
    floor_g.conaffinity = 1
    n_seg = 36
    r_mid = (hp["inner_r"] + hp["wall"] / 2) * IN
    seg_half_len = np.pi * r_mid / n_seg * 1.05
    for i in range(n_seg):
        a = 2 * np.pi * i / n_seg
        g = _box(wb, f"hopper_wall_{i}",
                 [hp["cx"] * IN + r_mid * np.cos(a), hp["cy"] * IN + r_mid * np.sin(a),
                  (bottom_in + hp["depth"] / 2) * IN],
                 [hp["wall"] / 2 * IN, seg_half_len, hp["depth"] / 2 * IN],
                 [0.25, 0.45, 0.75, 1])
        g.quat = [np.cos(a / 2), 0, 0, np.sin(a / 2)]

    # conservative frame: a solid slab over the whole hopper-base footprint,
    # bottom face at FRAME_TOP - FRAME_SLAB (62 in)
    slab_bottom = FRAME_TOP_IN - FRAME_SLAB_IN
    if FRAME_SLAB_IN > 0:
        _box(wb, "hopper_frame_slab",
             [(hb["x0"] + hb["w"] / 2) * IN, (hb["y0"] + hb["d"] / 2) * IN,
              (slab_bottom + FRAME_SLAB_IN / 2) * IN],
             [hb["w"] / 2 * IN, hb["d"] / 2 * IN, FRAME_SLAB_IN / 2 * IN],
             [0.9, 0.55, 0.2, 0.45])

    sb = SOURCE_BIN
    cx, cy = (sb["x0"] + sb["w"] / 2) * IN, (sb["y0"] + sb["d"] / 2) * IN
    hx, hy, hz, t = sb["w"] / 2 * IN, sb["d"] / 2 * IN, sb["h"] / 2 * IN, sb["wall"] / 2 * IN
    blue = [0.20, 0.40, 0.80, 1]
    _box(wb, "bin_floor", [cx, cy, t], [hx, hy, t], blue)
    _box(wb, "bin_wall_w", [cx - hx + t, cy, hz], [t, hy, hz], blue)
    _box(wb, "bin_wall_e", [cx + hx - t, cy, hz], [t, hy, hz], blue)
    _box(wb, "bin_wall_s", [cx, cy - hy + t, hz], [hx, t, hz], blue)
    _box(wb, "bin_wall_n", [cx, cy + hy - t, hz], [hx, t, hz], blue)

    light = wb.add_light()
    light.pos = [2.4, 1.6, 4.0]
    light.dir = [0, 0, -1]
    light.diffuse = [0.8, 0.8, 0.8]


def _lowest_robot_point(model: mujoco.MjModel) -> float:
    """Lowest z of any robot collision/visual geometry (wheels), with base_link at z=0."""
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    low = np.inf
    for g in range(model.ngeom):
        if model.geom_bodyid[g] == 0:
            continue
        R, p = data.geom_xmat[g].reshape(3, 3), data.geom_xpos[g]
        t = model.geom_type[g]
        if t == mujoco.mjtGeom.mjGEOM_MESH:
            m = model.geom_dataid[g]
            v = model.mesh_vert[model.mesh_vertadr[m]: model.mesh_vertadr[m] + model.mesh_vertnum[m]]
            low = min(low, float(np.min((v @ R.T + p)[:, 2])))
        elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:
            r, h = model.geom_size[g][:2]
            az = abs(R[2, 2])
            low = min(low, float(p[2] - r * np.sqrt(max(0.0, 1 - az**2)) - h * az))
        elif t == mujoco.mjtGeom.mjGEOM_BOX:
            corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
            low = min(low, float(np.min((corners * model.geom_size[g]) @ R.T + p, axis=0)[2]))
        elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
            low = min(low, float(p[2] - model.geom_size[g][0]))
        else:
            low = min(low, float(p[2] - model.geom_rbound[g]))
    return low


def build_model() -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(_prepared_urdf()))
    spec.compiler.discardvisual = False  # URDF visuals already dropped; keep our official visual meshes
    spec.option.timestep = 0.002
    spec.option.gravity = [0, 0, -9.81]
    base = spec.worldbody.first_body()
    base.pos = [0.0, 0.0, 0.0]
    # mobile base: planar x / y / yaw joints (the swerve chassis is driven as a whole)
    for name, jtype, axis in (("base_x", mujoco.mjtJoint.mjJNT_SLIDE, [1, 0, 0]),
                              ("base_y", mujoco.mjtJoint.mjJNT_SLIDE, [0, 1, 0]),
                              ("base_yaw", mujoco.mjtJoint.mjJNT_HINGE, [0, 0, 1])):
        j = base.add_joint()
        j.name, j.type, j.axis = name, jtype, axis
    for g in spec.geoms:
        g.contype = 1
        g.conaffinity = 2  # robot<->workcell only; self-collision is checked geometrically in ik.py
        g.group = 3        # official collision hulls: hidden in renders
        g.rgba = [0.85, 0.85, 0.88, 0.5]
    _add_official_visuals(spec)
    for side in ("r", "l"):
        site = spec.body(f"gripper_{side}_base_link").add_site()
        site.name = f"tcp_{side}"
        site.pos = [0, 0, TCP_OFFSET_FROM_GRIPPER_BASE]
        site.size = [0.01, 0, 0]
    # put the wheels on the floor (base_link origin sits above the wheel contact point)
    probe = spec.compile()
    base.pos = [0.0, 0.0, -_lowest_robot_point(probe)]
    _add_workcell(spec)

    vis = spec.visual
    vis.global_.offwidth = 1280
    vis.global_.offheight = 720
    model = spec.compile()
    (BUILD / "g2_workcell.xml").write_text(spec.to_xml())
    return model


def robot_hinge_joints(model: mujoco.MjModel) -> list[str]:
    names = []
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
            names.append(model.joint(j).name)
    return names


if __name__ == "__main__":
    m = build_model()
    print("bodies", m.nbody, "joints", m.njnt, "geoms", m.ngeom, "eq", m.neq)
    print([n for n in robot_hinge_joints(m) if re.match(r"idx", n)])
