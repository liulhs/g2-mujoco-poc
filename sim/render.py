"""Offscreen rendering helpers for the G2 workcell."""
from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np

VIEWS = {
    # name: (lookat xyz, distance, azimuth, elevation)
    "overview": ((2.45, 2.25, 0.9), 4.6, -60.0, -32.0),
    "top": ((2.45, 2.25, 0.5), 5.2, -90.0, -89.0),
    "hopper": ((1.5, 2.5, 1.35), 2.3, -20.0, -12.0),
    "bin": ((3.25, 2.1, 0.8), 2.4, -120.0, -25.0),
}


def make_camera(view: str) -> mujoco.MjvCamera:
    lookat, dist, az, el = VIEWS[view]
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.distance = dist
    cam.azimuth = az
    cam.elevation = el
    return cam


def snapshot(model: mujoco.MjModel, data: mujoco.MjData, view: str, path: Path,
             width: int = 1280, height: int = 720) -> Path:
    with mujoco.Renderer(model, height=height, width=width) as r:
        r.update_scene(data, camera=make_camera(view))
        img = r.render()
    iio.imwrite(path, img)
    return path


def grid(model: mujoco.MjModel, data: mujoco.MjData, path: Path,
         views: tuple[str, ...] = ("overview", "top", "hopper", "bin")) -> Path:
    tiles = []
    with mujoco.Renderer(model, height=360, width=640) as r:
        for v in views:
            r.update_scene(data, camera=make_camera(v))
            tiles.append(r.render().copy())
    top = np.concatenate(tiles[:2], axis=1)
    bottom = np.concatenate(tiles[2:4], axis=1)
    iio.imwrite(path, np.concatenate([top, bottom], axis=0))
    return path
