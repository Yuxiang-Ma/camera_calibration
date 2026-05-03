"""End-to-end fused-scene visualization.

Takes pre-captured ``CameraFrame``s (each with ``.depth`` set), looks up
saved extrinsics, fuses RGB-D into a world PCD, optionally overlays
caller-supplied geometries, and shows the result in Rerun (default) with
Open3D fallback.

Depth source is intentionally outside this package's scope — populate
``frame.depth`` yourself (hardware depth, FoundationStereo, anything else).
"""
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np

from cam_calib.adapters.base import CameraFrame
from cam_calib.core.extrinsics_io import load_cam_extrinsics
from cam_calib.viz.pointcloud import aggregate_world_pointcloud


PathLike = Union[str, Path]


def fuse_and_show(
    frames: List[CameraFrame],
    extrinsics_dir: PathLike,
    *,
    extra_pointclouds: Iterable[Tuple[str, np.ndarray]] = (),
    use_rerun: bool = True,
    save_rrd_dir: Optional[PathLike] = None,
    boundaries: Optional[dict] = None,
    voxel_size: Optional[float] = None,
) -> None:
    """Fuse RGB-D from ``frames`` into a world PCD and display it.

    Each frame must have:
      - ``image`` (BGR uint8)
      - ``K`` (3x3 intrinsics)
      - ``depth`` (float32 meters, aligned to color)
      - a saved ``<serial>.yaml`` in ``extrinsics_dir``
    """
    if not frames:
        raise ValueError("no frames provided")

    serials, colors_rgb, depths, Ks, extr = [], [], [], [], []
    for f in frames:
        T_cam_world = load_cam_extrinsics(f.serial, extrinsics_dir)
        if T_cam_world is None:
            raise FileNotFoundError(
                f"no extrinsics yaml for {f.serial} in {extrinsics_dir}; "
                f"run `cam-calib calibrate` first"
            )
        if f.depth is None:
            raise ValueError(
                f"frame for {f.serial} has no depth — adapters need "
                f"enable_depth=True"
            )
        serials.append(f.serial)
        colors_rgb.append(f.image[..., ::-1])  # BGR → RGB for viz
        depths.append(f.depth)
        Ks.append(f.K)
        extr.append(T_cam_world)

    colors_arr = np.stack(colors_rgb)
    depths_arr = np.stack(depths)
    Ks_arr = np.stack(Ks)
    extr_arr = np.stack(extr)

    positions, rgb = aggregate_world_pointcloud(
        colors_arr,
        depths_arr,
        Ks_arr,
        extr_arr,
        boundaries=boundaries,
        voxel_size=voxel_size,
    )
    print(f"fused {len(serials)} cameras → {positions.shape[0]} points")

    if use_rerun:
        try:
            from cam_calib.viz.rerun_viz import show_world_pointcloud as rr_show
            cameras_zip = list(zip(extr_arr, Ks_arr, colors_rgb))
            rr_show(
                positions,
                rgb,
                extra_pointclouds=extra_pointclouds,
                cameras=cameras_zip,
                save_rrd_dir=save_rrd_dir,
            )
            return
        except ImportError:
            print("rerun-sdk not installed; falling back to Open3D")

    from cam_calib.viz.open3d_viz import show_world_pointcloud as o3d_show
    o3d_show(positions, rgb, extra_pointclouds=extra_pointclouds)
