"""End-to-end fused-scene visualization.

Pulls one frame per camera, fuses to a world-frame PCD, optionally overlays
caller-supplied robot meshes, and shows the result in Rerun (default) with
Open3D fallback.
"""
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

import numpy as np

from cam_calib.adapters.base import CameraSource
from cam_calib.core.extrinsics_io import load_cam_extrinsics
from cam_calib.viz.pointcloud import aggregate_world_pointcloud


PathLike = Union[str, Path]


def fuse_and_show(
    cameras: Iterable[CameraSource],
    extrinsics_dir: PathLike,
    *,
    depths_per_camera: dict,
    extra_pointclouds: Iterable[Tuple[str, np.ndarray]] = (),
    use_rerun: bool = True,
    save_rrd_dir: Optional[PathLike] = None,
    boundaries: Optional[dict] = None,
    voxel_size: Optional[float] = None,
) -> None:
    """Fuse RGB-D from ``cameras`` into a world PCD and display it.

    ``depths_per_camera`` is a ``{serial: (H, W) depth in meters}`` dict
    supplied by the caller — depth source (hardware, FoundationStereo, etc.)
    is intentionally outside this package's scope.
    """
    cams = list(cameras)
    if not cams:
        raise ValueError("no cameras provided")

    serials = []
    colors = []
    Ks = []
    extr = []
    images_bgr = []
    for cam in cams:
        frame = cam.get_frame()
        T_cam_world = load_cam_extrinsics(frame.serial, extrinsics_dir)
        if T_cam_world is None:
            raise FileNotFoundError(
                f"no extrinsics yaml for {frame.serial} in {extrinsics_dir}"
            )
        if frame.serial not in depths_per_camera:
            raise KeyError(f"depths_per_camera missing entry for {frame.serial}")
        serials.append(frame.serial)
        # OpenCV is BGR; aggregate_world_pointcloud is colorspace-agnostic but
        # caller will likely want RGB in viz — flip channel order.
        rgb = frame.image[..., ::-1]
        colors.append(rgb)
        images_bgr.append(frame.image)
        Ks.append(frame.K)
        extr.append(T_cam_world)

    colors_arr = np.stack(colors)
    depths_arr = np.stack([depths_per_camera[s] for s in serials])
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

    if use_rerun:
        try:
            from cam_calib.viz.rerun_viz import show_world_pointcloud as rr_show
            cameras_zip = list(zip(extr_arr, Ks_arr, [c[..., ::-1] for c in colors]))
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
