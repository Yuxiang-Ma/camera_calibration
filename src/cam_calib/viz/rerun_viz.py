"""Rerun viewer for fused calibration scenes.

Logs the world-frame point cloud, optional extra geometries (e.g. robot
meshes built by the caller), per-camera frustums, and saves an .rrd recording
for offline inspection.
"""
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

import numpy as np

try:
    import rerun as rr  # type: ignore
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "rerun-sdk not installed. Install with: pip install camera-calibration[viz]"
    ) from e

from cam_calib.core.geometry import invert_se3


PathLike = Union[str, Path]


def show_world_pointcloud(
    positions: np.ndarray,
    colors: Optional[np.ndarray] = None,
    *,
    extra_pointclouds: Iterable[Tuple[str, np.ndarray]] = (),
    cameras: Optional[Iterable[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = None,
    save_rrd_dir: Optional[PathLike] = None,
    app_id: str = "cam_calib",
) -> Optional[Path]:
    """Spawn a Rerun viewer with the fused scene.

    Args:
        positions, colors: fused world PCD (colors in [0, 1] or [0, 255])
        extra_pointclouds: ``(label, (N, 3) positions)`` overlay geometries
        cameras: iterable of ``(T_cam_world, K, image)`` for frustum overlay;
            ``image`` may be (H, W, 3) BGR or RGB
        save_rrd_dir: if given, save the recording here as
            ``calibration_<timestamp>.rrd`` and return its path
    """
    rr.init(app_id, spawn=True)
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    rr.log("world/origin", rr.Transform3D(translation=[0, 0, 0]), static=True)
    rr.log(
        "world/origin/axes",
        rr.Arrows3D(
            origins=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            vectors=[[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        ),
        static=True,
    )

    if positions.shape[0] > 0:
        col = None
        if colors is not None and len(colors) > 0:
            col = colors
            if col.dtype != np.uint8:
                col = (np.clip(col, 0.0, 1.0) * 255).astype(np.uint8)
        rr.log(
            "world/camera_pointcloud",
            rr.Points3D(positions=positions, colors=col, radii=0.001),
        )

    for label, extra_pts in extra_pointclouds:
        rr.log(
            f"world/{label}",
            rr.Points3D(
                positions=extra_pts,
                colors=[[255, 200, 100]] * len(extra_pts),
                radii=0.002,
            ),
        )

    if cameras:
        for i, (T_cam_world, K, image) in enumerate(cameras):
            T_cam_to_world = invert_se3(T_cam_world)
            H, W = image.shape[:2]
            rr.log(
                f"world/camera_{i}",
                rr.Transform3D(
                    translation=T_cam_to_world[:3, 3],
                    mat3x3=T_cam_to_world[:3, :3],
                ),
            )
            rr.log(
                f"world/camera_{i}/image",
                rr.Pinhole(image_from_camera=K, width=W, height=H),
            )
            rr.log(f"world/camera_{i}/image", rr.Image(image))

    if save_rrd_dir is not None:
        save_dir = Path(save_rrd_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        rrd_path = save_dir / f"calibration_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.rrd"
        rr.save(str(rrd_path))
        return rrd_path
    return None
