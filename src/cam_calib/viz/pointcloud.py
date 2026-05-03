"""Multi-camera point-cloud aggregation in world frame.

Pure numpy. No Open3D import here — output is plain ``(positions, colors)``
arrays that the open3d/rerun visualizers can wrap.
"""
from typing import Optional, Tuple

import numpy as np

from cam_calib.core.geometry import invert_se3


def deproject_depth(
    depth: np.ndarray,
    K: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Back-project a depth image to (N, 3) points in the camera frame.

    ``depth`` is in meters. ``mask`` is an optional boolean array selecting
    which pixels to keep.
    """
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    if mask is None:
        mask = np.ones_like(depth, dtype=bool)
    vs, us = np.where(mask)
    z = depth[vs, us]
    x = (us - cx) * z / fx
    y = (vs - cy) * z / fy
    return np.stack([x, y, z], axis=1)


def aggregate_world_pointcloud(
    colors: np.ndarray,
    depths: np.ndarray,
    Ks: np.ndarray,
    T_cam_world_stack: np.ndarray,
    *,
    depth_min: float = 0.0,
    depth_max: float = 2.0,
    boundaries: Optional[dict] = None,
    voxel_size: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fuse RGB-D from N cameras into a single world-frame point cloud.

    Args:
        colors: (N, H, W, 3) uint8 or float
        depths: (N, H, W) meters
        Ks: (N, 3, 3) intrinsics
        T_cam_world_stack: (N, 4, 4) world→camera transforms
        depth_min/depth_max: per-pixel depth filter
        boundaries: optional ``{x_lower, x_upper, y_lower, y_upper, z_lower, z_upper}``
        voxel_size: if set, voxel-downsample the result via Open3D

    Returns ``(positions (M, 3), colors (M, 3) in [0, 1])``.
    """
    if colors.dtype != np.float32 and colors.dtype != np.float64:
        colors_n = colors.astype(np.float32) / 255.0
    else:
        colors_n = colors

    N = colors_n.shape[0]
    all_pts = []
    all_cols = []
    for i in range(N):
        depth = depths[i]
        mask = (depth > depth_min) & (depth < depth_max)
        cam_pts = deproject_depth(depth, Ks[i], mask)
        if cam_pts.shape[0] == 0:
            continue
        T_world_cam = invert_se3(T_cam_world_stack[i])
        homog = np.concatenate([cam_pts.T, np.ones((1, cam_pts.shape[0]))], axis=0)
        world_pts = (T_world_cam @ homog)[:3, :].T
        col = colors_n[i][mask]

        if boundaries:
            keep = (
                (world_pts[:, 0] > boundaries["x_lower"])
                & (world_pts[:, 0] < boundaries["x_upper"])
                & (world_pts[:, 1] > boundaries["y_lower"])
                & (world_pts[:, 1] < boundaries["y_upper"])
                & (world_pts[:, 2] > boundaries["z_lower"])
                & (world_pts[:, 2] < boundaries["z_upper"])
            )
            world_pts = world_pts[keep]
            col = col[keep]
        all_pts.append(world_pts)
        all_cols.append(col)

    if not all_pts:
        return np.empty((0, 3)), np.empty((0, 3))

    positions = np.concatenate(all_pts, axis=0)
    rgb = np.concatenate(all_cols, axis=0)

    if voxel_size is not None:
        try:
            import open3d as o3d  # type: ignore
        except ImportError as e:
            raise ImportError(
                "voxel_size requested but open3d not installed. "
                "Install with: pip install camera-calibration[viz]"
            ) from e
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(positions)
        pcd.colors = o3d.utility.Vector3dVector(rgb)
        pcd = pcd.voxel_down_sample(voxel_size)
        positions = np.asarray(pcd.points)
        rgb = np.asarray(pcd.colors)

    return positions, rgb
