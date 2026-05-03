"""Open3D viewer for fused point clouds + optional extra geometries."""
from typing import Iterable, Optional, Tuple

import numpy as np

try:
    import open3d as o3d  # type: ignore
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "open3d not installed. Install with: pip install cam-calib[viz]"
    ) from e


def make_pointcloud(positions: np.ndarray, colors: Optional[np.ndarray] = None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(positions)
    if colors is not None and len(colors) > 0:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def show_world_pointcloud(
    positions: np.ndarray,
    colors: Optional[np.ndarray] = None,
    *,
    extra_pointclouds: Iterable[Tuple[str, np.ndarray]] = (),
    axis_size: float = 0.1,
) -> None:
    """Open a blocking Open3D window with the fused PCD + an origin frame.

    ``extra_pointclouds`` is an iterable of ``(label, (N, 3) positions)`` —
    typically robot meshes that the caller pre-built (the package itself does
    not call any robot SDK).
    """
    pcd = make_pointcloud(positions, colors)
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size)
    geometries = [pcd, origin]
    for _, extra_pts in extra_pointclouds:
        extra = o3d.geometry.PointCloud()
        extra.points = o3d.utility.Vector3dVector(extra_pts)
        geometries.append(extra)
    o3d.visualization.draw_geometries(geometries)
