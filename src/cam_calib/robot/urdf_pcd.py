"""Build a world-frame point cloud from a URDF + joint angles + base pose.

This is a *minimal* helper for overlay viz. It does not replace a proper
kinematics library — for anything more than visualization, use the FK code in
your robotics stack and pass the result to ``viz`` directly.
"""
from pathlib import Path
from typing import Sequence, Union

import numpy as np

try:
    import yourdfpy  # type: ignore
    import trimesh  # type: ignore
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "yourdfpy/trimesh not installed. "
        "Install with: pip install camera-calibration[robot-viz]"
    ) from e


PathLike = Union[str, Path]


def urdf_to_pcd(
    urdf_path: PathLike,
    joint_angles: Sequence[float],
    T_base_world: np.ndarray,
    *,
    samples_per_link: int = 1500,
) -> np.ndarray:
    """Sample mesh points for every link, transform to world frame.

    Args:
        urdf_path: path to a URDF (yourdfpy will resolve mesh paths relative
            to it via ``ROS_PACKAGE_PATH`` or relative ``package://`` URIs)
        joint_angles: joint values in URDF order (revolute/prismatic only)
        T_base_world: (4, 4) transform from robot base to world frame
        samples_per_link: target sample count per link mesh

    Returns ``(N, 3)`` points in world frame.
    """
    if T_base_world.shape != (4, 4):
        raise ValueError(f"T_base_world must be (4, 4), got {T_base_world.shape}")

    robot = yourdfpy.URDF.load(str(urdf_path))
    cfg = {n: float(q) for n, q in zip(robot.actuated_joint_names, joint_angles)}
    robot.update_cfg(cfg)

    all_pts = []
    for link_name, link in robot.link_map.items():
        T_base_link = robot.get_transform(link_name)
        for visual in link.visuals or []:
            mesh = _mesh_from_visual(robot, visual)
            if mesh is None or len(mesh.vertices) == 0:
                continue
            # Apply visual.origin if present
            T_link_visual = (
                np.asarray(visual.origin) if visual.origin is not None else np.eye(4)
            )
            T_base_visual = T_base_link @ T_link_visual
            pts, _ = trimesh.sample.sample_surface(mesh, samples_per_link)
            homog = np.concatenate([pts.T, np.ones((1, pts.shape[0]))], axis=0)
            base_pts = (T_base_visual @ homog)[:3, :].T
            all_pts.append(base_pts)

    if not all_pts:
        return np.empty((0, 3))

    base_cloud = np.concatenate(all_pts, axis=0)
    homog = np.concatenate([base_cloud.T, np.ones((1, base_cloud.shape[0]))], axis=0)
    return (T_base_world @ homog)[:3, :].T


def _mesh_from_visual(robot, visual):
    """Resolve a yourdfpy visual to a ``trimesh.Trimesh``, or None."""
    geom = getattr(visual, "geometry", None)
    if geom is None:
        return None
    mesh_attr = getattr(geom, "mesh", None)
    if mesh_attr is None or getattr(mesh_attr, "filename", None) is None:
        return None
    fname = mesh_attr.filename
    resolved = robot._filename_handler(fname) if hasattr(robot, "_filename_handler") else fname
    try:
        loaded = trimesh.load(resolved, force="mesh")
    except Exception:
        return None
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    if mesh_attr.scale is not None:
        loaded.apply_scale(np.asarray(mesh_attr.scale))
    return loaded
