"""URDF-based mesh sampling for overlay viz.

Optional. Uses ``yourdfpy`` + ``trimesh`` (the ``[robot-viz]`` extra). No
robot SDK ever imported here. Joint angles, base poses, and URDF paths are
supplied by the caller.
"""
from cam_calib.robot.urdf_pcd import urdf_to_pcd

__all__ = ["urdf_to_pcd"]
