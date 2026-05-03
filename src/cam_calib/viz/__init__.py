"""Optional visualization helpers.

Open3D and Rerun are required only when the corresponding helpers are imported.
Install with the ``[viz]`` extra.
"""
from cam_calib.viz.pointcloud import (
    aggregate_world_pointcloud,
    deproject_depth,
)

__all__ = ["aggregate_world_pointcloud", "deproject_depth"]
