"""Persist robot base poses to YAML.

This package never queries a robot — base poses are passed in by the caller
(typically hardcoded per-lab). The workflow only handles the
load-or-write-default convention used by the original ReKep flow.
"""
from pathlib import Path
from typing import Optional, Union

import numpy as np

from cam_calib.core.robot_io import (
    load_robot_base_pose,
    save_robot_base_pose,
)


PathLike = Union[str, Path]


def seed_or_load_robot_base_pose(
    default_T_base_world: np.ndarray,
    robot_extrinsics_dir: PathLike,
    *,
    robot: str,
    lab: str,
    arm: Optional[str] = None,
    description: Optional[str] = None,
) -> np.ndarray:
    """Load saved base pose if present; otherwise write ``default`` and use it.

    Mirrors the original behavior of ``test_calibration`` in ReKep's
    ``multi_camera_manager.py``.
    """
    saved = load_robot_base_pose(
        robot_extrinsics_dir, robot=robot, lab=lab, arm=arm
    )
    if saved is not None:
        return saved
    save_robot_base_pose(
        default_T_base_world,
        robot_extrinsics_dir,
        robot=robot,
        lab=lab,
        arm=arm,
        description=description,
    )
    return default_T_base_world.copy()
