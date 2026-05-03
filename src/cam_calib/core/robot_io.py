"""Read/write per-robot per-lab base-pose YAMLs.

Filename convention matches the original ReKep layout:
    <arm>_<robot>_<lab>_base_pose_in_world.yaml   (e.g. left_franka_adelson_…)
or for a single-arm setup:
    <robot>_<lab>_base_pose_in_world.yaml         (e.g. franka_adelson_…)
"""
from pathlib import Path
from typing import Optional, Union

import numpy as np
import yaml

from cam_calib.core.extrinsics_io import _matrix_yaml_payload


PathLike = Union[str, Path]


def robot_base_pose_filename(robot: str, lab: str, arm: Optional[str] = None) -> str:
    """Compose the YAML filename for a robot base pose."""
    prefix = f"{arm}_{robot}" if arm else robot
    return f"{prefix}_{lab}_base_pose_in_world.yaml"


def save_robot_base_pose(
    T_base_world: np.ndarray,
    robot_extrinsics_dir: PathLike,
    *,
    robot: str,
    lab: str,
    arm: Optional[str] = None,
    description: Optional[str] = None,
) -> Path:
    """Write ``T_base_world`` to the canonical filename. Returns the path."""
    if T_base_world.shape != (4, 4):
        raise ValueError(f"T_base_world must be (4, 4), got {T_base_world.shape}")
    robot_extrinsics_dir = Path(robot_extrinsics_dir)
    robot_extrinsics_dir.mkdir(parents=True, exist_ok=True)
    fname = robot_base_pose_filename(robot, lab, arm)
    if description is None:
        nice_arm = f"{arm.capitalize()} " if arm else ""
        description = (
            f"{nice_arm}{robot.capitalize()} robot base to world transform ({lab} lab)"
        )
    payload = _matrix_yaml_payload(T_base_world, description)
    yaml_path = robot_extrinsics_dir / fname
    with open(yaml_path, "w") as f:
        yaml.dump(payload, f, default_flow_style=False, sort_keys=False)
    return yaml_path


def load_robot_base_pose(
    robot_extrinsics_dir: PathLike,
    *,
    robot: str,
    lab: str,
    arm: Optional[str] = None,
) -> Optional[np.ndarray]:
    """Return ``T_base_world`` for the requested arm, or None if missing."""
    fname = robot_base_pose_filename(robot, lab, arm)
    yaml_path = Path(robot_extrinsics_dir) / fname
    if not yaml_path.exists():
        return None
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return np.array(data["matrix"], dtype=data.get("dtype", "float64"))
