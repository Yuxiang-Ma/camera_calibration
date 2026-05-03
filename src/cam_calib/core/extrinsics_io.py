"""Read/write per-camera extrinsic YAMLs.

YAML schema is unchanged from the original ReKep format, so existing
``data/cam_extrinsics/<serial>.yaml`` files load round-trip identically.
"""
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import yaml


PathLike = Union[str, Path]


def _matrix_yaml_payload(matrix: np.ndarray, description: str) -> dict:
    return {
        "matrix": matrix.tolist(),
        "shape": list(matrix.shape),
        "dtype": str(matrix.dtype),
        "description": description,
    }


def save_cam_extrinsics(
    serial: str,
    T_cam_world: np.ndarray,
    extrinsics_dir: PathLike,
    *,
    description: Optional[str] = None,
) -> Path:
    """Write ``<extrinsics_dir>/<serial>.yaml``. Returns the file path.

    ``T_cam_world`` must be a (4, 4) world→camera transform such that
    ``P_cam = T_cam_world @ P_world``.
    """
    if T_cam_world.shape != (4, 4):
        raise ValueError(f"T_cam_world must be (4, 4), got {T_cam_world.shape}")
    extrinsics_dir = Path(extrinsics_dir)
    extrinsics_dir.mkdir(parents=True, exist_ok=True)
    if description is None:
        description = (
            f"World-to-Camera transformation matrix (T_cam_world) for "
            f"RealSense {serial}. Usage: P_cam = matrix @ P_world."
        )
    payload = _matrix_yaml_payload(T_cam_world, description)
    yaml_path = extrinsics_dir / f"{serial}.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(payload, f, default_flow_style=False, sort_keys=False)
    return yaml_path


def load_cam_extrinsics(serial: str, extrinsics_dir: PathLike) -> Optional[np.ndarray]:
    """Load T_cam_world for ``serial``, or None if no file exists."""
    yaml_path = Path(extrinsics_dir) / f"{serial}.yaml"
    if not yaml_path.exists():
        return None
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return np.array(data["matrix"], dtype=data.get("dtype", "float64"))


def list_cam_extrinsics(extrinsics_dir: PathLike) -> Dict[str, np.ndarray]:
    """Return ``{serial: T_cam_world}`` for every YAML in ``extrinsics_dir``."""
    extrinsics_dir = Path(extrinsics_dir)
    out: Dict[str, np.ndarray] = {}
    if not extrinsics_dir.exists():
        return out
    for yaml_path in sorted(extrinsics_dir.glob("*.yaml")):
        serial = yaml_path.stem
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        out[serial] = np.array(data["matrix"], dtype=data.get("dtype", "float64"))
    return out
