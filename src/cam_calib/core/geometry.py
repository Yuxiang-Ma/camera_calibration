"""SE(3) helpers and the world/board convention used by the live workflow."""
from typing import Optional

import cv2
import numpy as np


# World origin = ChArUco board origin, with Z pointing INTO the board surface
# (i.e. world Z is negative camera Z when the camera looks down at the board).
# This matches the long-standing convention from ReKep's calibrate_extrinsics.
DEFAULT_T_world_board: np.ndarray = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def invert_se3(T: np.ndarray) -> np.ndarray:
    """Closed-form SE(3) inverse for a (4, 4) homogeneous transform."""
    if T.shape != (4, 4):
        raise ValueError(f"expected (4, 4), got {T.shape}")
    R = T[:3, :3]
    t = T[:3, 3]
    Tinv = np.eye(4, dtype=T.dtype)
    Tinv[:3, :3] = R.T
    Tinv[:3, 3] = -R.T @ t
    return Tinv


def se3_from_rvec_tvec(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Build a 4x4 from an OpenCV (rvec, tvec) pair.

    Accepts shapes (3,), (3, 1), or (1, 3) for either input.
    """
    R, _ = cv2.Rodrigues(np.asarray(rvec).reshape(3, 1))
    t = np.asarray(tvec).reshape(3)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T
