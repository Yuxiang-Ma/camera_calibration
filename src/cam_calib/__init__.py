"""Camera extrinsic calibration via ChArUco/ArUco.

Public surface re-exports the most common entry points for convenience.
"""
# OpenCV is intentionally NOT a runtime dependency in pyproject.toml because
# `opencv-python` and `opencv-contrib-python` collide in shared envs. Surface a
# clear error here if neither is installed (or if the installed flavor lacks
# the contrib aruco module).
try:
    import cv2  # noqa: F401
    if not hasattr(cv2, "aruco"):
        raise ImportError(
            "OpenCV is installed without the contrib `aruco` module. "
            "Install opencv-contrib-python (e.g. "
            "`pip install 'opencv-contrib-python>=4.7,<5'`) or "
            "`pip install cam-calib[opencv]`."
        )
except ImportError as _cv_err:  # pragma: no cover
    raise ImportError(
        "cam_calib requires OpenCV with the ArUco contrib module. "
        "Install with `pip install cam-calib[opencv]` or "
        "`pip install 'opencv-contrib-python>=4.7,<5'`."
    ) from _cv_err

from cam_calib.core import (
    CharucoBoardSpec,
    CharucoDetection,
    BoardPose,
    MarkerPose,
    DEFAULT_BOARD,
    detect_board,
    estimate_board_pose,
    detect_marker_pose,
    save_cam_extrinsics,
    load_cam_extrinsics,
    list_cam_extrinsics,
    save_robot_base_pose,
    load_robot_base_pose,
)

__version__ = "0.1.0"

__all__ = [
    "CharucoBoardSpec",
    "CharucoDetection",
    "BoardPose",
    "MarkerPose",
    "DEFAULT_BOARD",
    "detect_board",
    "estimate_board_pose",
    "detect_marker_pose",
    "save_cam_extrinsics",
    "load_cam_extrinsics",
    "list_cam_extrinsics",
    "save_robot_base_pose",
    "load_robot_base_pose",
    "__version__",
]
