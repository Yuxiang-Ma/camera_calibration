"""Camera extrinsic calibration via ChArUco/ArUco.

Public surface re-exports the most common entry points for convenience.
"""
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
