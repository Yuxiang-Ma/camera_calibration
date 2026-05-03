"""Pure-Python core: ChArUco/ArUco detection, geometry, YAML I/O.

No camera, no visualization, no robot SDK. Numpy in, numpy out (plus YAML on
disk for the persistence helpers).
"""
from cam_calib.core.types import (
    CharucoBoardSpec,
    CharucoDetection,
    BoardPose,
    MarkerPose,
)
from cam_calib.core.charuco import (
    DEFAULT_BOARD,
    detect_board,
    estimate_board_pose,
)
from cam_calib.core.aruco import detect_marker_pose
from cam_calib.core.geometry import (
    DEFAULT_T_world_board,
    invert_se3,
    se3_from_rvec_tvec,
)
from cam_calib.core.extrinsics_io import (
    save_cam_extrinsics,
    load_cam_extrinsics,
    list_cam_extrinsics,
)
from cam_calib.core.robot_io import (
    save_robot_base_pose,
    load_robot_base_pose,
    robot_base_pose_filename,
)

__all__ = [
    "CharucoBoardSpec",
    "CharucoDetection",
    "BoardPose",
    "MarkerPose",
    "DEFAULT_BOARD",
    "detect_board",
    "estimate_board_pose",
    "detect_marker_pose",
    "DEFAULT_T_world_board",
    "invert_se3",
    "se3_from_rvec_tvec",
    "save_cam_extrinsics",
    "load_cam_extrinsics",
    "list_cam_extrinsics",
    "save_robot_base_pose",
    "load_robot_base_pose",
    "robot_base_pose_filename",
]
