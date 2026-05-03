"""ChArUco board detection and pose estimation.

Pure functions over numpy images and intrinsics. No camera, no I/O.
"""
from typing import Optional

import cv2
import numpy as np

from cam_calib.core.geometry import se3_from_rvec_tvec
from cam_calib.core.types import (
    BoardPose,
    CharucoBoardSpec,
    CharucoDetection,
)


# Default board: matches `ReKep/vision/calibration/assets/charuco_board.pdf`
# (US Letter portrait, 10x7 squares, 32.5mm squares, 23.4mm markers, DICT_4X4_50).
DEFAULT_BOARD = CharucoBoardSpec(
    size=(10, 7),
    square_length=0.0325,
    marker_length=0.0234,
    aruco_dict=cv2.aruco.DICT_4X4_50,
    legacy_pattern=True,
)


def detect_board(
    image: np.ndarray,
    K: np.ndarray,
    dist: Optional[np.ndarray],
    board_spec: CharucoBoardSpec = DEFAULT_BOARD,
) -> Optional[CharucoDetection]:
    """Detect markers and interpolate ChArUco corners.

    Returns None if no markers or no corners are found.
    """
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    board, dictionary = board_spec.opencv_board()

    marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(
        image=image,
        dictionary=dictionary,
        parameters=None,
    )
    if marker_ids is None or len(marker_corners) == 0:
        return None

    retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        markerCorners=marker_corners,
        markerIds=marker_ids,
        image=image,
        board=board,
        cameraMatrix=K,
        distCoeffs=dist,
    )
    if charuco_corners is None or retval == 0:
        return None

    return CharucoDetection(
        corners=charuco_corners,
        ids=charuco_ids,
        marker_corners=list(marker_corners),
        marker_ids=marker_ids,
    )


def estimate_board_pose(
    detection: CharucoDetection,
    K: np.ndarray,
    dist: Optional[np.ndarray],
    board_spec: CharucoBoardSpec = DEFAULT_BOARD,
) -> Optional[BoardPose]:
    """Estimate board pose in the camera frame from a ChArUco detection."""
    board, _ = board_spec.opencv_board()

    rvec_init = None
    tvec_init = None
    retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
        detection.corners,
        detection.ids,
        board,
        K,
        dist,
        rvec_init,
        tvec_init,
    )
    if not retval:
        return None

    # Mean reprojection error over the detected corners.
    object_points = board.getChessboardCorners()[detection.ids, :]
    reprojected, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist)
    reprojected = reprojected.reshape(-1, 2)
    measured = detection.corners.reshape(-1, 2)
    err = float(np.sqrt(np.sum((reprojected - measured) ** 2, axis=1)).mean())

    T_board_cam = se3_from_rvec_tvec(rvec, tvec)
    return BoardPose(T_board_cam=T_board_cam, rvec=rvec, tvec=tvec, reprojection_error=err)


def annotate_detection(
    image: np.ndarray,
    detection: CharucoDetection,
) -> np.ndarray:
    """Return a copy of ``image`` with detected markers + ChArUco corners drawn."""
    out = image.copy()
    cv2.aruco.drawDetectedMarkers(out, detection.marker_corners, detection.marker_ids)
    cv2.aruco.drawDetectedCornersCharuco(
        image=out,
        charucoCorners=detection.corners,
        charucoIds=detection.ids,
    )
    return out


def project_world_axes(
    image: np.ndarray,
    T_cam_world: np.ndarray,
    K: np.ndarray,
    dist: Optional[np.ndarray],
    axis_length: float = 0.05,
) -> np.ndarray:
    """Overlay the world coordinate axes on ``image`` given T_cam_world."""
    out = image.copy()
    world_axes = np.float32(
        [
            [0.0, 0.0, 0.0],
            [axis_length, 0.0, 0.0],
            [0.0, axis_length, 0.0],
            [0.0, 0.0, axis_length],
        ]
    )
    pts_h = np.vstack([world_axes.T, np.ones((1, 4))])
    cam_pts = (T_cam_world @ pts_h).T[:, :3]
    image_points, _ = cv2.projectPoints(cam_pts, np.zeros(3), np.zeros(3), K, dist)
    image_points = image_points.astype(int)
    cv2.line(out, tuple(image_points[0].ravel()), tuple(image_points[1].ravel()), (0, 0, 255), 2)
    cv2.line(out, tuple(image_points[0].ravel()), tuple(image_points[2].ravel()), (0, 255, 0), 2)
    cv2.line(out, tuple(image_points[0].ravel()), tuple(image_points[3].ravel()), (255, 0, 0), 2)
    return out
