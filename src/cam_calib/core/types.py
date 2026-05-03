"""Lightweight dataclasses used across core."""
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class CharucoBoardSpec:
    """Geometry + ArUco dictionary for a printed ChArUco board.

    Lengths are in meters. ``size`` is (squares_x, squares_y) — the count of
    chessboard squares in each direction, matching ``cv2.aruco.CharucoBoard``.
    """
    size: tuple
    square_length: float
    marker_length: float
    aruco_dict: int = cv2.aruco.DICT_4X4_50
    legacy_pattern: bool = True

    def opencv_board(self):
        dictionary = cv2.aruco.getPredefinedDictionary(self.aruco_dict)
        board = cv2.aruco.CharucoBoard(
            size=self.size,
            squareLength=self.square_length,
            markerLength=self.marker_length,
            dictionary=dictionary,
        )
        if self.legacy_pattern:
            board.setLegacyPattern(True)
        return board, dictionary


@dataclass
class CharucoDetection:
    """ChArUco corners interpolated from a single image."""
    corners: np.ndarray            # (N, 1, 2) sub-pixel corner positions
    ids: np.ndarray                # (N, 1) corner IDs
    marker_corners: list           # raw ArUco marker corners (for viz)
    marker_ids: np.ndarray         # raw ArUco marker IDs (for viz)


@dataclass
class BoardPose:
    """ChArUco board pose in the camera frame."""
    T_board_cam: np.ndarray        # (4, 4) — board-frame points → camera frame
    rvec: np.ndarray               # (3, 1) Rodrigues rotation
    tvec: np.ndarray               # (3, 1) translation, meters
    reprojection_error: float      # mean pixel error of detected corners


@dataclass
class MarkerPose:
    """Single ArUco marker pose in the camera frame."""
    rvec: np.ndarray               # (3,)
    tvec: np.ndarray               # (3,)
    T_marker_cam: np.ndarray       # (4, 4)
