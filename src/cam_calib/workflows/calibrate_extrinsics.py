"""Live ChArUco extrinsic calibration workflow.

This is a thin orchestrator: pull a frame from each camera, run core detection
and pose estimation, write the YAML, optionally show a viz window. All math
lives in ``core``; this module just glues steps together.
"""
from pathlib import Path
from typing import Iterable, Optional, Union
import warnings

import numpy as np

from cam_calib.adapters.base import CameraFrame, CameraSource
from cam_calib.core.charuco import (
    DEFAULT_BOARD,
    annotate_detection,
    detect_board,
    estimate_board_pose,
    project_world_axes,
)
from cam_calib.core.extrinsics_io import save_cam_extrinsics
from cam_calib.core.geometry import DEFAULT_T_world_board
from cam_calib.core.types import CharucoBoardSpec


PathLike = Union[str, Path]


def calibrate_camera_from_frame(
    frame: CameraFrame,
    extrinsics_dir: PathLike,
    *,
    board_spec: CharucoBoardSpec = DEFAULT_BOARD,
    T_world_board: np.ndarray = DEFAULT_T_world_board,
    save: bool = True,
    visualize: bool = False,
) -> Optional[np.ndarray]:
    """Calibrate one camera from a single image. Returns T_cam_world or None.

    On detection or pose failure, prints a warning and returns None.
    """
    detection = detect_board(frame.image, frame.K, frame.dist, board_spec)
    if detection is None:
        warnings.warn(f"[{frame.serial}] no ChArUco markers detected")
        return None

    pose = estimate_board_pose(detection, frame.K, frame.dist, board_spec)
    if pose is None:
        warnings.warn(f"[{frame.serial}] ChArUco pose estimation failed")
        return None

    print(f"[{frame.serial}] corners={len(detection.corners)} "
          f"reproj_err={pose.reprojection_error:.3f}px")

    T_cam_world = pose.T_board_cam @ T_world_board

    if save:
        path = save_cam_extrinsics(frame.serial, T_cam_world, extrinsics_dir)
        print(f"[{frame.serial}] wrote {path}")

    if visualize:
        try:
            import cv2
            annotated = annotate_detection(frame.image, detection)
            annotated = project_world_axes(
                annotated, T_cam_world, frame.K, frame.dist
            )
            cv2.imshow(f"calibration_{frame.serial}", annotated)
            cv2.waitKey(1)
        except ImportError:
            pass

    return T_cam_world


def run_calibration_loop(
    cameras: Iterable[CameraSource],
    extrinsics_dir: PathLike,
    *,
    board_spec: CharucoBoardSpec = DEFAULT_BOARD,
    T_world_board: np.ndarray = DEFAULT_T_world_board,
    visualize: bool = True,
) -> None:
    """Loop forever, recalibrating from fresh frames until KeyboardInterrupt.

    Each iteration captures one frame per camera and writes a YAML. The user
    is expected to kill the loop (Ctrl+C) once reprojection error is low and
    stable across cameras.
    """
    cams = list(cameras)
    if not cams:
        raise ValueError("no cameras provided")

    print(f"Starting calibration loop ({len(cams)} cameras). Ctrl+C to stop.")
    try:
        while True:
            for cam in cams:
                frame = cam.get_frame()
                calibrate_camera_from_frame(
                    frame,
                    extrinsics_dir,
                    board_spec=board_spec,
                    T_world_board=T_world_board,
                    visualize=visualize,
                )
    except KeyboardInterrupt:
        print("\ncalibration loop interrupted by user")
