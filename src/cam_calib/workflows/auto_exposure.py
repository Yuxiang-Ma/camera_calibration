"""Per-camera exposure auto-tuning so multi-camera calibration is consistent.

Background: when several RealSense cameras stare at the same ChArUco board
from different angles, RealSense's stock auto-exposure converges to different
values per camera (each AE controller integrates over its own field of view,
not the board). The result is one camera looking blown-out and another dim,
which hurts ChArUco corner accuracy.

This module runs a small feedback loop per camera:

  1. Capture a frame.
  2. Try to detect the ChArUco board.
  3. If detected, measure mean luminance in the **board's bounding box**
     (the only region we care about for calibration). Otherwise, fall back
     to the global frame mean.
  4. Compare to ``target_mean``. If outside ``tolerance``, scale exposure by
     the proportional ratio (clamped to avoid wild swings) and retry.

The loop converges in a few iterations and ends up with each camera at its
own exposure value, but with the **board's brightness** matched across
cameras — exactly the consistency that calibration cares about.

The function works with anything quacking like ``SimpleRealSense``: it needs
``get_frame``, ``set_exposure(exposure_us, gain)``, and ``get_exposure``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

import cv2
import numpy as np

from cam_calib.core.charuco import DEFAULT_BOARD, detect_board
from cam_calib.core.types import CharucoBoardSpec


@runtime_checkable
class ExposureControllableCamera(Protocol):
    """Subset of ``CameraSource`` that exposes exposure controls."""
    serial: str
    def get_frame(self): ...
    def set_exposure(self, exposure_us: Optional[float] = None,
                     gain: Optional[float] = None) -> None: ...
    def get_exposure(self) -> float: ...


@dataclass
class AutoExposureResult:
    """What the tuner ended up with for a given camera."""
    serial: str
    converged: bool
    iterations: int
    final_exposure_us: float
    final_mean_luminance: float
    used_board_roi: bool


def auto_tune_charuco_exposure(
    camera: ExposureControllableCamera,
    *,
    target_mean: float = 120.0,
    tolerance: float = 10.0,
    max_iters: int = 8,
    settle_time: float = 0.25,
    initial_exposure_us: float = 200.0,
    initial_gain: float = 16.0,
    min_exposure_us: float = 1.0,
    max_exposure_us: float = 8000.0,
    board_spec: CharucoBoardSpec = DEFAULT_BOARD,
    verbose: bool = True,
) -> AutoExposureResult:
    """Adjust ``camera``'s exposure until the ChArUco board ROI is at target.

    Args:
        target_mean: desired mean grayscale luminance in [0, 255]. ~120 keeps
            the board mid-gray with good headroom on both ends.
        tolerance: stop when ``|measured - target| <= tolerance``.
        max_iters: cap on adjustment rounds.
        settle_time: seconds to wait between exposure changes (RealSense
            applies new options on the next captured frame).
        initial_exposure_us, initial_gain: starting point. Tuner only changes
            exposure; gain stays at its initial value.
        min/max_exposure_us: clamp range for the proportional update.
        board_spec: which ChArUco board to look for.

    Returns an ``AutoExposureResult``.
    """
    # Disable AE and start from a known baseline.
    camera.set_exposure(exposure_us=initial_exposure_us, gain=initial_gain)
    time.sleep(settle_time * 2)

    iterations = 0
    converged = False
    last_mean = float("nan")
    used_board_roi = False

    for i in range(max_iters):
        iterations = i + 1
        frame = camera.get_frame()
        gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)

        det = detect_board(frame.image, frame.K, frame.dist, board_spec)
        if det is not None and len(det.corners) >= 6:
            corners = det.corners.reshape(-1, 2)
            x0 = max(int(corners[:, 0].min()) - 5, 0)
            y0 = max(int(corners[:, 1].min()) - 5, 0)
            x1 = min(int(corners[:, 0].max()) + 5, gray.shape[1])
            y1 = min(int(corners[:, 1].max()) + 5, gray.shape[0])
            mean = float(gray[y0:y1, x0:x1].mean())
            roi_source = "board"
            used_board_roi = True
        else:
            mean = float(gray.mean())
            roi_source = "frame"

        last_mean = mean
        if verbose:
            print(
                f"[{camera.serial}] auto-exposure iter {i + 1}: "
                f"{roi_source} mean={mean:6.1f} (target {target_mean:.0f})"
            )

        if abs(mean - target_mean) <= tolerance:
            converged = True
            break

        # Proportional update — scale exposure by the brightness ratio,
        # clamped to ±2x per iteration so a couple bad measurements don't
        # explode the value.
        current = camera.get_exposure()
        ratio = target_mean / max(mean, 1.0)
        ratio = float(np.clip(ratio, 0.5, 2.0))
        new_exposure = float(np.clip(current * ratio, min_exposure_us, max_exposure_us))

        # If we're already pinned at a clamp and the error has the wrong sign,
        # the loop can't progress — bail.
        if new_exposure == current:
            if verbose:
                print(
                    f"[{camera.serial}] exposure pinned at {current:.0f} us; stopping"
                )
            break
        camera.set_exposure(exposure_us=new_exposure, gain=initial_gain)
        time.sleep(settle_time)

    final_exposure = camera.get_exposure()
    if verbose:
        outcome = "converged" if converged else "max_iters or pinned"
        print(
            f"[{camera.serial}] auto-exposure done ({outcome}): "
            f"exposure={final_exposure:.0f}us, mean={last_mean:.1f}"
        )

    return AutoExposureResult(
        serial=camera.serial,
        converged=converged,
        iterations=iterations,
        final_exposure_us=final_exposure,
        final_mean_luminance=last_mean,
        used_board_roi=used_board_roi,
    )


def proportional_step(
    current_exposure_us: float,
    measured_mean: float,
    target_mean: float,
    *,
    min_exposure_us: float = 1.0,
    max_exposure_us: float = 8000.0,
    max_step_ratio: float = 2.0,
) -> float:
    """Pure-numpy version of the inner adjustment, for unit testing.

    Returns the next exposure value to try.
    """
    ratio = target_mean / max(measured_mean, 1.0)
    ratio = float(np.clip(ratio, 1.0 / max_step_ratio, max_step_ratio))
    return float(np.clip(current_exposure_us * ratio, min_exposure_us, max_exposure_us))
