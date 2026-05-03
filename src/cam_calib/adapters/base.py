"""Camera-source protocol used by workflows.

Anything that can return ``(image, K, dist)`` for a single camera satisfies
``CameraSource``. ReKep's existing ``RealsenseDriver`` already exposes the
underlying pieces (``ring_buffer.get()``, ``get_intr_mat``, ``get_dist_coeff``)
so a thin adapter is straightforward.
"""
from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

import numpy as np


@dataclass
class StereoIRData:
    """IR stereo pair plus the geometry needed to compute & warp depth.

    Populated by adapters that support IR stereo capture (e.g. RealSense).
    Consumed by ``cam_calib.depth.foundation_stereo`` to run external stereo
    inference and warp the resulting depth into the color frame.
    """
    ir_left: np.ndarray            # (H, W) uint8 (or (H, W, 3) — adapter's choice)
    ir_right: np.ndarray
    K_ir: np.ndarray               # (3, 3) IR camera intrinsics
    baseline: float                # meters between IR sensors
    T_color_from_ir: np.ndarray    # (4, 4) IR→color extrinsic


@dataclass
class CameraFrame:
    """A single capture from one camera.

    ``depth`` and ``stereo_ir`` are independent optional channels:
      - ``depth``: hardware (or otherwise pre-computed) depth, color-aligned
      - ``stereo_ir``: raw IR pair + geometry, for external stereo inference
    """
    serial: str
    image: np.ndarray              # (H, W, 3) BGR uint8
    K: np.ndarray                  # (3, 3) intrinsics
    dist: Optional[np.ndarray]     # (5,) or None
    depth: Optional[np.ndarray] = None         # (H, W) float32 meters, color-aligned
    stereo_ir: Optional[StereoIRData] = None   # IR pair for FoundationStereo etc.


@runtime_checkable
class CameraSource(Protocol):
    """Minimal duck-typed camera interface."""

    serial: str

    def get_frame(self) -> CameraFrame:
        ...
