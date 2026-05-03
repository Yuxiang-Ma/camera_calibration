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
class CameraFrame:
    """A single capture from one camera.

    ``depth`` is optional — calibration only needs ``image``, but the
    visualize workflow needs RGB-D, so adapters that can supply depth set
    this field.
    """
    serial: str
    image: np.ndarray              # (H, W, 3) BGR uint8
    K: np.ndarray                  # (3, 3) intrinsics
    dist: Optional[np.ndarray]     # (5,) or None
    depth: Optional[np.ndarray] = None   # (H, W) float32 meters, aligned to color


@runtime_checkable
class CameraSource(Protocol):
    """Minimal duck-typed camera interface."""

    serial: str

    def get_frame(self) -> CameraFrame:
        ...
