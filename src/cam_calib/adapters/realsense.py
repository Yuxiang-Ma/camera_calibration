"""Lightweight pyrealsense2 adapter — opens, captures one frame, closes.

This is intentionally minimal: it does *not* duplicate ReKep's multiprocessing
streaming infrastructure. For long-running capture loops, write your own
adapter or pass an existing camera object that satisfies ``CameraSource``.

Requires the optional ``[realsense]`` extra (``pyrealsense2``).
"""
from typing import List, Optional

import numpy as np

from cam_calib.adapters.base import CameraFrame

try:
    import pyrealsense2 as rs  # type: ignore
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "pyrealsense2 not installed. "
        "Install with: pip install camera-calibration[realsense]"
    ) from e


class SimpleRealSense:
    """Open a single RealSense pipeline at ``serial`` and grab frames on demand."""

    def __init__(
        self,
        serial: str,
        resolution: tuple = (1280, 720),
        fps: int = 30,
        warmup_frames: int = 30,
    ):
        self.serial = serial
        self._resolution = resolution
        self._fps = fps
        self._warmup_frames = warmup_frames
        self._pipeline: Optional[rs.pipeline] = None
        self._intrinsics: Optional[np.ndarray] = None
        self._dist: Optional[np.ndarray] = None

    @staticmethod
    def list_connected_serials() -> List[str]:
        ctx = rs.context()
        return [d.get_info(rs.camera_info.serial_number) for d in ctx.devices]

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self) -> None:
        if self._pipeline is not None:
            return
        cfg = rs.config()
        cfg.enable_device(self.serial)
        cfg.enable_stream(
            rs.stream.color,
            self._resolution[0],
            self._resolution[1],
            rs.format.bgr8,
            self._fps,
        )
        self._pipeline = rs.pipeline()
        profile = self._pipeline.start(cfg)
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_stream.get_intrinsics()
        self._intrinsics = np.array(
            [
                [intr.fx, 0.0, intr.ppx],
                [0.0, intr.fy, intr.ppy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self._dist = np.array(intr.coeffs, dtype=np.float64)
        # Warm up — early frames have unstable auto-exposure.
        for _ in range(self._warmup_frames):
            self._pipeline.wait_for_frames()

    def stop(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None

    def get_frame(self) -> CameraFrame:
        if self._pipeline is None:
            raise RuntimeError("SimpleRealSense.start() must be called first")
        frames = self._pipeline.wait_for_frames()
        color = frames.get_color_frame()
        if not color:
            raise RuntimeError("RealSense returned no color frame")
        image = np.asanyarray(color.get_data())
        return CameraFrame(
            serial=self.serial,
            image=image,
            K=self._intrinsics,
            dist=self._dist,
        )
