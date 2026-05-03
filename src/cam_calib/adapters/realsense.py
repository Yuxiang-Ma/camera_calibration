"""Lightweight pyrealsense2 adapter — opens, captures one frame, closes.

This is intentionally minimal: it does *not* duplicate ReKep's multiprocessing
streaming infrastructure. For long-running capture loops, write your own
adapter or pass an existing camera object that satisfies ``CameraSource``.

Requires the optional ``[realsense]`` extra (``pyrealsense2``).
"""
from typing import List, Optional

import numpy as np

from cam_calib.adapters.base import CameraFrame, StereoIRData

try:
    import pyrealsense2 as rs  # type: ignore
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "pyrealsense2 not installed. "
        "Install with: pip install cam-calib[realsense]"
    ) from e


def _intr_to_K(intr) -> np.ndarray:
    return np.array(
        [
            [intr.fx, 0.0, intr.ppx],
            [0.0, intr.fy, intr.ppy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _rs_extrinsics_to_T(ext) -> np.ndarray:
    """Convert pyrealsense2 ``rs.extrinsics`` (rotation row-major + translation)
    to a 4x4 SE(3)."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.array(ext.rotation, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.array(ext.translation, dtype=np.float64)
    return T


class SimpleRealSense:
    """Open a single RealSense pipeline at ``serial`` and grab frames on demand.

    Capture options:
      - ``enable_depth=True``: stream + align hardware depth to color.
        ``CameraFrame.depth`` is populated (float32 meters).
      - ``enable_infrared_stereo=True``: stream both IR sensors and gather
        the geometry (IR intrinsics, baseline, IR→color extrinsic) needed
        to run external stereo inference (e.g. FoundationStereo).
        ``CameraFrame.stereo_ir`` is populated.

    Both flags can be combined — useful when you want to fall back from a
    network depth service to hardware depth without re-opening the pipeline.
    """

    def __init__(
        self,
        serial: str,
        resolution: tuple = (1280, 720),
        fps: int = 30,
        warmup_frames: int = 30,
        enable_depth: bool = False,
        enable_infrared_stereo: bool = False,
    ):
        self.serial = serial
        self._resolution = resolution
        self._fps = fps
        self._warmup_frames = warmup_frames
        self._enable_depth = enable_depth
        self._enable_ir_stereo = enable_infrared_stereo

        self._pipeline: Optional[rs.pipeline] = None
        self._intrinsics: Optional[np.ndarray] = None
        self._dist: Optional[np.ndarray] = None
        self._depth_scale: Optional[float] = None
        self._align: Optional[rs.align] = None

        # IR stereo geometry — populated in start() if enabled
        self._K_ir: Optional[np.ndarray] = None
        self._baseline: Optional[float] = None
        self._T_color_from_ir: Optional[np.ndarray] = None

        # Color sensor handle — captured in start() for runtime control
        self._color_sensor = None

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
        if self._enable_depth:
            cfg.enable_stream(
                rs.stream.depth,
                self._resolution[0],
                self._resolution[1],
                rs.format.z16,
                self._fps,
            )
        if self._enable_ir_stereo:
            # Index 1 = left IR, index 2 = right IR (D4xx convention)
            cfg.enable_stream(
                rs.stream.infrared, 1,
                self._resolution[0], self._resolution[1],
                rs.format.y8, self._fps,
            )
            cfg.enable_stream(
                rs.stream.infrared, 2,
                self._resolution[0], self._resolution[1],
                rs.format.y8, self._fps,
            )

        self._pipeline = rs.pipeline()
        profile = self._pipeline.start(cfg)
        self._color_sensor = profile.get_device().first_color_sensor()

        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_stream.get_intrinsics()
        self._intrinsics = _intr_to_K(intr)
        self._dist = np.array(intr.coeffs, dtype=np.float64)

        if self._enable_depth:
            depth_sensor = profile.get_device().first_depth_sensor()
            self._depth_scale = float(depth_sensor.get_depth_scale())
            self._align = rs.align(rs.stream.color)

        if self._enable_ir_stereo:
            ir_left_stream = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
            ir_right_stream = profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()
            self._K_ir = _intr_to_K(ir_left_stream.get_intrinsics())
            extr_ir = ir_left_stream.get_extrinsics_to(ir_right_stream)
            # Baseline magnitude (IR-IR translation; D4xx is along +X)
            self._baseline = float(np.linalg.norm(np.array(extr_ir.translation)))
            extr_ir_to_color = ir_left_stream.get_extrinsics_to(color_stream)
            self._T_color_from_ir = _rs_extrinsics_to_T(extr_ir_to_color)

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
        if self._align is not None:
            frames = self._align.process(frames)

        color = frames.get_color_frame()
        if not color:
            raise RuntimeError("RealSense returned no color frame")
        image = np.asanyarray(color.get_data())

        depth_arr: Optional[np.ndarray] = None
        if self._enable_depth:
            depth = frames.get_depth_frame()
            if depth:
                raw = np.asanyarray(depth.get_data())
                depth_arr = raw.astype(np.float32) * self._depth_scale

        stereo_ir: Optional[StereoIRData] = None
        if self._enable_ir_stereo:
            ir_left = frames.get_infrared_frame(1)
            ir_right = frames.get_infrared_frame(2)
            if ir_left and ir_right:
                stereo_ir = StereoIRData(
                    ir_left=np.asanyarray(ir_left.get_data()).copy(),
                    ir_right=np.asanyarray(ir_right.get_data()).copy(),
                    K_ir=self._K_ir,
                    baseline=self._baseline,
                    T_color_from_ir=self._T_color_from_ir,
                )

        return CameraFrame(
            serial=self.serial,
            image=image,
            K=self._intrinsics,
            dist=self._dist,
            depth=depth_arr,
            stereo_ir=stereo_ir,
        )

    # ----- runtime sensor controls (used by exposure auto-tuning) -----

    def _require_color_sensor(self):
        if self._color_sensor is None:
            raise RuntimeError("camera not started; call .start() first")
        return self._color_sensor

    def set_exposure(
        self,
        exposure_us: Optional[float] = None,
        gain: Optional[float] = None,
    ) -> None:
        """Set color exposure (microseconds) and gain.

        Pass ``None`` for both to re-enable auto-exposure. Setting either
        disables auto-exposure.
        """
        sensor = self._require_color_sensor()
        if exposure_us is None and gain is None:
            sensor.set_option(rs.option.enable_auto_exposure, 1)
            return
        sensor.set_option(rs.option.enable_auto_exposure, 0)
        if exposure_us is not None:
            sensor.set_option(rs.option.exposure, float(exposure_us))
        if gain is not None:
            sensor.set_option(rs.option.gain, float(gain))

    def get_exposure(self) -> float:
        return float(self._require_color_sensor().get_option(rs.option.exposure))

    def get_gain(self) -> float:
        return float(self._require_color_sensor().get_option(rs.option.gain))

    def set_auto_exposure(self, enabled: bool) -> None:
        self._require_color_sensor().set_option(
            rs.option.enable_auto_exposure, 1 if enabled else 0
        )

    def set_white_balance(self, kelvin: Optional[float] = None) -> None:
        """Set color white balance in Kelvin. ``None`` re-enables auto-WB."""
        sensor = self._require_color_sensor()
        if kelvin is None:
            sensor.set_option(rs.option.enable_auto_white_balance, 1)
            return
        sensor.set_option(rs.option.enable_auto_white_balance, 0)
        sensor.set_option(rs.option.white_balance, float(kelvin))
