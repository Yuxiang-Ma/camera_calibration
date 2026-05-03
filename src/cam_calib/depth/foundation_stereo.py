"""HTTP client for the FoundationStereo inference server.

Targets https://github.com/williamshen-nz/FoundationStereo (run with
``pixi run server``; default listen address ``http://0.0.0.0:1234``).

Wire format (from the upstream server):
  POST /infer
    multipart/form-data:
      left_image  : PNG bytes
      right_image : PNG bytes
      fx, fy, cx, cy : floats — IR camera intrinsics
      baseline       : float  — IR baseline in meters
      scale          : float  — image scale (default 1.0)
      hiera          : int    — hierarchical refinement (default 0)
      valid_iters    : int    — refinement iterations (default 32)
    response: NPZ bytes with key "depth" (H, W) float, meters

The FoundationStereo server returns depth in the **left IR** camera frame.
Use ``warp_depth_ir_to_color`` to project it onto the color image grid.
"""
from __future__ import annotations

import io
import logging
import time
import warnings
from typing import Optional

import cv2
import numpy as np

from cam_calib.adapters.base import CameraFrame, StereoIRData


_log = logging.getLogger(__name__)


DEFAULT_FS_URL = "http://localhost:1234"


class FoundationStereoUnavailable(RuntimeError):
    """Raised when the FoundationStereo server can't be reached or returns an error."""


class FoundationStereoClient:
    """Thin requests-based client for the FoundationStereo /infer endpoint.

    Use ``is_reachable()`` to decide whether to ask cameras for IR streams up
    front (USB-bandwidth concern). Use ``infer_color_aligned_depth(frame)``
    per-frame to get depth on the camera's color grid.
    """

    def __init__(
        self,
        url: str = DEFAULT_FS_URL,
        timeout: float = 30.0,
        scale: float = 1.0,
        hiera: int = 0,
        valid_iters: int = 32,
    ):
        try:
            import requests  # type: ignore  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "FoundationStereo client requires `requests`. "
                "Install with: pip install requests"
            ) from e
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.scale = scale
        self.hiera = hiera
        self.valid_iters = valid_iters

    def is_reachable(self, timeout: float = 2.0) -> bool:
        """Quick TCP/HTTP probe. Returns True on any HTTP response from the host."""
        import requests
        try:
            requests.head(self.url, timeout=timeout)
            return True
        except Exception:
            return False

    def infer_depth_ir(
        self,
        ir_left: np.ndarray,
        ir_right: np.ndarray,
        K_ir: np.ndarray,
        baseline: float,
    ) -> np.ndarray:
        """Run inference; return depth in the IR (left) camera frame, in meters."""
        import requests

        # Server expects PNGs; cv2.imencode handles both grayscale and BGR.
        # Convert single-channel IR to 3-channel so RGB-trained models behave.
        left = ir_left if ir_left.ndim == 3 else np.stack([ir_left] * 3, axis=-1)
        right = ir_right if ir_right.ndim == 3 else np.stack([ir_right] * 3, axis=-1)
        ok_l, left_bytes = cv2.imencode(".png", left)
        ok_r, right_bytes = cv2.imencode(".png", right)
        if not (ok_l and ok_r):
            raise FoundationStereoUnavailable("failed to encode IR PNGs")

        files = {
            "left_image": ("left.png", left_bytes.tobytes(), "image/png"),
            "right_image": ("right.png", right_bytes.tobytes(), "image/png"),
        }
        data = {
            "fx": float(K_ir[0, 0]),
            "fy": float(K_ir[1, 1]),
            "cx": float(K_ir[0, 2]),
            "cy": float(K_ir[1, 2]),
            "baseline": float(baseline),
            "scale": self.scale,
            "hiera": self.hiera,
            "valid_iters": self.valid_iters,
        }

        t0 = time.perf_counter()
        try:
            resp = requests.post(
                f"{self.url}/infer", files=files, data=data, timeout=self.timeout
            )
        except Exception as e:
            raise FoundationStereoUnavailable(
                f"could not reach FoundationStereo at {self.url}: {e}"
            ) from e

        if resp.status_code != 200:
            raise FoundationStereoUnavailable(
                f"FoundationStereo returned {resp.status_code}: {resp.text[:200]}"
            )

        depth = np.load(io.BytesIO(resp.content))["depth"]
        _log.info(
            "FoundationStereo depth %s in %.2fs",
            depth.shape,
            time.perf_counter() - t0,
        )
        return depth.astype(np.float32, copy=False)

    def infer_color_aligned_depth(self, frame: CameraFrame) -> np.ndarray:
        """End-to-end: IR pair → FS depth → warp to ``frame``'s color grid.

        Requires ``frame.stereo_ir`` to be populated (RealSense adapter does
        this when ``enable_infrared_stereo=True``).
        """
        if frame.stereo_ir is None:
            raise FoundationStereoUnavailable(
                f"frame for {frame.serial} has no stereo_ir data; "
                f"open camera with enable_infrared_stereo=True"
            )
        ir = frame.stereo_ir
        depth_ir = self.infer_depth_ir(
            ir.ir_left, ir.ir_right, ir.K_ir, ir.baseline
        )
        H, W = frame.image.shape[:2]
        return warp_depth_ir_to_color(
            depth_ir, ir.K_ir, ir.T_color_from_ir, frame.K, (W, H)
        )


def warp_depth_ir_to_color(
    depth_ir: np.ndarray,
    K_ir: np.ndarray,
    T_color_from_ir: np.ndarray,
    K_color: np.ndarray,
    color_size: tuple,
    fill_iters: int = 5,
) -> np.ndarray:
    """Forward-project IR-frame depth onto the color pixel grid.

    Algorithm matches ReKep's ``_depth_ir_to_color``: 4-neighbour splatting
    with z-buffer min, then iterative min-filter to fill 1-2 pixel holes.
    """
    H_ir, W_ir = depth_ir.shape
    W_c, H_c = color_size

    fx_ir, fy_ir = K_ir[0, 0], K_ir[1, 1]
    cx_ir, cy_ir = K_ir[0, 2], K_ir[1, 2]
    fx_c, fy_c = K_color[0, 0], K_color[1, 1]
    cx_c, cy_c = K_color[0, 2], K_color[1, 2]

    R = T_color_from_ir[:3, :3]
    t = T_color_from_ir[:3, 3]

    us, vs = np.meshgrid(np.arange(W_ir), np.arange(H_ir))
    z = depth_ir
    valid = (z > 0) & np.isfinite(z)
    x_ir = (us - cx_ir) / fx_ir * z
    y_ir = (vs - cy_ir) / fy_ir * z
    pts_ir = np.stack([x_ir[valid], y_ir[valid], z[valid]], axis=1)

    pts_color = (R @ pts_ir.T).T + t
    z_c = pts_color[:, 2]
    u_c = pts_color[:, 0] / z_c * fx_c + cx_c
    v_c = pts_color[:, 1] / z_c * fy_c + cy_c

    depth_out = np.full((H_c, W_c), np.inf, dtype=np.float32)
    for du, dv in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        ui = np.floor(u_c).astype(np.int32) + du
        vi = np.floor(v_c).astype(np.int32) + dv
        in_bounds = (ui >= 0) & (ui < W_c) & (vi >= 0) & (vi < H_c) & (z_c > 0)
        idx = vi[in_bounds] * W_c + ui[in_bounds]
        np.minimum.at(depth_out.ravel(), idx, z_c[in_bounds])

    depth_out[depth_out == np.inf] = 0.0

    for _ in range(fill_iters):
        holes = depth_out == 0
        if not holes.any():
            break
        kernel = np.ones((3, 3), dtype=np.float32)
        dilated = cv2.erode(
            np.where(depth_out > 0, depth_out, np.finfo(np.float32).max),
            kernel,
        )
        depth_out = np.where(
            holes & (dilated < np.finfo(np.float32).max), dilated, depth_out
        )

    return depth_out


def resolve_depth_for_frame(
    frame: CameraFrame,
    fs_client: Optional[FoundationStereoClient],
) -> np.ndarray:
    """Convenience: try FoundationStereo first, fall back to ``frame.depth``.

    Returns the depth array. Raises if neither path produces depth.
    """
    if fs_client is not None and frame.stereo_ir is not None:
        try:
            return fs_client.infer_color_aligned_depth(frame)
        except FoundationStereoUnavailable as e:
            warnings.warn(
                f"[{frame.serial}] FoundationStereo failed ({e}); "
                f"falling back to frame.depth",
                RuntimeWarning,
            )
    if frame.depth is None:
        raise RuntimeError(
            f"frame for {frame.serial} has no depth fallback "
            f"(open the camera with enable_depth=True)"
        )
    return frame.depth
