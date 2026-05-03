"""Tests for the FoundationStereo client + IR→color warp.

We don't talk to a real server here — only:
  - is_reachable() returns False on a clearly-down URL
  - resolve_depth_for_frame falls back to frame.depth when no client
  - warp_depth_ir_to_color produces a sensible color-frame depth from a
    synthetic IR-frame depth
"""
import numpy as np
import pytest

from cam_calib.adapters.base import CameraFrame, StereoIRData
from cam_calib.depth.foundation_stereo import (
    FoundationStereoClient,
    FoundationStereoUnavailable,
    resolve_depth_for_frame,
    warp_depth_ir_to_color,
)


def test_is_reachable_false_on_dead_port():
    # 65500 is in the ephemeral range; nothing listens here in CI.
    client = FoundationStereoClient("http://127.0.0.1:65500")
    assert client.is_reachable(timeout=0.5) is False


def test_resolve_depth_uses_frame_depth_when_no_client():
    depth = np.full((10, 20), 1.5, dtype=np.float32)
    frame = CameraFrame(
        serial="x",
        image=np.zeros((10, 20, 3), dtype=np.uint8),
        K=np.eye(3),
        dist=None,
        depth=depth,
    )
    out = resolve_depth_for_frame(frame, fs_client=None)
    np.testing.assert_array_equal(out, depth)


def test_resolve_depth_raises_when_no_fallback():
    frame = CameraFrame(
        serial="x",
        image=np.zeros((10, 20, 3), dtype=np.uint8),
        K=np.eye(3),
        dist=None,
        depth=None,
    )
    with pytest.raises(RuntimeError, match="no depth fallback"):
        resolve_depth_for_frame(frame, fs_client=None)


def test_resolve_depth_falls_back_on_fs_error():
    """If fs_client is set but the call fails, fall back to frame.depth."""
    class FailingClient:
        def infer_color_aligned_depth(self, frame):
            raise FoundationStereoUnavailable("simulated network failure")

    fallback = np.full((4, 4), 0.7, dtype=np.float32)
    frame = CameraFrame(
        serial="x",
        image=np.zeros((4, 4, 3), dtype=np.uint8),
        K=np.eye(3),
        dist=None,
        depth=fallback,
        stereo_ir=StereoIRData(
            ir_left=np.zeros((4, 4), dtype=np.uint8),
            ir_right=np.zeros((4, 4), dtype=np.uint8),
            K_ir=np.eye(3),
            baseline=0.05,
            T_color_from_ir=np.eye(4),
        ),
    )
    out = resolve_depth_for_frame(frame, fs_client=FailingClient())
    np.testing.assert_array_equal(out, fallback)


def test_warp_identity_extrinsic_same_intrinsics_recovers_depth():
    """If color and IR are the same camera (T = I, same K, same size), the
    warp should recover the input depth almost exactly (modulo splatting)."""
    H, W = 60, 80
    K = np.array([[100.0, 0.0, W / 2], [0.0, 100.0, H / 2], [0.0, 0.0, 1.0]])
    # Smooth ramp depth — easy to verify.
    depth_ir = 0.5 + 0.01 * np.tile(np.arange(W, dtype=np.float32), (H, 1))

    out = warp_depth_ir_to_color(
        depth_ir, K, np.eye(4), K, color_size=(W, H), fill_iters=0
    )
    # Most pixels should match within numerical tolerance after splatting.
    valid = out > 0
    assert valid.sum() > 0.95 * H * W
    np.testing.assert_allclose(out[valid], depth_ir[valid], atol=0.02)


def test_warp_translation_shifts_depth():
    """A pure +X translation between IR and color should shift depth in -u."""
    H, W = 60, 80
    K = np.array([[100.0, 0.0, W / 2], [0.0, 100.0, H / 2], [0.0, 0.0, 1.0]])
    depth_ir = np.full((H, W), 1.0, dtype=np.float32)
    T_color_from_ir = np.eye(4)
    T_color_from_ir[0, 3] = 0.1  # 10 cm to the right in color frame
    out = warp_depth_ir_to_color(
        depth_ir, K, T_color_from_ir, K, color_size=(W, H), fill_iters=0
    )
    # Expected pixel shift: dx_pixels = fx * tx / Z = 100 * 0.1 / 1.0 = 10 px
    # i.e. the warped depth should appear shifted to higher u (right).
    left_band = out[:, :5]
    right_band = out[:, 15:25]
    assert (left_band == 0).mean() > 0.5
    assert (right_band > 0).mean() > 0.9
