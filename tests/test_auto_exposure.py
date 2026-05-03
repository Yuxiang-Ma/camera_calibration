"""Pure-numpy tests for the exposure auto-tuner's adjustment math.

Hardware-dependent code (the feedback loop with a real camera) is not
exercised here — only the core proportional-step rule.
"""
import numpy as np

from cam_calib.workflows.auto_exposure import proportional_step


def test_step_doubles_when_image_half_target():
    # measured = 60, target = 120 → ratio 2.0, exposure should double
    out = proportional_step(100.0, 60.0, 120.0)
    assert out == 200.0


def test_step_halves_when_image_double_target():
    # measured = 240, target = 120 → ratio 0.5
    out = proportional_step(100.0, 240.0, 120.0)
    assert out == 50.0


def test_step_clamps_at_max_step_ratio():
    # measured = 1, target = 120 → naive ratio = 120, clamped to 2.0 → 2x
    out = proportional_step(100.0, 1.0, 120.0, max_step_ratio=2.0)
    assert out == 200.0


def test_step_clamps_at_max_exposure():
    out = proportional_step(5000.0, 60.0, 120.0, max_exposure_us=8000.0)
    assert out == 8000.0


def test_step_clamps_at_min_exposure():
    out = proportional_step(2.0, 240.0, 120.0, min_exposure_us=1.0)
    assert out == 1.0


def test_step_no_change_when_at_target():
    out = proportional_step(100.0, 120.0, 120.0)
    assert out == 100.0


def test_step_handles_zero_measured():
    # Pure-black frame shouldn't divide-by-zero — should clamp to max_step_ratio
    out = proportional_step(100.0, 0.0, 120.0, max_step_ratio=2.0)
    assert out == 200.0
