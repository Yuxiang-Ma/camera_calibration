"""End-user entry points that compose core + adapter + (optional) viz."""
from cam_calib.workflows.calibrate_extrinsics import (
    calibrate_camera_from_frame,
    run_calibration_loop,
)
from cam_calib.workflows.set_robot_base_pose import (
    seed_or_load_robot_base_pose,
)
# visualize_fused is intentionally not imported eagerly: it pulls viz extras.

__all__ = [
    "calibrate_camera_from_frame",
    "run_calibration_loop",
    "seed_or_load_robot_base_pose",
]
