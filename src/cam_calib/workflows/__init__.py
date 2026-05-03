"""End-user entry points that compose core + adapter + (optional) viz."""
from cam_calib.workflows.auto_exposure import (
    AutoExposureResult,
    auto_tune_charuco_exposure,
)
from cam_calib.workflows.calibrate_extrinsics import (
    calibrate_camera_from_frame,
    run_calibration_loop,
)
from cam_calib.workflows.set_robot_base_pose import (
    seed_or_load_robot_base_pose,
)
from cam_calib.workflows.handeye_dataset import (
    RobotPoseSource,
    compute_T_cb,
    interactive_capture_session,
    resolve_next_view_id,
    write_pose_atomic,
    write_view_metadata,
)
# visualize_fused is intentionally not imported eagerly: it pulls viz extras.

__all__ = [
    "AutoExposureResult",
    "auto_tune_charuco_exposure",
    "calibrate_camera_from_frame",
    "run_calibration_loop",
    "seed_or_load_robot_base_pose",
    "RobotPoseSource",
    "compute_T_cb",
    "interactive_capture_session",
    "resolve_next_view_id",
    "write_pose_atomic",
    "write_view_metadata",
]
