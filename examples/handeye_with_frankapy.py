"""Hand-eye dataset collection — FrankaPy + Klampt + RealSense reference example.

This file is **reference only** — it is not part of the cam_calib package and is
not imported by anything. It demonstrates how to plug a real robot setup into
``cam_calib.workflows.handeye_dataset.interactive_capture_session``.

Required (none of these are cam_calib dependencies):
    - frankapy   (FrankaPy joint reads)
    - klampt     (URDF FK for link_transforms_base)
    - pyrealsense2  (cameras, via cam_calib[realsense])
    - opencv-contrib-python  (cam_calib[opencv])

Usage:
    python examples/handeye_with_frankapy.py \\
        --dataset-dir /path/to/output \\
        --arm right --lab adelson \\
        --extrinsics-dir /path/to/cam_extrinsics \\
        --robot-extrinsics-dir /path/to/robot_extrinsics

Note that the original tac_foundation/ReKep ships its own copy of this script
under ``ReKep/scripts/collect_handeye_dataset.py`` that uses ReKep's
``MultiCameraManager`` and ``RobotKinHelper``. This example shows the same
idea built on cam_calib primitives directly.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from cam_calib.adapters.realsense import SimpleRealSense
from cam_calib.adapters.base import CameraFrame
from cam_calib.core.extrinsics_io import load_cam_extrinsics
from cam_calib.core.robot_io import load_robot_base_pose
from cam_calib.depth.foundation_stereo import (
    DEFAULT_FS_URL,
    FoundationStereoClient,
)
from cam_calib.workflows.handeye_dataset import interactive_capture_session


# ---------- RobotPoseSource: FrankaPy + Klampt -----------------------------


class FrankaPyRobotPoseSource:
    """Reconnects to FrankaPy each iteration, then samples qpos + URDF FK.

    The reconnect-per-iteration pattern is needed when ``franka-interface`` is
    restarted between poses (e.g. Desk Guidance mode requires a restart to
    re-arm the robot). A single long-lived FrankaPy connection would silently
    go stale across that restart.

    Args:
        arm: ``"left"`` or ``"right"``.
        urdf_path: path to the robot's URDF (resolved by ``yourdfpy`` / Klampt).
        connect_fn: callable that returns a fresh FrankaPy ``arm`` handle. Must
            expose ``get_joints() -> (7,) array``.
        kin_helper: any object with ``robot_name`` and ``robot_model`` such that
            ``robot_model.configFromDrivers(qpos)``, ``setConfig(...)``,
            ``numLinks()``, and ``link(i).getName() / .getTransform()`` work.
            ReKep's ``RobotKinHelper`` is the canonical implementation.
    """

    def __init__(self, *, connect_fn, kin_helper):
        self._connect_fn = connect_fn
        self._kin_helper = kin_helper
        self._arm = None

    def prepare_sample(self) -> None:
        # Drop any stale connection before reconnecting.
        self._arm = None
        self._arm = self._connect_fn()

    def acquire_sample(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        if self._arm is None:
            raise RuntimeError("call prepare_sample() before acquire_sample()")
        qpos = np.asarray(self._arm.get_joints(), dtype=np.float64)
        if qpos.shape != (7,):
            raise RuntimeError(f"unexpected qpos shape {qpos.shape}; want (7,)")
        link_xforms = _link_transforms_base(self._kin_helper, qpos)
        return qpos, link_xforms


def _klampt_xform_to_numpy(xform) -> np.ndarray:
    """Klampt's (R_flat, t) → 4x4 numpy. R_flat is column-major flattened."""
    R_flat, t = xform
    R = np.asarray(R_flat, dtype=np.float64).reshape(3, 3).T
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=np.float64)
    return T


def _link_transforms_base(kin_helper, qpos: np.ndarray) -> Dict[str, np.ndarray]:
    """Return ``{link_name: T_link_in_base}`` for every link in the URDF."""
    qpos = np.asarray(qpos, dtype=np.float64)
    model = kin_helper.robot_model
    if kin_helper.robot_name == "franka":
        cfg = model.configFromDrivers(qpos)
    else:
        cfg = list(np.concatenate([
            getattr(kin_helper, "base_joint_offset", []),
            qpos,
            getattr(kin_helper, "ee_joint_offset", []),
        ]))
    model.setConfig(cfg)
    out: Dict[str, np.ndarray] = {}
    for i in range(model.numLinks()):
        link = model.link(i)
        out[link.getName()] = _klampt_xform_to_numpy(link.getTransform())
    return out


# ---------- multi-camera capture callback ----------------------------------


def make_realsense_capture(
    cameras: Dict[str, SimpleRealSense],
) -> "callable":
    """Return a ``capture_multi_frame`` callable for a dict of running cameras.

    Each camera should be started (``cam.start()``) before the loop runs and
    stopped after; this callable just calls ``get_frame()`` on each.
    """
    def _capture() -> Dict[str, CameraFrame]:
        return {serial: cam.get_frame() for serial, cam in cameras.items()}
    return _capture


# ---------- entry point ----------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--cameras", required=True, help="Comma-separated serials.")
    parser.add_argument("--arm", default="right", choices=["left", "right"])
    parser.add_argument("--lab", default="adelson")
    parser.add_argument("--robot", default="franka")
    parser.add_argument("--extrinsics-dir", required=True, type=Path,
                        help="Directory of cam_extrinsics/<serial>.yaml files.")
    parser.add_argument("--robot-extrinsics-dir", required=True, type=Path,
                        help="Directory of robot base pose YAMLs.")
    parser.add_argument("--urdf-path", required=True, type=Path)
    parser.add_argument("--resolution", default="1280x720")
    parser.add_argument("--foundation-stereo-url", default=DEFAULT_FS_URL)
    parser.add_argument("--no-foundation-stereo", action="store_true")
    args = parser.parse_args(argv)

    serials = [s.strip() for s in args.cameras.split(",") if s.strip()]
    width, height = (int(x) for x in args.resolution.lower().split("x"))

    # 1) Extrinsics -- fail fast if anything's missing.
    T_cam_world = {}
    for s in serials:
        T = load_cam_extrinsics(s, args.extrinsics_dir)
        if T is None:
            raise SystemExit(f"missing cam_extrinsics for {s} in {args.extrinsics_dir}")
        T_cam_world[s] = T
    T_world_base = load_robot_base_pose(
        args.robot_extrinsics_dir, robot=args.robot, lab=args.lab, arm=args.arm
    )
    if T_world_base is None:
        raise SystemExit(
            f"missing robot base pose for {args.arm}/{args.robot}/{args.lab} "
            f"in {args.robot_extrinsics_dir}"
        )

    # 2) Robot pose source. The user supplies a Klampt-based kin_helper and a
    # `connect_fn` callable. In tac_foundation this looks like:
    #
    #     from ReKep.environment.robot_utils import connect_frankapy_arm
    #     from ReKep.vision.utils.common.kin_utils import RobotKinHelper
    #     kin_helper = RobotKinHelper("franka", tool_name=None,
    #                                 robot_model_variant="fer")
    #     connect_fn = lambda: connect_frankapy_arm(args.arm, with_gripper=False,
    #                                               gripper_type="franka")
    #
    # Replace with your own equivalents below.
    raise SystemExit(
        "This is a reference example; replace the SystemExit above with your "
        "own `connect_fn` + `kin_helper` and re-run. See the comment block "
        "for what they should look like."
    )

    # pose_source = FrankaPyRobotPoseSource(connect_fn=connect_fn,
    #                                       kin_helper=kin_helper)

    # 3) Cameras (color + IR for FoundationStereo).
    cameras = {
        s: SimpleRealSense(
            serial=s,
            resolution=(width, height),
            enable_depth=True,
            enable_infrared_stereo=not args.no_foundation_stereo,
        )
        for s in serials
    }
    fs_client = None
    if not args.no_foundation_stereo:
        fs_client = FoundationStereoClient(args.foundation_stereo_url)
        if not fs_client.is_reachable():
            print(f"[handeye] FoundationStereo at {args.foundation_stereo_url} "
                  f"unreachable; using hardware depth.")
            fs_client = None

    for cam in cameras.values():
        cam.start()
    time.sleep(2.0)

    try:
        intrinsics = {
            s: (
                cameras[s].get_frame().K,
                cameras[s].get_frame().dist if cameras[s].get_frame().dist is not None
                else np.zeros(5),
            )
            for s in serials
        }
        n_saved = interactive_capture_session(
            dataset_dir=args.dataset_dir,
            serials=serials,
            capture_multi_frame=make_realsense_capture(cameras),
            robot_pose_source=pose_source,  # noqa: F821
            T_cam_world=T_cam_world,
            T_world_base=T_world_base,
            image_hw=(height, width),
            intrinsics=intrinsics,
            fs_client=fs_client,
        )
        print(f"[handeye] saved {n_saved} pose(s).")
    finally:
        for cam in cameras.values():
            cam.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
