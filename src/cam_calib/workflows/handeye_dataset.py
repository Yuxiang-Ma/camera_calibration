"""Hand-eye dataset collection — robot-agnostic.

For each robot pose, save (rgb, depth, qpos, link_transforms_base) per camera
into a per-view directory tree:

    dataset_dir/
      view_<NNN>_<serial>/
        T_cb.yaml, T_cam_world.yaml, intrinsics.yaml
        pose_<NNN>/{rgb.png, depth.npy, qpos.npy, link_transforms_base.npz}

The package itself never imports a robot SDK or a kinematics library. Callers
plug in:
  - ``capture_multi_frame``: ``Callable[[], dict[serial, CameraFrame]]`` that
    returns one synced multi-camera frame. Wire this up against any backend
    (``cam_calib.adapters.SimpleRealSense``, ReKep's ``MultiCameraManager``, a
    mock, …).
  - ``robot_pose_source``: implements the ``RobotPoseSource`` protocol —
    typically a thin wrapper around FrankaPy + a URDF FK helper.
"""
from __future__ import annotations

import os
import re
import shutil
import sys
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional, Protocol, Tuple, Union, runtime_checkable

import numpy as np
import yaml

from cam_calib.adapters.base import CameraFrame
from cam_calib.depth.foundation_stereo import (
    FoundationStereoClient,
    FoundationStereoUnavailable,
    resolve_depth_for_frame,
)


PathLike = Union[str, Path]


_VIEW_DIR_RE = re.compile(r"^view_(\d+)_.+$")


@runtime_checkable
class RobotPoseSource(Protocol):
    """Pluggable source of (qpos, link FK) samples.

    The capture loop calls ``prepare_sample`` before each ``acquire_sample``
    so the implementer can do per-iteration setup (e.g. reconnect to a robot
    daemon that gets restarted between poses). Both methods may raise; the
    loop catches and surfaces them with separate error messages.
    """

    def prepare_sample(self) -> None:
        """Per-iteration setup. Implement as a no-op if not needed."""

    def acquire_sample(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Return ``(qpos, {link_name: T_link_in_base (4, 4)})``."""


# ---------- helpers ---------------------------------------------------------


def resolve_next_view_id(dataset_dir: PathLike) -> int:
    """Return the next free view integer.

    Scans ``dataset_dir`` for subdirs matching ``view_<NNN>_<serial>`` and
    returns ``max(NNN) + 1``. Returns 1 if no such subdirs exist. Creates
    ``dataset_dir`` if it doesn't exist.
    """
    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    max_id = 0
    for entry in dataset_dir.iterdir():
        if not entry.is_dir():
            continue
        match = _VIEW_DIR_RE.match(entry.name)
        if match:
            max_id = max(max_id, int(match.group(1)))
    return max_id + 1


def compute_T_cb(T_cam_world: np.ndarray, T_world_base: np.ndarray) -> np.ndarray:
    """Camera-from-base transform: ``P_cam = T_cb @ P_base``.

    Derivation:
        P_cam   = T_cam_world @ P_world
        P_world = T_world_base @ P_base
        ⇒ T_cb  = T_cam_world @ T_world_base
    """
    return T_cam_world @ T_world_base


def _write_matrix_yaml(path: Path, matrix: np.ndarray, description: str) -> None:
    payload = {
        "matrix": [list(map(float, row)) for row in np.asarray(matrix)],
        "shape": list(matrix.shape),
        "dtype": "float64",
        "description": description,
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def write_view_metadata(
    view_dir: PathLike,
    *,
    T_cb: np.ndarray,
    T_cam_world: np.ndarray,
    intrinsics_K: np.ndarray,
    dist_coeffs: np.ndarray,
    image_hw: Tuple[int, int],
) -> None:
    """Write ``T_cb.yaml``, ``T_cam_world.yaml``, ``intrinsics.yaml`` into ``view_dir``."""
    view_dir = Path(view_dir)
    view_dir.mkdir(parents=True, exist_ok=True)
    _write_matrix_yaml(
        view_dir / "T_cb.yaml", T_cb,
        "Camera-from-base transform (T_cb). Usage: P_cam = T_cb @ P_base.",
    )
    _write_matrix_yaml(
        view_dir / "T_cam_world.yaml", T_cam_world,
        "World-to-camera transform (T_cam_world). "
        "Provenance: copied from cam_extrinsics/<serial>.yaml.",
    )
    intr_payload = {
        "K": [list(map(float, row)) for row in np.asarray(intrinsics_K)],
        "dist_coeffs": list(map(float, np.asarray(dist_coeffs).ravel())),
        "image_hw": [int(image_hw[0]), int(image_hw[1])],
        "description": (
            "Color-frame intrinsics. If FoundationStereo depth is used, it is "
            "already warped to this frame."
        ),
    }
    (view_dir / "intrinsics.yaml").write_text(yaml.safe_dump(intr_payload, sort_keys=False))


def write_pose_atomic(
    pose_dir: PathLike,
    *,
    rgb: np.ndarray,
    depth: np.ndarray,
    qpos: np.ndarray,
    link_transforms_base: Mapping[str, np.ndarray],
) -> None:
    """Write all four files atomically: tmp dir → rename to final.

    On any failure the tmp dir is removed and ``pose_dir`` is not created.
    """
    # cv2 is needed to write rgb.png — import lazily so callers that only use
    # the read helpers don't need OpenCV.
    import cv2

    pose_dir = Path(pose_dir)
    tmp_dir = pose_dir.with_name(pose_dir.name + ".tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=False)
    try:
        cv2.imwrite(str(tmp_dir / "rgb.png"), rgb)
        np.save(tmp_dir / "depth.npy", depth.astype(np.float32))
        np.save(tmp_dir / "qpos.npy", np.asarray(qpos, dtype=np.float64))
        np.savez(
            tmp_dir / "link_transforms_base.npz",
            **{k: np.asarray(v, dtype=np.float64) for k, v in link_transforms_base.items()},
        )
        os.rename(tmp_dir, pose_dir)
    except Exception:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


# ---------- high-level loop -------------------------------------------------


CaptureMultiFrame = Callable[[], Dict[str, CameraFrame]]


def _resolve_depths(
    frames: Dict[str, CameraFrame],
    fs_client: Optional[FoundationStereoClient],
) -> Dict[str, np.ndarray]:
    """Run FoundationStereo where available, fall back to hardware depth."""
    out: Dict[str, np.ndarray] = {}
    for serial, frame in frames.items():
        if fs_client is not None:
            try:
                depth = resolve_depth_for_frame(frame, fs_client)
            except FoundationStereoUnavailable:
                depth = frame.depth
        else:
            depth = frame.depth
        if depth is None:
            raise RuntimeError(
                f"Camera {serial} produced no depth (FoundationStereo unavailable "
                f"and frame.depth is None). Either enable hardware depth on the "
                f"capture callback or wire up a reachable FoundationStereo client."
            )
        out[serial] = depth.astype(np.float32)
    return out


def _print_pose_summary(pose_id: int, qpos: np.ndarray, depths: Dict[str, np.ndarray]) -> None:
    first_serial = next(iter(depths))
    d0 = depths[first_serial]
    finite = d0[np.isfinite(d0) & (d0 > 0)]
    if finite.size:
        d_lo, d_hi = float(finite.min()), float(finite.max())
        d_str = f"[{d_lo:.2f}, {d_hi:.2f}] m"
    else:
        d_str = "[nan, nan] m"
    qpos_str = "[" + ", ".join(f"{x:+.3f}" for x in qpos) + "]"
    print(f"\033[92m✓ pose_{pose_id:03d} saved\033[0m  qpos={qpos_str}  depth={d_str}")


def interactive_capture_session(
    *,
    dataset_dir: PathLike,
    serials: list,
    capture_multi_frame: CaptureMultiFrame,
    robot_pose_source: RobotPoseSource,
    T_cam_world: Mapping[str, np.ndarray],
    T_world_base: np.ndarray,
    image_hw: Tuple[int, int],
    intrinsics: Optional[Mapping[str, Tuple[np.ndarray, np.ndarray]]] = None,
    fs_client: Optional[FoundationStereoClient] = None,
    prompt_message: Optional[str] = None,
    input_fn: Callable[[str], str] = input,
    output_stream=None,
) -> int:
    """Run the interactive capture loop. Returns the number of poses saved.

    Args:
        dataset_dir: root directory; ``view_<id>_<serial>`` subdirs are created.
        serials: cameras to record. Must be keys in ``T_cam_world`` and the
            dict returned by ``capture_multi_frame``.
        capture_multi_frame: returns a synced ``{serial: CameraFrame}`` dict.
        robot_pose_source: implements ``prepare_sample`` + ``acquire_sample``.
        T_cam_world: ``{serial: (4, 4) world→camera}`` for each camera.
        T_world_base: ``(4, 4) base→world`` for the robot.
        image_hw: ``(H, W)`` to record in per-view metadata.
        intrinsics: optional ``{serial: (K (3,3), dist_coeffs (5,))}``. If
            omitted, ``K`` and ``dist`` are read from the first
            ``CameraFrame`` returned by ``capture_multi_frame``.
        fs_client: if set, attempt FoundationStereo per frame; falls back to
            ``CameraFrame.depth`` on per-frame failure. If ``None`` (default),
            ``CameraFrame.depth`` is used directly.
        prompt_message: override the default per-iteration prompt.
        input_fn: injectable for testing (default: builtin ``input``).
        output_stream: where status lines go. Defaults to ``sys.stdout``.
    """
    if output_stream is None:
        output_stream = sys.stdout

    def _say(msg: str) -> None:
        print(msg, file=output_stream)

    dataset_dir = Path(dataset_dir)
    missing_xform = [s for s in serials if s not in T_cam_world]
    if missing_xform:
        raise ValueError(f"T_cam_world missing for cameras: {missing_xform}")

    # Pre-compute T_cb for each camera.
    T_cb = {s: compute_T_cb(T_cam_world[s], T_world_base) for s in serials}

    # Resolve view id and create view dirs.
    view_id = resolve_next_view_id(dataset_dir)
    view_dirs = {s: dataset_dir / f"view_{view_id:03d}_{s}" for s in serials}
    for d in view_dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    _say(f"[handeye] view_id={view_id:03d}, dirs:")
    for s, d in view_dirs.items():
        _say(f"          {s} → {d}")

    # Per-view metadata. If intrinsics weren't passed, capture one frame to
    # learn them. This wastes one capture but keeps the API simple for the
    # common case of "just give me a callback."
    if intrinsics is None:
        seed_frames = capture_multi_frame()
        missing = [s for s in serials if s not in seed_frames]
        if missing:
            raise RuntimeError(f"capture_multi_frame returned no frames for: {missing}")
        intrinsics = {
            s: (seed_frames[s].K, seed_frames[s].dist if seed_frames[s].dist is not None else np.zeros(5))
            for s in serials
        }
    for s in serials:
        K, dist = intrinsics[s]
        write_view_metadata(
            view_dirs[s],
            T_cb=T_cb[s],
            T_cam_world=T_cam_world[s],
            intrinsics_K=K,
            dist_coeffs=dist,
            image_hw=image_hw,
        )
    _say("[handeye] per-view metadata written")

    if prompt_message is None:
        prompt_message = "[robot ready?] Press Enter for pose_{pose_id:03d}, q to quit > "

    pose_id = 1
    n_saved = 0
    try:
        while True:
            user_in = input_fn(prompt_message.format(pose_id=pose_id)).strip().lower()
            if user_in in ("q", "quit", "exit"):
                break

            # 1) Robot prepare (e.g. reconnect FrankaPy)
            try:
                robot_pose_source.prepare_sample()
            except Exception as exc:
                _say(f"\033[93m⚠ Robot prepare failed: {type(exc).__name__}: {exc}\033[0m")
                _say("\033[93m  Pose number not advanced.\033[0m")
                continue

            # 2) Capture cameras + robot pose
            try:
                frames = capture_multi_frame()
                missing = [s for s in serials if s not in frames]
                if missing:
                    raise RuntimeError(f"capture_multi_frame returned no frames for: {missing}")
                depths = _resolve_depths({s: frames[s] for s in serials}, fs_client)
                qpos, link_xforms = robot_pose_source.acquire_sample()
            except Exception as exc:
                _say(f"\033[93m⚠ Capture failed: {type(exc).__name__}: {exc}\033[0m")
                _say("\033[93m  Pose number not advanced.\033[0m")
                continue

            # 3) Atomic writes per camera
            while any((view_dirs[s] / f"pose_{pose_id:03d}").exists() for s in serials):
                _say(f"  pose_{pose_id:03d} already exists, skipping to next id")
                pose_id += 1

            try:
                for s in serials:
                    write_pose_atomic(
                        view_dirs[s] / f"pose_{pose_id:03d}",
                        rgb=frames[s].image,
                        depth=depths[s],
                        qpos=qpos,
                        link_transforms_base=link_xforms,
                    )
            except Exception as exc:
                _say(f"\033[91m✗ Write failed: {type(exc).__name__}: {exc}\033[0m")
                _say("\033[91m  Pose number not advanced.\033[0m")
                continue

            _print_pose_summary(pose_id, qpos, depths)
            n_saved += 1
            pose_id += 1
    except KeyboardInterrupt:
        _say("\n[handeye] Interrupted.")

    _say(f"[handeye] saved {n_saved} pose(s) to:")
    for s, d in view_dirs.items():
        _say(f"          {d}")
    return n_saved
