"""Tests for cam_calib.workflows.handeye_dataset.

Covers the four pure helpers and the high-level interactive_capture_session.
No real cameras / robot — capture and pose source are injected as callables.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest
import yaml

from cam_calib.adapters.base import CameraFrame
from cam_calib.workflows.handeye_dataset import (
    compute_T_cb,
    interactive_capture_session,
    resolve_next_view_id,
    write_pose_atomic,
    write_view_metadata,
)


# ---------- resolve_next_view_id --------------------------------------------


def test_resolve_next_view_id_empty_dir(tmp_path):
    assert resolve_next_view_id(tmp_path) == 1


def test_resolve_next_view_id_picks_max_plus_one(tmp_path):
    (tmp_path / "view_001_408322072302").mkdir()
    (tmp_path / "view_001_134322071848").mkdir()
    (tmp_path / "view_003_408322072302").mkdir()
    (tmp_path / "view_007_408322072302").mkdir()
    assert resolve_next_view_id(tmp_path) == 8


def test_resolve_next_view_id_ignores_unrelated_dirs(tmp_path):
    (tmp_path / "view_002_408322072302").mkdir()
    (tmp_path / "view_xyz_408322072302").mkdir()  # malformed
    (tmp_path / "scratch").mkdir()
    (tmp_path / "view_005").mkdir()  # missing serial → ignored
    assert resolve_next_view_id(tmp_path) == 3


def test_resolve_next_view_id_creates_dir_if_missing(tmp_path):
    target = tmp_path / "does_not_exist_yet"
    assert resolve_next_view_id(target) == 1
    assert target.is_dir()


# ---------- compute_T_cb ----------------------------------------------------


def test_compute_T_cb_camera_at_world_origin():
    T_cam_world = np.eye(4)
    T_world_base = np.eye(4)
    T_world_base[:3, 3] = [1.0, 0.0, 0.0]
    T_cb = compute_T_cb(T_cam_world, T_world_base)
    base_origin_in_cam = T_cb @ np.array([0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(base_origin_in_cam[:3], [1.0, 0.0, 0.0], atol=1e-9)


def test_compute_T_cb_camera_translated_in_world():
    T_cam_world = np.eye(4)
    T_cam_world[:3, 3] = [-2.0, 0.0, 0.0]
    T_world_base = np.eye(4)
    T_world_base[:3, 3] = [1.0, 0.0, 0.0]
    T_cb = compute_T_cb(T_cam_world, T_world_base)
    base_origin_in_cam = T_cb @ np.array([0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(base_origin_in_cam[:3], [-1.0, 0.0, 0.0], atol=1e-9)


# ---------- write_view_metadata ---------------------------------------------


def test_write_view_metadata_writes_three_yamls(tmp_path):
    view_dir = tmp_path / "view_001_408322072302"
    T_cb = np.eye(4); T_cb[0, 3] = 0.5
    T_cam_world = np.eye(4); T_cam_world[1, 3] = 0.25
    K = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros(5, dtype=np.float64)

    write_view_metadata(
        view_dir,
        T_cb=T_cb,
        T_cam_world=T_cam_world,
        intrinsics_K=K,
        dist_coeffs=dist,
        image_hw=(720, 1280),
    )

    cb = yaml.safe_load((view_dir / "T_cb.yaml").read_text())
    np.testing.assert_allclose(np.asarray(cb["matrix"]), T_cb)

    cw = yaml.safe_load((view_dir / "T_cam_world.yaml").read_text())
    np.testing.assert_allclose(np.asarray(cw["matrix"]), T_cam_world)

    intr = yaml.safe_load((view_dir / "intrinsics.yaml").read_text())
    np.testing.assert_allclose(np.asarray(intr["K"]), K)
    np.testing.assert_allclose(np.asarray(intr["dist_coeffs"]), dist)
    assert intr["image_hw"] == [720, 1280]


# ---------- write_pose_atomic -----------------------------------------------


def test_write_pose_atomic_creates_all_four_files(tmp_path):
    pose_dir = tmp_path / "pose_001"
    rgb = (np.random.rand(64, 80, 3) * 255).astype(np.uint8)
    depth = np.random.rand(64, 80).astype(np.float32) * 2.0
    qpos = np.array([0.1, -0.7, 0.0, -2.3, 0.0, 1.5, 0.7], dtype=np.float64)
    link_xforms = {"link0": np.eye(4), "panda_link7": np.eye(4) * 2}

    write_pose_atomic(
        pose_dir,
        rgb=rgb,
        depth=depth,
        qpos=qpos,
        link_transforms_base=link_xforms,
    )

    assert pose_dir.is_dir()
    assert (pose_dir / "rgb.png").exists()
    assert (pose_dir / "depth.npy").exists()
    assert (pose_dir / "qpos.npy").exists()
    assert (pose_dir / "link_transforms_base.npz").exists()

    np.testing.assert_allclose(np.load(pose_dir / "depth.npy"), depth)
    np.testing.assert_allclose(np.load(pose_dir / "qpos.npy"), qpos)
    loaded = np.load(pose_dir / "link_transforms_base.npz")
    np.testing.assert_allclose(loaded["link0"], np.eye(4))
    np.testing.assert_allclose(loaded["panda_link7"], np.eye(4) * 2)


def test_write_pose_atomic_cleans_tmp_on_failure(tmp_path, monkeypatch):
    pose_dir = tmp_path / "pose_002"
    rgb = (np.random.rand(64, 80, 3) * 255).astype(np.uint8)
    depth = np.random.rand(64, 80).astype(np.float32)
    qpos = np.zeros(7)

    def fail_savez(*args, **kwargs):
        raise RuntimeError("simulated disk failure")

    monkeypatch.setattr("numpy.savez", fail_savez)

    with pytest.raises(RuntimeError, match="simulated"):
        write_pose_atomic(
            pose_dir,
            rgb=rgb,
            depth=depth,
            qpos=qpos,
            link_transforms_base={"link0": np.eye(4)},
        )

    assert not pose_dir.exists()
    assert not (pose_dir.parent / "pose_002.tmp").exists()


# ---------- interactive_capture_session -------------------------------------


def _make_frame(serial: str, *, depth_value: float = 1.0) -> CameraFrame:
    return CameraFrame(
        serial=serial,
        image=(np.random.rand(32, 48, 3) * 255).astype(np.uint8),
        K=np.array([[300.0, 0, 24], [0, 300.0, 16], [0, 0, 1]], dtype=np.float64),
        dist=np.zeros(5, dtype=np.float64),
        depth=np.full((32, 48), depth_value, dtype=np.float32),
    )


class _FakeRobotPoseSource:
    def __init__(self, qposes, link_xforms, *, prepare_fail_iters=()):
        self._qposes = list(qposes)
        self._link_xforms = link_xforms
        self._prepare_fail_iters = set(prepare_fail_iters)
        self.prepare_calls = 0
        self.acquire_calls = 0

    def prepare_sample(self) -> None:
        self.prepare_calls += 1
        if self.prepare_calls in self._prepare_fail_iters:
            raise RuntimeError(f"simulated prepare failure on call {self.prepare_calls}")

    def acquire_sample(self):
        if not self._qposes:
            raise RuntimeError("no more qposes queued")
        q = self._qposes.pop(0)
        self.acquire_calls += 1
        return q, self._link_xforms


def _capture_factory(serials: List[str], depth_value: float = 1.0):
    def _capture():
        return {s: _make_frame(s, depth_value=depth_value) for s in serials}
    return _capture


def _scripted_input(answers):
    answers = list(answers)

    def _input(_prompt: str) -> str:
        if not answers:
            return "q"
        return answers.pop(0)

    return _input


def test_session_saves_two_poses_and_writes_metadata(tmp_path):
    serials = ["AAA", "BBB"]
    qpos = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7])
    pose_source = _FakeRobotPoseSource(
        qposes=[qpos.copy(), qpos.copy() + 0.01],
        link_xforms={"link0": np.eye(4), "panda_link7": np.eye(4) * 2},
    )
    T_cam_world = {s: np.eye(4) for s in serials}
    T_world_base = np.eye(4)
    T_world_base[:3, 3] = [0.5, 0, 0]

    out = io.StringIO()
    n = interactive_capture_session(
        dataset_dir=tmp_path,
        serials=serials,
        capture_multi_frame=_capture_factory(serials),
        robot_pose_source=pose_source,
        T_cam_world=T_cam_world,
        T_world_base=T_world_base,
        image_hw=(32, 48),
        input_fn=_scripted_input(["", "", "q"]),
        output_stream=out,
    )

    assert n == 2
    assert pose_source.prepare_calls == 2
    assert pose_source.acquire_calls == 2
    for s in serials:
        view_dir = tmp_path / f"view_001_{s}"
        assert (view_dir / "T_cb.yaml").exists()
        assert (view_dir / "T_cam_world.yaml").exists()
        assert (view_dir / "intrinsics.yaml").exists()
        assert (view_dir / "pose_001" / "rgb.png").exists()
        assert (view_dir / "pose_002" / "qpos.npy").exists()
        np.testing.assert_allclose(np.load(view_dir / "pose_001" / "qpos.npy"), qpos)


def test_session_does_not_advance_pose_id_on_prepare_failure(tmp_path):
    serials = ["AAA"]
    qpos = np.zeros(7)
    pose_source = _FakeRobotPoseSource(
        qposes=[qpos.copy(), qpos.copy()],
        link_xforms={"link0": np.eye(4)},
        prepare_fail_iters={1},  # first prepare blows up
    )
    T_cam_world = {s: np.eye(4) for s in serials}
    out = io.StringIO()

    n = interactive_capture_session(
        dataset_dir=tmp_path,
        serials=serials,
        capture_multi_frame=_capture_factory(serials),
        robot_pose_source=pose_source,
        T_cam_world=T_cam_world,
        T_world_base=np.eye(4),
        image_hw=(32, 48),
        input_fn=_scripted_input(["", "", "q"]),
        output_stream=out,
    )

    assert n == 1
    assert (tmp_path / f"view_001_AAA" / "pose_001").exists()
    assert not (tmp_path / f"view_001_AAA" / "pose_002").exists()
    assert "Robot prepare failed" in out.getvalue()


def test_session_skips_existing_pose_id(tmp_path):
    serials = ["AAA"]
    qpos = np.zeros(7)
    pose_source = _FakeRobotPoseSource(
        qposes=[qpos.copy()],
        link_xforms={"link0": np.eye(4)},
    )
    # Pre-create a view_001_AAA/pose_001 dir so the loop has to skip past it.
    (tmp_path / "view_001_AAA").mkdir(parents=True)
    (tmp_path / "view_001_AAA" / "pose_001").mkdir()
    out = io.StringIO()

    n = interactive_capture_session(
        dataset_dir=tmp_path,
        serials=serials,
        capture_multi_frame=_capture_factory(serials),
        robot_pose_source=pose_source,
        T_cam_world={s: np.eye(4) for s in serials},
        T_world_base=np.eye(4),
        image_hw=(32, 48),
        input_fn=_scripted_input(["", "q"]),
        output_stream=out,
    )

    # view_001 already had pose_001 → next id starts at view_002 because the
    # session also calls resolve_next_view_id at the top.
    # Actually: pre-existing view_001_AAA → resolve_next_view_id returns 2.
    assert n == 1
    assert (tmp_path / "view_002_AAA" / "pose_001").exists()


def test_session_raises_when_T_cam_world_missing(tmp_path):
    with pytest.raises(ValueError, match="T_cam_world missing"):
        interactive_capture_session(
            dataset_dir=tmp_path,
            serials=["AAA", "BBB"],
            capture_multi_frame=_capture_factory(["AAA", "BBB"]),
            robot_pose_source=_FakeRobotPoseSource([np.zeros(7)], {}),
            T_cam_world={"AAA": np.eye(4)},  # BBB missing
            T_world_base=np.eye(4),
            image_hw=(32, 48),
            input_fn=_scripted_input(["q"]),
            output_stream=io.StringIO(),
        )
