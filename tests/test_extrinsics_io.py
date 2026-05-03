import numpy as np
import yaml

from cam_calib.core.extrinsics_io import (
    list_cam_extrinsics,
    load_cam_extrinsics,
    save_cam_extrinsics,
)


def test_save_load_round_trip(tmp_path):
    T = np.eye(4)
    T[:3, 3] = [1.5, -0.25, 2.0]
    save_cam_extrinsics("123456789", T, tmp_path)
    loaded = load_cam_extrinsics("123456789", tmp_path)
    np.testing.assert_allclose(loaded, T)


def test_yaml_schema_matches_legacy(tmp_path):
    T = np.eye(4)
    save_cam_extrinsics("abc", T, tmp_path)
    with open(tmp_path / "abc.yaml") as f:
        data = yaml.safe_load(f)
    assert "matrix" in data
    assert data["shape"] == [4, 4]
    assert data["dtype"].startswith("float")
    assert "T_cam_world" in data["description"]


def test_list_cam_extrinsics_sorted(tmp_path):
    T = np.eye(4)
    save_cam_extrinsics("c", T, tmp_path)
    save_cam_extrinsics("a", T, tmp_path)
    save_cam_extrinsics("b", T, tmp_path)
    out = list_cam_extrinsics(tmp_path)
    assert list(out.keys()) == ["a", "b", "c"]


def test_load_missing_returns_none(tmp_path):
    assert load_cam_extrinsics("does_not_exist", tmp_path) is None


def test_legacy_yaml_loads(tmp_path):
    """Verify a YAML written by the original ReKep code still loads."""
    legacy = {
        "matrix": [
            [1.0, 0.0, 0.0, 0.1],
            [0.0, 1.0, 0.0, 0.2],
            [0.0, 0.0, 1.0, 0.3],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "shape": [4, 4],
        "dtype": "float64",
        "description": "World-to-Camera transformation matrix (T_cam_world) ...",
    }
    yaml_path = tmp_path / "999.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(legacy, f, default_flow_style=False, sort_keys=False)
    T = load_cam_extrinsics("999", tmp_path)
    assert T.shape == (4, 4)
    np.testing.assert_allclose(T[:3, 3], [0.1, 0.2, 0.3])
