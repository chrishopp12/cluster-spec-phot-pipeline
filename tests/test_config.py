"""Tests for the YAML config system."""

import pytest
from pathlib import Path

from cluster_pipeline.config import (
    load_config,
    save_config,
    merge_config,
    get_default_config,
    ensure_config,
)


# ------------------------------------
# load_config / save_config round-trip
# ------------------------------------

def test_load_missing_file(tmp_path):
    """Loading from a nonexistent directory returns empty dict."""
    cfg = load_config(tmp_path / "nonexistent")
    assert cfg == {}


def test_load_empty_file(tmp_path):
    """Loading an empty config.yaml returns empty dict."""
    (tmp_path / "config.yaml").write_text("")
    assert load_config(tmp_path) == {}


def test_round_trip(tmp_path):
    """Save then load should return the same data."""
    original = {
        "identifier": "RMJ_1327",
        "ra": 201.85,
        "dec": 53.78,
        "redshift": 0.434,
        "z_range": [0.419, 0.449],
        "fov": 6.0,
        "psf": 8.0,
        "bcgs": {
            1: {"ra": 201.851, "dec": 53.782, "z": 0.441},
            2: {"ra": 201.792, "dec": 53.801, "z": 0.438},
        },
        "subclusters": [
            {"bcg_id": 1, "label": "1", "color": "white", "radius_mpc": 2.0},
            {"bcg_id": 2, "label": "2", "color": "tab:green", "radius_mpc": 2.5},
        ],
        "xray": {
            "filename": "comb-adaptimsky-400-1100.fits",
            "contour_levels": [0.5, 0, 12],
        },
    }

    save_config(tmp_path, original)
    loaded = load_config(tmp_path)

    assert loaded["identifier"] == "RMJ_1327"
    assert loaded["ra"] == pytest.approx(201.85)
    assert loaded["z_range"] == [0.419, 0.449]
    assert loaded["xray"]["filename"] == "comb-adaptimsky-400-1100.fits"
    assert len(loaded["subclusters"]) == 2
    assert loaded["subclusters"][0]["bcg_id"] == 1


def test_save_creates_directory(tmp_path):
    """save_config should create the directory if it doesn't exist."""
    deep_path = tmp_path / "a" / "b" / "c"
    save_config(deep_path, {"test": True})
    assert (deep_path / "config.yaml").exists()
    assert load_config(deep_path) == {"test": True}


def test_save_has_comment_header(tmp_path):
    """Saved file should start with a comment header."""
    save_config(tmp_path, {"foo": "bar"})
    text = (tmp_path / "config.yaml").read_text()
    assert text.startswith("# Cluster Pipeline configuration")


# ------------------------------------
# merge_config
# ------------------------------------

def test_merge_cli_wins():
    """CLI override should replace YAML value."""
    yaml_cfg = {"fov": 6.0, "psf": 8.0}
    cli = {"psf": 12.0}
    merged = merge_config(yaml_cfg, cli)
    assert merged["psf"] == 12.0
    assert merged["fov"] == 6.0  # untouched


def test_merge_none_ignored():
    """CLI keys with None value should not override YAML."""
    yaml_cfg = {"fov": 6.0, "psf": 8.0}
    cli = {"fov": None, "psf": 10.0}
    merged = merge_config(yaml_cfg, cli)
    assert merged["fov"] == 6.0  # None ignored
    assert merged["psf"] == 10.0


def test_merge_nested_dict():
    """Nested dicts should be recursively merged."""
    yaml_cfg = {
        "xray": {"filename": "old.fits", "contour_levels": [0.5, 0, 12]},
    }
    cli = {
        "xray": {"contour_levels": [1.0, 0, 8]},
    }
    merged = merge_config(yaml_cfg, cli)
    assert merged["xray"]["filename"] == "old.fits"  # kept
    assert merged["xray"]["contour_levels"] == [1.0, 0, 8]  # replaced


def test_merge_does_not_mutate_inputs():
    """merge_config should not modify the input dicts."""
    yaml_cfg = {"fov": 6.0, "xray": {"psf": 8.0}}
    cli = {"fov": 10.0, "xray": {"psf": 12.0}}
    merge_config(yaml_cfg, cli)
    assert yaml_cfg["fov"] == 6.0
    assert yaml_cfg["xray"]["psf"] == 8.0


def test_merge_new_keys_added():
    """CLI can introduce keys not present in YAML."""
    yaml_cfg = {"fov": 6.0}
    cli = {"bandwidth": 0.15}
    merged = merge_config(yaml_cfg, cli)
    assert merged["bandwidth"] == 0.15
    assert merged["fov"] == 6.0


# ------------------------------------
# get_default_config
# ------------------------------------

def test_defaults_have_expected_keys():
    """Default config should contain core keys."""
    defaults = get_default_config()
    assert "fov" in defaults
    assert "psf" in defaults
    assert "xray" in defaults
    assert "filename" in defaults["xray"]


# ------------------------------------
# ensure_config
# ------------------------------------

def test_ensure_fills_defaults(tmp_path):
    """ensure_config on an empty directory should return defaults."""
    cfg = ensure_config(tmp_path)
    assert cfg["fov"] == 6.0
    assert cfg["psf"] == 8.0
    assert "xray" in cfg


def test_ensure_yaml_overrides_defaults(tmp_path):
    """YAML values should override defaults."""
    save_config(tmp_path, {"psf": 12.0})
    cfg = ensure_config(tmp_path)
    assert cfg["psf"] == 12.0
    assert cfg["fov"] == 6.0  # default


def test_ensure_cli_overrides_yaml(tmp_path):
    """CLI overrides should win over both YAML and defaults."""
    save_config(tmp_path, {"psf": 12.0, "fov": 8.0})
    cfg = ensure_config(tmp_path, overrides={"psf": 15.0})
    assert cfg["psf"] == 15.0    # CLI wins
    assert cfg["fov"] == 8.0     # YAML wins over default


def test_ensure_does_not_save(tmp_path):
    """ensure_config should NOT write to disk."""
    cfg = ensure_config(tmp_path, overrides={"psf": 99.0})
    assert cfg["psf"] == 99.0
    # But the file should not exist (we didn't call save_config)
    assert not (tmp_path / "config.yaml").exists()
