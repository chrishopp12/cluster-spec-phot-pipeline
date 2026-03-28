"""
YAML-based per-cluster configuration.

Two-layer system:
  1. Persistent YAML file at ~/Clusters/{id}/config.yaml — single source of truth
  2. Ephemeral CLI overrides — merged at runtime, discarded unless --save is used

Usage:
    cfg = load_config(cluster_path)              # read YAML (or empty dict)
    cfg = merge_config(cfg, cli_overrides)       # CLI wins for non-None keys
    save_config(cluster_path, cfg)               # write back (only with --save)
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


CONFIG_FILENAME = "config.yaml"


# ------------------------------------
# Load / Save
# ------------------------------------

def load_config(cluster_path: str | Path) -> dict[str, Any]:
    """Load a cluster's config.yaml, returning an empty dict if it doesn't exist.

    Parameters
    ----------
    cluster_path : str or Path
        Path to the cluster's data directory (e.g., ~/Clusters/RMJ_1327/).

    Returns
    -------
    dict
        Parsed YAML contents, or ``{}`` if the file is missing or empty.
    """
    config_file = Path(cluster_path) / CONFIG_FILENAME
    if not config_file.exists():
        return {}

    text = config_file.read_text()
    if not text.strip():
        return {}

    data = yaml.safe_load(text)
    return data if isinstance(data, dict) else {}


def save_config(cluster_path: str | Path, config: dict[str, Any]) -> Path:
    """Write a cluster config dict to config.yaml.

    Creates the directory and file if they don't exist.  Adds a comment
    header so the file is self-documenting.

    Parameters
    ----------
    cluster_path : str or Path
        Path to the cluster's data directory.
    config : dict
        Configuration to persist.

    Returns
    -------
    Path
        Path to the written config file.
    """
    config_file = Path(cluster_path) / CONFIG_FILENAME
    config_file.parent.mkdir(parents=True, exist_ok=True)

    header = (
        "# Cluster Pipeline configuration\n"
        "# Edit this file or use `cluster-pipeline run ... --save` to update.\n"
        "# CLI overrides are ephemeral unless --save is passed.\n"
        "\n"
    )

    body = yaml.dump(
        config,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
    )

    config_file.write_text(header + body)
    return config_file


# ------------------------------------
# Merge
# ------------------------------------

def merge_config(
    yaml_config: dict[str, Any],
    cli_overrides: dict[str, Any],
) -> dict[str, Any]:
    """Merge CLI overrides into a YAML config.  CLI wins for any non-None key.

    Performs a shallow merge at the top level and a recursive merge for
    nested dicts (e.g., ``xray``).  Lists are replaced wholesale, not appended.

    Parameters
    ----------
    yaml_config : dict
        Base configuration loaded from YAML.
    cli_overrides : dict
        CLI-provided overrides.  Keys with ``None`` values are ignored
        (treated as "not provided").

    Returns
    -------
    dict
        Merged configuration.  The input dicts are not mutated.
    """
    merged = copy.deepcopy(yaml_config)

    for key, value in cli_overrides.items():
        if value is None:
            continue

        # Recursive merge for nested dicts (e.g., xray settings)
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = value

    return merged


# ------------------------------------
# Defaults
# ------------------------------------

def get_default_config() -> dict[str, Any]:
    """Return a config dict populated with package defaults.

    This is used as the base when creating a new config.yaml for the
    first time.  Values come from ``constants.py``.
    """
    from cluster_pipeline.constants import (
        DEFAULT_BANDWIDTH,
        DEFAULT_COLOR_TYPE,
        DEFAULT_CONTOUR_LEVELS,
        DEFAULT_FOV_ARCMIN,
        DEFAULT_FOV_FULL_ARCMIN,
        DEFAULT_PHOT_LEVELS,
        DEFAULT_PHOT_SKIP,
        DEFAULT_PSF_ARCSEC,
        DEFAULT_RADIUS_MPC,
        DEFAULT_SURVEY,
        DEFAULT_XRAY_FILENAME,
        DEFAULT_Z_PAD,
    )

    return {
        "fov": DEFAULT_FOV_ARCMIN,
        "fov_full": DEFAULT_FOV_FULL_ARCMIN,
        "psf": DEFAULT_PSF_ARCSEC,
        "survey": DEFAULT_SURVEY,
        "color_type": DEFAULT_COLOR_TYPE,
        "bandwidth": DEFAULT_BANDWIDTH,
        "phot_levels": DEFAULT_PHOT_LEVELS,
        "phot_skip": DEFAULT_PHOT_SKIP,
        "z_pad": DEFAULT_Z_PAD,
        "default_radius_mpc": DEFAULT_RADIUS_MPC,
        "xray": {
            "filename": DEFAULT_XRAY_FILENAME,
            "contour_levels": list(DEFAULT_CONTOUR_LEVELS),
        },
    }


def ensure_config(
    cluster_path: str | Path,
    *,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load config, fill missing keys with defaults, optionally apply overrides.

    This is the main entry point for pipeline stages that need configuration.
    It does NOT save — call ``save_config`` explicitly (i.e., only with --save).

    Parameters
    ----------
    cluster_path : str or Path
        Cluster data directory.
    overrides : dict, optional
        CLI overrides to apply on top of YAML + defaults.

    Returns
    -------
    dict
        Complete configuration with all defaults filled in.
    """
    defaults = get_default_config()
    yaml_cfg = load_config(cluster_path)

    # Defaults first, then YAML on top, then CLI on top
    cfg = merge_config(defaults, yaml_cfg)
    if overrides:
        cfg = merge_config(cfg, overrides)

    return cfg
