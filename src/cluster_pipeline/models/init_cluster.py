"""Cluster initialization: discover and populate cluster metadata.

This module handles the messy work of finding cluster data from
multiple sources (config.yaml, clusters.csv, NED/SIMBAD) so the
Cluster class itself can stay clean.

Usage:
    cluster = cluster_init("RMJ 1327")
    cluster = cluster_init("RMJ 1327", base_path="/path/to/data")
    cluster = cluster_init("RMJ 1327", ra=201.85, dec=53.78)  # override
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from cluster_pipeline.models.cluster import Cluster, DEFAULT_BASE_PATH
from cluster_pipeline.config import load_config, ensure_config
from cluster_pipeline.constants import (
    DEFAULT_FOV_ARCMIN,
    DEFAULT_FOV_FULL_ARCMIN,
    DEFAULT_RA_OFFSET_ARCMIN,
    DEFAULT_DEC_OFFSET_ARCMIN,
    DEFAULT_SURVEY,
    DEFAULT_COLOR_TYPE,
    DEFAULT_PHOT_LEVELS,
    DEFAULT_PHOT_SKIP,
    DEFAULT_CONTOUR_LEVELS,
    DEFAULT_PSF_ARCSEC,
    DEFAULT_BANDWIDTH,
)


MASTER_CSV_NAME = "clusters.csv"


def cluster_init(
    identifier: str,
    *,
    base_path: Path | str | None = None,
    verbose: bool = True,
    **overrides: Any,
) -> Cluster:
    """Create a fully populated Cluster from available sources.

    Resolution order (first non-missing value wins):
        1. Explicit overrides (keyword arguments)
        2. config.yaml (per-cluster YAML in the cluster directory)
        3. clusters.csv (lightweight index at base_path)
        4. External queries (NED/SIMBAD for coords/redshift)
        5. Package defaults (for analysis parameters only)

    Parameters
    ----------
    identifier : str
        Cluster name or identifier.
    base_path : Path or str, optional
        Base directory for cluster data. Defaults to ~/XSorter/Clusters/.
    verbose : bool
        Print progress messages.
    **overrides
        Any Cluster field to override (ra, dec, redshift, fov, etc.).

    Returns
    -------
    Cluster
        Fully populated cluster object.
    """
    bp = Path(base_path) if base_path else DEFAULT_BASE_PATH

    # Start with a bare cluster (just identifier + base_path)
    cluster = Cluster(identifier=identifier, base_path=bp)

    # Collect values from each source (later sources don't overwrite earlier ones)
    values: dict[str, Any] = {}

    # --- Source 1: Explicit overrides (highest priority) ---
    for key, val in overrides.items():
        if val is not None:
            values[key] = val

    # --- Source 2: config.yaml ---
    cfg = load_config(cluster.cluster_path)
    if cfg:
        if verbose and cfg:
            print(f"  [config.yaml] loaded from {cluster.config_path}")
        _merge_source(values, cfg, verbose=verbose, label="config.yaml")

    # --- Source 3: clusters.csv ---
    csv_path = bp / MASTER_CSV_NAME
    if csv_path.exists():
        csv_values = _read_cluster_csv(csv_path, identifier)
        if csv_values:
            _merge_source(values, csv_values, verbose=verbose, label="clusters.csv")

    # --- Source 4: External queries (only for missing identity fields) ---
    if "name" not in values:
        values["name"] = identifier  # Use identifier as name if nothing better

    if "ra" not in values or "dec" not in values:
        try:
            from cluster_pipeline.utils.resolvers import get_coordinates
            coords = get_coordinates(identifier)
            if "ra" not in values:
                values["ra"] = float(coords.ra.deg)
            if "dec" not in values:
                values["dec"] = float(coords.dec.deg)
            if verbose:
                print(f"  [query] coords: {values.get('ra')}, {values.get('dec')}")
        except Exception as e:
            if verbose:
                print(f"  [query] coords lookup failed: {e}")

    if "redshift" not in values:
        try:
            from cluster_pipeline.utils.resolvers import get_redshift
            values["redshift"] = float(get_redshift(identifier))
            if verbose:
                print(f"  [query] redshift: {values['redshift']}")
        except Exception as e:
            if verbose:
                print(f"  [query] redshift lookup failed: {e}")

    # --- Source 5: Defaults (analysis parameters only) ---
    _apply_defaults(values)

    # --- Apply all collected values to the cluster ---
    _apply_values(cluster, values)

    if verbose:
        print(f"  Cluster initialized: {cluster.identifier} (z={cluster.redshift})")

    return cluster


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _merge_source(
    values: dict[str, Any],
    source: dict[str, Any],
    *,
    verbose: bool = False,
    label: str = "",
) -> None:
    """Merge values from a source dict, only filling keys not already set."""
    # Handle nested 'analysis' section from config.yaml
    if "analysis" in source and isinstance(source["analysis"], dict):
        for k, v in source["analysis"].items():
            if k not in values and v is not None:
                values[k] = v
                if verbose:
                    print(f"  [{label}] {k}: {v}")
        source = {k: v for k, v in source.items() if k != "analysis"}

    # Handle nested 'xray' section
    if "xray" in source and isinstance(source["xray"], dict):
        xray = source["xray"]
        if "psf" in xray and "psf" not in values:
            values["psf"] = xray["psf"]
        if "contour_levels" in xray and "contour_levels" not in values:
            values["contour_levels"] = tuple(xray["contour_levels"])
        source = {k: v for k, v in source.items() if k != "xray"}

    for key, val in source.items():
        if key in ("bcgs", "subclusters", "groups"):
            continue  # These are handled separately, not Cluster fields
        if key not in values and val is not None and not _is_missing(val):
            values[key] = val
            if verbose:
                print(f"  [{label}] {key}: {val}")


def _read_cluster_csv(csv_path: Path, identifier: str) -> dict[str, Any]:
    """Read a single cluster's row from clusters.csv."""
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return {}

    if "identifier" not in df.columns:
        return {}

    row = df.loc[df["identifier"].astype(str).str.strip() == str(identifier).strip()]
    if row.empty:
        return {}

    return {k: v for k, v in row.iloc[0].to_dict().items() if k != "identifier"}


def _apply_defaults(values: dict[str, Any]) -> None:
    """Fill in package defaults for analysis parameters (not identity fields)."""
    defaults = {
        "fov": DEFAULT_FOV_ARCMIN,
        "fov_full": DEFAULT_FOV_FULL_ARCMIN,
        "ra_offset": DEFAULT_RA_OFFSET_ARCMIN,
        "dec_offset": DEFAULT_DEC_OFFSET_ARCMIN,
        "survey": DEFAULT_SURVEY,
        "color_type": DEFAULT_COLOR_TYPE,
        "phot_levels": DEFAULT_PHOT_LEVELS,
        "phot_skip": DEFAULT_PHOT_SKIP,
        "contour_levels": DEFAULT_CONTOUR_LEVELS,
        "psf": DEFAULT_PSF_ARCSEC,
        "bandwidth": DEFAULT_BANDWIDTH,
    }
    for key, default in defaults.items():
        if key not in values:
            values[key] = default


def _apply_values(cluster: Cluster, values: dict[str, Any]) -> None:
    """Set values on the Cluster object with type coercion."""
    # Float fields
    for key in ("ra", "dec", "ra_offset", "dec_offset", "redshift", "redshift_err",
                "z_min", "z_max", "richness", "richness_err", "fov", "fov_full",
                "psf", "bandwidth"):
        if key in values:
            setattr(cluster, key, _safe_float(values[key]))

    # Int fields
    for key in ("phot_levels", "phot_skip"):
        if key in values:
            setattr(cluster, key, _safe_int(values[key]))

    # String fields
    for key in ("name", "survey", "color_type"):
        if key in values and values[key] is not None:
            setattr(cluster, key, str(values[key]))

    # Tuple fields
    if "contour_levels" in values:
        cl = values["contour_levels"]
        if isinstance(cl, (list, tuple)) and len(cl) == 3:
            cluster.contour_levels = (float(cl[0]), float(cl[1]), float(cl[2]))
        elif isinstance(cl, str):
            parts = cl.replace("(", "").replace(")", "").split(",")
            if len(parts) == 3:
                cluster.contour_levels = (float(parts[0]), float(parts[1]), float(parts[2]))


def _is_missing(val: Any) -> bool:
    """Check if a value is None, empty string, or NaN."""
    if val is None:
        return True
    if isinstance(val, str) and val.strip() == "":
        return True
    try:
        if isinstance(val, float) and np.isnan(val):
            return True
    except (TypeError, ValueError):
        pass
    return False


def _safe_float(val: Any) -> float | None:
    if _is_missing(val):
        return None
    try:
        v = float(val)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _safe_int(val: Any) -> int | None:
    if _is_missing(val):
        return None
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return None
