#!/usr/bin/env python3
"""
photometry.py

Stage 2: Archival Photometric Catalog Retrieval
---------------------------------------------------------

Queries archival photometric catalogs around a cluster position from:
  - PanSTARRS DR1 (via VizieR)
  - Legacy DR10 (via NOIRLab TAP and Q3C_RADIAL_QUERY)

Computes g-r, r-i, and g-i colors and luminosity weights in each band.
Surveys are NOT combined — Legacy and PanSTARRS are archived separately
since their photometric systems are not directly interchangeable.

Data products (archived per survey, before any deduplication):
  - Photometry/photometry_legacy.csv      Legacy DR10 photometry
  - Photometry/photometry_panstarrs.csv   PanSTARRS DR1 photometry

Column convention:
  RA, Dec, gmag, rmag, imag, phot_source, lum_weight_g, lum_weight_r,
  lum_weight_i, g_r, r_i, g_i

Requirements:
  - astropy, astroquery, numpy, pandas

Notes:
  - Each survey is queried independently; a failure in one does not prevent
    the other from running.
  - Missing magnitudes are stored as NaN (never -9999 or inf).
  - Luminosity weights are unnormalized flux proportional (10^(-0.4 * mag)).
  - Legacy uses paginated TAP queries to handle large catalogs.
"""

from __future__ import annotations

import os

import pandas as pd
import numpy as np

import astropy.units as u
from astroquery.vizier import Vizier
from astroquery.utils.tap.core import TapPlus

from cluster_pipeline.models.cluster import Cluster
from cluster_pipeline.utils import coerce_to_numeric

# ------------------------------------
# Constants
# ------------------------------------
DEFAULT_RADIUS_ARCMIN = 10.0
DEFAULT_LEGACY_STEP = 2000
DEFAULT_LEGACY_MAX_ROWS = 150000

PANSTARRS_CATALOG = "II/349/ps1"
LEGACY_URL = "https://datalab.noirlab.edu/tap"
LEGACY_TABLE = "ls_dr10.tractor"

# Standard column names
MAG_COLS = ("gmag", "rmag", "imag")
LUM_WEIGHT_COLS = ("lum_weight_g", "lum_weight_r", "lum_weight_i")
COLOR_COLS = ("g_r", "r_i", "g_i")
COLUMNS = ["RA", "Dec", *MAG_COLS, "phot_source", *LUM_WEIGHT_COLS, *COLOR_COLS]

# Sentinel values to replace with NaN
MAGNITUDE_SENTINELS = [-9999.0, -9999, -999.0, -999, 99.0, 99.99]


# ====================================================================
# Public interface
# ====================================================================

def run_photometry(
    cluster: Cluster,
    *,
    surveys: list[str] | None = None,
    radius_arcmin: float = DEFAULT_RADIUS_ARCMIN,
    retrieve: bool = True,
) -> dict[str, pd.DataFrame]:
    """Query archival photometry and return per-survey catalogs.

    Each survey is queried independently — a failure in one does not
    prevent the other from running.

    Parameters
    ----------
    cluster : Cluster
        Must have ``coords`` set (ra, dec) and valid ``photometry_path``.
    surveys : list of str, optional
        Which surveys to query. Default: ``["legacy", "panstarrs"]``.
    radius_arcmin : float
        Search radius in arcminutes. [default: 10.0]
    retrieve : bool
        If True, query the survey APIs. If False, load from existing CSVs
        on disk. [default: True]

    Returns
    -------
    catalogs : dict[str, pd.DataFrame]
        Keyed by survey name (``"legacy"``, ``"panstarrs"``).
        Each DataFrame has columns:
        ``RA, Dec, gmag, rmag, imag, phot_source, lum_weight_g,
        lum_weight_r, lum_weight_i, g_r, r_i, g_i``.

    Notes
    -----
    - Surveys are NOT combined — their photometric systems are not
      directly interchangeable.
    - Each survey is archived individually to
      ``Photometry/photometry_{survey}.csv``.
    - Missing magnitudes are stored as NaN (sentinels like -9999 are cleaned).
    """
    if cluster.coords is None:
        raise ValueError("Cluster coordinates not set. Cannot query photometry.")

    if surveys is None:
        surveys = ["legacy", "panstarrs"]

    coord = cluster.coords
    cluster.ensure_directories()

    print(f"\nPhotometry: RA={coord.ra.deg:.5f}, Dec={coord.dec.deg:.5f}, "
          f"radius={radius_arcmin}', surveys={surveys}")

    catalogs: dict[str, pd.DataFrame] = {}

    for survey in surveys:
        out_path = os.path.join(cluster.photometry_path, f"photometry_{survey}.csv")

        if retrieve:
            df = _query_survey(survey, coord, radius_arcmin)
        else:
            df = _load_from_disk(out_path, survey)

        # Compute derived columns and clean sentinels
        df = _clean_magnitudes(df)
        df = _add_derived_columns(df)

        # Archive
        if retrieve and not df.empty:
            df.to_csv(out_path, index=False)
            print(f"  {survey}: {len(df)} sources → {out_path}")
        elif not retrieve:
            print(f"  {survey}: {len(df)} sources loaded from {out_path}")

        catalogs[survey] = df

    return catalogs


# ====================================================================
# Query functions
# ====================================================================

def query_panstarrs(
    coord,
    radius_arcmin: float,
) -> pd.DataFrame:
    """Query PanSTARRS DR1 photometry via VizieR.

    Parameters
    ----------
    coord : SkyCoord
        Central search coordinate.
    radius_arcmin : float
        Search radius in arcminutes.

    Returns
    -------
    pd.DataFrame
        Photometry catalog with columns: ``RA, Dec, gmag, rmag, imag, phot_source``.
        Empty DataFrame if query fails or returns no results.

    Notes
    -----
    - Uses VizieR catalog II/349/ps1 (PanSTARRS DR1 mean photometry).
    - Row limit is set to -1 (unlimited) to retrieve all sources.
    """
    print(f"  PanSTARRS query...")

    vizier = Vizier(
        columns=["RAJ2000", "DEJ2000", "gmag", "rmag", "imag"],
        row_limit=-1,
    )

    try:
        result = vizier.query_region(
            coord,
            radius=radius_arcmin * u.arcmin,
            catalog=PANSTARRS_CATALOG,
        )
    except Exception as e:
        print(f"  PanSTARRS query failed ({type(e).__name__}): {e}")
        return pd.DataFrame(columns=COLUMNS)

    if not result:
        print("  PanSTARRS: no results found")
        return pd.DataFrame(columns=COLUMNS)

    df = result[0].to_pandas().rename(columns={"RAJ2000": "RA", "DEJ2000": "Dec"})
    df["phot_source"] = "panstarrs"

    return _standardize(df)


def query_legacy(
    coord,
    radius_arcmin: float,
    *,
    step: int = DEFAULT_LEGACY_STEP,
    max_rows: int = DEFAULT_LEGACY_MAX_ROWS,
) -> pd.DataFrame:
    """Query Legacy DR10 photometry via NOIRLab TAP.

    Parameters
    ----------
    coord : SkyCoord
        Central search coordinate.
    radius_arcmin : float
        Search radius in arcminutes.
    step : int
        Page size for paginated query. [default: 2000]
    max_rows : int
        Maximum total rows to retrieve. [default: 150000]

    Returns
    -------
    pd.DataFrame
        Photometry catalog with columns: ``RA, Dec, gmag, rmag, imag, phot_source``.
        Empty DataFrame if query fails or returns no results.

    Notes
    -----
    - Uses Q3C_RADIAL_QUERY for efficient spatial indexing.
    - Only returns brick_primary=1 sources (avoids duplicate detections).
    - Paginated to handle large catalogs.
    """
    print(f"  Legacy DR10 query...")

    tap = TapPlus(url=LEGACY_URL)
    ra, dec = float(coord.ra.deg), float(coord.dec.deg)
    radius_deg = radius_arcmin / 60.0

    chunks: list[pd.DataFrame] = []

    for offset in range(0, int(max_rows), int(step)):
        query = f"""
        SELECT ra, dec, mag_g, mag_r, mag_i
        FROM {LEGACY_TABLE}
        WHERE brick_primary = 1
        AND 't' = q3c_radial_query(ra, dec, {ra:.6f}, {dec:.6f}, {radius_deg:.6f})
        ORDER BY ra
        LIMIT {int(step)}
        OFFSET {int(offset)}
        """

        try:
            job = tap.launch_job(query)
            chunk = job.get_results().to_pandas()
        except Exception as e:
            print(f"  Legacy query failed at offset {offset} ({type(e).__name__}): {e}")
            break

        if chunk.empty:
            break

        chunks.append(chunk)

        if len(chunk) < step:
            break  # Last page

    if not chunks:
        print("  Legacy: no results found")
        return pd.DataFrame(columns=COLUMNS)

    df = pd.concat(chunks, ignore_index=True)
    df = df.rename(columns={"ra": "RA", "dec": "Dec", "mag_g": "gmag", "mag_r": "rmag", "mag_i": "imag"})
    df["phot_source"] = "legacy"

    return _standardize(df)


# ====================================================================
# Derived columns
# ====================================================================

def _add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add luminosity weights and colors to a photometry DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must have ``gmag``, ``rmag``, ``imag`` columns (may contain NaN).

    Returns
    -------
    pd.DataFrame
        DataFrame with added columns: ``lum_weight_g, lum_weight_r,
        lum_weight_i, g_r, r_i, g_i``.

    Notes
    -----
    - Luminosity weight = 10^(-0.4 * mag), proportional to flux (AB system).
    - Colors are simple magnitude differences. NaN propagates naturally.
    """
    out = df.copy()

    # Ensure magnitudes are numeric
    for col in MAG_COLS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # Luminosity weights (all bands — downstream uses lum_weight_r for density)
    for mag_col, weight_col in zip(MAG_COLS, LUM_WEIGHT_COLS):
        if mag_col in out.columns:
            out[weight_col] = 10.0 ** (-0.4 * out[mag_col])
        else:
            out[weight_col] = np.nan

    # Colors
    out["g_r"] = out.get("gmag", np.nan) - out.get("rmag", np.nan)
    out["r_i"] = out.get("rmag", np.nan) - out.get("imag", np.nan)
    out["g_i"] = out.get("gmag", np.nan) - out.get("imag", np.nan)

    return out


# ====================================================================
# Internal helpers
# ====================================================================

def _query_survey(name: str, coord, radius_arcmin: float) -> pd.DataFrame:
    """Dispatch to the appropriate query function by survey name."""
    if name == "legacy":
        return query_legacy(coord, radius_arcmin)
    elif name == "panstarrs":
        return query_panstarrs(coord, radius_arcmin)
    else:
        print(f"  Unknown survey '{name}', skipping")
        return pd.DataFrame(columns=COLUMNS)


def _load_from_disk(path: str, survey: str) -> pd.DataFrame:
    """Load a photometry CSV from disk, or return empty DataFrame."""
    if os.path.exists(path):
        df = pd.read_csv(path)
        return _standardize(df, source=survey)
    print(f"  {survey}: file not found at {path}")
    return pd.DataFrame(columns=COLUMNS)


def _standardize(df: pd.DataFrame, source: str | None = None) -> pd.DataFrame:
    """Ensure standard column names, types, and ordering.

    Parameters
    ----------
    df : pd.DataFrame
        Raw photometry DataFrame.
    source : str or None
        Phot source label to set (e.g., "legacy", "panstarrs").

    Returns
    -------
    pd.DataFrame
        Standardized DataFrame with NaN for missing columns, numeric types
        enforced, and rows with missing coordinates dropped.
    """
    out = df.copy()

    # Ensure all expected columns exist
    for col in COLUMNS:
        if col not in out.columns:
            out[col] = np.nan

    # Numeric coercion
    numeric_cols = ["RA", "Dec", *MAG_COLS]
    out = coerce_to_numeric(out, numeric_cols)

    # Drop rows with missing coordinates
    out = out.dropna(subset=["RA", "Dec"]).reset_index(drop=True)

    # Set source
    if source is not None:
        out["phot_source"] = source

    return out


def _clean_magnitudes(df: pd.DataFrame) -> pd.DataFrame:
    """Replace magnitude sentinel values (-9999, inf, etc.) with NaN.

    Parameters
    ----------
    df : pd.DataFrame
        Photometry DataFrame with magnitude columns.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with sentinels replaced by NaN.
    """
    out = df.copy()

    for col in MAG_COLS:
        if col not in out.columns:
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce")
        # Replace sentinel values
        out.loc[out[col].isin(MAGNITUDE_SENTINELS), col] = np.nan
        # Replace inf
        out[col] = out[col].replace([np.inf, -np.inf], np.nan)

    return out
