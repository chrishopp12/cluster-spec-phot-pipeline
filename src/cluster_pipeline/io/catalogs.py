#!/usr/bin/env python3
"""
catalogs.py

Central I/O for Cluster Catalog Files
---------------------------------------------------------

Handles reading and writing of all CSV-based data products for a cluster.
Provides path construction for red-sequence catalogs, bulk loading of the
three core DataFrames (combined redshifts, matched photometry, members),
and the authoritative BCG reader/filter functions.

Key functions:
  - get_redseq_filename()   Construct red-sequence catalog paths by survey/color
  - load_dataframes()       Load the three core catalogs into DataFrames
  - read_bcg_csv()          Authoritative BCG CSV reader (with type coercion)
  - select_bcgs()           Filter BCGs by subcluster membership
  - bcg_basic_info()        Summary info for a BCG subset
  - load_photo_coords()     Extract RA/Dec arrays from photometric catalogs

Requirements:
  - pandas, numpy

Notes:
  - All numeric columns are coerced via coerce_to_numeric() on read.
  - Missing values are NaN throughout; no sentinel values.
  - BCG reading goes through read_bcg_csv() to ensure consistent types.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import numpy as np

from cluster_pipeline.utils import coerce_to_numeric

if TYPE_CHECKING:
    from cluster_pipeline.models.cluster import Cluster


def get_redseq_filename(
        photometry_folder: str,
        survey: str,
        color_type: str
    ) -> str:
    """
    Construct the expected filename for a red sequence catalog CSV based on survey and color band.

    Parameters
    ----------
    photometry_folder : str
        Path to the folder containing photometric catalogs for the cluster.
    survey : str
        Survey identifier, case-insensitive. Supported values: "Legacy" or "PanSTARRS".
    color_type : str
        Color band combination for the red sequence selection.
        Supported formats: "g-r", "r-i", or "g-i" (with or without dashes or underscores, case-insensitive).

    Returns
    -------
    filename : str
        Full path to the corresponding red sequence CSV file, e.g.,
        '.../redseq_legacy_gr.csv' or '.../redseq_panstarrs_gi.csv'

    Raises
    ------
    ValueError
        If an unrecognized survey or color_type is provided.
    """

    # Normalize inputs
    survey_map = {"panstarrs": "panstarrs", "legacy": "legacy"}
    color_map = {"g-r": "gr", "r-i": "ri", "g-i": "gi",
                 "gr": "gr", "ri": "ri", "gi": "gi"}

    survey_str = survey.lower()
    color_str = color_type.lower().replace("_", "-")
    if survey_str not in survey_map:
        raise ValueError(f"Unknown survey: {survey}")
    if color_str not in color_map:
        raise ValueError(f"Unknown color_type: {color_type}")

    filename = f"redseq_{survey_map[survey_str]}_{color_map[color_str]}.csv"
    return os.path.join(photometry_folder, filename)


def load_dataframes(
        cluster: "Cluster",
        survey: str | None = None,
        color_type: str | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load spectroscopic and photometric dataframes for a given cluster.

    Parameters
    ----------
    cluster : "Cluster"
        Cluster object containing paths to data.
    survey : str, optional
        Photometric survey name (default: cluster default).
    color_type : str, optional
        Color type for red sequence selection (default: cluster default).

    Returns
    -------
    spec_df : pd.DataFrame
        combined_redshifts.csv: All redshifts, no photometry.
    member_df : pd.DataFrame
        redseq csv: All members as determined by cmd OR spectroscopy.
    bcg_df : pd.DataFrame
        BCG candidates catalog.
    """

    spec_df = pd.read_csv(cluster.spec_file)
    member_df = pd.read_csv(cluster.get_phot_file(survey=survey, color_type=color_type))
    bcg_df = pd.read_csv(cluster.bcg_file)

    return spec_df, member_df, bcg_df



def read_bcg_csv(
    cluster: "Cluster",
    *,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Read the per-cluster BCG CSV and return the full DataFrame.

    This function does NOT filter rows or select columns.
    It is the single authoritative reader for BCGs.csv.

    Required columns
    ----------------
    - RA
    - Dec   (Dec or DEC accepted)

    Optional / inherited columns
    ----------------------------
    - BCG_priority
    - BCG_probability
    - z, sigma_z, spec_source
    - gmag, rmag, imag, g_r, r_i, g_i, lum_weight_*
    - phot_source
    - any future columns

    Parameters
    ----------
    cluster : Cluster
        Must have attribute `bcg_file`.
    verbose : bool
        Print diagnostics.

    Returns
    -------
    df : pd.DataFrame
        Full BCG table. Empty DataFrame if file missing or empty.
    """
    path = Path(cluster.bcg_file)
    if not path.exists():
        if verbose:
            print(f"BCG file not found: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    if df.empty:
        return df

    # Normalize Dec column name
    if "Dec" not in df.columns and "DEC" in df.columns:
        df = df.rename(columns={"DEC": "Dec"})

    numeric_cols = [
        "RA", "Dec", "BCG_priority", "BCG_probability",
        "z", "sigma_z",
        "gmag", "rmag", "imag",
        "g_r", "r_i", "g_i",
        "lum_weight_g", "lum_weight_r", "lum_weight_i",
        ]

    df = coerce_to_numeric(df, numeric_cols)

    return df


def select_bcgs(
    bcg_df: pd.DataFrame,
    *,
    bcg_id: int | None = None,
    rm_only: bool = False,
) -> pd.DataFrame:
    """
    Select BCG rows from a full BCG DataFrame.

    Parameters
    ----------
    bcg_df : pd.DataFrame
        Full BCG DataFrame, as returned by `read_bcg_csv()`.
    bcg_id : int or None
        If specified, select only the row with this BCG ID.
    rm_only : bool
        If True, select only rows with BCG_priority 1-5 (inclusive).

    Returns
    -------
    pd.DataFrame
         Filtered BCG DataFrame according to the specified criteria. Sorted by BCG_priority if that column exists.
    """
    if bcg_df.empty:
        return bcg_df

    out = bcg_df.copy()

    if "BCG_priority" in out.columns:
        if rm_only:
            out = out[(out["BCG_priority"] >= 1) & (out["BCG_priority"] <= 5)]
        if bcg_id is not None:
            out = out[out["BCG_priority"] == int(bcg_id)]

    return out.sort_values("BCG_priority") if "BCG_priority" in out.columns else out


def bcg_basic_info(
    bcg_df: pd.DataFrame,
) -> list[tuple[float, float, float | None, float | None]]:
    """
    Convert BCG DataFrame to (ra, dec, z, P) tuples.

    Parameters
    ----------
    bcg_df : pd.DataFrame
        BCG DataFrame with at least 'RA' and 'Dec' columns.

    Returns
    -------
    list of tuples
        List of (ra, dec, z, P) tuples for each BCG.
        z and P are None if not present or NaN.
    """
    out = []
    for _, r in bcg_df.iterrows():
        ra = float(r["RA"])
        dec = float(r["Dec"])
        z = None if pd.isna(r.get("z")) else float(r["z"])
        p = None if pd.isna(r.get("BCG_probability")) else float(r["BCG_probability"])
        out.append((ra, dec, z, p))
    return out


def load_photo_coords(
        csv_file: str,
        ra_col: str = "RA",
        dec_col: str = "Dec",
        mag_col: str = "lum_weight_r"
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads RA and Dec from a CSV file.

    Parameters
    ----------
    csv_file : str
        Path to CSV file.
    ra_col, dec_col, mag_col : str
        Column names for RA, Dec, and magnitude.

    Returns
    -------
    ra : np.ndarray
        Right Ascension values in degrees.
    dec : np.ndarray
        Declination values in degrees.
    """
    df = pd.read_csv(csv_file)
    df_clean = df[(df[mag_col]) > 0]
    # df_clean = df[(df["on_red_sequence"])]

    return df_clean[ra_col].values, df_clean[dec_col].values, df_clean[mag_col].values
