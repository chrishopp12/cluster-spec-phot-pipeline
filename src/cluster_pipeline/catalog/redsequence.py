#!/usr/bin/env python3
"""
redsequence.py

Stage 4: Red Sequence Fitting and Cluster Member Selection
----------------------------------------------------------
Fits the red sequence in color-magnitude space using spectroscopically
confirmed cluster members, then selects photometric members that fall
on that sequence.  Builds a deduplicated cluster member catalog
combining spectroscopic and photometric members.

Data products:
  - Members/cluster_members.csv          Deduplicated member catalog
  - Members/redseq_{survey}_{color}.csv  Per-fit intermediate results

Column conventions (output member catalog):
  RA, Dec, z, sigma_z, spec_source, gmag, rmag, imag, g_r, r_i, g_i,
  lum_weight_r, phot_source, member_type

Notes:
  - The red sequence is fit with iterative sigma-clipped linear
    regression on the (magnitude, color) plane of spec-z members.
  - Photometric members are sources whose color lies within
    ``color_band`` of the fit line, after applying a minimum
    magnitude cut.
  - ``member_type`` is "spec" for spectroscopic members (z in the
    cluster redshift range) or "phot" for red-sequence-only sources.
  - When a source has both a spec match and a red-sequence detection,
    the spec version is kept (with all photometric columns attached).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from cluster_pipeline.models.cluster import Cluster
from cluster_pipeline.utils import coerce_to_numeric, get_color_mag_functions
from cluster_pipeline.utils.coordinates import make_skycoord
from cluster_pipeline.catalog.matching import match_skycoords_unique
from cluster_pipeline.constants import DEFAULT_MAG_MIN, DEFAULT_COLOR_BAND


# ---------------------------------------------------------------
# Constants
# ---------------------------------------------------------------
DEFAULT_MAG = "rmag"

RA_COL = "RA"
DEC_COL = "Dec"
Z_COL = "z"
SIGMA_Z_COL = "sigma_z"
SPEC_SOURCE_COL = "spec_source"
PHOT_SOURCE_COL = "phot_source"

# Columns retained in the final member catalog
MEMBER_COLS = [
    "RA", "Dec", "z", "sigma_z", "spec_source",
    "gmag", "rmag", "imag", "g_r", "r_i", "g_i",
    "lum_weight_r", "phot_source", "member_type",
]


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def _check_required_columns(
        df: pd.DataFrame,
        required_cols: set[str],
        df_name: str = "DataFrame"
    ) -> bool:
    """
    Check if the DataFrame contains all required columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to check.
    required_cols : set of str
        Set of required column names.
    df_name : str
        Name of the DataFrame for error messages.

    Returns
    -------
    bool
        True if all required columns are present, False otherwise.
    """
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        missing_str = ", ".join(sorted(missing_cols))
        print(f"{df_name} is missing required columns: {missing_str}")
        return False
    return True


def _filter_by_redshift(
        df: pd.DataFrame,
        zmin: float,
        zmax: float
    ) -> pd.DataFrame:
    """
    Filter DataFrame to only include rows with redshift within [zmin, zmax].

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a 'z' column.
    zmin : float
        Minimum redshift.
    zmax : float
        Maximum redshift.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with redshifts in the specified range.
    """
    return df[(df[Z_COL] >= zmin) & (df[Z_COL] <= zmax)].copy()


# ---------------------------------------------------------------
# Red Sequence Fitting and Member Selection
# ---------------------------------------------------------------

def iterative_linear_fit(
        x: np.ndarray,
        y: np.ndarray,
        *,
        sigma_clip: float = 2.0,
        max_iter: int = 10
    ) -> tuple[LinearRegression | None, np.ndarray | None]:
    """
    Iteratively fits a linear model using sigma-clipping to remove outliers.

    Parameters
    ----------
    x : np.ndarray
        1D array of magnitudes.
    y : np.ndarray
        1D array of colors.
    sigma_clip : float
        Sigma threshold for clipping residuals. [default: 2.0]
    max_iter : int
        Maximum number of iterations. [default: 10]

    Returns
    -------
    model : LinearRegression | None
        Trained linear regression model, or None if no fit could be performed.
    inlier_mask : np.ndarray | None
        Boolean mask of inlier points after clipping, or None if no fit could be performed.
    """
    x = np.asarray(x).reshape(-1, 1)
    y = np.asarray(y)

    mask = np.ones_like(y, dtype=bool)
    if x[mask].shape[0] == 0:
        print("Warning: No valid data to fit red sequence!")
        return None, None

    model: LinearRegression | None = None
    for _ in range(max_iter):
        model = LinearRegression().fit(x[mask], y[mask])
        y_pred = model.predict(x[mask])
        residuals = y[mask] - y_pred
        std = np.std(residuals)

        # If std is zero, all points are identical
        if std == 0:
            break

        new_mask = np.abs(residuals) < sigma_clip * std

        # Break if no new points are removed
        if np.all(new_mask):
            break

        # Update mask for next iteration
        full_mask = mask.copy()
        full_mask[mask] = new_mask
        mask = full_mask

    return model, mask


def fit_red_sequence(
    matched_df: pd.DataFrame,
    *,
    zmin: float,
    zmax: float,
    color_type: str = "g-r",
) -> tuple[float, float]:
    """
    Fits a linear red sequence to spectroscopic members and returns the fit parameters.

    Parameters
    ----------
    matched_df : pd.DataFrame
        DataFrame with photometry added for all matched spec sources.
    zmin, zmax : float
        Redshift bounds for selecting spectroscopic members.
    color_type : str, optional
        Color combination to use: "g-r", "r-i", or "g-i" [default: "g-r"]

    Returns
    -------
    a : float
        Slope of the red sequence fit (NaN if fit failed).
    b : float
        Intercept of the red sequence fit (NaN if fit failed).
    """
    required_cols, color_label, mag_label, color_func, mag_func = get_color_mag_functions(color_type)

    # Drop rows missing required mags
    spec_members = _filter_by_redshift(matched_df, zmin, zmax)
    fit_data = spec_members.dropna(subset=required_cols)

    if fit_data.empty:
        print("No valid spec-z members with required magnitudes.")
        return np.nan, np.nan

    x = mag_func(fit_data).to_numpy().reshape(-1, 1)
    y = color_func(fit_data).to_numpy()  # y can remain 1D

    model, _ = iterative_linear_fit(x, y)
    if model is None:
        print("Red sequence fit failed.")
        return np.nan, np.nan
    a = float(model.coef_[0])
    b = float(model.intercept_)
    print(f"Fit: {color_label} = {a:.3f} * {mag_label} + {b:.3f}")

    return a, b


def assign_red_sequence(
    df: pd.DataFrame,
    *,
    required_cols: list[str],
    mag_func,
    color_func,
    a: float,
    b: float,
    color_band: float,
) -> pd.DataFrame:
    """
    Adds/overwrites:
      - delta_color: color - (a*mag + b)
      - on_red_sequence: abs(delta_color) < color_band
    Only computed for rows with full required photometry; others get NaN/False.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    required_cols : list of str
        List of required magnitude columns for the chosen color_type.
    mag_func : callable
        Function to compute magnitude from the DataFrame.
    color_func : callable
        Function to compute color from the DataFrame.
    a : float
        Slope of red sequence fit.
    b : float
        Intercept of red sequence fit.
    color_band : float
        Max allowed deviation in color to be considered on the red sequence.

    Returns
    -------
    pd.DataFrame
        DataFrame with added/updated 'delta_color' and 'on_red_sequence' columns.
    """
    out = df.copy()

    # Ensure columns exist
    out["delta_color"] = np.nan
    out["on_red_sequence"] = False

    # Only rows with complete photometry can be evaluated
    has_phot = out[required_cols].notna().all(axis=1)
    if has_phot.any():
        mag = mag_func(out.loc[has_phot]).to_numpy()
        color = color_func(out.loc[has_phot]).to_numpy()
        delta = color - (a * mag + b)

        out.loc[has_phot, "delta_color"] = delta
        out.loc[has_phot, "on_red_sequence"] = (np.abs(delta) < color_band).astype(bool)

    return out


def apply_membership_cuts(
    df: pd.DataFrame,
    *,
    zmin: float,
    zmax: float,
) -> pd.DataFrame:
    """
    Keep:
      - any row with spec-z in [zmin,zmax]  (member regardless of color)
      - OR any row with no spec-z and on_red_sequence==True (phot candidate)
    Drop:
      - any row with spec-z outside range (never a member)

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    zmin : float
        Minimum redshift.
    zmax : float
        Maximum redshift.

    Returns
    -------
    pd.DataFrame
        DataFrame filtered to only include members as per the criteria.
    """
    out = df.copy()

    if Z_COL not in out.columns:
        print(f"Warning: '{Z_COL}' column not found in DataFrame. No spec-z cuts applied.")
        return out.loc[out["on_red_sequence"].astype(bool)].copy()

    has_z = out[Z_COL].notna()
    in_range = has_z & (out[Z_COL] >= zmin) & (out[Z_COL] <= zmax)
    out_of_range = has_z & ~in_range

    # 1) hard drop spec-z out of range
    out = out.loc[~out_of_range].copy()

    # 2) keep in-range spec OR (no spec and on RS)
    has_z = out[Z_COL].notna()
    keep = has_z | (~has_z & out["on_red_sequence"].astype(bool))

    return out.loc[keep].copy()


def apply_mag_cut(
    df: pd.DataFrame,
    *,
    mag_func: callable,
    mag_min: float = DEFAULT_MAG_MIN,
) -> pd.DataFrame:
    """
    Apply minimum magnitude cut to DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    mag_func : callable
        Function to compute magnitude from the DataFrame.
    mag_min : float, optional
        Minimum magnitude to keep [default: 16.0].

    Returns
    -------
    pd.DataFrame
        DataFrame filtered to only include rows with magnitude > mag_min.
    """
    out = df.copy()

    mag = mag_func(out)
    has_mag = mag.notna()
    bright_enough = mag > mag_min
    keep = has_mag & bright_enough

    return out.loc[keep].copy()


def fit_and_select_red_sequence(
    cluster: Cluster,
    matched_df: pd.DataFrame,
    full_df: pd.DataFrame,
    *,
    color_band: float = 0.2,
    color_type: str = "g-r",
    survey: str = "Legacy",
    mag_min: float = DEFAULT_MAG_MIN,
    zmin: float | None = None,
    zmax: float | None = None,
) -> pd.DataFrame | None:
    """
    Fits a linear red sequence to spectroscopic members and selects photometric matches.

    Parameters
    ----------
    cluster : Cluster
        Cluster object containing all necessary metadata (paths, coordinates, etc.).
    matched_df : pd.DataFrame
        DataFrame with photometry added for all matched spec sources.
    full_df : pd.DataFrame
        DataFrame of all photometric sources.
    color_band : float, optional
        Max allowed deviation in color to be selected [default: 0.2 mag]
    color_type : str, optional
        Color combination to use: "g-r", "r-i", or "g-i" [default: "g-r"]
    survey : str, optional
        Survey name for labeling (e.g., "PanSTARRS", "Legacy") [default: "Legacy"]
    mag_min : float, optional
        Minimum magnitude to consider [default: 16.0]
    zmin, zmax : float, optional
        Redshift bounds; defaults to cluster.z_min/z_max if None.

    Returns
    -------
    selected_df : pd.DataFrame or None
        Photometric members on the red sequence (+spec-z matches),
        or None if fitting/selection failed.
    """
    # Use cluster z_min/z_max unless overridden
    zmin = zmin if zmin is not None else cluster.z_min
    zmax = zmax if zmax is not None else cluster.z_max
    print(f"Using zmin = {zmin}, zmax = {zmax}")

    # Define bands based on color choice
    try:
        required_cols, color_label, mag_label, color_func, mag_func = get_color_mag_functions(color_type)
    except ValueError as exc:
        print(f"{survey} {color_type}: {exc}")
        return None

    # Verify required columns
    needed_matched = {RA_COL, DEC_COL, Z_COL, *required_cols}

    if not _check_required_columns(matched_df, needed_matched, "matched_df"):
        print(f"Skipping {survey} {color_type}: missing required columns in matched_df.")
        return None

    a, b = fit_red_sequence(
        matched_df,
        color_type=color_type,
        zmin=zmin,
        zmax=zmax,
    )

    full_catalog = assign_red_sequence(
        full_df,
        required_cols=required_cols,
        mag_func=mag_func,
        color_func=color_func,
        a=a,
        b=b,
        color_band=color_band,
    )

    full_catalog = apply_mag_cut(
        full_catalog,
        mag_func=mag_func,
        mag_min=mag_min,
    )

    # Merge spec-z info from matched catalog into full photometry catalog
    # so apply_membership_cuts can distinguish spec vs phot-only members
    spec_cols = [Z_COL, SIGMA_Z_COL, SPEC_SOURCE_COL]
    available_spec_cols = [c for c in spec_cols if c in matched_df.columns]
    if available_spec_cols:
        matched_coords = make_skycoord(matched_df[RA_COL], matched_df[DEC_COL])
        full_coords = make_skycoord(full_catalog[RA_COL], full_catalog[DEC_COL])
        full_idx, matched_idx, _ = match_skycoords_unique(
            full_coords, matched_coords, match_tol_arcsec=3.0,
        )
        for col in available_spec_cols:
            if col not in full_catalog.columns:
                full_catalog[col] = pd.Series(
                    index=full_catalog.index, dtype=matched_df[col].dtype
                )
            full_catalog.iloc[full_idx, full_catalog.columns.get_loc(col)] = (
                matched_df.iloc[matched_idx][col].values
            )

    selected_df = apply_membership_cuts(
        full_catalog,
        zmin=zmin,
        zmax=zmax,
    )

    if selected_df.empty:
        print(f"Skipping {survey} {color_type}: no photometric members after cuts.")
        return None

    return selected_df


# ---------------------------------------------------------------
# Member Catalog Construction
# ---------------------------------------------------------------

def build_member_catalog(
    spec_df: pd.DataFrame,
    redseq_df: pd.DataFrame,
    z_min: float,
    z_max: float,
    *,
    match_tol_arcsec: float = 3.0,
) -> pd.DataFrame:
    """Build a deduplicated cluster member catalog from spec + phot members.

    Spec members: sources in ``spec_df`` with z between ``z_min`` and
    ``z_max``.  Phot members: sources in ``redseq_df`` that have no
    spectroscopic counterpart within ``match_tol_arcsec``.  One row per
    sky position; ``member_type`` column indicates selection method.

    Parameters
    ----------
    spec_df : pd.DataFrame
        Combined spectroscopic catalog (e.g. combined_redshifts.csv).
        Must have RA, Dec, z columns.
    redseq_df : pd.DataFrame
        Red sequence selected catalog (output of ``fit_and_select_red_sequence``).
        Must have RA, Dec columns.
    z_min, z_max : float
        Cluster redshift range for selecting spectroscopic members.
    match_tol_arcsec : float
        Maximum separation (arcsec) for matching a phot source to a spec
        source.  Matched phot sources are dropped (spec version kept).

    Returns
    -------
    pd.DataFrame
        Deduplicated member catalog with ``member_type`` column
        ("spec" or "phot").
    """
    # --- Spec members: z in range ---
    if spec_df.empty or Z_COL not in spec_df.columns:
        spec_members = pd.DataFrame(columns=spec_df.columns if not spec_df.empty else [RA_COL, DEC_COL])
    else:
        spec_members = _filter_by_redshift(spec_df, z_min, z_max)

    spec_members = spec_members.copy()
    spec_members["member_type"] = "spec"

    if redseq_df is None or redseq_df.empty:
        # Only spec members available
        return _finalize_member_columns(spec_members)

    # --- Phot members: red sequence sources without a spec match ---
    redseq = redseq_df.copy()

    if spec_members.empty:
        # No spec members to deduplicate against — all redseq are phot
        redseq["member_type"] = "phot"
        return _finalize_member_columns(pd.concat([spec_members, redseq], ignore_index=True))

    # Cross-match to find red sequence sources that already have a spec member
    spec_coords = make_skycoord(spec_members[RA_COL].values, spec_members[DEC_COL].values)
    redseq_coords = make_skycoord(redseq[RA_COL].values, redseq[DEC_COL].values)

    spec_idx_arr, redseq_idx_arr, _ = match_skycoords_unique(
        spec_coords, redseq_coords, match_tol_arcsec=match_tol_arcsec,
    )

    # Keep only red-sequence sources that do NOT have a spec match
    phot_only_mask = np.ones(len(redseq), dtype=bool)
    if len(redseq_idx_arr) > 0:
        phot_only_mask[redseq_idx_arr] = False
    phot_members = redseq.loc[phot_only_mask].copy()
    phot_members["member_type"] = "phot"

    # --- Merge spec members with phot-only columns from redseq ---
    # For spec members that DO have a redseq match, pull in any
    # photometric columns they might be missing (gmag, rmag, etc.)
    if len(redseq_idx_arr) > 0:
        # Fill missing phot columns in spec members from matched redseq rows
        phot_fill_cols = ["gmag", "rmag", "imag", "g_r", "r_i", "g_i",
                          "lum_weight_r", PHOT_SOURCE_COL]
        for col in phot_fill_cols:
            if col in redseq.columns:
                spec_vals = spec_members.iloc[spec_idx_arr].get(col)
                redseq_vals = redseq.iloc[redseq_idx_arr][col].values
                if spec_vals is not None:
                    # Fill only where spec is missing
                    needs_fill = spec_vals.isna().values
                    fill_positions = spec_idx_arr[needs_fill]
                    fill_values = redseq_vals[needs_fill]
                    if len(fill_positions) > 0:
                        spec_members.iloc[fill_positions, spec_members.columns.get_loc(col)] = fill_values
                else:
                    # Column doesn't exist in spec_members — initialize
                    # with source dtype so string cols don't become float64
                    spec_members[col] = pd.Series(
                        index=spec_members.index, dtype=redseq[col].dtype
                    )
                    spec_members.iloc[spec_idx_arr, spec_members.columns.get_loc(col)] = redseq_vals

    members = pd.concat([spec_members, phot_members], ignore_index=True)
    return _finalize_member_columns(members)


def _finalize_member_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the member catalog has all expected columns and canonical order.

    Missing columns are filled with NaN (or appropriate defaults).
    Extra columns are dropped.

    Parameters
    ----------
    df : pd.DataFrame
        Raw merged member catalog.

    Returns
    -------
    pd.DataFrame
        Cleaned catalog with columns in ``MEMBER_COLS`` order.
    """
    out = df.copy()
    for col in MEMBER_COLS:
        if col not in out.columns:
            out[col] = np.nan
    return out[MEMBER_COLS].reset_index(drop=True)


# ---------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------

def run_redsequence(
    cluster: Cluster,
    *,
    matched_dfs: dict[str, pd.DataFrame] | None = None,
    survey: str | None = None,
    color_type: str | None = None,
) -> pd.DataFrame:
    """Fit the red sequence and build a deduplicated cluster member catalog.

    Reads: matched catalogs (from Stage 3) or loads from disk
    Produces:
        Members/cluster_members.csv -- deduplicated member catalog
        Members/redseq_{survey}_{color}.csv -- per-fit intermediate results

    Parameters
    ----------
    cluster : Cluster
        Cluster object with resolved z_range and valid paths.
    matched_dfs : dict[str, DataFrame], optional
        Per-survey matched catalogs keyed by survey name (e.g.
        ``{"legacy": df}``).  If None, loads from disk at
        ``Photometry/{survey}_matched.csv``.
    survey : str, optional
        Which survey to use. Default: ``cluster.survey`` (typically
        ``"legacy"``).
    color_type : str, optional
        Which color to fit. Default: ``cluster.color_type`` (typically
        ``"gr"``).

    Returns
    -------
    members_df : DataFrame
        Deduplicated member catalog with columns:
        RA, Dec, z, sigma_z, spec_source, gmag, rmag, imag, g_r, r_i, g_i,
        lum_weight_r, phot_source, member_type
    """
    survey = survey or cluster.survey
    color_type = color_type or cluster.color_type
    z_min, z_max = cluster.z_min, cluster.z_max

    # Normalize survey to lowercase for file paths
    survey_lower = survey.lower()

    # ---- Load matched catalog (spec+phot crossmatch) ----
    if matched_dfs is not None and survey_lower in matched_dfs:
        matched_df = matched_dfs[survey_lower]
    else:
        matched_path = os.path.join(
            cluster.photometry_path, f"{survey_lower}_matched.csv"
        )
        if not os.path.isfile(matched_path):
            raise FileNotFoundError(
                f"Matched catalog not found: {matched_path}\n"
                "Run Stage 3 (matching) first, or pass matched_dfs."
            )
        matched_df = pd.read_csv(matched_path)
        print(f"Loaded matched catalog: {matched_path} ({len(matched_df)} rows)")

    # ---- Load full photometry catalog ----
    full_phot_path = os.path.join(
        cluster.photometry_path, f"photometry_{survey_lower}.csv"
    )
    if not os.path.isfile(full_phot_path):
        raise FileNotFoundError(
            f"Full photometry catalog not found: {full_phot_path}\n"
            "Run Stage 2 (photometry) first."
        )
    full_df = pd.read_csv(full_phot_path)
    print(f"Loaded full photometry: {full_phot_path} ({len(full_df)} rows)")

    # ---- Load combined spectroscopic catalog ----
    spec_path = cluster.spec_file  # Redshifts/combined_redshifts.csv
    if not os.path.isfile(spec_path):
        raise FileNotFoundError(
            f"Combined redshift catalog not found: {spec_path}\n"
            "Run Stage 3 (matching) first."
        )
    spec_df = pd.read_csv(spec_path)
    print(f"Loaded spectroscopic catalog: {spec_path} ({len(spec_df)} rows)")

    # Coerce numeric columns that may have been read as strings
    numeric_cols = ["gmag", "rmag", "imag", "g_r", "r_i", "g_i", "lum_weight_r", "z", "sigma_z"]
    for df in (matched_df, full_df, spec_df):
        present = [c for c in numeric_cols if c in df.columns]
        if present:
            coerce_to_numeric(df, present)

    # ---- Fit red sequence and select members ----
    redseq_df = fit_and_select_red_sequence(
        cluster,
        matched_df=matched_df,
        full_df=full_df,
        color_band=DEFAULT_COLOR_BAND,
        color_type=color_type,
        survey=survey,
        zmin=z_min,
        zmax=z_max,
    )

    # ---- Build deduplicated member catalog ----
    members_df = build_member_catalog(
        spec_df=spec_df,
        redseq_df=redseq_df if redseq_df is not None else pd.DataFrame(),
        z_min=z_min,
        z_max=z_max,
    )

    n_spec = (members_df["member_type"] == "spec").sum()
    n_phot = (members_df["member_type"] == "phot").sum()
    print(f"Member catalog: {len(members_df)} total ({n_spec} spec, {n_phot} phot)")

    # ---- Write outputs ----
    os.makedirs(cluster.members_path, exist_ok=True)

    members_path = os.path.join(cluster.members_path, "cluster_members.csv")
    members_df.to_csv(members_path, index=False)
    print(f"Wrote {members_path}")

    # Normalize color_type for filename (strip hyphens: "g-r" -> "gr")
    color_tag = color_type.replace("-", "")
    redseq_path = os.path.join(
        cluster.members_path, f"redseq_{survey_lower}_{color_tag}.csv"
    )
    if redseq_df is not None and not redseq_df.empty:
        redseq_df.to_csv(redseq_path, index=False)
        print(f"Wrote {redseq_path}")
    else:
        print(f"No red sequence members to write for {survey} {color_type}.")

    return members_df
