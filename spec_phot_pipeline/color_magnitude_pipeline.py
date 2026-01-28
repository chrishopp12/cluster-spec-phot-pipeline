#!/usr/bin/env python3
"""
color_magnitude_pipeline.py

Red Sequence Fitting and Photometric Selection Pipeline
---------------------------------------------------------

This script crossmatches photometric and redshift catalogs, fits the cluster red sequence,
selects photometric red sequence members, and produces color-magnitude and spatial plots.

Requirements:
    - astropy
    - astroquery
    - sklearn
    - numpy
    - pandas
    - matplotlib
    - cluster.py (local)
    - my_utils.py (local)
    - color_magnitude_plotting.py (local)

Usage:
    python color_magnitude_pipeline.py CLUSTER_ID [options]

Example:
    python color_magnitude_pipeline.py "Abell 2355" --zmin 0.215 --zmax 0.245

Options:
    --zmin         <float>    Minimum cluster redshift           [default: 0.0]
    --zmax         <float>    Maximum cluster redshift           [default: 1.5]
    -p, --path     <str>      Science directory                  [default: cwd]
    --plot-cmd     <bool>     Create CMD plots                   [default: true]
    --plot-spatial <bool>     Create RA-DEC member map           [default: true]
    --save-plots   <bool>     Save plots to file                 [default: true]

Notes:
    - "CLUSTER_ID" positional arg can be input as a cluster identifier mapped to a cluster name (eg. '1219' or '0881900301') or the
    full RMJ/Abell name (possibly others).
"""
from __future__ import annotations

import os
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from astropy.coordinates import SkyCoord
import astropy.units as u

from sklearn.linear_model import LinearRegression

from cluster import Cluster
from my_utils import str2bool, finalize_figure, load_dataframes, get_color_mag_functions, split_members_by_spec
from spec_phot_pipeline.color_magnitude_plotting import plot_cmd, plot_spatial


# ------------------------------------
# Defaults/ Constants
# ------------------------------------
DEFAULT_MAG = 'rmag'

RA_COL = 'RA'
DEC_COL = 'Dec'
Z_COL = 'z'
SIGMA_Z_COL = 'sigma_z'
SPEC_SOURCE_COL = 'spec_source'
PHOT_SOURCE_COL = 'phot_source'

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

# ------------------------------------
# Helpers
# ------------------------------------

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


# ------------------------------------
# Red Sequence Fitting and Member Selection
# ------------------------------------

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
) -> tuple[float | None, float | None, pd.DataFrame | None, pd.DataFrame | None]:
    """
    Fits a linear red sequence to spectroscopic members and returns the fit parameters.

    Parameters
    ----------
    matched_df : pd.DataFrame
        DataFrame with photometry added for all matched spec sources.
    color_type : str, optional
        Color combination to use: "g-r", "r-i", or "g-i" [default: "g-r"]
    zmin, zmax: float
        Redshift bounds; defaults to cluster.z_min/z_max if None.

    Returns
    -------
    a, b : float
        Slope and intercept of red sequence fit.
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
        return np.nan, np.nan, pd.DataFrame(), pd.DataFrame()
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
    mag_min: float = 16.0,
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
    ylim: tuple[float, float] = (-4.0, 4.0),
    xlim: tuple[float, float] = (10.0, 27.5),
    color_type: str = "g-r",
    survey: str = "Legacy",
    mag_min: float = 16.0,
    zmin: float | None = None,
    zmax: float | None = None,
    plot_cmd_flag: bool = True,
    plot_spatial_flag: bool = True,
    show_plots: bool = True,
    save_plots: bool = False,
    save_path: str | None = None,
):
    """
    Fits a linear red sequence to spectroscopic members and selects photometric matches.

    Parameters
    ----------
    cluster : Cluster
        Cluster object containing all necessary metadata (paths, coordinates, etc.).
    matched_df : pd.DataFrame
        DataFrame with photometry added for all matched spec sources.
    full_df : pd.DataFrame
        DataFrame of all sources.
    color_band : float, optional
        Max allowed deviation in color to be selected [default: 0.2 mag]
    ylim, xlim : tuple of float, optional
        Y-axis (color), X-axis (magnitude) plot limits [default: (-4.0, 4.0), (10.0, 27.5)]
    color_type : str, optional
        Color combination to use: "g-r", "r-i", or "g-i" [default: "g-r"]
    survey : str, optional
        Survey name for plot labeling (e.g., "PanSTARRS", "Legacy") [default: "Legacy"]
    mag_min : float, optional
        Minimum magnitude to consider [default: 16.0]
    zmin, zmax: float, optional
        Redshift bounds; defaults to cluster.z_min/z_max if None.
    plot_cmd_flag, plot_spatial_flag: bool, optional
        If True, plot CMD and/or sky map. [default: True]
    show_plots, save_plots : bool, optional
        Plot controls passed through to ``finalize_figure``.
    save_path : str or None, optional
        Directory (or full file path) for plot saving.

    Returns
    -------
    selected_df : pd.DataFrame
        Photometric members on the red sequence (+spec-z matches).
    """

    # Use cluster z_min/z_max unless overridden
    zmin = zmin if zmin is not None else cluster.z_min
    zmax = zmax if zmax is not None else cluster.z_max
    print(f"Using zmin = {zmin}, zmax = {zmax}")

    # Define bands based on color choice
    try:
        required_cols, color_label, mag_label, color_func, mag_func = get_color_mag_functions(color_type)
    except ValueError as exc:
        print(f'{survey} {color_type}: {exc}')
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

    selected_df = apply_membership_cuts(
        full_catalog,
        zmin=zmin,
        zmax=zmax,
    )

    if selected_df.empty:
        print(f"Skipping {survey} {color_type}: no photometric members after cuts.")
        return None

    spec_all, phot_all = split_members_by_spec(
        full_catalog,
        z_col=Z_COL,
    )

    phot_all = phot_all.dropna(subset=required_cols).copy()


    # --- Diagnostic Plotting ---
    if plot_cmd_flag:
        plot_cmd(
            member_df=selected_df,
            phot_df=phot_all,
            a=a,
            b=b,
            survey=survey,
            color_type=color_type,
            save_name="Diagnostic",
            ylim=ylim,
            xlim=xlim,
            show_plots=show_plots,
            save_plots=save_plots,
            save_path=save_path
        )

    if plot_spatial_flag:
        plot_spatial(
            member_df=selected_df,
            phot_df=phot_all,
            zmin=zmin,
            zmax=zmax,
            survey=survey,
            color_type=color_type,
            save_name="Diagnostic",
            with_cmap=False,
            show_plots=show_plots,
            save_plots=save_plots,
            save_path=save_path
        )


    return selected_df


# ------------------------------------
# Plotting Utilities
# ------------------------------------

def plot_color_magnitude(
        phot_df,
        color_col,
        mag_col,
        title=None,
        ax=None,
        plot_legend=False,
        color="gray",
        ylim=(-4, 4),
        xlim=(10, 27.5),
        spec_df=None,
        z_min=None,
        z_max=None,
        overlay_label="Spec-z members"
    ) -> None:
    """
    Plots a single color-magnitude diagram with optional overlay of spec-z members.

    Parameters
    ----------
    phot_df : pd.DataFrame
        Catalog with color and magnitude columns.
    color_col : str
        Name of the color column (e.g., 'g_r').
    mag_col : str
        Name of the magnitude column (e.g., 'rmag').
    title : str, optional
        Plot title.
    ax : matplotlib.axes.Axes, optional
        Axis to plot on (if None, creates new figure).
    plot_legend : bool
        If True, adds legend to plot [default: False]
    label : str, optional
        Label for the photometric scatter plot.
    color : str, optional
        Color for the photometric points. [default: "gray"]
    xlim : tuple, optional
        Limits for the magnitude axis (x-axis) [default: (10, 27.5)]
    ylim : tuple, optional
        Limits for the color axis (y-axis) [default: (-4,4)]
    spec_df : pd.DataFrame, optional
        Matched spectroscopic sources to overlay.
    z_min, z_max : float, optional
        Redshift bounds for filtering `spec_df` before plotting.
    overlay_label : str, optional
        Legend label for spec-z overlay points [default: "Spec-z members"]
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    # Main photometric scatter
    valid = phot_df[[color_col, mag_col]].dropna()
    ax.scatter(valid[mag_col], valid[color_col], s=5, color=color, label="All photometric sources")

    # Optional overlay
    if spec_df is not None:
        spec_valid = spec_df.copy()
        if z_min is not None and z_max is not None and "z" in spec_valid.columns:
            spec_valid = spec_valid[(spec_valid["z"] >= z_min) & (spec_valid["z"] <= z_max)]
        spec_valid = spec_valid.dropna(subset=[color_col, mag_col])
        if not spec_valid.empty:
            ax.scatter(spec_valid[mag_col], spec_valid[color_col],
                       s=20, color="crimson", marker="o", label=overlay_label, edgecolor="black")

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_ylabel(color_col.replace("_", " - "))
    ax.set_xlabel(f"{mag_col[0]} magnitude")

    if title:
        ax.set_title(title)
    if plot_legend:
        ax.legend()


def plot_all_color_magnitude(
        cluster: Cluster,
        panstarrs_df: pd.DataFrame,
        legacy_df: pd.DataFrame,
        panstarrs_spec: pd.DataFrame | None = None,
        legacy_spec: pd.DataFrame | None = None,
        z_min: float | None = None,
        z_max: float | None = None,
        show_plots: bool = True,
        save_plots: bool = False,
        save_path: str | None = None
    ) -> None:
    """
    Plot a grid of six color-magnitude diagrams (CMDs) for Pan-STARRS and Legacy photometric catalogs.

    Each row shows three CMDs for one catalog:
        - g-r vs r
        - r-i vs i
        - g-i vs i

    Optionally overlays spectroscopically confirmed cluster members for each panel, 
    restricted to a redshift range if z_min and z_max are provided.

    Parameters
    ----------
    cluster : Cluster
        Cluster object containing all necessary metadata (paths, coordinates, etc.).
    panstarrs_df : pd.DataFrame
        Pan-STARRS photometric catalog, must include color columns (g_r, r_i, g_i) and mag columns (rmag, imag).
    legacy_df : pd.DataFrame
        Legacy photometric catalog, must include color columns (g_r, r_i, g_i) and mag columns (rmag, imag).
    panstarrs_spec : pd.DataFrame | None, optional
        Pan-STARRS catalog of spectroscopic matches to overlay [default: None].
    legacy_spec : pd.DataFrame | None, optional
        Legacy catalog of spectroscopic matches to overlay [default: None].
    z_min, z_max : float | None, optional
        If provided, only overlays spec-z members within this redshift range.
    show_plots, save_plots : bool
        Plot controls passed through to ``finalize_figure``.
    save_path : str | None, optional
        Directory (or full file path) for plot saving.

    Returns
    -------
    None
        The function displays or saves the 2x3 grid of CMD plots.
    """

    # Use cluster z_min/z_max unless overridden
    z_min = z_min if z_min is not None else cluster.z_min
    z_max = z_max if z_max is not None else cluster.z_max

    fig, axs = plt.subplots(2, 3, figsize=(16, 12), sharex=False, sharey=False)

    # Pan-STARRS
    plot_color_magnitude(panstarrs_df, "g_r", "rmag", "Pan-STARRS: g-r vs r",
                         ax=axs[0, 0], plot_legend=True, spec_df=panstarrs_spec, z_min=z_min, z_max=z_max)
    plot_color_magnitude(panstarrs_df, "r_i", "imag", "Pan-STARRS: r-i vs i",
                         ax=axs[0, 1], spec_df=panstarrs_spec, z_min=z_min, z_max=z_max)
    plot_color_magnitude(panstarrs_df, "g_i", "imag", "Pan-STARRS: g-i vs i",
                         ax=axs[0, 2], spec_df=panstarrs_spec, z_min=z_min, z_max=z_max)

    # Legacy
    plot_color_magnitude(legacy_df, "g_r", "rmag", "Legacy: g-r vs r",
                         ax=axs[1, 0], spec_df=legacy_spec, z_min=z_min, z_max=z_max)
    plot_color_magnitude(legacy_df, "r_i", "imag", "Legacy: r-i vs i",
                         ax=axs[1, 1], spec_df=legacy_spec, z_min=z_min, z_max=z_max)
    plot_color_magnitude(legacy_df, "g_i", "imag", "Legacy: g-i vs i",
                         ax=axs[1, 2], spec_df=legacy_spec, z_min=z_min, z_max=z_max)

    finalize_figure(fig, show_plots=show_plots,
                    save_plots=save_plots,
                    save_path=save_path,
                    filename="CMDs.pdf")


def print_cluster_info(
        cluster: Cluster,
    ) -> None:
    """
    Prints detailed cluster information including coordinates, redshift, source counts, BCGs, and member statistics.

    Parameters
    ----------
    cluster : Cluster
        Cluster object containing all necessary metadata (paths, coordinates, etc.).
    """
    member_zrange = (cluster.z_min, cluster.z_max)

    # Load catalogs
    ned_path = os.path.join(cluster.redshift_path, "ned.csv")
    sdss_path = os.path.join(cluster.redshift_path, "sdss.csv")
    desi_path = os.path.join(cluster.redshift_path, "desi.csv")
    deimos_path = os.path.join(cluster.redshift_path, "deimos.csv")
    archival_z_path = os.path.join(cluster.redshift_path, "archival_z.csv")

    ned = pd.read_csv(ned_path)
    sdss = pd.read_csv(sdss_path)
    desi = pd.read_csv(desi_path)
    deimos = pd.read_csv(deimos_path)
    archival_z = pd.read_csv(archival_z_path)


    combined, red_sequence, bcg_df = load_dataframes(cluster)
    BCGs = []
    for _, row in bcg_df.iterrows():
        ra = row.get(RA_COL, None)
        dec = row.get(DEC_COL, None)
        z = row.get(Z_COL, None) if pd.notna(row.get(Z_COL, None)) else None
        P = row.get("BCG_probability", None) if pd.notna(row.get("BCG_probability", None)) else None
        BCGs.append((ra, dec, z, P))


    # Count sources from each survey
    n_ned = len(ned)
    n_sdss = len(sdss)
    n_desi = len(desi)
    n_deimos = len(deimos)
    n_combined = len(combined)
    new_deimos = len(combined)-len(archival_z)  # New spectroscopic redshifts only
    n_archival = len(archival_z)

    # Total unique sources
    coords_all = SkyCoord(ra=combined['RA'].values * u.deg, dec=combined['Dec'].values * u.deg)
    _, uniq_idx = np.unique(coords_all.to_string('decimal'), return_index=True)
    total_unique = len(uniq_idx)

    # Member count in range
    z_members = combined['z'][(combined['z'] > member_zrange[0]) & (combined['z'] < member_zrange[1])]
    n_members = len(z_members)

    # Photometric members
    phot_members = len(red_sequence) - n_members


    print(f"\n=== Cluster Info Report: {cluster.name} ===")
    print(f"Location: RA = {cluster.coords.ra.deg:.5f}, Dec = {cluster.coords.dec.deg:.5f}")
    print(f"Redshift: z = {cluster.redshift:.5f}")
    print("\nSource Counts:")
    print(f"  NED:   {n_ned}")
    print(f"  SDSS:  {n_sdss}")
    print(f"  DESI:  {n_desi}")
    print(f"  Total Combined: {n_combined}")
    print(f"  Total Deimos: {n_deimos}")
    print(f"  New Deimos: {new_deimos}")
    print(f"  Total unique (post-merge): {total_unique}")
    print(f"  Archival: {n_archival}")
    print(f"  Cluster members in {member_zrange[0]:.3f} < z < {member_zrange[1]:.3f}: {n_members}")
    print(f"  Photometric members (red sequence): {phot_members}")

    print("\nBCG Candidates:")

    for i, (ra, dec, z, P) in enumerate(BCGs):
        z_str = f"{z:.4f}" if z is not None else "unknown"
        P_str = f"P={P:.5f}" if P is not None else "P=unknown"
        print(f"BCG {i+1}: RA = {ra:.5f}, Dec = {dec:.5f}, z = {z_str}, {P_str}")


    print("=== End of Report ===\n")


def run_cmd_pipeline(
        cluster,
        plot_cmd_flag,
        plot_spatial_flag,
        save_plots,
        show_plots
    ) -> None:

    """
    Run the color-magnitude diagram (CMD) and red sequence selection pipeline for a galaxy cluster.

    This function:
      - Loads photometric and redshift catalogs for a cluster.
      - Crossmatches photometric catalogs (Pan-STARRS and Legacy) to redshift data.
      - Plots all color-magnitude diagrams (CMDs) for both catalogs.
      - Fits red sequence models for multiple color indices, selecting photometric cluster members.
      - Saves resulting red sequence catalogs and (optionally) plots.

    Parameters
    ----------
    redshift_folder : str
        Path to folder containing the redshift catalog (e.g., '.../Redshifts').
    photometry_folder : str
        Path to folder containing photometry catalogs (e.g., '.../Photometry').
    zmin : float
        Lower redshift bound for cluster member selection.
    zmax : float
        Upper redshift bound for cluster member selection.
    plot_cmd_flag : bool
        If True, display or save CMD plots for each red sequence fit.
    plot_spatial_flag : bool
        If True, display or save spatial (RA-Dec) plots for each fit.
    save_flag : bool
        If True, save output plots to the 'Images/' subdirectory of the photometry folder. Otherwise, show interactively.

    Returns
    -------
    None
        All results are saved to CSV and/or as plots in the specified directories.
    """

    # All photometric catalogs
    # panstarrs_df = pd.read_csv(os.path.join(cluster.photometry_path, "photometry_PanSTARRS.csv"))
    # legacy_df = pd.read_csv(os.path.join(cluster.photometry_path, "photometry_legacy.csv"))

    # Deduplicated combined spec/ phot catalogs
    panstarrs_df= pd.read_csv(os.path.join(cluster.redshift_path, "panstarrs_catalog.csv"))
    legacy_df = pd.read_csv(os.path.join(cluster.redshift_path, "legacy_catalog.csv"))


    # Matched catalogs
    matched_panstarrs = pd.read_csv(os.path.join(cluster.photometry_path, "panstarrs_matched.csv"))
    matched_legacy = pd.read_csv(os.path.join(cluster.photometry_path, "legacy_matched.csv"))

    img_dir = os.path.join(cluster.photometry_path, "Images")
    os.makedirs(img_dir, exist_ok=True)

    plot_all_color_magnitude(
                            cluster=cluster,
                            panstarrs_df=panstarrs_df,
                            legacy_df=legacy_df,
                            panstarrs_spec=matched_panstarrs,
                            legacy_spec=matched_legacy,
                            show_plots=show_plots,
                            save_plots=save_plots,
                            save_path=(os.path.join(img_dir, "CMDs.pdf")))

    selected_PS_gr = fit_and_select_red_sequence(cluster, matched_df = matched_panstarrs, full_df = panstarrs_df, color_band = 0.15, color_type="g-r", survey="PanSTARRS",
                                                                    plot_cmd_flag=plot_cmd_flag, plot_spatial_flag=plot_spatial_flag, save_plots=save_plots, show_plots=show_plots, save_path=img_dir)
    if selected_PS_gr is not None:
        selected_PS_gr.to_csv(os.path.join(cluster.photometry_path, "redseq_panstarrs_gr.csv"), index=False)

    selected_PS_ri = fit_and_select_red_sequence(cluster, matched_df = matched_panstarrs, full_df = panstarrs_df, color_band = 0.15, color_type="r-i", survey="PanSTARRS",
                                                                   plot_cmd_flag=plot_cmd_flag, plot_spatial_flag=plot_spatial_flag, save_plots=save_plots, show_plots=show_plots,save_path=img_dir)
    if selected_PS_ri is not None:
        selected_PS_ri.to_csv(os.path.join(cluster.photometry_path, "redseq_panstarrs_ri.csv"), index=False)

    selected_PS_gi = fit_and_select_red_sequence(cluster, matched_df = matched_panstarrs, full_df = panstarrs_df, color_band = 0.15, color_type="g-i", survey="PanSTARRS",
                                                                   plot_cmd_flag=plot_cmd_flag, plot_spatial_flag=plot_spatial_flag, save_plots=save_plots, show_plots=show_plots, save_path=img_dir)
    if selected_PS_gi is not None:
        selected_PS_gi.to_csv(os.path.join(cluster.photometry_path, "redseq_panstarrs_gi.csv"), index=False)

    selected_Leg_gr = fit_and_select_red_sequence(cluster, matched_df = matched_legacy, full_df = legacy_df, color_band = 0.15, color_type="g-r", survey="Legacy",
                                                                      plot_cmd_flag=plot_cmd_flag, plot_spatial_flag=plot_spatial_flag, save_plots=save_plots, show_plots=show_plots, save_path=img_dir)
    if selected_Leg_gr is not None:
        selected_Leg_gr.to_csv(os.path.join(cluster.photometry_path, "redseq_legacy_gr.csv"), index=False)

    selected_Leg_ri = fit_and_select_red_sequence(cluster, matched_df = matched_legacy, full_df = legacy_df, color_band = 0.15, color_type="r-i", survey="Legacy",
                                                                      plot_cmd_flag=plot_cmd_flag, plot_spatial_flag=plot_spatial_flag, save_plots=save_plots, show_plots=show_plots, save_path=img_dir)
    if selected_Leg_ri is not None:
        selected_Leg_ri.to_csv(os.path.join(cluster.photometry_path, "redseq_legacy_ri.csv"), index=False)

    selected_Leg_gi = fit_and_select_red_sequence(cluster, matched_df = matched_legacy, full_df = legacy_df, color_band = 0.15, color_type="g-i", survey="Legacy",
                                                                      plot_cmd_flag=plot_cmd_flag, plot_spatial_flag=plot_spatial_flag, save_plots=save_plots, show_plots=show_plots, save_path=img_dir)
    if selected_Leg_gi is not None:
        selected_Leg_gi.to_csv(os.path.join(cluster.photometry_path, "redseq_legacy_gi.csv"), index=False)

    print_cluster_info(cluster)


def main():
    parser = argparse.ArgumentParser(description="Run red sequence photometry pipeline")
    parser.add_argument(
        "cluster_id",
        type=str,
        help="Cluster ID (e.g. '1327') or full RMJ/Abell name."
    )
    parser.add_argument(
        "--zmin",
        type=float,
        default=0.0,
        help="Minimum cluster redshift"
    )
    parser.add_argument(
        "--zmax",
        type=float,
        default=1.5,
        help="Maximum cluster redshift"
    )
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        default=os.getcwd(),
        help="Science directory path (default: current directory)"
    )
    parser.add_argument(
        '--plot-cmd',
        type=str2bool,
        nargs='?',
        const=True,
        default=True,
        help="Whether to plot CMD. (default: True). Accepts true/false, yes/no, y/n, t/f, 1/0."
    )
    parser.add_argument(
        '--plot-spatial',
        type=str2bool,
        nargs='?',
        const=True,
        default=True,
        help="Whether to plot RA-DEC member map. (default: True). Accepts true/false, yes/no, y/n, t/f, 1/0."
    )
    parser.add_argument(
        '--save-plots',
        type=str2bool,
        nargs='?',
        const=True,
        default=True,
        help="Whether to save plots to file. (default: True). Accepts true/false, yes/no, y/n, t/f, 1/0."
    )
    parser.add_argument(
        '--show-plots',
        type=str2bool,
        nargs='?',
        const=True,
        default=True,
        help="Whether to show plots interactively. (default: True). Accepts true/false, yes/no, y/n, t/f, 1/0."
    )
    args = parser.parse_args()

    cluster = Cluster(args.cluster_id, base_path=args.path)
    cluster.populate()

    if args.zmin is not None:
        cluster.z_min = args.zmin
    if args.zmax is not None:
        cluster.z_max = args.zmax
    if cluster.z_min is None:
        cluster.z_min = cluster.redshift - 0.015
    if cluster.z_max is None:
        cluster.z_max = cluster.redshift + 0.015

    run_cmd_pipeline(
        cluster=cluster,
        plot_cmd_flag=args.plot_cmd,
        plot_spatial_flag=args.plot_spatial,
        save_plots=args.save_plots,
        show_plots=args.show_plots
    )

if __name__ == "__main__":
    main()

