#!/usr/bin/env python3
"""
make_catalogs_pipeline.py

Redshift Catalog Builder and Exporter
---------------------------------------------------------

This script consolidates spectroscopic redshift measurements from DEIMOS 'zdump.txt'
files and archival redshift catalogs, merges them with de-duplication by position,
and generates both a machine-readable catalog and a LaTeX deluxetable preview.

Requirements:
    - numpy
    - pandas
    - astropy
    - cluster.py (local)
    - my_utils.py (local)

Usage:
    python make_catalogs_pipeline.py CLUSTER_ID [options]

Example:
    python make_catalogs_pipeline.py "Abell 2355" --path "/path/to/XMM Scripts"

Options:
    -p, --path      <str>   Base path to science directories            [default: cwd]
    --spec-folder   <str>   Subdirectory for redshift files             [default: /Redshifts]
    --save-tex      <bool>  Whether to write LaTeX and ASCII tables     [default: True]
    --print-tex     <bool>  Whether to print LaTeX tables to console    [default: False]

Example:
    python make_catalogs_pipeline.py "Abell 2355" --path "/path/to/XMM Scripts"

Notes:
    - Output _matched.csv contains rows for all sources with both redshift and photometry data.
    - Output _catalog.csv contains rows for all sources, with photometric or spectroscopic data where available, NaNs for lacking data.  
"""
from __future__ import annotations

import os
import glob
import argparse
import ast

import numpy as np
import pandas as pd

from astropy.coordinates import SkyCoord
import astropy.units as u

from cluster import Cluster
from my_utils import str2bool, emit_latex, query_redmapper_bcg_candidates


# ------------------------------------
# Defaults/ Constants
# ------------------------------------
DEFAULT_TOL_ARCSEC = 3.0  # Matching tolerance in arcseconds
DEFAULT_MAG = 'rmag'

RA_COL = 'RA'
DEC_COL = 'Dec'
Z_COL = 'z'
SIGMA_Z_COL = 'sigma_z'
SPEC_SOURCE_COL = 'spec_source'
PHOT_SOURCE_COL = 'phot_source'



# ------------------------------------
# Helpers
# ------------------------------------

def _coerce_to_numeric(
        df: pd.DataFrame,
        columns: list[str] | tuple[str, ...]
    ) -> pd.DataFrame:
    """
    Coerces specified columns of a DataFrame to numeric, forcing errors to NaN.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list[str] | tuple[str, ...]
        List of column names to coerce.

    Returns
    -------
    out : pd.DataFrame
        Modified DataFrame with specified columns coerced to numeric.
    """
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors='coerce')
    return out


def _standardize_redshift_df(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure redshift DataFrame has standard columns and types.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with redshift data.

    Returns
    -------
    pd.DataFrame
        Standardized DataFrame with columns: RA, Dec, z, sigma_z, source.
    """
    if df.empty:
        return pd.DataFrame(columns=[RA_COL, DEC_COL, Z_COL, SIGMA_Z_COL, SPEC_SOURCE_COL])
    
    out = df.copy()
    out = out.rename(columns={c: c.strip() for c in out.columns})

    if SPEC_SOURCE_COL not in out.columns and 'source' in out.columns:
        out = out.rename(columns={'source': SPEC_SOURCE_COL})

    # Standardize zerr, sigma_z column name
    if SIGMA_Z_COL not in out.columns:
        if 'zerr' in out.columns:
            out = out.rename(columns={'zerr': SIGMA_Z_COL})
        else:
            out[SIGMA_Z_COL] = np.nan
        
    # Ensure all required columns are present
    for col in [RA_COL, DEC_COL, Z_COL, SIGMA_Z_COL, SPEC_SOURCE_COL]:
        if col not in out.columns:
            out[col] = np.nan if col != SPEC_SOURCE_COL else 'Unknown'

    out = _coerce_to_numeric(out, [RA_COL, DEC_COL, Z_COL, SIGMA_Z_COL])
    out = out.dropna(subset=[RA_COL, DEC_COL, Z_COL]).reset_index(drop=True)

    return out


def _standardize_phot_df(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure photometry DataFrame has standard columns and types.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with photometry data.

    Returns
    -------
    pd.DataFrame
        Standardized DataFrame with columns: RA, Dec.
    """
    if df.empty:
        return pd.DataFrame(columns=[RA_COL, DEC_COL])
    
    out = df.copy()
    out = out.rename(columns={c: c.strip() for c in out.columns})

    # Ensure RA and Dec columns are present
    for col in [RA_COL, DEC_COL]:
        if col not in out.columns:
            out[col] = np.nan

    out = _coerce_to_numeric(out, [RA_COL, DEC_COL])
    out = out.dropna(subset=[RA_COL, DEC_COL]).reset_index(drop=True)

    return out


# ------------------------------------
# Redshift Catalog Functions
# ------------------------------------

def load_spectroscopic_redshifts(cluster: Cluster) -> pd.DataFrame:
    """Loads redshift data from zdump.txt files.

    Parameters
    ----------
    cluster : Cluster
        Cluster object containing all necessary metadata (paths, coordinates, etc.).

    Returns
    --------
    new_spectroscopy : pd.DataFrame
        DataFrame containing the spectroscopy from zdump Deimos files.
    """

    new_spectroscopy = []

    # Loop through all zdump.txt files in the folder
    for file_path in glob.glob(os.path.join(cluster.redshift_path, "zdump*.txt")):
        try:
            spec_entry = np.loadtxt(file_path, unpack=True)
            if spec_entry.ndim == 1:
                spec_entry = spec_entry.reshape(4,-1)
            for ra, dec, z, sigma_z in zip(*spec_entry):
                new_spectroscopy.append(
                    {
                        RA_COL: float(ra),
                        DEC_COL: float(dec),
                        Z_COL: float(z),
                        SIGMA_Z_COL: float(sigma_z),
                        SPEC_SOURCE_COL: 'Deimos',
                    }
                )
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    new_spec_df = pd.DataFrame(new_spectroscopy)
    new_spec_df = _standardize_redshift_df(new_spec_df)

    print(f"Loaded {len(new_spec_df)} new spectroscopic redshifts from zdump files.")

    return new_spec_df


def load_archival_redshifts(
        cluster: Cluster,
        new_spec_df: pd.DataFrame,
        *,
        match_tol_arcsec: float = DEFAULT_TOL_ARCSEC
    ) -> pd.DataFrame:
    """
    Load archival redshifts and cross-match them with new spectroscopic data.

    Parameters
    ----------
    cluster : Cluster
        Cluster object containing all necessary metadata (paths, coordinates, etc.).
    new_spec_df : pd.DataFrame
        DataFrame containing new spectroscopic redshift measurements.
    match_tol_arcsec : float
        Matching tolerance in arcseconds. [default: DEFAULT_TOL_ARCSEC]

    Returns
    -------
    combined_df : pd.DataFrame
        DataFrame containing combined RA, Dec, redshift, and redshift uncertainty.
    """

    archival_path = os.path.join(cluster.redshift_path, "archival_z.txt")
    try:
        arc_spec_df = pd.read_csv(archival_path, sep=r"\s+")
    except FileNotFoundError:
        print(f"Archival redshift file not found: {archival_path}")
        arc_spec_df = pd.DataFrame(columns=[RA_COL, DEC_COL, Z_COL, SIGMA_Z_COL, SPEC_SOURCE_COL])

    arc_spec_df = _standardize_redshift_df(arc_spec_df)
    new_spec_df = _standardize_redshift_df(new_spec_df)

    print(f"Loaded {len(arc_spec_df)} archival redshifts.")
  
    # If no spectroscopic redshifts were passed in, return archival data only
    if new_spec_df.empty:
        print("No input spectroscopic data provided. Returning archival data only.")
        return arc_spec_df

    # Match archival sources to input sources
    coords_arc = SkyCoord(ra=arc_spec_df[RA_COL].values * u.deg, dec=arc_spec_df[DEC_COL].values * u.deg)
    coords_new = SkyCoord(ra=new_spec_df[RA_COL].values * u.deg, dec=new_spec_df[DEC_COL].values * u.deg)

    arc_idx, _ , _ = match_skycoords_unique(
        coords_arc,
        coords_new,
        match_tol_arcsec=match_tol_arcsec,
    )

    archival_unmatched = arc_spec_df.drop(index=arc_idx).reset_index(drop=True)
    combined_df = pd.concat([new_spec_df, archival_unmatched], ignore_index=True)

    # For reporting unmatched stats
    archival_n = len(arc_spec_df)
    new_n = len(new_spec_df)
    matched_n = len(arc_idx)
    unmatched_n = archival_n - matched_n
    combined_n = len(combined_df)
    print(f"Spectroscopic input sources: {new_n}")
    print(f"Archival matches (within {match_tol_arcsec} arcsec): {matched_n}")
    print(f"Archival unmatched sources: {unmatched_n}")
    print(f"Total redshifts after merge: {combined_n}")
    print()

    return combined_df


# ------------------------------------
# Crossmatching Functions
# ------------------------------------

def match_skycoords_unique(
        coords1: SkyCoord,
        coords2: SkyCoord,
        *,
        match_tol_arcsec: float = DEFAULT_TOL_ARCSEC,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Match two sets of SkyCoord objects with unique assignments.

    Parameters
    ----------
    coords1 : SkyCoord
        First set of coordinates to match.
    coords2 : SkyCoord
        Second set of coordinates to match against.
    match_tol_arcsec : float
        Matching tolerance in arcseconds.

    Returns
    -------
    matched_indices1 : np.ndarray
        Indices in coords1 that have matches in coords2.
    matched_indices2 : np.ndarray
        Corresponding indices in coords2 that match coords1.
    matched_separations : np.ndarray
        Angular separations of the matched pairs.
    """
    if len(coords1) == 0 or len(coords2) == 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float)

    idx2, sep2d, _ = coords1.match_to_catalog_sky(coords2)
    match_mask = sep2d < (match_tol_arcsec * u.arcsec)

    matched_indices1 = np.where(match_mask)[0]
    matched_idx2 = idx2[match_mask]
    matched_sep2d = sep2d[match_mask].arcsec

    # Unique assignment: keep only the closest match for each coords2 entry
    seen_coords2 = {}
    for i, idx2_val, sep in zip(matched_indices1, matched_idx2, matched_sep2d):
        if idx2_val not in seen_coords2 or sep < seen_coords2[idx2_val][1]:
            seen_coords2[idx2_val] = (i, sep)

    pairs = [(i1, i2, sept) for i2, (i1, sept) in seen_coords2.items()]
    pairs.sort(key=lambda x: x[0])  # Sort by coords1 index

    matched_1 = np.array([p[0] for p in pairs], dtype=int)
    matched_2 = np.array([p[1] for p in pairs], dtype=int)
    matched_sep = np.array([p[2] for p in pairs], dtype=float)

    return matched_1, matched_2, matched_sep



def crossmatch_photometry_with_redshifts(
        phot_df: pd.DataFrame,
        redshift_df: pd.DataFrame,
        *,
        match_tol_arcsec: float = DEFAULT_TOL_ARCSEC
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Crossmatch photometric catalog with a redshift catalog and return:
    1. Photometric matches only
    2. Complete redshift catalog with matched photometry (NaNs if unmatched)

    Parameters
    ----------
    phot_df : pd.DataFrame
        Photometry catalog with 'RA' and 'Dec' columns (in degrees).
    redshift_df : pd.DataFrame
        df containing redshift data, backwards compatible with previous CSV format.
    match_tol_arcsec : float
        Matching radius in arcseconds. [default 3.0]

    Returns
    -------
    matched : pd.DataFrame
        Only redshift rows with photometric matches (joined to photometry).
    combined : pd.DataFrame
        All photometry with redshifts appended where matched (NaNs otherwise).
    """
    phot_df = _standardize_phot_df(phot_df)
    redshift_df = _standardize_redshift_df(redshift_df)

    spec_cols = [col for col in redshift_df.columns if col not in [RA_COL, DEC_COL]]
    phot_cols = [col for col in phot_df.columns if col not in [RA_COL, DEC_COL]]
    
    # Edge cases
    if redshift_df.empty and phot_df.empty:
        print("Both photometry and redshift catalogs are empty. Returning empty DataFrames.")
        empty_matched = _standardize_redshift_df(pd.DataFrame())
        empty_final = _standardize_phot_df(pd.DataFrame())
        return empty_matched, empty_final
    elif redshift_df.empty:
        print("Redshift catalog is empty. Returning empty matched DataFrame and full photometry catalog.")
        empty_matched = _standardize_redshift_df(pd.DataFrame())
        return empty_matched, phot_df.copy()
    elif phot_df.empty:
        print("Photometry catalog is empty. Final catalog will contain redshifts only.")
        matched_catalog = redshift_df.copy()
        final_catalog = redshift_df.copy()
        return matched_catalog, final_catalog

    spec_coords = SkyCoord(ra=redshift_df[RA_COL].values * u.deg, dec=redshift_df[DEC_COL].values * u.deg)
    phot_coords = SkyCoord(ra=phot_df[RA_COL].values * u.deg, dec=phot_df[DEC_COL].values * u.deg)

    spec_idx, phot_idx, sep = match_skycoords_unique(
        spec_coords,
        phot_coords,
        match_tol_arcsec=match_tol_arcsec,
    )

    # matched_catalog: Subset of redshifts with photometric matches
    matched_catalog = redshift_df.copy()
    for col in phot_cols:
        if col not in matched_catalog.columns:
            matched_catalog[col] = phot_df[col].iloc[0:0].copy()

    if len(spec_idx) > 0 and phot_cols:
        matched_catalog.loc[spec_idx, phot_cols] = phot_df.loc[phot_idx, phot_cols].to_numpy()

    matched_catalog = matched_catalog.iloc[spec_idx].reset_index(drop=True)

    # final_catalog: phot_df + spec columns, plus unmatched spec rows
    final_catalog = phot_df.copy()
    for col in spec_cols:
        if col not in final_catalog.columns:
            final_catalog[col] = redshift_df[col].iloc[0:0].copy()

    if len(phot_idx) > 0 and spec_cols:
        final_catalog.loc[phot_idx, spec_cols] = redshift_df.loc[spec_idx, spec_cols].to_numpy()

    matched_spec_set = set(spec_idx.tolist())
    unmatched_spec_idx = [i for i in range(len(redshift_df)) if i not in matched_spec_set]

    if unmatched_spec_idx:
        spec_extra = redshift_df.iloc[unmatched_spec_idx].copy()

        for col in final_catalog.columns:
            if col not in spec_extra.columns:
                spec_extra[col] = np.nan

        spec_extra = spec_extra[final_catalog.columns]  # align columns
        final_catalog = pd.concat([final_catalog, spec_extra], ignore_index=True)

    return matched_catalog, final_catalog


def create_matched_catalogs(
        cluster: Cluster,
        panstarrs_df: pd.DataFrame,
        legacy_df: pd.DataFrame,
        redshift_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Crossmatch both Pan-STARRS and Legacy photometry catalogs with the redshift catalog.
    Saves results as CSVs in the photometry folder.

    Parameters
    ----------
    cluster : Cluster
        Cluster object containing all necessary metadata (paths, coordinates, etc.).
    panstarrs_df, legacy_df : pd.DataFrame
        Photometry catalogs.
    redshift_df : pd.DataFrame
        DataFrame containing combined redshift data.

    Returns
    -------
    matched_panstarrs, matched_legacy : pd.DataFrame
        Crossmatched Pan-STARRS and Legacy catalogs with redshifts appended.
    """

    if PHOT_SOURCE_COL not in panstarrs_df.columns:
        panstarrs_df[PHOT_SOURCE_COL] = 'panstarrs'
    if PHOT_SOURCE_COL not in legacy_df.columns:
        legacy_df[PHOT_SOURCE_COL] = 'legacy'

    matched_panstarrs, catalog_panstarrs = crossmatch_photometry_with_redshifts(panstarrs_df, redshift_df)
    matched_path = os.path.join(cluster.photometry_path, "panstarrs_matched.csv")
    matched_panstarrs.to_csv(matched_path, index=False)

    catalog_path = os.path.join(cluster.redshift_path, "panstarrs_catalog.csv")
    catalog_panstarrs.to_csv(catalog_path, index=False)
    print(f"Matched PanSTARRS photometry saved to {matched_path}")

    matched_legacy, catalog_legacy = crossmatch_photometry_with_redshifts(legacy_df, redshift_df)
    matched_path = os.path.join(cluster.photometry_path, "legacy_matched.csv")
    matched_legacy.to_csv(matched_path, index=False)
    catalog_path = os.path.join(cluster.redshift_path, "legacy_catalog.csv")
    catalog_legacy.to_csv(catalog_path, index=False)
    print(f"Matched Legacy photometry saved to {matched_path}")

    return matched_panstarrs, matched_legacy, catalog_panstarrs, catalog_legacy

# ------------------------------------
# BCG Matching Functions
# ------------------------------------

def get_BCG(
        cluster: Cluster,
        redshift_df: pd.DataFrame,
        *,
        match_tol_arcsec: float = DEFAULT_TOL_ARCSEC,
        manual_BCGs: list[tuple[float, float]] | None = None,
        verbose: bool = True,
    ) -> list[tuple[float, float, float | None, float | None]]:
    """
    Finds BCG coordinates and matches them with the closest redshifted galaxy within a given separation.
    Includes BCG selection probabilities (P0, P1, ..., P4) if available. Also allows for manual BCG input.

    Parameters
    ----------
    cluster : Cluster
        Cluster object containing all necessary metadata (paths, coordinates, etc.).
    redshift_df : pd.DataFrame
        DataFrame containing redshift information for all galaxies.
    match_tol_arcsec : float
        Maximum allowed separation (in arcseconds) to assign a redshift to a BCG.
    manual_BCGs : list of tuple, optional
        List of manually added BCGs in (RA, Dec) format. Redshift will be matched if possible; P will be None.

    Returns
    -------
    bcg_candidates : list of tuple
        List of BCGs with matched redshifts and probabilities, in the format (RA, Dec, z, P).

    """
    redshift_df = _standardize_redshift_df(redshift_df)

    if redshift_df.empty:
        print("Redshift catalog is empty. Returning BCGs without redshift matches.")
        galaxy_coords = SkyCoord(ra=[] * u.deg, dec=[] * u.deg, frame='icrs')
        z = np.array([], dtype=float)
    else:
        galaxy_coords = SkyCoord(
            ra=redshift_df[RA_COL].values * u.deg,
            dec=redshift_df[DEC_COL].values * u.deg,
            frame='icrs'
        )
        z = redshift_df[Z_COL].values


    # Load BCG candidates from CSV
    bcg_candidates: list[tuple[float, float, float | None, float | None]] = [] # (ra, dec, z, P)

    # Candidates.csv
    try:
        data = pd.read_csv(cluster.bcg_csv_file)
    except FileNotFoundError:
        print(f"File not found: {cluster.bcg_csv_file}")
        data = pd.DataFrame()


    # If cluster exists in CSV
    if not data.empty and cluster.name in data['NAME'].values:
        bcg_data = data.loc[data['NAME'] == cluster.name]

        # Dynamically collect only valid (non-empty/non-NaN) BCGs
        ra_list, dec_list, prob_list = [], [], []
        for i in range(5):
            ra_val = bcg_data[f'RA_CEN{i}'].values[0] if f'RA_CEN{i}' in bcg_data.columns else None
            dec_val = bcg_data[f'DEC_CEN{i}'].values[0] if f'DEC_CEN{i}' in bcg_data.columns else None
            # Consider valid if neither is nan/None/empty
            if (
                ra_val not in [None, '', ' '] and dec_val not in [None, '', ' ']
                and pd.notna(ra_val) and pd.notna(dec_val)
            ):
                ra_list.append(float(ra_val))
                dec_list.append(float(dec_val))
                # Probability column (may be missing)
                p_col = f'P{i}'
                if p_col in bcg_data.columns:
                    prob_list.append(bcg_data[p_col].values[0])
                else:
                    prob_list.append(None)

        for ra_bcg, dec_bcg, p in zip(ra_list, dec_list, prob_list):
            bcg_candidates.append((float(ra_bcg), float(dec_bcg), None, p))

    else:
        print(f"Cluster '{cluster.name}' not found in CSV.")

    # Append manual BCGs
    if manual_BCGs:
        for ra_manual, dec_manual in manual_BCGs:
            bcg_candidates.append((float(ra_manual), float(dec_manual), None, None))

    if not bcg_candidates:
    # Fallback #1: try redMaPPer centering candidates near the cluster center
        if cluster.coords is not None:
            rm_cands = query_redmapper_bcg_candidates(
                cluster.coords,
                radius_arcmin=10.0,
                verbose=verbose,
            )
            if rm_cands:
                print("Using redMaPPer centering candidates as BCG list.")
                bcg_candidates = rm_cands

    if not bcg_candidates:
        # Fallback #2: use the cluster center as a "best available" BCG.
        if cluster.coords is None:
            raise ValueError(
                f"No BCG candidates available for {cluster.name!r}, and cluster.coords is None."
            )

        ra_fallback = float(cluster.coords.ra.deg)
        dec_fallback = float(cluster.coords.dec.deg)
        z_fallback = float(cluster.redshift) if cluster.redshift is not None else None
        p_fallback = 1.0

        print("No BCG candidates available; using cluster.coords as fallback BCG.")
        bcg_candidates = [(ra_fallback, dec_fallback, z_fallback, p_fallback)]
        if verbose:
            print(f"Fallback BCG: RA={ra_fallback}, Dec={dec_fallback}, z={z_fallback}, P={p_fallback}")



    # Match BCGs to redshifts

    bcg_coords = SkyCoord(
        ra=[bcg[0] for bcg in bcg_candidates] * u.deg,
        dec=[bcg[1] for bcg in bcg_candidates] * u.deg,
        frame="icrs"
    )

    if len(galaxy_coords) == 0:
        print("No galaxies with redshifts available for matching.")
        return [(ra, dec, None, p) for ra, dec, _, p in bcg_candidates]
    
    bcg_idx, gal_idx, sep = match_skycoords_unique(
        bcg_coords,
        galaxy_coords,
        match_tol_arcsec=match_tol_arcsec,
    )

    sep_bcg = [None] * len(bcg_candidates)


    for b_idx, g_idx, s in zip(bcg_idx, gal_idx, sep):
        bcg_candidates[b_idx] = (bcg_candidates[b_idx][0], bcg_candidates[b_idx][1], z[g_idx], bcg_candidates[b_idx][3])
        sep_bcg[b_idx] = s


    # Verbose reporting
    if verbose:
        print(f"\n--- BCG Matching Report for {cluster.name} ---")
        for i, (ra, dec, z_bcg, prob) in enumerate(bcg_candidates):
            sep_info = f", closest sep = {sep_bcg[i]:.2f}\"" if sep_bcg[i] is not None else ", no match"
            z_str = f"{z_bcg:.5f}" if z_bcg is not None else "unknown"
            print(f"BCG {i+1}: RA = {ra:.5f}, Dec = {dec:.5f}, z = {z_str}{sep_info}")

            if sep_bcg[i] is None:
                print(f"WARNING: No galaxy found within {match_tol_arcsec} arcsec!")

        print("--- End of Report ---\n")

    return bcg_candidates


def save_BCG_catalog(
        cluster: Cluster,
        phot_df: pd.DataFrame,
        bcgs: list[tuple],
        *,
        match_tol_arcsec: float = DEFAULT_TOL_ARCSEC,
        n_bcgs: int = 5,
        verbose: bool = True,
    ) -> pd.DataFrame:
    """
    Match BCG candidates from `candidate_mergers.csv` to legacy redshift+photometry catalog,
    append P and match separation, and save to CSV. Always includes all BCGs, even if no match is found.

    Parameters
    ----------
    cluster : Cluster
        Cluster object containing all necessary metadata (paths, coordinates, etc.).
    phot_df : pd.DataFrame
        Photometric catalog DataFrame (e.g., legacy or PanSTARRS) containing 'RA', 'Dec', 'z', 'sigma_z', 'source', and photometry columns.
    bcgs : list of tuples
        List of BCG coordinates and redshifts, where each tuple is (ra, dec, z, P).
    match_tol_arcsec : float
        Maximum allowed separation for BCG-galaxy matching (arcsec).
    n_bcgs : int
        Number of closest matches to print for each BCG (for verbose output).
    verbose : bool
        If True, print detailed matching information to console.

    Returns
    -------
    output_df : pd.DataFrame
        DataFrame containing the matched BCGs with their photometric information.
    """
    phot_df = _standardize_phot_df(phot_df)

    if phot_df.empty:
        print("Photometric catalog is empty. Cannot match BCGs. Returning BCG table without photometry.")
        output_rows = []
        for i, (ra_bcg, dec_bcg, z_match, P) in enumerate(bcgs):
            row_data = {
                RA_COL: ra_bcg,
                DEC_COL: dec_bcg,
                "BCG_separation_arcsec": np.nan,
                "BCG_priority": i + 1,
                "BCG_input_RA": ra_bcg,
                "BCG_input_Dec": dec_bcg,
                "BCG_probability": P,
            }
            output_rows.append(row_data)
        output_df = pd.DataFrame(output_rows)
        output_csv = os.path.join(cluster.redshift_path, 'BCGs.csv')
        output_df.to_csv(output_csv, index=False)
        print(f"Saved {len(output_df)} BCG(s) to: {output_csv} (no photometry available)")
        return output_df

    # Load full redshift+photometry catalog
    phot_coords = SkyCoord(
        phot_df[RA_COL].values * u.deg,
        phot_df[DEC_COL].values * u.deg,
        frame='icrs'
    )

    if verbose:
        for i, bcg in enumerate(bcgs):
            print(f"Processing BCG {i+1}: RA={bcg[0]}, Dec={bcg[1]}, z={bcg[2]}, P={bcg[3]}")

    output_rows: list[dict] = []
    for i, (ra_bcg, dec_bcg, z_match, P) in enumerate(bcgs):
        bcg_coord = SkyCoord(ra_bcg * u.deg, dec_bcg * u.deg, frame='icrs')
        sep = bcg_coord.separation(phot_coords)
        min_sep = sep.min().arcsec
        closest_idx = sep.argmin()
        closest_ra = phot_coords[closest_idx].ra.deg
        closest_dec = phot_coords[closest_idx].dec.deg
        
        row_data = {}
        # Get sorted index of separations
        sorted_indices = np.argsort(sep.arcsec)
        if verbose:
            print(f"Closest match for BCG {i+1}: RA={closest_ra:.6f}, Dec={closest_dec:.6f}, Sep={min_sep:.3f} arcsec")
            print(f"\nTop {n_bcgs} closest matches for BCG {i+1} (RA={ra_bcg:.6f}, Dec={dec_bcg:.6f}):")
        for j in range(n_bcgs):
            idx = sorted_indices[j]
            if verbose:
                print(f"  {j+1}. RA={phot_coords[idx].ra.deg:.6f}, Dec={phot_coords[idx].dec.deg:.6f}, Sep={sep[idx].arcsec:.3f} arcsec")


        if min_sep <= match_tol_arcsec:
            match_idx = sep.argmin()
            matched_row = phot_df.iloc[match_idx]

            # Copy all photometric catalog columns
            row_data.update(matched_row.to_dict())

            # Add separation info
            row_data["BCG_separation_arcsec"] = min_sep
        else:
            # No match: fill legacy fields with NaN
            row_data["RA"] = ra_bcg
            row_data["Dec"] = dec_bcg
            row_data["BCG_separation_arcsec"] = np.nan
            row_data["z"] = z_match if z_match is not None else np.nan

        # Always include metadata
        row_data["BCG_priority"] = i + 1
        row_data["BCG_input_RA"] = ra_bcg
        row_data["BCG_input_Dec"] = dec_bcg
        row_data["BCG_probability"] = P




        output_rows.append(row_data)


    output_csv = os.path.join(cluster.redshift_path, 'BCGs.csv')
    output_df = pd.DataFrame(output_rows)
    output_df.to_csv(output_csv, index=False)
    print(f"Saved {len(output_df)} BCG(s) to: {output_csv} (including unmatched)")
    return output_df


# ------------------------------------
# Table Functions
# ------------------------------------

def export_redshift_table(
        cluster: Cluster,
        combined_df: pd.DataFrame,
        *,
        save_tex: bool = True,
        print_tex: bool = False,
    ) -> None:
    """
    Generate a LaTeX deluxetable (first 5 rows) and a machine-readable ASCII table.

    Parameters
    ----------
    cluster : Cluster
        Cluster object containing all necessary metadata (paths, coordinates, etc.).
    save_tex : bool
        If True, save to <folder>/redshifts_table.tex; else don't save.
    print_tex : bool
        If True, print the LaTeX string to the console.
    """


    # Ensure expected column names (or rename safely if already correct)
    combined_df = combined_df.rename(columns={c: c.strip() for c in combined_df.columns})
    expected = [RA_COL, DEC_COL, Z_COL, SIGMA_Z_COL, SPEC_SOURCE_COL]
    if list(combined_df.columns[:5]) != expected:
        # if the first five columns correspond but are misnamed, force labels for table
        combined_df = combined_df.rename(columns=dict(zip(combined_df.columns[:5], expected)))

    clean_id = cluster.identifier.replace(" ", "")
    header = (
        "\\begin{deluxetable}{ccccc}\n"
        f"\\tablecaption{{{clean_id} Spectroscopic Redshifts}}\\label{{tab:{clean_id}_redshifts}}\n"
        "\\tablehead{\n"
        "  \\colhead{RA [deg]} & \\colhead{Dec [deg]} & \\colhead{Redshift} & "
        "\\colhead{Error} & \\colhead{Source} \\\n"
        "}\n"
        "\\centering\n"
        "\\startdata\n"
    )

    # First 5 rows as a preview
    body_lines = []
    for _, row in combined_df.head(5).iterrows():
        ra = f"{row[RA_COL]:.5f}"
        dec = f"{row[DEC_COL]:.5f}"
        z = f"{row[Z_COL]:.5f}"

        sigma = row[SIGMA_Z_COL]
        sigma_str = "" if (sigma == 0.0) else f"{sigma:.1e}"

        source = row[SPEC_SOURCE_COL] if pd.notnull(row[SPEC_SOURCE_COL]) else "Unknown"
        body_lines.append(f"{ra} & {dec} & {z} & {sigma_str} & {source} \\\\ \\hline")

    footer = "\\enddata\n\\end{deluxetable}"

    latex_str = header + "\n".join(body_lines) + "\n" + footer

    # Always print; optionally save
    path = os.path.join(cluster.redshift_path, "redshifts_table.tex")
    emit_latex(latex_str, save_tex=save_tex, print_tex=print_tex, save_path=path)

    # Also keep the full machine-readable table alongside
    combined_df.to_csv(f"{cluster.redshift_path}/redshifts_table.dat", sep="\t", index=False)
    print(f"Machine-readable table written to {cluster.redshift_path}/redshifts_table.dat")


def print_bcg_deluxetable(
        cluster: Cluster,
        bcg_df: pd.DataFrame,
        *,
        mag_col: str = DEFAULT_MAG,
        save_tex: bool = True,
        print_tex: bool = False
    ) -> None:
    """
    Print a LaTeX deluxetable with BCG info and optionally save it as a .tex file.

    Parameters
    ----------
    cluster : Cluster
        Cluster object containing all necessary metadata (paths, coordinates, etc.).
    bcg_df : pd.DataFrame
        DataFrame containing BCG information
    mag_col : str
        Column name for the magnitude to show (default: 'rmag').
    save_tex : bool
        If True, save to <folder>/BCGs_table.tex; else don't save.
    print_tex : bool
        If True, print the LaTeX string to the console.
    """

    path = os.path.join(cluster.redshift_path, "BCGs_table.tex")

    # Build LaTeX
    clean_id = cluster.identifier.replace(" ", "")
    lines = []
    lines.append(r"\begin{deluxetable}{cccccc}[hb]")
    lines.append(f"    \\caption{{{clean_id} BCG Information}}\\label{{tab:{clean_id}_BCG}}")
    lines.append("    \\tablehead{")
    lines.append("        \\colhead{BCG} & \\colhead{Probability} & \\colhead{Redshift} & "
                 f"\\colhead{{{mag_col}\\tablenotemark{{\\footnotesize d}}}} & "
                 "\\colhead{RA {[}deg{]}} & \\colhead{Dec {[}deg{]}} ")
    lines.append("    }")
    lines.append("    \\centering")
    lines.append("    \\startdata")

    for _, row in bcg_df.sort_values("BCG_priority").iterrows():
        priority = int(row["BCG_priority"])
        mag = row.get(mag_col, None)
        z = row.get(Z_COL, None)
        prob = row.get("BCG_probability", None)
        ra = row.get("BCG_input_RA", row.get(RA_COL))
        dec = row.get("BCG_input_Dec", row.get(DEC_COL))
        source = row.get(SPEC_SOURCE_COL, "---")

        z_str = f"{z:.3f}" if pd.notnull(z) else "---"
        prob_str = f"{prob:.4f}" if pd.notnull(prob) else "---"
        mag_str = f"{mag:.2f}" if pd.notnull(mag) else "---"

        lines.append(
            f"    {priority}  & {prob_str} & {z_str}\\tablenotemark{{\\footnotesize a}} "
            f"& {mag_str} & {ra:.5f} & {dec:.5f} & {source}\\\\ \\hline"
        )

    lines.append("    \\enddata")
    lines.append("\\tablenotetext{a}{SDSS DR18 \\citep{almeida2023eighteenth}}")
    lines.append("\\tablenotetext{b}{DESI DR1 \\citep{abdul2025data}}")
    lines.append("\\tablenotetext{c}{DEIMOS (This work)}")
    lines.append("\\tablenotetext{d}{DESI Legacy Survey DR10 \\citep{dey2019overview}}")
    lines.append("\\end{deluxetable}")

    latex_str = "\n".join(lines)
    emit_latex(latex_str, save_tex=save_tex, print_tex=print_tex, save_path=path)


def print_cluster_table(cluster: Cluster) -> None:
    """
    Prints a formatted report of cluster information.

    Parameters
    ----------
    cluster : Cluster
        Cluster object with attributes:
        - 'RA': Right Ascension in degrees
        - 'Dec': Declination in degrees
        - 'z': Redshift
        - 'z_err': Redshift error
        - 'richness': Cluster richness
        - 'richness_err': Richness error
    """


    print(f"\\textbf{{{cluster.name}}} & {cluster.redshift:.4f} & {cluster.redshift_err:.4f} & {cluster.richness:.2f} & {cluster.richness_err:.2f} & {cluster.coords.ra.deg:.4f} & {cluster.coords.dec.deg:.4f} \\\\ \\hline")


# ------------------------------------
# Main Pipeline Function
# ------------------------------------

def build_redshift_catalog(
        cluster: Cluster,
        *,
        max_sep_arcsec: float = DEFAULT_TOL_ARCSEC,
        save_tex: bool = True,
        print_tex: bool = False,
        with_legacy: bool = True,
        manual_list: list[tuple[float, float]] | None = None
    ) -> None:
    """
    Main pipeline function to build redshift catalogs, match photometry, identify BCGs, and export results.

    Parameters
    ----------
    cluster : Cluster
        Cluster object containing relevant attributes and paths.
    max_sep_arcsec : float, optional
        Maximum separation in arcseconds for matching BCGs (default is DEFAULT_TOL_ARCSEC).
    export_tex : bool, optional
        Whether to export LaTeX/ASCII tables (default is True).
    print_tex : bool, optional
        Whether to print LaTeX tables to console (default is False).
    with_legacy : bool, optional
        Whether to use Legacy photometric catalog in BCG matching (default is True).
    manual_list : list of tuple of float, optional
        Manual list of BCG coordinates (default is None).
    """

    # Save new deimos spectroscopy
    new_spectroscopy_df = load_spectroscopic_redshifts(cluster)
    deimos_path = os.path.join(cluster.redshift_path, "deimos.csv")
    new_spectroscopy_df.to_csv(deimos_path, index=False)
    print(f"New DEIMOS redshift catalog saved to {deimos_path}")

    # Create combined_redshifts.csv
    combined_spec_df = load_archival_redshifts(cluster, new_spectroscopy_df)
    combined_path = os.path.join(cluster.redshift_path, "combined_redshifts.csv")
    combined_spec_df.to_csv(combined_path, index=False)
    print(f"Combined redshift catalog saved to {combined_path}")

    # Create redshift matched photometric catalogs
    panstarrs_df=pd.read_csv(f"{cluster.photometry_path}/photometry_PanSTARRS.csv")
    legacy_df=pd.read_csv(f"{cluster.photometry_path}/photometry_legacy.csv")

    _matched_panstarrs, _matched_legacy, catalog_panstarrs, catalog_legacy = create_matched_catalogs(cluster, panstarrs_df=panstarrs_df, legacy_df=legacy_df, redshift_df=combined_spec_df)

    if with_legacy:
        phot_catalog = catalog_legacy
    else:
        phot_catalog = catalog_panstarrs
    BCGs = get_BCG(cluster, combined_spec_df, manual_BCGs=manual_list)
    bcg_df = save_BCG_catalog(cluster, phot_catalog, BCGs, match_tol_arcsec=max_sep_arcsec)

    if cluster.richness is not None and cluster.redshift is not None:
  
        print_cluster_table(cluster)


    print_bcg_deluxetable(cluster, bcg_df, mag_col='rmag', save_tex=save_tex, print_tex=print_tex)
    export_redshift_table(cluster, combined_spec_df, save_tex=save_tex, print_tex=print_tex)



def main():
    parser = argparse.ArgumentParser(description="Redshift catalog builder and exporter")
    parser.add_argument(
        "cluster_id",
        type=str,
        help="Cluster ID (e.g. '1327') or full RMJ/Abell name."
    )
    parser.add_argument(
            "-p",
            "--path",
            type=str,
            default=None,
            help="Base path for cluster science directories (default: cluster.py BASE_PATH)"
    )
    parser.add_argument(
        "--manual-list",
        type=str,
        default=None,
        help="Manual list of BCG coordinates as a string representation of a list of tuples, e.g. '[(184.800825, 50.9097139), (184.756871, 50.9070661)]'"
    )
    parser.add_argument(
        "--max-sep",
        type=float,
        default=DEFAULT_TOL_ARCSEC,
        help=f"Maximum separation (arcsec) for BCG-galaxy matching [default: {DEFAULT_TOL_ARCSEC}]"
    )
    parser.add_argument(
        "--with-legacy",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Use Legacy photometric catalog in BCG matching (False=PanSTARRS) [default: True]"
    )
    parser.add_argument(
        "--save-tex",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Export LaTeX/ASCII tables [default: True]"
    )
    parser.add_argument(
        "--print-tex",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Print LaTeX tables to console [default: False]"
    )
    args = parser.parse_args()

    cluster = Cluster(args.cluster_id, base_path=args.path)
    cluster.populate()

    if args.manual_list is None:
        manual_list = None
    elif isinstance(args.manual_list, str):
        manual_list = ast.literal_eval(args.manual_list)
    else:
        manual_list = args.manual_list 

    build_redshift_catalog(
        cluster,
        save_tex=args.save_tex,
        print_tex=args.print_tex,
        max_sep_arcsec=args.max_sep,
        manual_list=manual_list,
    )

if __name__ == "__main__":
    main()
