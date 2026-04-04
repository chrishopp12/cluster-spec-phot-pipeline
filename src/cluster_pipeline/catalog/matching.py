#!/usr/bin/env python3
"""
matching.py

Stage 3: Catalog Matching and BCG Identification
---------------------------------------------------------

Merges archival spectroscopy with user DEIMOS spectra (DEIMOS highest
priority) and manual redshift files (lowest priority), crossmatches with
photometry, and identifies BCG candidates.

Data products:
  - Redshifts/combined_redshifts.csv   Merged spec catalog (DEIMOS + archival + manual)
  - Photometry/{survey}_matched.csv    Per-survey crossmatch (spec+phot only)
  - BCGs.csv                           BCG catalog with spec + phot columns

Column conventions:
  - Spectroscopic: RA, Dec, z, sigma_z, spec_source
  - Photometric:   RA, Dec, gmag, rmag, imag, phot_source,
                   lum_weight_g, lum_weight_r, lum_weight_i, g_r, r_i, g_i

Notes:
  - The combined_redshifts catalog keeps ALL spec sources (matched and
    unmatched to photometry).  The matched catalogs contain ONLY sources
    with both spec and phot data.
  - BCGs.csv is self-contained: it includes photometry columns so
    downstream code doesn't need to re-crossmatch.
  - LaTeX table exports live in io/tables.py, not here.
"""
from __future__ import annotations

import os
import glob
from typing import Any

import numpy as np
import pandas as pd

from astropy.coordinates import SkyCoord
import astropy.units as u

from cluster_pipeline.models.cluster import Cluster
from cluster_pipeline.models.bcg import BCG
from cluster_pipeline.utils import coerce_to_numeric
from cluster_pipeline.utils.coordinates import make_skycoord, skycoord_from_df
from cluster_pipeline.utils.resolvers import get_redmapper_bcg_candidates
from cluster_pipeline.config import load_config
from cluster_pipeline.io.catalogs import read_bcg_csv, select_bcgs

# ---------------------------------------------------------------
# Column constants
# ---------------------------------------------------------------
COLUMNS_SPEC = ["RA", "Dec", "z", "sigma_z", "spec_source"]
COLUMNS_PHOT = [
    "RA", "Dec", "gmag", "rmag", "imag", "phot_source",
    "lum_weight_g", "lum_weight_r", "lum_weight_i", "g_r", "r_i", "g_i",
]
DEFAULT_TOL_ARCSEC = 3.0


# ---------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------

def _standardize_redshift_df(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a redshift DataFrame has the standard spec columns and types.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with redshift data.

    Returns
    -------
    pd.DataFrame
        Standardized DataFrame with columns: RA, Dec, z, sigma_z, spec_source.
        Rows missing RA, Dec, or z are dropped.
    """
    if df.empty:
        return pd.DataFrame(columns=COLUMNS_SPEC)

    out = df.copy()
    out = out.rename(columns={c: c.strip() for c in out.columns})

    # Normalize column aliases
    if "spec_source" not in out.columns and "source" in out.columns:
        out = out.rename(columns={"source": "spec_source"})
    if "sigma_z" not in out.columns:
        if "zerr" in out.columns:
            out = out.rename(columns={"zerr": "sigma_z"})
        else:
            out["sigma_z"] = np.nan

    for col in COLUMNS_SPEC:
        if col not in out.columns:
            out[col] = np.nan if col != "spec_source" else "Unknown"

    out = coerce_to_numeric(out, ["RA", "Dec", "z", "sigma_z"])
    out = out.dropna(subset=["RA", "Dec", "z"]).reset_index(drop=True)
    return out


def _standardize_phot_df(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a photometry DataFrame has RA/Dec and numeric types.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with photometry data.

    Returns
    -------
    pd.DataFrame
        Standardized DataFrame, rows missing RA or Dec dropped.
    """
    if df.empty:
        return pd.DataFrame(columns=["RA", "Dec"])

    out = df.copy()
    out = out.rename(columns={c: c.strip() for c in out.columns})

    for col in ["RA", "Dec"]:
        if col not in out.columns:
            out[col] = np.nan

    out = coerce_to_numeric(out, ["RA", "Dec"])
    out = out.dropna(subset=["RA", "Dec"]).reset_index(drop=True)
    return out


# ---------------------------------------------------------------
# Core matching
# ---------------------------------------------------------------

def match_skycoords_unique(
    coords1: SkyCoord,
    coords2: SkyCoord,
    *,
    match_tol_arcsec: float = DEFAULT_TOL_ARCSEC,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match two SkyCoord arrays with unique (one-to-one) assignments.

    Each source in *coords2* can be claimed by at most one source in
    *coords1*.  When multiple coords1 entries match the same coords2
    entry, the closest pair wins.

    Parameters
    ----------
    coords1 : SkyCoord
        First set of coordinates ("query" catalog).
    coords2 : SkyCoord
        Second set of coordinates ("reference" catalog).
    match_tol_arcsec : float
        Maximum allowed separation in arcseconds.

    Returns
    -------
    matched_idx1 : np.ndarray of int
        Indices into coords1 that have a match.
    matched_idx2 : np.ndarray of int
        Corresponding indices into coords2.
    matched_sep : np.ndarray of float
        Angular separations of each matched pair in arcseconds.
    """
    if len(coords1) == 0 or len(coords2) == 0:
        return (
            np.array([], dtype=int),
            np.array([], dtype=int),
            np.array([], dtype=float),
        )

    idx2, sep2d, _ = coords1.match_to_catalog_sky(coords2)
    mask = sep2d < (match_tol_arcsec * u.arcsec)

    matched_i1 = np.where(mask)[0]
    matched_i2 = idx2[mask]
    matched_sep2d = sep2d[mask].arcsec

    # Unique assignment: keep only the closest match for each coords2 entry
    seen: dict[int, tuple[int, float]] = {}
    for i1, i2, sep in zip(matched_i1, matched_i2, matched_sep2d):
        if i2 not in seen or sep < seen[i2][1]:
            seen[i2] = (i1, sep)

    pairs = sorted(
        [(i1, i2, sep) for i2, (i1, sep) in seen.items()],
        key=lambda x: x[0],
    )

    if not pairs:
        return (
            np.array([], dtype=int),
            np.array([], dtype=int),
            np.array([], dtype=float),
        )

    m1 = np.array([p[0] for p in pairs], dtype=int)
    m2 = np.array([p[1] for p in pairs], dtype=int)
    ms = np.array([p[2] for p in pairs], dtype=float)
    return m1, m2, ms


# ---------------------------------------------------------------
# Merge spectra
# ---------------------------------------------------------------

def _merge_spectra(
    archival_df: pd.DataFrame,
    deimos_df: pd.DataFrame,
    *,
    match_tol_arcsec: float = DEFAULT_TOL_ARCSEC,
    verbose: bool = True,
) -> pd.DataFrame:
    """Merge archival and DEIMOS spectra, with DEIMOS as highest priority.

    Archival sources within *match_tol_arcsec* of a DEIMOS source are
    removed (the DEIMOS version is kept).  Unmatched archival sources
    are appended.

    Parameters
    ----------
    archival_df : pd.DataFrame
        Archival spectroscopic catalog (RA, Dec, z, sigma_z, spec_source).
    deimos_df : pd.DataFrame
        DEIMOS spectroscopic catalog (same columns).
    match_tol_arcsec : float
        Positional tolerance for deduplication.
    verbose : bool
        Print merge statistics.

    Returns
    -------
    pd.DataFrame
        Combined catalog with DEIMOS rows first, then unmatched archival.
    """
    archival_df = _standardize_redshift_df(archival_df)
    deimos_df = _standardize_redshift_df(deimos_df)

    if deimos_df.empty:
        if verbose:
            print("No DEIMOS spectra. Returning archival catalog only.")
        return archival_df.copy()

    if archival_df.empty:
        if verbose:
            print("No archival spectra. Returning DEIMOS catalog only.")
        return deimos_df.copy()

    # Find archival sources that duplicate a DEIMOS source
    coords_arc = skycoord_from_df(archival_df)
    coords_deimos = skycoord_from_df(deimos_df)

    arc_idx, _, _ = match_skycoords_unique(
        coords_arc, coords_deimos, match_tol_arcsec=match_tol_arcsec,
    )

    archival_unique = archival_df.drop(index=arc_idx).reset_index(drop=True)
    combined = pd.concat([deimos_df, archival_unique], ignore_index=True)

    if verbose:
        n_arc = len(archival_df)
        n_deimos = len(deimos_df)
        n_dup = len(arc_idx)
        print(f"DEIMOS sources: {n_deimos}")
        print(f"Archival sources: {n_arc}")
        print(f"  Duplicates removed (within {match_tol_arcsec}\"): {n_dup}")
        print(f"  Archival unique: {n_arc - n_dup}")
        print(f"Combined total: {len(combined)}")

    return combined


# ---------------------------------------------------------------
# Append manual spectra (lowest priority)
# ---------------------------------------------------------------

def _append_manual_spectra(
    combined_df: pd.DataFrame,
    manual_df: pd.DataFrame,
    *,
    match_tol_arcsec: float = DEFAULT_TOL_ARCSEC,
    verbose: bool = True,
) -> pd.DataFrame:
    """Append manual spectra at lowest priority.

    Manual sources within *match_tol_arcsec* of any existing source in
    *combined_df* are dropped.  Only unmatched manual sources are appended.

    Parameters
    ----------
    combined_df : pd.DataFrame
        Existing merged catalog (DEIMOS + archival).
    manual_df : pd.DataFrame
        Manual redshift catalog (RA, Dec, z, sigma_z, spec_source).
    match_tol_arcsec : float
        Positional tolerance for deduplication.
    verbose : bool
        Print merge statistics.

    Returns
    -------
    pd.DataFrame
        Combined catalog with manual-unique rows appended.
    """
    combined_df = _standardize_redshift_df(combined_df)
    manual_df = _standardize_redshift_df(manual_df)

    if manual_df.empty:
        return combined_df.copy()

    if combined_df.empty:
        if verbose:
            print(f"No existing spectra. Returning {len(manual_df)} manual sources only.")
        return manual_df.copy()

    coords_manual = skycoord_from_df(manual_df)
    coords_combined = skycoord_from_df(combined_df)

    manual_dup_idx, _, _ = match_skycoords_unique(
        coords_manual, coords_combined, match_tol_arcsec=match_tol_arcsec,
    )

    manual_unique = manual_df.drop(index=manual_dup_idx).reset_index(drop=True)
    result = pd.concat([combined_df, manual_unique], ignore_index=True)

    if verbose:
        n_manual = len(manual_df)
        n_dup = len(manual_dup_idx)
        print(f"Manual sources: {n_manual}")
        print(f"  Duplicates removed (within {match_tol_arcsec}\"): {n_dup}")
        print(f"  Manual unique: {n_manual - n_dup}")
        print(f"Combined total: {len(result)}")

    return result


# ---------------------------------------------------------------
# Crossmatch photometry
# ---------------------------------------------------------------

def crossmatch_photometry_with_redshifts(
    phot_df: pd.DataFrame,
    redshift_df: pd.DataFrame,
    *,
    match_tol_arcsec: float = DEFAULT_TOL_ARCSEC,
) -> pd.DataFrame:
    """Crossmatch a photometry catalog with a redshift catalog.

    Returns ONLY sources that have BOTH a spectroscopic redshift and
    a photometric match (true matches).  Sources with spec but no phot
    are in combined_redshifts.csv; this function does not include them.

    Parameters
    ----------
    phot_df : pd.DataFrame
        Photometry catalog with RA, Dec, and magnitude columns.
    redshift_df : pd.DataFrame
        Spectroscopic catalog with RA, Dec, z, sigma_z, spec_source.
    match_tol_arcsec : float
        Maximum separation for a valid match.

    Returns
    -------
    pd.DataFrame
        Matched catalog: spec columns joined with phot columns for each
        matched pair.  Uses spec RA/Dec as the positional reference.
    """
    phot_df = _standardize_phot_df(phot_df)
    redshift_df = _standardize_redshift_df(redshift_df)

    if redshift_df.empty or phot_df.empty:
        # Return an empty DataFrame with the union of column names
        all_cols = list(dict.fromkeys(
            list(redshift_df.columns) + [c for c in phot_df.columns if c not in ("RA", "Dec")]
        ))
        return pd.DataFrame(columns=all_cols)

    spec_coords = skycoord_from_df(redshift_df)
    phot_coords = skycoord_from_df(phot_df)

    spec_idx, phot_idx, _ = match_skycoords_unique(
        spec_coords, phot_coords, match_tol_arcsec=match_tol_arcsec,
    )

    if len(spec_idx) == 0:
        all_cols = list(dict.fromkeys(
            list(redshift_df.columns) + [c for c in phot_df.columns if c not in ("RA", "Dec")]
        ))
        return pd.DataFrame(columns=all_cols)

    # Build matched catalog: start from spec rows, add phot columns
    matched = redshift_df.iloc[spec_idx].reset_index(drop=True)
    phot_matched = phot_df.iloc[phot_idx].reset_index(drop=True)

    phot_cols = [c for c in phot_matched.columns if c not in ("RA", "Dec")]
    for col in phot_cols:
        matched[col] = phot_matched[col].values

    return matched


# ---------------------------------------------------------------
# BCG retrieval
# ---------------------------------------------------------------

def get_BCGs(
    cluster: Cluster,
    *,
    radius_arcmin: float = 5.0,
    verbose: bool = True,
) -> list[BCG]:
    """Retrieve BCG candidates, trying sources in priority order.

    Priority order:
      1. config.yaml ``BCGs`` section
      2. Existing BCGs.csv on disk
      3. redMaPPer VizieR query
      4. Cluster center as a single fallback BCG

    Parameters
    ----------
    cluster : Cluster
        Cluster object with paths and coordinates.
    radius_arcmin : float
        Search radius for redMaPPer query.
    verbose : bool
        Print diagnostics.

    Returns
    -------
    list[BCG]
        BCG candidate objects (typically 1--5).

    Raises
    ------
    ValueError
        If no BCGs can be determined and cluster.coords is None.
    """
    # --- 1. Try config.yaml ---
    cfg = load_config(cluster.cluster_path)
    bcg_section = cfg.get("BCGs", cfg.get("bcgs", None))

    if bcg_section and isinstance(bcg_section, dict):
        bcgs = []
        for bcg_id, bcg_cfg in bcg_section.items():
            if isinstance(bcg_cfg, dict) and "ra" in bcg_cfg and "dec" in bcg_cfg:
                bcgs.append(BCG.from_config(int(bcg_id), bcg_cfg))
        if bcgs:
            if verbose:
                print(f"Loaded {len(bcgs)} BCG(s) from config.yaml.")
            return bcgs

    # --- 2. Try BCGs.csv ---
    bcg_df = read_bcg_csv(cluster, verbose=verbose)
    if not bcg_df.empty:
        bcg_df = select_bcgs(bcg_df)
        bcgs = []
        for _, row in bcg_df.iterrows():
            bcgs.append(BCG.from_dataframe_row(row))
        if bcgs:
            if verbose:
                print(f"Loaded {len(bcgs)} BCG(s) from BCGs.csv.")
            return bcgs

    # --- 3. Try redMaPPer ---
    if cluster.coords is not None:
        rm_tuples = get_redmapper_bcg_candidates(
            cluster.coords, radius_arcmin=radius_arcmin, verbose=verbose,
        )
        if rm_tuples:
            bcgs = []
            for i, (ra, dec, z, prob) in enumerate(rm_tuples, start=1):
                bcgs.append(BCG(
                    bcg_id=i, ra=ra, dec=dec, z=z,
                    rank=i, probability=prob, label=str(i),
                ))
            if verbose:
                print(f"Loaded {len(bcgs)} BCG(s) from redMaPPer query.")
            return bcgs

    # --- 4. Fallback: cluster center ---
    if cluster.coords is None:
        raise ValueError(
            f"Cannot retrieve BCGs for {cluster.identifier!r}: "
            "no config, CSV, redMaPPer match, or cluster coordinates."
        )
    ra = float(cluster.coords.ra.deg)
    dec = float(cluster.coords.dec.deg)
    z = float(cluster.redshift) if cluster.redshift is not None else None
    if verbose:
        print("No BCG candidates found; using cluster center as fallback.")
    return [BCG(bcg_id=1, ra=ra, dec=dec, z=z, rank=1, probability=1.0, label="1")]


# ---------------------------------------------------------------
# BCG matching helpers
# ---------------------------------------------------------------

def _print_top_n_matches(
    *,
    label: str,
    bcg_i: int,
    ra_bcg: float,
    dec_bcg: float,
    cat_coords: SkyCoord,
    sep2d: u.Quantity,
    cat_df: pd.DataFrame,
    n: int,
    source_col: str,
    z_col: str = "z",
) -> None:
    """Print the top-N closest catalog matches for one BCG (verbose helper).

    Parameters
    ----------
    label : str
        Catalog label for display (e.g. "spectroscopic" or "photometric").
    bcg_i : int
        1-based BCG index.
    ra_bcg, dec_bcg : float
        BCG coordinates in degrees.
    cat_coords : SkyCoord
        Catalog coordinates aligned with *cat_df* rows.
    sep2d : astropy Quantity
        Separations aligned with *cat_coords* / *cat_df*.
    cat_df : pd.DataFrame
        Catalog DataFrame.
    n : int
        Number of closest matches to print.
    source_col : str
        Column name for source label.
    z_col : str
        Column name for redshift.
    """
    if len(cat_coords) == 0:
        print(f"  No catalog coordinates available for {label} matching.")
        return

    order = np.argsort(sep2d.to_value(u.arcsec))
    top = order[: min(n, len(order))]

    print(
        f"\nTop {len(top)} closest {label} matches for BCG {bcg_i} "
        f"(RA={ra_bcg:.6f}, Dec={dec_bcg:.6f}):"
    )
    for rank, idx in enumerate(top, start=1):
        idx = int(idx)
        row = cat_df.iloc[idx]
        print(
            f"  {rank}. RA={cat_coords[idx].ra.deg:.6f}, "
            f"Dec={cat_coords[idx].dec.deg:.6f}, "
            f"Sep={sep2d[idx].to_value(u.arcsec):.3f}\", "
            f"z={row.get(z_col, 'N/A')}, "
            f"{source_col}={row.get(source_col, 'N/A')}"
        )


def _match_bcgs_spec(
    bcgs: list[BCG],
    redshift_df: pd.DataFrame,
    *,
    match_tol_arcsec: float = DEFAULT_TOL_ARCSEC,
    n_print: int = 5,
    verbose: bool = True,
) -> list[BCG]:
    """Enrich BCG objects with the closest spectroscopic match.

    For each BCG, finds the closest spec source within tolerance and
    copies z / sigma_z onto the BCG object.  Does NOT modify RA/Dec.

    Parameters
    ----------
    bcgs : list[BCG]
        BCG candidates (modified in-place and returned).
    redshift_df : pd.DataFrame
        Combined spectroscopic catalog.
    match_tol_arcsec : float
        Maximum separation in arcseconds.
    n_print : int
        Number of top matches to print per BCG (verbose mode).
    verbose : bool
        Print matching diagnostics.

    Returns
    -------
    list[BCG]
        Same BCG objects, now enriched with spec data where matched.
    """
    redshift_df = _standardize_redshift_df(redshift_df)

    if not bcgs or redshift_df.empty:
        if verbose:
            reason = "No BCGs" if not bcgs else "Empty redshift catalog"
            print(f"{reason} — skipping spectroscopic BCG matching.")
        return bcgs

    spec_coords = skycoord_from_df(redshift_df)
    bcg_coords = make_skycoord([b.ra for b in bcgs], [b.dec for b in bcgs])

    bcg_idx, gal_idx, sep_arcsec = match_skycoords_unique(
        bcg_coords, spec_coords, match_tol_arcsec=match_tol_arcsec,
    )
    match_map = {
        int(bi): (int(gi), float(sep))
        for bi, gi, sep in zip(bcg_idx, gal_idx, sep_arcsec)
    }

    if verbose:
        print("\n--- BCG Spectroscopic Matching ---")

    for i, bcg in enumerate(bcgs):
        if verbose:
            bcg_coord = make_skycoord([bcg.ra], [bcg.dec])
            sep_all = bcg_coord.separation(spec_coords)
            _print_top_n_matches(
                label="spectroscopic", bcg_i=bcg.bcg_id,
                ra_bcg=bcg.ra, dec_bcg=bcg.dec,
                cat_coords=spec_coords, sep2d=sep_all,
                cat_df=redshift_df, n=n_print, source_col="spec_source",
            )

        if i in match_map:
            gi, sep = match_map[i]
            row = redshift_df.iloc[gi]
            bcg.z = float(row["z"]) if pd.notna(row.get("z")) else bcg.z
            bcg.sigma_z = float(row["sigma_z"]) if pd.notna(row.get("sigma_z")) else bcg.sigma_z
            if verbose:
                z_str = f"{bcg.z:.5f}" if bcg.z is not None else "N/A"
                print(
                    f"BCG {bcg.bcg_id} matched spec source at sep={sep:.2f}\", "
                    f"z={z_str}, source={row.get('spec_source', 'N/A')}"
                )
        elif verbose:
            bcg_coord = make_skycoord([bcg.ra], [bcg.dec])
            nearest = bcg_coord.separation(spec_coords).min().arcsec
            print(
                f"BCG {bcg.bcg_id} has no spec match within {match_tol_arcsec}\". "
                f"Nearest source at {nearest:.2f}\"."
            )

    if verbose:
        print("--- End spec matching ---\n")

    return bcgs


def _match_bcgs_phot(
    bcgs: list[BCG],
    phot_df: pd.DataFrame,
    *,
    match_tol_arcsec: float = DEFAULT_TOL_ARCSEC,
    n_print: int = 5,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Match BCGs to photometry and return enriched row dicts.

    Each returned dict has BCG identity columns plus any photometry
    columns from the closest match.  These dicts are used to build
    BCGs.csv.

    Parameters
    ----------
    bcgs : list[BCG]
        BCG candidates (already enriched with spec data).
    phot_df : pd.DataFrame
        Photometric catalog.
    match_tol_arcsec : float
        Maximum separation in arcseconds.
    n_print : int
        Number of top matches to print per BCG (verbose mode).
    verbose : bool
        Print matching diagnostics.

    Returns
    -------
    list[dict[str, Any]]
        One dict per BCG, with keys: BCG_priority, RA, Dec,
        BCG_probability, z, sigma_z, plus photometry columns if matched.
    """
    phot_df = _standardize_phot_df(phot_df)

    rows: list[dict[str, Any]] = []
    for bcg in bcgs:
        rows.append({
            "BCG_priority": bcg.bcg_id,
            "RA": bcg.ra,
            "Dec": bcg.dec,
            "BCG_probability": bcg.probability,
            "z": bcg.z,
            "sigma_z": bcg.sigma_z,
        })

    if not bcgs or phot_df.empty:
        if verbose:
            reason = "No BCGs" if not bcgs else "Empty photometry catalog"
            print(f"{reason} — skipping photometric BCG matching.")
        return rows

    phot_coords = skycoord_from_df(phot_df)
    bcg_coords = make_skycoord([b.ra for b in bcgs], [b.dec for b in bcgs])

    # Nearest-neighbor match for each BCG
    nn_idx, sep_nn, _ = bcg_coords.match_to_catalog_sky(phot_coords)

    if verbose:
        print("\n--- BCG Photometric Matching ---")

    for i, bcg in enumerate(bcgs):
        if verbose:
            bcg_coord = make_skycoord([bcg.ra], [bcg.dec])
            sep_all = bcg_coord.separation(phot_coords)
            _print_top_n_matches(
                label="photometric", bcg_i=bcg.bcg_id,
                ra_bcg=bcg.ra, dec_bcg=bcg.dec,
                cat_coords=phot_coords, sep2d=sep_all,
                cat_df=phot_df, n=n_print, source_col="phot_source",
            )

        if sep_nn[i] < (match_tol_arcsec * u.arcsec):
            matched_row = phot_df.iloc[int(nn_idx[i])]
            # Add phot columns without overwriting BCG identity columns
            for col in phot_df.columns:
                if col in ("RA", "Dec", "BCG_priority", "BCG_probability"):
                    continue
                val = matched_row[col]
                # Only set if not already present or is missing
                if col not in rows[i] or rows[i][col] is None or (
                    isinstance(rows[i].get(col), float) and pd.isna(rows[i][col])
                ):
                    rows[i][col] = val

            if verbose:
                sep_as = sep_nn[i].arcsec
                print(
                    f"BCG {bcg.bcg_id} matched phot source at sep={sep_as:.2f}\", "
                    f"source={matched_row.get('phot_source', 'N/A')}"
                )
        elif verbose:
            print(
                f"BCG {bcg.bcg_id} has no phot match within {match_tol_arcsec}\". "
                f"Nearest at {sep_nn[i].arcsec:.2f}\"."
            )

    if verbose:
        print("--- End phot matching ---\n")

    return rows


# ---------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------

def run_matching(
    cluster: Cluster,
    *,
    archival_df: pd.DataFrame | None = None,
    deimos_df: pd.DataFrame | None = None,
    manual_df: pd.DataFrame | None = None,
    phot_dfs: dict[str, pd.DataFrame] | None = None,
    match_tol_arcsec: float = DEFAULT_TOL_ARCSEC,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], list[BCG]]:
    """Run the full Stage 3 matching pipeline.

    Parameters
    ----------
    cluster : Cluster
        Cluster object with paths and coordinates.
    archival_df : pd.DataFrame, optional
        Archival spectroscopic catalog.  If *None*, loads from
        ``Redshifts/archival_z.csv``.
    deimos_df : pd.DataFrame, optional
        DEIMOS spectroscopic catalog.  If *None*, loads from
        ``Redshifts/deimos.csv``.
    manual_df : pd.DataFrame, optional
        Manual redshift catalog.  If *None*, auto-discovers
        ``Redshifts/manual_*.csv`` files.
    phot_dfs : dict[str, pd.DataFrame], optional
        Per-survey photometry catalogs, keyed by survey name
        (e.g. ``{"panstarrs": df, "legacy": df}``).  If *None*, loads
        from ``Photometry/photometry_{survey}.csv`` for all CSVs found.
    match_tol_arcsec : float
        Positional matching tolerance in arcseconds.
    verbose : bool
        Print progress and diagnostics.

    Returns
    -------
    combined_df : pd.DataFrame
        Merged spectroscopic catalog (DEIMOS + archival + manual, deduplicated).
    matched_dfs : dict[str, pd.DataFrame]
        Per-survey crossmatched catalogs (only true spec+phot matches).
    bcgs : list[BCG]
        BCG candidate objects enriched with spec and phot data.

    Notes
    -----
    Deduplication priority: DEIMOS (highest) > archival > manual (lowest).

    Side effects (file writes):
      - ``Redshifts/combined_redshifts.csv``
      - ``Photometry/{survey}_matched.csv`` for each survey
      - ``BCGs.csv``
    """
    # ----------------------------------------------------------
    # 1. Load inputs if not provided
    # ----------------------------------------------------------
    if archival_df is None:
        archival_path = os.path.join(cluster.redshift_path, "archival_z.csv")
        if os.path.isfile(archival_path):
            archival_df = pd.read_csv(archival_path)
            if verbose:
                print(f"Loaded {len(archival_df)} archival redshifts from disk.")
        else:
            archival_df = pd.DataFrame(columns=COLUMNS_SPEC)
            if verbose:
                print(f"No archival_z.csv found at {archival_path}.")

    if deimos_df is None:
        deimos_path = os.path.join(cluster.redshift_path, "deimos.csv")
        if os.path.isfile(deimos_path):
            deimos_df = pd.read_csv(deimos_path)
            if verbose:
                print(f"Loaded {len(deimos_df)} DEIMOS redshifts from disk.")
        else:
            deimos_df = pd.DataFrame(columns=COLUMNS_SPEC)
            if verbose:
                print(f"No deimos.csv found at {deimos_path}.")

    if manual_df is None:
        manual_files = sorted(glob.glob(
            os.path.join(cluster.redshift_path, "manual_*.csv")
        ))
        if manual_files:
            manual_dfs = [pd.read_csv(f) for f in manual_files]
            manual_df = pd.concat(manual_dfs, ignore_index=True)
            if verbose:
                for f in manual_files:
                    print(f"Loaded manual redshifts from {os.path.basename(f)}")
        else:
            manual_df = pd.DataFrame(columns=COLUMNS_SPEC)

    if phot_dfs is None:
        phot_dfs = _discover_phot_files(cluster, verbose=verbose)

    # ----------------------------------------------------------
    # 2. Merge spectra: DEIMOS highest, then archival, then manual
    # ----------------------------------------------------------
    combined_df = _merge_spectra(
        archival_df, deimos_df,
        match_tol_arcsec=match_tol_arcsec, verbose=verbose,
    )

    if not manual_df.empty:
        combined_df = _append_manual_spectra(
            combined_df, manual_df,
            match_tol_arcsec=match_tol_arcsec, verbose=verbose,
        )

    combined_path = os.path.join(cluster.redshift_path, "combined_redshifts.csv")
    combined_df.to_csv(combined_path, index=False)
    if verbose:
        print(f"Wrote combined_redshifts.csv ({len(combined_df)} sources).\n")

    # ----------------------------------------------------------
    # 3. Crossmatch with each photometry survey
    # ----------------------------------------------------------
    matched_dfs: dict[str, pd.DataFrame] = {}
    for survey, phot_df in phot_dfs.items():
        matched = crossmatch_photometry_with_redshifts(
            phot_df, combined_df, match_tol_arcsec=match_tol_arcsec,
        )
        matched_dfs[survey] = matched

        out_path = os.path.join(cluster.photometry_path, f"{survey}_matched.csv")
        matched.to_csv(out_path, index=False)
        if verbose:
            print(f"{survey}: {len(matched)} spec+phot matches -> {out_path}")

    if verbose:
        print()

    # ----------------------------------------------------------
    # 4. BCG identification and matching
    # ----------------------------------------------------------
    bcgs = get_BCGs(cluster, verbose=verbose)

    # Enrich with spec data
    _match_bcgs_spec(
        bcgs, combined_df,
        match_tol_arcsec=match_tol_arcsec, verbose=verbose,
    )

    # Enrich with phot data and build BCGs.csv rows
    # Use the preferred survey photometry for BCG matching.
    # Prefer the cluster's configured survey; fall back to first available.
    preferred_survey = cluster.survey.lower() if cluster.survey else "legacy"
    bcg_phot_df = _pick_bcg_phot(phot_dfs, preferred_survey)

    bcg_rows = _match_bcgs_phot(
        bcgs, bcg_phot_df,
        match_tol_arcsec=match_tol_arcsec, verbose=verbose,
    )

    # Write BCGs.csv
    bcg_out_df = pd.DataFrame(bcg_rows)
    bcg_out_df.to_csv(cluster.bcg_file, index=False)
    if verbose:
        print(f"Wrote BCGs.csv ({len(bcg_out_df)} BCGs) -> {cluster.bcg_file}\n")

    return combined_df, matched_dfs, bcgs


# ---------------------------------------------------------------
# Private helpers for run_matching
# ---------------------------------------------------------------

def _discover_phot_files(
    cluster: Cluster,
    *,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """Auto-discover photometry CSVs in the Photometry directory.

    Looks for files matching ``photometry_{survey}.csv``.

    Parameters
    ----------
    cluster : Cluster
        Cluster object with ``photometry_path``.
    verbose : bool
        Print which files were found.

    Returns
    -------
    dict[str, pd.DataFrame]
        Survey name -> photometry DataFrame.
    """
    import glob

    pattern = os.path.join(cluster.photometry_path, "photometry_*.csv")
    phot_dfs: dict[str, pd.DataFrame] = {}

    for fpath in sorted(glob.glob(pattern)):
        fname = os.path.basename(fpath)
        # Extract survey name from "photometry_{survey}.csv"
        survey = fname.replace("photometry_", "").replace(".csv", "").lower()
        try:
            df = pd.read_csv(fpath)
            phot_dfs[survey] = df
            if verbose:
                print(f"Found photometry: {fname} ({len(df)} sources)")
        except Exception as e:
            if verbose:
                print(f"Warning: could not read {fpath}: {e}")

    if not phot_dfs and verbose:
        print(f"No photometry_*.csv files found in {cluster.photometry_path}.")

    return phot_dfs


def _pick_bcg_phot(
    phot_dfs: dict[str, pd.DataFrame],
    preferred_survey: str,
) -> pd.DataFrame:
    """Select the photometry DataFrame to use for BCG matching.

    Parameters
    ----------
    phot_dfs : dict[str, pd.DataFrame]
        Available photometry catalogs keyed by survey name.
    preferred_survey : str
        Preferred survey name (lowercase).

    Returns
    -------
    pd.DataFrame
        The selected photometry DataFrame, or an empty DataFrame if none.
    """
    if preferred_survey in phot_dfs:
        return phot_dfs[preferred_survey]

    # Fall back to first available
    if phot_dfs:
        first_key = next(iter(phot_dfs))
        return phot_dfs[first_key]

    return pd.DataFrame(columns=["RA", "Dec"])
