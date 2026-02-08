#!/usr/bin/env python3
"""
archival_phot_pipeline.py

Archival Photometry Pipeline
---------------------------------------------------------

This script queries archival photometric data around a given cluster from:
  - PanSTARRS (via Visier)
  - Legacy DR10 (via NOIRLab TAP and Q3C_RADIAL_QUERY)

It calculates g-r, g-i, and r-i colors and luminosity weighting
in each filter. The outputs are then:
  - Individual CSVs for each catalog (Legacy, PanSTARRS)
  - A merged catalog, with no removal of double-entries.

Requirements:
    - astropy
    - astroquery
    - numpy
    - pandas
    - cluster.py (local)
    - my_utils.py (local)

Usage:
    python archival_phot_pipeline.py CLUSTER_ID [options]

Example:
    python archival_phot_pipeline.py "Abell 2355" -r 15.0 --get-legacy True --get-panstarrs False

Options:
    -r, --radius    <float>    Search radius in arcminutes        [default: 10.0]
    -p, --path      <str>      Output directory path              [default: cluster BASE_PATH]
    --get-legacy    <bool>     Build Legacy catalog from query    [default: true]
    --get-panstarrs <bool>     Build PanSTARRS catalog from query [default: true]

Notes:
    "CLUSTER_ID" positional arg can be input as a cluster identifier mapped to a cluster name (eg. '1219' or '0881900301') or the
    full RMJ/Abell name (possibly others).
"""
from __future__ import annotations

import os
import argparse

import pandas as pd
import numpy as np

from astropy.coordinates import SkyCoord
import astropy.units as u

from astroquery.vizier import Vizier
from astroquery.utils.tap.core import TapPlus

from cluster import Cluster
from my_utils import str2bool, coerce_to_numeric


# ------------------------------------
# Defaults/ Constants
# ------------------------------------
DEFAULT_RADIUS_ARCMIN = 10.0
DEFAULT_LEGACY_STEP = 2000
DEFAULT_LEGACY_MAX_ROWS = 150000

PANSTARRS_CATALOG = "II/349/ps1"
LEGACY_CATALOG = "ls_dr10.tractor"
LEGACY_URL = "https://datalab.noirlab.edu/tap"

RA_COL = 'RA'
DEC_COL = 'Dec'
SOURCE_COL = 'phot_source'
MAG_COLS = ('gmag', 'rmag', 'imag')
LUM_WEIGHT_COLS = ('lum_weight_g', 'lum_weight_r', 'lum_weight_i')
COLOR_COLS = ('g_r', 'r_i', 'g_i')

BASE_COLS = (RA_COL, DEC_COL, *MAG_COLS, SOURCE_COL)
DERIVED_COLS = (*LUM_WEIGHT_COLS, *COLOR_COLS)
ALL_COLS = (*BASE_COLS, *DERIVED_COLS)

PANSTARRS_OUTFILE = "photometry_PanSTARRS.csv"
LEGACY_OUTFILE = "photometry_legacy.csv"
COMBINED_OUTFILE = "photometry_combined.csv"


# ------------------------------------
# Helpers
# ------------------------------------

def _build_phot_df() -> pd.DataFrame:
    """
    Builds an empty photometry DataFrame with required columns.

    Returns
    -------
    df : pd.DataFrame
        Empty DataFrame with columns for RA, Dec, magnitudes.
    """
    columns = list(ALL_COLS)
    return pd.DataFrame(columns=columns)

def _standardize_phot_df(
        df: pd.DataFrame,
        *,
        source: str | None = None,
) -> pd.DataFrame:
    """
    Standardizes a photometry DataFrame to have required columns and source.

    Parameters
    ----------
    df : pd.DataFrame
        Input photometry DataFrame.
    source : str | None
        Source catalog name ('legacy' or 'panstarrs').

    Returns
    -------
    output : pd.DataFrame
        Standardized DataFrame with required columns and source.
    """
    out = df.copy()

    # Ensure required columns exist
    for col in ALL_COLS:
        if col not in out.columns:
            out[col] = np.nan

    # Ensure numeric types, drop missing coords
    out = coerce_to_numeric(out, (RA_COL, DEC_COL, *MAG_COLS))
    out = out.dropna(subset=[RA_COL, DEC_COL]).reset_index(drop=True)

    # Set source column
    if source is not None:
        out[SOURCE_COL] = source

    # Ensure stable order
    output = out.loc[:, list(ALL_COLS)]

    return output


def _open_or_build_phot_df(file_path: str) -> pd.DataFrame:
    """
    Opens a photometry CSV if it exists, else builds an empty DataFrame.

    Parameters
    ----------
    file_path : str
        Path to the photometry CSV file.

    Returns
    -------
    df : pd.DataFrame
        Photometry DataFrame.
    """
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df = _standardize_phot_df(df)
    else:
        df = _build_phot_df()
    return df


def _save_photometry(
        df: pd.DataFrame,
        out_file: str,
    ) -> None:
    """
    Saves the photometry DataFrame to a CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        Photometry DataFrame to save.
    out_file : str
        Output file path.
    """
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    df = _standardize_phot_df(df)
    df.to_csv(out_file, index=False)
    print(f"Photometry saved to {out_file} ({len(df)} rows)")


# ------------------------------------
# Query Functions
# ------------------------------------

def query_panstarrs_phot(
        coord: SkyCoord,
        radius_arcmin: float,
    ) -> pd.DataFrame:
    """
    Queries Pan-STARRS photometry around a given SkyCoord.

    Parameters
    ----------
    coord : SkyCoord
        Center coordinates for the query.
    radius_arcmin : float
        Radius of the query region in arcminutes.

    Returns
    -------
    panstarrs_df : pd.DataFrame
        DataFrame containing Pan-STARRS photometry with RA, Dec, gmag, rmag, imag.
    """
    print("Querying Pan-STARRS via Vizier...")

    vizier = Vizier(columns=['RAJ2000', 'DEJ2000', 'gmag', 'rmag', 'imag'],
                    column_filters={}, row_limit=-1)
    
    try:
        result = vizier.query_region(
            coord,
            radius=radius_arcmin * u.arcmin,
            catalog=PANSTARRS_CATALOG,
        )
    except Exception as e:
        print(f"Pan-STARRS query failed: {e}")
        return _build_phot_df()
    
    if not result:
        print("No Pan-STARRS results found.")
        return _build_phot_df()
    
    panstarrs_df = result[0].to_pandas().rename(columns={'RAJ2000': RA_COL, 'DEJ2000': DEC_COL})
    panstarrs_df = _standardize_phot_df(panstarrs_df, source='panstarrs')

    return panstarrs_df


def query_legacy_phot(
        coord: SkyCoord,
        radius_arcmin: float,
        *,
        step: int = DEFAULT_LEGACY_STEP,
        max_rows: int = DEFAULT_LEGACY_MAX_ROWS,
    ) -> pd.DataFrame:
    """
    Queries Legacy DR10 photometry around a given SkyCoord.

    Parameters
    ----------
    coord : SkyCoord
        Center coordinates for the query.
    radius_arcmin : float
        Radius of the query region in arcminutes.
    step : int
        Step size for paginated query. [default: DEFAULT_LEGACY_STEP]
    max_rows : int
        Max total rows for query. [default: DEFAULT_LEGACY_MAX_ROWS]

    Returns
    -------
    legacy_df : pd.DataFrame
        DataFrame containing Legacy DR10 photometry with RA, Dec, gmag, rmag, imag.
    """
    print("Querying Legacy DR10 via NOIRLab TAP...")

    tap = TapPlus(url=LEGACY_URL)
    ra, dec = float(coord.ra.deg), float(coord.dec.deg)
    radius_deg = float(radius_arcmin) / 60.0

    chunks: list[pd.DataFrame] = []

    for offset in range(0, int(max_rows), int(step)):
        print(f"Querying rows {offset + 1} to {offset + step}...")

        query = f"""
        SELECT ra, dec, mag_g, mag_r, mag_i
        FROM {LEGACY_CATALOG}
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
            print(f"Legacy query failed at offset {offset}: {e}")
            break
        if chunk.empty:
            print("No more results.")
            break
        chunks.append(chunk)

    if not chunks:
        print("No Legacy DR10 results found.")
        return _build_phot_df()
    

    legacy_df = pd.concat(chunks, ignore_index=True)
    legacy_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    legacy_df = legacy_df.rename(
        columns={
            'ra': RA_COL,
            'dec': DEC_COL,
            'mag_g': 'gmag',
            'mag_r': 'rmag',
            'mag_i': 'imag',
        }
    )
    legacy_df = _standardize_phot_df(legacy_df, source='legacy')

    return legacy_df


# ------------------------------------
# Derived Columns Functions
# ------------------------------------

def add_luminosity_weights(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes luminosity weights for each photometric band (g, r, i) and adds them as columns to the DataFrame.

    For each magnitude column present ('gmag', 'rmag', 'imag'), computes:
        lum_weight_band = 10 ** (-0.4 * mag_band)
    The output columns are 'lum_weight_g', 'lum_weight_r', and 'lum_weight_i'.
    If a required magnitude column is missing, the luminosity weight for that band is set to NaN.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing photometric data. Should include one or more of:
        'gmag', 'rmag', 'imag' (magnitudes in g, r, i bands).

    Returns
    -------
    out : pd.DataFrame
        Modified DataFrame with new columns:
        'lum_weight_g', 'lum_weight_r', 'lum_weight_i'

    Notes
    -----
    - Assumes AB magnitude system.
    - Output columns are always present, with NaN for missing bands.
    - The luminosity weights are unnormalized and proportional to flux in each band.
    """
    out = df.copy()

    for mag, weight in zip(MAG_COLS, LUM_WEIGHT_COLS):
        if mag in out.columns:
            out[weight] = 10 ** (-0.4 * out[mag])
        else:
            out[weight] = np.nan
   
    return out


def add_colors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add color columns 'g_r','r_i', and 'g-i' to a photometric DataFrame.
    Missing columns are assigned NaN values.

    Parameters
    ----------
    df : pd.DataFrame
        Photometric catalog with gmag, rmag, imag.

    Returns
    -------
    out : pd.DataFrame
        Modified DataFrame with 'g_r' and 'r_i' columns.
    """
    out = df.copy()

    out['g_r'] = out['gmag'] - out['rmag']
    out['r_i'] = out['rmag'] - out['imag']
    out['g_i'] = out['gmag'] - out['imag']

    return out


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds derived photometric columns to the DataFrame:
      - Luminosity weights in g, r, i bands
      - Colors: g-r, r-i, g-i

    Parameters
    ----------
    df : pd.DataFrame
        Input photometric DataFrame.

    Returns
    -------
    out : pd.DataFrame
        Modified DataFrame with added derived columns.
    """
    out = df.copy()
    out = coerce_to_numeric(out, list(MAG_COLS))

    out = add_luminosity_weights(out)
    out = add_colors(out)
    return out


def combine_photometry(
        legacy_df: pd.DataFrame,
        pan_df: pd.DataFrame,
    ) -> pd.DataFrame:
    """
    Combine Legacy and Pan-STARRS photometry into a single CSV, keeping all entries.
    Adds a 'source' column to indicate catalog provenance.

    Parameters
    ----------
    legacy_df : pd.DataFrame
        Legacy photometry DataFrame.
    pan_df : pd.DataFrame
        Pan-STARRS photometry DataFrame.

    Returns
    -------
    combined_df : pd.DataFrame
        Concatenated DataFrame with all required columns.
    """
    
    # Standardize both DataFrames
    legacy = _standardize_phot_df(legacy_df)
    pan = _standardize_phot_df(pan_df)

    combined = pd.concat([legacy, pan], ignore_index=True)

    return combined


# ------------------------------------
# Main Pipeline Function
# ------------------------------------

def run_photometry_pipeline(
        cluster: Cluster,
        *,
        radius_arcmin: float = DEFAULT_RADIUS_ARCMIN,
        retrieve_legacy: bool = True,
        retrieve_panstarrs: bool = True,
    ) -> None:
    """
    Pipeline to query Pan-STARRS and Legacy photometry, and output individual and combined catalogs.

    Parameters
    ----------
    cluster : Cluster
        Cluster object containing all necessary metadata (paths, coordinates, etc.).
    radius_arcmin : float
        Search radius in arcminutes.
    retrieve_legacy : bool
        If True, query Legacy, else load existing CSV.
    retrieve_panstarrs : bool
        If True, query Pan-STARRS, else load existing CSV.
    """
    os.makedirs(cluster.photometry_path, exist_ok=True)
    panstarrs_path = os.path.join(cluster.photometry_path, PANSTARRS_OUTFILE)
    legacy_path = os.path.join(cluster.photometry_path, LEGACY_OUTFILE)
    combined_path = os.path.join(cluster.photometry_path, COMBINED_OUTFILE)

    coord = cluster.coords
    print(f"Using coordinates: RA = {coord.ra.deg:.5f}, Dec = {coord.dec.deg:.5f}")

    # Queries
    if retrieve_panstarrs:
        pan_df = query_panstarrs_phot(coord, radius_arcmin)
    else:
        pan_df = _open_or_build_phot_df(panstarrs_path)
        print(f"Pan-STARRS photometry loaded from {panstarrs_path} ({len(pan_df)} rows)")

    if retrieve_legacy:
        legacy_df = query_legacy_phot(coord, radius_arcmin)
    else:
        legacy_df = _open_or_build_phot_df(legacy_path)
        print(f"Legacy DR10 photometry loaded from {legacy_path} ({len(legacy_df)} rows)")

    # Compute color and luminosity columns
    pan_df = add_derived_columns(pan_df)
    legacy_df = add_derived_columns(legacy_df)

    # Save individual catalogs
    if retrieve_panstarrs:
        _save_photometry(pan_df, panstarrs_path)
    if retrieve_legacy:
        _save_photometry(legacy_df, legacy_path)

    # Combine and save
    combined_df = combine_photometry(legacy_df, pan_df)
    _save_photometry(combined_df, combined_path)


def main():
    parser = argparse.ArgumentParser(description="Run archival photometry query for a galaxy cluster.")
    parser.add_argument(
        "cluster_id",
        type=str,
        help="Cluster ID (e.g. '1327') or full RMJ/Abell name."
    )
    parser.add_argument(
        '-r',
        '--radius',
        type=float,
        default=DEFAULT_RADIUS_ARCMIN,
        help=f"Search radius in arcminutes (default: {DEFAULT_RADIUS_ARCMIN})"
    )
    parser.add_argument(
        '-p',
        '--path',
        type=str,
        default=None,
        help="Base path for cluster science directories (default: cluster.py BASE_PATH)"
    )
    parser.add_argument(
        '--get-legacy',
        type=str2bool,
        nargs='?',
        const=True,
        default=True,
        help="Whether to build Legacy catalog from query. (default: True). Accepts true/false, yes/no, y/n, t/f, 1/0."
    )
    parser.add_argument(
        '--get-panstarrs',
        type=str2bool,
        nargs='?',
        const=True,
        default=True,
        help="Build Pan-STARRS catalog from query (default: True). Accepts true/false, yes/no, y/n, t/f, 1/0."
    )
    args = parser.parse_args()

    cluster = Cluster(args.cluster_id, base_path=args.path)
    cluster.populate()

    run_photometry_pipeline(
        cluster=cluster,
        radius_arcmin=args.radius,
        retrieve_legacy=args.get_legacy,
        retrieve_panstarrs=args.get_panstarrs
    )


if __name__ == "__main__":
    main()
