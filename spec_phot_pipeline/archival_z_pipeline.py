#!/usr/bin/env python3
"""
archival_z_pipeline.py

Archival Redshift Pipeline
---------------------------------------------------------

This script queries archival redshift data around a given cluster from:
  - NED (via astroquery.ned)
  - SDSS DR18 (via CasJobs REST API)
  - DESI DR1 (via NOIRLab TAP and Q3C_RADIAL_QUERY)

It deduplicates overlapping results and outputs:
  - Individual CSVs for each catalog (NED, SDSS, DESI)
  - A merged catalog (archival_z.txt and archival_z.csv)

Requirements:
    - astropy
    - numpy
    - pandas
    - requests
    - astroquery
    - cluster.py (local)

Usage:
    python archival_z_pipeline.py CLUSTER_ID [options]

Options:
    -r, --radius    <float>    Search radius in arcminutes      [default: 10.0]
    -t, --tolerance <float>    Duplicate match tolerance in deg [default: 1/3600.0]
    -p, --path      <str>      Output directory path            [default: cluster BASE_PATH]
    -u, --user      <str>      CasJobs username                 [default: prompt or env CASJOBS_USER]
    --password      <str>      CasJobs password                 [default: prompt or env CASJOBS_PW]

Notes:
    "CLUSTER_ID" positional arg can be input as a cluster identifier mapped to a cluster name (eg. '1219' or '0881900301') or the
    full RMJ/Abell name (possibly others).
"""
from __future__ import annotations

import os
import argparse
import time
from getpass import getpass
from io import StringIO

import pandas as pd
import numpy as np
import requests

from astropy.coordinates import SkyCoord
import astropy.units as u

from astroquery.ipac.ned import Ned
from astroquery.ipac.ned import Conf as NedConf
from astroquery.utils.tap.core import TapPlus

from cluster import Cluster


# ------------------------------------
# Defaults/ Constants
# ------------------------------------
DEFAULT_RADIUS_ARCMIN = 10.0
DEFAULT_TOLERANCE_DEG = 1.0 / 3600.0  # 1 arcsec
DEFAULT_MAX_RETRIES = 5
DEFAULT_INITIAL_WAIT = 5  # seconds
DEFAULT_TIMEOUT = 300  # seconds
DEFAULT_SDSS_TOP_N = 100

SDSS_URL = "https://skyserver.sdss.org/CasJobs/RestApi/contexts/DR18/query"
DESI_URL = "https://datalab.noirlab.edu/tap"

DF_COLUMNS = ['RA', 'Dec', 'z', 'zerr']


# ------------------------------------
# Query Functions
# ------------------------------------
def query_ned(
        coord: SkyCoord,
        radius_arcmin: float,
        *,
        max_retries: int = DEFAULT_MAX_RETRIES,
        initial_wait: int = DEFAULT_INITIAL_WAIT,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> pd.DataFrame:
    """
    Queries NED for SLS-type spectroscopic redshifts around a coordinate.

    Parameters
    ----------
    coord : SkyCoord
        Central search coordinate.
    radius_arcmin : float
        Radius in arcminutes.
    max_retries : int, optional
        Maximum number of retries on failure. [default: DEFAULT_MAX_RETRIES]
    initial_wait : int, optional
        Initial wait time between retries in seconds. [default: DEFAULT_INITIAL_WAIT]

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: RA, Dec, z, zerr
    """
    # TODO: Get references, ned_table = Ned.get_table(name, table='references'), but you get
    # multiple entries per query and would need to query each galaxy. 

    NedConf.timeout = timeout
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Querying NED (attempt {attempt}/{max_retries}) for RA={coord.ra.deg}, Dec={coord.dec.deg}...")
            result = Ned.query_region(coord, radius=radius_arcmin * u.arcmin)
            if result is None or len(result) == 0:
                print("NED query returned no results.")
                return pd.DataFrame(columns=DF_COLUMNS)
    
            filtered = result[result['Redshift Flag'] == 'SLS'].copy()
            if len(filtered) == 0:
                print("NED query returned no SLS-type redshifts.")
                return pd.DataFrame(columns=DF_COLUMNS)
            
            ra = pd.to_numeric(filtered['RA'], errors='coerce')
            dec = pd.to_numeric(filtered['DEC'], errors='coerce')
            z = pd.to_numeric(filtered['Redshift'], errors='coerce')

            df = pd.DataFrame({
                'RA': ra,
                'Dec': dec,
                'z': z,
                'zerr': 0.0,  # NED does not provide redshift errors
            })
            df = df.dropna(subset=['RA', 'Dec', 'z']).reset_index(drop=True)
            return df

        except Exception as e:
            print(f"Attempt {attempt} failed: {e}")
            if attempt == max_retries:
                print("All attempts failed. Raising exception.")
                raise
            wait_time = initial_wait * (2 ** (attempt - 1))  # Exponential backoff: 5s, 10s, 20s, ...
            print(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)


def _sdss_ra_wrap(
        ra_deg: float,
        radius_deg: float
    ) -> str:
    """
    Helper function to handle RA wrapping for SDSS queries.

    Parameters
    ----------
    ra_deg : float
        Right ascension in degrees.
    radius_deg : float
        Search radius in degrees.

    Returns
    -------
    str
        SQL WHERE clause for RA with wrapping handled.

    """
    ra_min = ra_deg - radius_deg
    ra_max = ra_deg + radius_deg

    if 0.0 <= ra_min and ra_max <= 360.0:
        return f"s.ra BETWEEN {ra_min} AND {ra_max}"

    # Wrap case
    ra_min_wrapped = ra_min % 360.0
    ra_max_wrapped = ra_max % 360.0
    # Example: ra=0.2, radius=1 -> ra_min=-0.8->359.2, ra_max=1.2
    # Clause should be (ra >= 359.2 OR ra <= 1.2)
    return f"(s.ra >= {ra_min_wrapped} OR s.ra <= {ra_max_wrapped})"


def query_sdss(
        coord: SkyCoord,
        radius_arcmin: float,
        *,
        user: str,
        password: str,
        top_n: int = DEFAULT_SDSS_TOP_N,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> pd.DataFrame:
    """
    Queries SDSS DR18 using CasJobs REST API.

    Parameters
    ----------
    coord : SkyCoord
        Central search coordinate.
    radius_arcmin : float
        Radius in arcminutes.
    user : str
        CasJobs username.
    password : str
        CasJobs password.
    top_n : int, optional
        Number of top results to return. [default: DEFAULT_SDSS_TOP_N]
    timeout : int, optional
        Request timeout in seconds. [default: DEFAULT_TIMEOUT]

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: RA, Dec, z, zerr
    """
    ra_deg = float(coord.ra.deg)
    dec_deg = float(coord.dec.deg)
    radius_deg = float(radius_arcmin) / 60.0

    ra_clause = _sdss_ra_wrap(ra_deg, radius_deg)
    dec_min = dec_deg - radius_deg
    dec_max = dec_deg + radius_deg

    sql = f"""
    SELECT TOP {int(top_n)}
        s.ra, s.dec, s.z, s.zerr
    FROM dr18..specObjAll AS s
    WHERE {ra_clause}
    AND s.dec BETWEEN {dec_min} AND {dec_max}
    AND s.z IS NOT NULL
    """

    url = SDSS_URL
    headers = {"Accept": "text/plain", "Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "query": sql,
        "taskname": "quick",
        "userid": user,
        "password": password,
        "format": "csv"
    }

    print(f"Querying SDSS DR18 for RA={ra_deg:.5f}, Dec={dec_deg:.5f}...")

    try:
        response = requests.post(url, headers=headers, data=data, timeout=timeout)
        response.raise_for_status()

        # CasJobs returns CSV text directly
        df = pd.read_csv(StringIO(response.text))
        if df.empty:
            print("SDSS query returned no results.")
            return pd.DataFrame(columns=DF_COLUMNS)
        
        df['ra'] = pd.to_numeric(df['ra'], errors="coerce")
        df['dec'] = pd.to_numeric(df['dec'], errors="coerce")
        df['z'] = pd.to_numeric(df['z'], errors="coerce")
        df['zerr'] = pd.to_numeric(df['zerr'], errors="coerce")

        df = df.dropna(subset=['ra', 'dec', 'z']).reset_index(drop=True)
        
        coords = SkyCoord(df['ra'].to_numpy(), df['dec'].to_numpy(), unit='deg')
        df = df[coords.separation(coord).arcmin < radius_arcmin]
        df = df.rename(columns={'ra': 'RA', 'dec': 'Dec'})

        return df[DF_COLUMNS].reset_index(drop=True)
    
    except Exception as e:
        print(f"SDSS query failed: {e}")
        return pd.DataFrame(columns=DF_COLUMNS)


def query_desi_redshifts(
        coord: SkyCoord,
        radius_arcmin: float
    ) -> pd.DataFrame:
    """
    Queries DESI DR1 using NOIRLab TAP service and Q3C_RADIAL_QUERY.

    Parameters
    ----------
    coord : SkyCoord
        Central search coordinate.
    radius_arcmin : float
        Radius in arcminutes.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: RA, Dec, z, zerr
    """
    ra_deg = float(coord.ra.deg)
    dec_deg = float(coord.dec.deg)
    radius_deg = radius_arcmin / 60.0

    print(f"Querying DESI DR1 for RA={ra_deg:.5f}, Dec={dec_deg:.5f}...")

    tap = TapPlus(url=DESI_URL)

    query = f"""
    SELECT mean_fiber_ra, mean_fiber_dec, z, zerr
    FROM desi_dr1.zpix
    WHERE 't' = Q3C_RADIAL_QUERY(mean_fiber_ra, mean_fiber_dec, {ra_deg}, {dec_deg}, {radius_deg})
    AND spectype = 'GALAXY'
    AND z IS NOT NULL
    """
    try:
        job = tap.launch_job(query)
        df = job.get_results().to_pandas()
        if df.empty:
            print("DESI query returned no results.")
            return pd.DataFrame(columns=DF_COLUMNS)
        
        df = df.rename(columns={'mean_fiber_ra': 'RA', 'mean_fiber_dec': 'Dec'})
        df['RA'] = pd.to_numeric(df['RA'], errors='coerce')
        df['Dec'] = pd.to_numeric(df['Dec'], errors='coerce')
        df['z'] = pd.to_numeric(df['z'], errors='coerce')
        df['zerr'] = pd.to_numeric(df['zerr'], errors='coerce')
        df = df.dropna(subset=['RA', 'Dec', 'z']).reset_index(drop=True)

        return df[DF_COLUMNS].reset_index(drop=True)
    
    except Exception as e:
        print(f"DESI query failed: {e}")
        return pd.DataFrame(columns=DF_COLUMNS)


# ------------------------------------
# Deduplication Functions
# ------------------------------------
def filter_duplicates(
        df: pd.DataFrame,
        ref_coords: SkyCoord | None,
        tol_deg: float,
    ) -> pd.DataFrame:
    """
    Filters a DataFrame by removing rows within tol_deg of any ref_coords.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with RA and Dec columns.
    ref_coords : SkyCoord | None
        Reference catalog as SkyCoord.
    tol_deg : float
        Angular tolerance in degrees.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with duplicates removed.
    """
    # If nothing to compare against, just return the input unchanged
    if df.empty:
        return df.reset_index(drop=True)
    
    if ref_coords is None or len(ref_coords) == 0:
        return df.reset_index(drop=True)

    missing = {"RA", "Dec"} - set(df.columns)
    if missing:
        raise KeyError(f"DataFrame missing required columns: {sorted(missing)}")

    df_coords = SkyCoord(df['RA'].to_numpy(), df['Dec'].to_numpy(), unit='deg')
    _, sep2d, _ = df_coords.match_to_catalog_sky(ref_coords)
    keep_mask = sep2d > (tol_deg * u.deg)
    return df[keep_mask].reset_index(drop=True)


def deduplicate_self(
        df: pd.DataFrame,
        tol_deg: float,
    ) -> pd.DataFrame:
    """
    Removes near-duplicate rows within a DataFrame using SkyCoord separations.
    Keeps the first instance of each group of duplicates.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'RA' and 'Dec' columns.
    tol_deg : float
        Angular separation (degrees) below which objects are considered duplicates.

    Returns
    -------
    pd.DataFrame
        Deduplicated DataFrame.
    """
    if df.empty or len(df) == 1:
        return df.copy()

    missing = {'RA', 'Dec'} - set(df.columns)
    if missing:
        raise KeyError(f"DataFrame missing required columns: {sorted(missing)}")
    
    coords = SkyCoord(df['RA'].to_numpy(), df['Dec'].to_numpy(), unit='deg')
    keep = np.ones(len(df), dtype=bool)

    for i in range(len(df)-1):
        if not keep[i]:
            continue  # already marked as duplicate
        sep = coords[i].separation(coords[i+1:]).deg
        # Mark any objects within tol_deg as duplicates (except the first)
        dup_indices = np.where(sep < tol_deg)[0] + (i + 1)
        keep[dup_indices] = False

    return df[keep].reset_index(drop=True)


# ------------------------------------
# Main Pipeline Function
# ------------------------------------
def run_redshift_pipeline(
        cluster: Cluster,
        user: str,
        password: str,
        *,
        radius_arcmin: float = DEFAULT_RADIUS_ARCMIN,
        tol_deg: float = DEFAULT_TOLERANCE_DEG,
    ) -> None:
    """
    Main pipeline to query redshifts from NED, SDSS, and DESI, remove duplicates,
    and output a combined catalog with priority: DESI > SDSS > NED.

    Parameters
    ----------
    cluster : Cluster
        Cluster object containing all necessary cluster metadata (paths, coordinates, etc.).
    user : str
        CasJobs username.
    password : str
        CasJobs password.
    radius_arcmin : float [default: DEFAULT_RADIUS_ARCMIN]
        Search radius in arcminutes.
    tol_deg : float [default: DEFAULT_TOLERANCE_DEG]
        Angular tolerance for deduplication (degrees).
    """
    os.makedirs(cluster.cluster_path, exist_ok=True)
    os.makedirs(cluster.redshift_path, exist_ok=True)

    output_ned = os.path.join(cluster.redshift_path, "ned.csv")
    output_sdss = os.path.join(cluster.redshift_path, "sdss.csv")
    output_desi = os.path.join(cluster.redshift_path, "desi.csv")
    output_combined_txt = os.path.join(cluster.redshift_path, "archival_z.txt")
    output_combined_csv = os.path.join(cluster.redshift_path, "archival_z.csv")

    coord = cluster.coords
    print(f"Using coordinates: RA = {coord.ra.deg:.5f}, Dec = {coord.dec.deg:.5f}")
    print(f"Search radius: {radius_arcmin} arcmin")
    print(f"Deduplication tolerance: {tol_deg} deg")

    # --- Query catalogs ---
    ned_df = query_ned(coord, radius_arcmin)
    ned_df['spec_source'] = 'NED'
    ned_df.to_csv(output_ned, index=False)
    print(f"NED: {len(ned_df)} objects saved to {output_ned}")

    sdss_df = query_sdss(coord, radius_arcmin, user=user, password=password)
    sdss_df['spec_source'] = 'SDSS'
    sdss_df.to_csv(output_sdss, index=False)
    print(f"SDSS: {len(sdss_df)} objects saved to {output_sdss}")

    desi_df = query_desi_redshifts(coord, radius_arcmin)
    desi_df['spec_source'] = 'DESI'
    desi_df.to_csv(output_desi, index=False)
    print(f"DESI: {len(desi_df)} objects saved to {output_desi}")

    # --- Deduplication ---
    desi_cleaned = deduplicate_self(desi_df, tol_deg)
    removed = len(desi_df) - len(desi_cleaned)
    print(f"DESI: {removed} duplicates removed due to DESI duplication")

    sdss_cleaned = deduplicate_self(sdss_df, tol_deg)
    removed = len(sdss_df) - len(sdss_cleaned)
    print(f"SDSS: {removed} duplicates removed due to SDSS duplication")

    ned_cleaned = deduplicate_self(ned_df, tol_deg)
    removed = len(ned_df) - len(ned_cleaned)
    print(f"NED: {removed} duplicates removed due to NED duplication")

    # Convert coordinates for crossmatching (NED and SDSS unused currently)
    _ned_coords = SkyCoord(ned_cleaned["RA"], ned_cleaned["Dec"], unit='deg') if not ned_cleaned.empty else SkyCoord([], [], unit='deg')
    _sdss_coords = SkyCoord(sdss_cleaned["RA"], sdss_cleaned["Dec"], unit='deg') if not sdss_cleaned.empty else SkyCoord([], [], unit='deg')
    desi_coords = SkyCoord(desi_cleaned["RA"], desi_cleaned["Dec"], unit='deg') if not desi_cleaned.empty else SkyCoord([], [], unit='deg')

    # --- Filter lower-priority catalogs against higher-priority ones ---
    sdss_filtered = filter_duplicates(sdss_cleaned, desi_coords, tol_deg)
    sdss_coords_filtered = SkyCoord(sdss_filtered["RA"], sdss_filtered["Dec"], unit='deg') if not sdss_filtered.empty else SkyCoord([], [], unit='deg')
    removed = len(sdss_cleaned) - len(sdss_filtered)
    print(f"SDSS: {removed} duplicates removed due to DESI overlap")

    combined_coords = SkyCoord(
        np.concatenate([desi_coords.ra.deg, sdss_coords_filtered.ra.deg]),
        np.concatenate([desi_coords.dec.deg, sdss_coords_filtered.dec.deg]),
        unit='deg'
    ) if not desi_cleaned.empty or not sdss_filtered.empty else SkyCoord([], [], unit='deg')


    ned_filtered = filter_duplicates(ned_cleaned, combined_coords, tol_deg)
    removed = len(ned_cleaned) - len(ned_filtered)
    print(f"NED: {removed} duplicates removed due to DESI or SDSS overlap")

    # Combine in priority order: DESI > SDSS > NED
    final_df = pd.concat([desi_cleaned, sdss_filtered, ned_filtered], ignore_index=True)
    final_df.to_csv(output_combined_txt, sep=" ", index=False, header=True)
    final_df.to_csv(output_combined_csv, index=False)
    print(f"Final combined catalog: {len(final_df)} sources")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run archival redshift query for a galaxy cluster.")
    parser.add_argument(
        "cluster_id",
        type=str,
        help="Cluster ID (e.g. '1327') or full RMJ/Abell name."
    )
    parser.add_argument(
        "-r",
        "--radius",
        type=float,
        default=DEFAULT_RADIUS_ARCMIN,
        help=f"Search radius in arcminutes (default: {DEFAULT_RADIUS_ARCMIN})")
    parser.add_argument(
        "-t",
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE_DEG,
        help=f"Duplicate match tolerance in degrees (default: {DEFAULT_TOLERANCE_DEG} = {DEFAULT_TOLERANCE_DEG * 3600} arcsec)")
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        default=None,
        help="Base path for cluster science directories (default: cluster.py BASE_PATH)")
    parser.add_argument(
        "-u",
        "--user",
        type=str,
        default=None,
        help="CasJobs username (default: prompt or env CASJOBS_USER)")
    parser.add_argument(
        "--password",
        type=str,
        default=None,
        help="CasJobs password (default: prompt or env CASJOBS_PW)")
    args = parser.parse_args()

    cluster = Cluster(args.cluster_id, base_path=args.path)
    cluster.populate()

    user = args.user or os.environ.get("CASJOBS_USER") or input("CasJobs username: ")
    password = args.password or os.environ.get("CASJOBS_PW") or getpass("CasJobs password: ")

    run_redshift_pipeline(cluster, user, password, radius_arcmin=args.radius, tol_deg=args.tolerance)

if __name__ == "__main__":
    main()
