#!/usr/bin/env python3
"""
spectroscopy.py

Stage 1: Spectroscopic Redshift Retrieval
---------------------------------------------------------

Queries archival spectroscopic redshift catalogs around a cluster position and
loads user-supplied Keck/DEIMOS spectra. Deduplicates with priority
DESI > SDSS > NED, then appends user spectra as the highest-fidelity source.

Archival sources:
  - NED (via astroquery.ned)
  - SDSS DR18 (via CasJobs REST API)
  - DESI DR1 (via NOIRLab TAP and Q3C_RADIAL_QUERY)

User spectra:
  - Keck/DEIMOS zdump*.txt files (4-column: RA, Dec, z, sigma_z)
  - Manual redshift files: manual_z_{source}.txt (same 4-column format)
    The {source} portion of the filename becomes the spec_source value.

Data products (all archived individually before merging):
  - Redshifts/ned.csv              Raw NED query result
  - Redshifts/sdss.csv             Raw SDSS query result
  - Redshifts/desi.csv             Raw DESI query result
  - Redshifts/deimos.csv           User DEIMOS spectra (separate from archival)
  - Redshifts/manual_{source}.csv  Manual redshifts per source
  - Redshifts/archival_z.csv       Merged + deduplicated archival catalog (NO user spectra)

The archival_z.csv file is used by other group members in their own pipelines.
DEIMOS spectra are merged with archival data in Stage 3 (catalog matching),
where combined_redshifts.csv is produced with DEIMOS at highest priority.

Column convention:
  RA, Dec, z, sigma_z, spec_source

Requirements:
  - astropy, astroquery, numpy, pandas, requests

Notes:
  - Each archival source is queried independently; a failure in one does not
    prevent others from running.
  - SDSS is skipped if CasJobs credentials are not provided.
  - Deduplication uses positional matching with configurable angular tolerance.
  - User spectra (DEIMOS) are appended after archival deduplication and are
    NOT deduplicated against archival sources (they are the highest-priority data).
"""

from __future__ import annotations

import os
import glob
import time
from io import StringIO

import pandas as pd
import numpy as np
import requests

from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.ipac.ned import Ned
from astroquery.utils.tap.core import TapPlus

from cluster_pipeline.models.cluster import Cluster
from cluster_pipeline.utils.coordinates import make_skycoord

# ------------------------------------
# Constants
# ------------------------------------
DEFAULT_RADIUS_ARCMIN = 10.0
DEFAULT_TOLERANCE_DEG = 2.0 / 3600.0  # 2 arcsec
DEFAULT_MAX_RETRIES = 5
DEFAULT_INITIAL_WAIT = 5  # seconds
DEFAULT_TIMEOUT = 300  # seconds
DEFAULT_SDSS_TOP_N = 100

SDSS_URL = "https://skyserver.sdss.org/CasJobs/RestApi/contexts/DR18/query"
DESI_URL = "https://datalab.noirlab.edu/tap"

COLUMNS = ["RA", "Dec", "z", "sigma_z", "spec_source"]


# ====================================================================
# Public interface
# ====================================================================

def run_spectroscopy(
    cluster: Cluster,
    *,
    sources: list[str] | None = None,
    radius_arcmin: float = DEFAULT_RADIUS_ARCMIN,
    tolerance_deg: float = DEFAULT_TOLERANCE_DEG,
    casjobs_user: str | None = None,
    casjobs_password: str | None = None,
    load_deimos: bool = True,
    load_manual: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Query archival spectroscopic redshifts and load user spectra.

    Each archival source is queried independently — a failure in one does not
    prevent the others from running. User spectra (DEIMOS) and manual
    redshift files are loaded and archived separately.

    Parameters
    ----------
    cluster : Cluster
        Must have ``coords`` set (ra, dec) and valid ``redshift_path``.
    sources : list of str, optional
        Which archival sources to query. Default: ``["ned", "sdss", "desi"]``.
        SDSS is automatically skipped if credentials are not available.
    radius_arcmin : float
        Search radius in arcminutes. [default: 10.0]
    tolerance_deg : float
        Angular tolerance for positional deduplication in degrees. [default: 2 arcsec]
    casjobs_user : str or None
        SDSS CasJobs username. Falls back to env ``CASJOBS_USER``.
    casjobs_password : str or None
        SDSS CasJobs password. Falls back to env ``CASJOBS_PW``.
    load_deimos : bool
        Whether to load user DEIMOS spectra from zdump*.txt files. [default: True]
    load_manual : bool
        Whether to load manual redshift files from manual_z_*.txt. [default: True]

    Returns
    -------
    archival_df : pd.DataFrame
        Deduplicated archival catalog (NED + SDSS + DESI only).
        Columns: ``RA, Dec, z, sigma_z, spec_source``.
        Priority: DESI > SDSS > NED.
    deimos_df : pd.DataFrame
        User DEIMOS spectra (separate, not merged into archival).
        Columns: ``RA, Dec, z, sigma_z, spec_source``.
        Empty DataFrame if no zdump files or ``load_deimos=False``.
    manual_df : pd.DataFrame
        Manual redshifts from ``manual_z_{source}.txt`` files.
        Columns: ``RA, Dec, z, sigma_z, spec_source``.
        Empty DataFrame if no manual files or ``load_manual=False``.

    Notes
    -----
    - Individual source catalogs archived to Redshifts/{source}.csv before
      any deduplication.
    - Archival catalog written to Redshifts/archival_z.csv (no user spectra).
    - DEIMOS spectra written to Redshifts/deimos.csv.
    - Manual spectra written to Redshifts/manual_{source}.csv per source file.
    - Merging archival + DEIMOS + manual into combined_redshifts.csv happens in
      Stage 3 (catalog matching), where DEIMOS gets highest priority and manual
      gets lowest.
    """
    if cluster.coords is None:
        raise ValueError("Cluster coordinates not set. Cannot query spectroscopy.")

    if sources is None:
        sources = ["ned", "sdss", "desi"]

    coord = cluster.coords
    cluster.ensure_directories()

    print(f"\nSpectroscopy: RA={coord.ra.deg:.5f}, Dec={coord.dec.deg:.5f}, "
          f"radius={radius_arcmin}', sources={sources}")

    # --- Resolve SDSS credentials ---
    user = casjobs_user or os.environ.get("CASJOBS_USER")
    password = casjobs_password or os.environ.get("CASJOBS_PW")

    # --- Query each archival source independently ---
    catalogs: dict[str, pd.DataFrame] = {}

    if "ned" in sources:
        catalogs["ned"] = _query_and_archive(
            "NED", cluster.redshift_path,
            lambda: query_ned(coord, radius_arcmin),
        )

    if "sdss" in sources:
        if user and password:
            catalogs["sdss"] = _query_and_archive(
                "SDSS", cluster.redshift_path,
                lambda: query_sdss(coord, radius_arcmin, user=user, password=password),
            )
        else:
            print("  SDSS: skipped (no CasJobs credentials)")

    if "desi" in sources:
        catalogs["desi"] = _query_and_archive(
            "DESI", cluster.redshift_path,
            lambda: query_desi(coord, radius_arcmin),
        )

    # --- Deduplicate archival sources: DESI > SDSS > NED ---
    archival_df = _deduplicate_catalogs(catalogs, tolerance_deg)

    # --- Save archival catalog (no user spectra) ---
    archival_path = os.path.join(cluster.redshift_path, "archival_z.csv")
    archival_df.to_csv(archival_path, index=False)
    print(f"  Archival catalog: {len(archival_df)} sources → {archival_path}")

    # --- Load user DEIMOS spectra (archived separately) ---
    deimos_df = pd.DataFrame(columns=COLUMNS)
    if load_deimos:
        deimos_df = load_user_spectra(cluster)

    # --- Load manual redshift files (archived separately) ---
    manual_df = pd.DataFrame(columns=COLUMNS)
    if load_manual:
        manual_df = load_manual_spectra(cluster)

    return archival_df, deimos_df, manual_df


# ====================================================================
# User spectra (DEIMOS)
# ====================================================================

def load_user_spectra(cluster: Cluster) -> pd.DataFrame:
    """Load Keck/DEIMOS spectroscopic redshifts from zdump*.txt files.

    Reads all files matching ``zdump*.txt`` in the cluster's Redshifts/
    directory. Each file is expected to have 4 columns: RA, Dec, z, sigma_z
    (whitespace-delimited, no header).

    Parameters
    ----------
    cluster : Cluster
        Cluster object with valid ``redshift_path``.

    Returns
    -------
    deimos_df : pd.DataFrame
        DataFrame with columns ``RA, Dec, z, sigma_z, spec_source``.
        Empty DataFrame if no zdump files found or all fail to parse.

    Notes
    -----
    - Archived to Redshifts/deimos.csv.
    - Source column is set to ``"Deimos"``.
    """
    rows = []

    zdump_pattern = os.path.join(cluster.redshift_path, "zdump*.txt")
    zdump_files = sorted(glob.glob(zdump_pattern))

    if not zdump_files:
        return pd.DataFrame(columns=COLUMNS)

    for file_path in zdump_files:
        try:
            data = np.loadtxt(file_path, unpack=True)
            if data.ndim == 1:
                data = data.reshape(4, -1)
            for ra, dec, z, sigma_z in zip(*data):
                rows.append({
                    "RA": float(ra),
                    "Dec": float(dec),
                    "z": float(z),
                    "sigma_z": float(sigma_z),
                    "spec_source": "Deimos",
                })
        except Exception as e:
            print(f"  Warning: failed to read {file_path} ({type(e).__name__}): {e}")

    deimos_df = pd.DataFrame(rows, columns=COLUMNS)

    if not deimos_df.empty:
        deimos_path = os.path.join(cluster.redshift_path, "deimos.csv")
        deimos_df.to_csv(deimos_path, index=False)
        print(f"  DEIMOS: {len(deimos_df)} spectra → {deimos_path}")

    return deimos_df


# ====================================================================
# Manual redshift files
# ====================================================================

def load_manual_spectra(cluster: Cluster) -> pd.DataFrame:
    """Load manual redshift files matching ``manual_z_{source}.txt``.

    Each file uses the same 4-column whitespace-delimited format as zdump
    files (RA, Dec, z, sigma_z — no header). The ``{source}`` portion of
    the filename is used as the ``spec_source`` value, so
    ``manual_z_hectospec.txt`` produces ``spec_source="hectospec"``.

    Each source file is archived to ``Redshifts/manual_{source}.csv``.

    Parameters
    ----------
    cluster : Cluster
        Cluster object with valid ``redshift_path``.

    Returns
    -------
    manual_df : pd.DataFrame
        DataFrame with columns ``RA, Dec, z, sigma_z, spec_source``.
        Empty DataFrame if no manual files found or all fail to parse.
    """
    all_rows = []

    pattern = os.path.join(cluster.redshift_path, "manual_z_*.txt")
    manual_files = sorted(glob.glob(pattern))

    if not manual_files:
        return pd.DataFrame(columns=COLUMNS)

    for file_path in manual_files:
        # Extract source name: manual_z_{source}.txt → source
        stem = os.path.splitext(os.path.basename(file_path))[0]
        source = stem[len("manual_z_"):]

        if not source:
            print(f"  Warning: skipping {file_path} (no source name after 'manual_z_')")
            continue

        rows = []
        try:
            data = np.loadtxt(file_path, unpack=True)
            if data.ndim == 1:
                data = data.reshape(4, -1)
            for ra, dec, z, sigma_z in zip(*data):
                rows.append({
                    "RA": float(ra),
                    "Dec": float(dec),
                    "z": float(z),
                    "sigma_z": float(sigma_z),
                    "spec_source": source,
                })
        except Exception as e:
            print(f"  Warning: failed to read {file_path} ({type(e).__name__}): {e}")
            continue

        source_df = pd.DataFrame(rows, columns=COLUMNS)
        if not source_df.empty:
            csv_path = os.path.join(cluster.redshift_path, f"manual_{source}.csv")
            source_df.to_csv(csv_path, index=False)
            print(f"  Manual ({source}): {len(source_df)} spectra → {csv_path}")
            all_rows.extend(rows)

    return pd.DataFrame(all_rows, columns=COLUMNS)


# ====================================================================
# Archival query functions
# ====================================================================

def query_ned(
    coord: SkyCoord,
    radius_arcmin: float,
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    initial_wait: int = DEFAULT_INITIAL_WAIT,
    timeout: int = DEFAULT_TIMEOUT,
) -> pd.DataFrame:
    """Query NED for spectroscopic galaxy redshifts around a coordinate.

    Parameters
    ----------
    coord : SkyCoord
        Central search coordinate.
    radius_arcmin : float
        Search radius in arcminutes.
    max_retries : int
        Maximum query attempts with exponential backoff. [default: 5]
    initial_wait : int
        Initial wait between retries in seconds. [default: 5]
    timeout : int
        NED query timeout in seconds. [default: 300]

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ``RA, Dec, z, sigma_z, spec_source``.
        Empty if no results or all attempts fail.

    Notes
    -----
    - Filters to spectroscopic (SLS flag) galaxy (Type='G') redshifts only.
    - NED does not provide per-object redshift uncertainties; sigma_z is set to 0.
    """
    Ned.TIMEOUT = timeout

    for attempt in range(1, max_retries + 1):
        try:
            print(f"  NED query (attempt {attempt}/{max_retries})...")
            result = Ned.query_region(coord, radius=radius_arcmin * u.arcmin)

            if result is None or len(result) == 0:
                return pd.DataFrame(columns=COLUMNS)

            # Filter: spectroscopic redshifts for galaxies only
            filtered = result[result["Redshift Flag"] == "SLS"].copy()
            if "Type" in filtered.colnames:
                filtered = filtered[filtered["Type"] == "G"].copy()

            if len(filtered) == 0:
                return pd.DataFrame(columns=COLUMNS)

            df = pd.DataFrame({
                "RA": pd.to_numeric(filtered["RA"], errors="coerce"),
                "Dec": pd.to_numeric(filtered["DEC"], errors="coerce"),
                "z": pd.to_numeric(filtered["Redshift"], errors="coerce"),
                "sigma_z": 0.0,
                "spec_source": "NED",
            })
            return df.dropna(subset=["RA", "Dec", "z"]).reset_index(drop=True)

        except Exception as e:
            print(f"  NED attempt {attempt} failed ({type(e).__name__}): {e}")
            if attempt == max_retries:
                print("  NED: all attempts failed")
                return pd.DataFrame(columns=COLUMNS)
            time.sleep(initial_wait * (2 ** (attempt - 1)))

    return pd.DataFrame(columns=COLUMNS)


def query_sdss(
    coord: SkyCoord,
    radius_arcmin: float,
    *,
    user: str,
    password: str,
    top_n: int = DEFAULT_SDSS_TOP_N,
    timeout: int = DEFAULT_TIMEOUT,
) -> pd.DataFrame:
    """Query SDSS DR18 via CasJobs REST API for galaxy redshifts.

    Parameters
    ----------
    coord : SkyCoord
        Central search coordinate.
    radius_arcmin : float
        Search radius in arcminutes.
    user : str
        CasJobs username.
    password : str
        CasJobs password.
    top_n : int
        Maximum number of results to return. [default: 100]
    timeout : int
        HTTP request timeout in seconds. [default: 300]

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ``RA, Dec, z, sigma_z, spec_source``.
        Empty if query fails or returns no results.

    Notes
    -----
    - SQL query uses a bounding box, then filters to true angular radius.
    - Only returns objects classified as GALAXY.
    """
    ra_deg = float(coord.ra.deg)
    dec_deg = float(coord.dec.deg)
    radius_deg = radius_arcmin / 60.0

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
    AND s.class = 'GALAXY'
    """

    print(f"  SDSS query...")
    try:
        response = requests.post(
            SDSS_URL,
            headers={"Accept": "text/plain", "Content-Type": "application/x-www-form-urlencoded"},
            data={"query": sql, "taskname": "quick", "userid": user, "password": password, "format": "csv"},
            timeout=timeout,
        )
        response.raise_for_status()

        df = pd.read_csv(StringIO(response.text))
        if df.empty:
            return pd.DataFrame(columns=COLUMNS)

        df = df.rename(columns={"ra": "RA", "dec": "Dec", "zerr": "sigma_z"})
        for col in ("RA", "Dec", "z", "sigma_z"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["RA", "Dec", "z"]).reset_index(drop=True)

        # SQL uses a box; filter to true angular radius
        df_coords = make_skycoord(df["RA"].to_numpy(), df["Dec"].to_numpy())
        df = df[df_coords.separation(coord).arcmin < radius_arcmin].reset_index(drop=True)

        df["spec_source"] = "SDSS"
        return df[COLUMNS].reset_index(drop=True)

    except Exception as e:
        print(f"  SDSS query failed ({type(e).__name__}): {e}")
        return pd.DataFrame(columns=COLUMNS)


def query_desi(
    coord: SkyCoord,
    radius_arcmin: float,
) -> pd.DataFrame:
    """Query DESI DR1 via NOIRLab TAP for galaxy redshifts.

    Parameters
    ----------
    coord : SkyCoord
        Central search coordinate.
    radius_arcmin : float
        Search radius in arcminutes.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ``RA, Dec, z, sigma_z, spec_source``.
        Empty if query fails or returns no results.

    Notes
    -----
    - Uses Q3C_RADIAL_QUERY for efficient spatial indexing.
    - Only returns objects with spectype='GALAXY'.
    """
    ra_deg = float(coord.ra.deg)
    dec_deg = float(coord.dec.deg)
    radius_deg = radius_arcmin / 60.0

    print(f"  DESI query...")
    try:
        tap = TapPlus(url=DESI_URL)
        query = f"""
        SELECT mean_fiber_ra, mean_fiber_dec, z, zerr
        FROM desi_dr1.zpix
        WHERE 't' = Q3C_RADIAL_QUERY(mean_fiber_ra, mean_fiber_dec, {ra_deg}, {dec_deg}, {radius_deg})
        AND spectype = 'GALAXY'
        AND z IS NOT NULL
        """
        job = tap.launch_job(query)
        df = job.get_results().to_pandas()

        if df.empty:
            return pd.DataFrame(columns=COLUMNS)

        df = df.rename(columns={"mean_fiber_ra": "RA", "mean_fiber_dec": "Dec", "zerr": "sigma_z"})
        for col in ("RA", "Dec", "z", "sigma_z"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["RA", "Dec", "z"]).reset_index(drop=True)

        df["spec_source"] = "DESI"
        return df[COLUMNS].reset_index(drop=True)

    except Exception as e:
        print(f"  DESI query failed ({type(e).__name__}): {e}")
        return pd.DataFrame(columns=COLUMNS)


# ====================================================================
# Deduplication
# ====================================================================

def filter_duplicates(
    df: pd.DataFrame,
    ref_coords: SkyCoord | None,
    tol_deg: float,
) -> pd.DataFrame:
    """Remove rows from ``df`` that fall within ``tol_deg`` of any ``ref_coords``.

    Parameters
    ----------
    df : pd.DataFrame
        Must have ``RA`` and ``Dec`` columns (degrees).
    ref_coords : SkyCoord or None
        Reference catalog positions. If None or empty, returns df unchanged.
    tol_deg : float
        Angular tolerance in degrees.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with positional duplicates removed.
    """
    if df.empty or ref_coords is None or len(ref_coords) == 0:
        return df.reset_index(drop=True)

    df_coords = make_skycoord(df["RA"].to_numpy(), df["Dec"].to_numpy())
    _, sep2d, _ = df_coords.match_to_catalog_sky(ref_coords)
    return df[sep2d > (tol_deg * u.deg)].reset_index(drop=True)


def deduplicate_self(df: pd.DataFrame, tol_deg: float) -> pd.DataFrame:
    """Remove near-duplicate rows within a DataFrame. Keeps first occurrence.

    Parameters
    ----------
    df : pd.DataFrame
        Must have ``RA`` and ``Dec`` columns (degrees).
    tol_deg : float
        Angular separation in degrees below which sources are considered duplicates.

    Returns
    -------
    pd.DataFrame
        Deduplicated DataFrame.
    """
    if df.empty or len(df) == 1:
        return df.copy()

    coords = make_skycoord(df["RA"].to_numpy(), df["Dec"].to_numpy())
    keep = np.ones(len(df), dtype=bool)

    for i in range(len(df) - 1):
        if not keep[i]:
            continue
        sep = coords[i].separation(coords[i + 1:]).deg
        dup_indices = np.where(sep < tol_deg)[0] + (i + 1)
        keep[dup_indices] = False

    return df[keep].reset_index(drop=True)


# ====================================================================
# Internal helpers
# ====================================================================

def _query_and_archive(
    name: str,
    output_dir: str,
    query_fn,
) -> pd.DataFrame:
    """Run a query function, archive the raw result to CSV, return the DataFrame.

    Parameters
    ----------
    name : str
        Source name (e.g., "NED", "SDSS", "DESI") — used for filename and logging.
    output_dir : str
        Directory to write the archived CSV.
    query_fn : callable
        Zero-argument callable that returns a DataFrame.

    Returns
    -------
    pd.DataFrame
        Query result (also archived to ``{output_dir}/{name.lower()}.csv``).
    """
    try:
        df = query_fn()
        output_path = os.path.join(output_dir, f"{name.lower()}.csv")
        df.to_csv(output_path, index=False)
        print(f"  {name}: {len(df)} sources → {output_path}")
        return df
    except Exception as e:
        print(f"  {name}: query failed ({type(e).__name__}: {e})")
        return pd.DataFrame(columns=COLUMNS)


def _deduplicate_catalogs(
    catalogs: dict[str, pd.DataFrame],
    tol_deg: float,
) -> pd.DataFrame:
    """Deduplicate across archival catalogs with priority DESI > SDSS > NED.

    Self-deduplicates each catalog first, then removes lower-priority sources
    that positionally match higher-priority ones.

    Parameters
    ----------
    catalogs : dict[str, DataFrame]
        Keyed by source name (lowercase). Only "desi", "sdss", "ned" are processed.
    tol_deg : float
        Angular tolerance for deduplication in degrees.

    Returns
    -------
    pd.DataFrame
        Merged, deduplicated catalog.
    """
    priority = ["desi", "sdss", "ned"]
    cleaned: dict[str, pd.DataFrame] = {}

    # Self-deduplicate each catalog
    for name in priority:
        if name not in catalogs:
            continue
        df = catalogs[name]
        before = len(df)
        df = deduplicate_self(df, tol_deg)
        removed = before - len(df)
        if removed > 0:
            print(f"  {name.upper()}: {removed} internal duplicates removed")
        cleaned[name] = df

    # Cross-deduplicate: lower-priority sources removed if they match higher-priority
    accumulated_coords: SkyCoord | None = None
    final_parts: list[pd.DataFrame] = []

    for name in priority:
        if name not in cleaned:
            continue
        df = cleaned[name]

        if accumulated_coords is not None:
            before = len(df)
            df = filter_duplicates(df, accumulated_coords, tol_deg)
            removed = before - len(df)
            if removed > 0:
                print(f"  {name.upper()}: {removed} removed (overlap with higher-priority catalog)")

        final_parts.append(df)

        # Build up the accumulated reference catalog
        if not df.empty:
            new_coords = make_skycoord(df["RA"].to_numpy(), df["Dec"].to_numpy())
            if accumulated_coords is None:
                accumulated_coords = new_coords
            else:
                accumulated_coords = make_skycoord(
                    np.concatenate([accumulated_coords.ra.deg, new_coords.ra.deg]),
                    np.concatenate([accumulated_coords.dec.deg, new_coords.dec.deg]),
                )

    if not final_parts:
        return pd.DataFrame(columns=COLUMNS)

    return pd.concat(final_parts, ignore_index=True)


def _sdss_ra_wrap(ra_deg: float, radius_deg: float) -> str:
    """Generate SQL WHERE clause for RA that handles the 0/360 degree wrap.

    Parameters
    ----------
    ra_deg : float
        Right ascension of the search center in degrees.
    radius_deg : float
        Search radius in degrees.

    Returns
    -------
    str
        SQL WHERE clause fragment for RA filtering.
    """
    ra_min = ra_deg - radius_deg
    ra_max = ra_deg + radius_deg

    if 0.0 <= ra_min and ra_max <= 360.0:
        return f"s.ra BETWEEN {ra_min} AND {ra_max}"

    ra_min_wrapped = ra_min % 360.0
    ra_max_wrapped = ra_max % 360.0
    return f"(s.ra >= {ra_min_wrapped} OR s.ra <= {ra_max_wrapped})"
