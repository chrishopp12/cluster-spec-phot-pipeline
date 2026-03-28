"""Stage 1: Archival spectroscopic redshift retrieval.

Queries NED, SDSS DR18, and DESI DR1 for spectroscopic redshifts
around a cluster position. Deduplicates with priority DESI > SDSS > NED.

Reads:  Cluster (coords, paths)
Produces:
    Redshifts/ned.csv       — archived raw NED query
    Redshifts/sdss.csv      — archived raw SDSS query
    Redshifts/desi.csv      — archived raw DESI query
    Redshifts/archival_z.csv — merged, deduplicated
Returns: DataFrame[RA, Dec, z, sigma_z, spec_source]
"""

from __future__ import annotations

import os
import time
from getpass import getpass
from io import StringIO

import pandas as pd
import numpy as np
import requests

from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.ipac.ned import Ned
from astroquery.utils.tap.core import TapPlus

from cluster_pipeline.models.cluster import Cluster

# ------------------------------------
# Constants
# ------------------------------------
DEFAULT_RADIUS_ARCMIN = 10.0
DEFAULT_TOLERANCE_DEG = 2.0 / 3600.0  # 2 arcsec
DEFAULT_MAX_RETRIES = 5
DEFAULT_INITIAL_WAIT = 5
DEFAULT_TIMEOUT = 300
DEFAULT_SDSS_TOP_N = 100

SDSS_URL = "https://skyserver.sdss.org/CasJobs/RestApi/contexts/DR18/query"
DESI_URL = "https://datalab.noirlab.edu/tap"

COLUMNS = ["RA", "Dec", "z", "sigma_z", "spec_source"]


# ------------------------------------
# Public interface
# ------------------------------------

def run_spectroscopy(
    cluster: Cluster,
    *,
    sources: list[str] | None = None,
    radius_arcmin: float = DEFAULT_RADIUS_ARCMIN,
    tolerance_deg: float = DEFAULT_TOLERANCE_DEG,
    casjobs_user: str | None = None,
    casjobs_password: str | None = None,
) -> pd.DataFrame:
    """Query archival spectroscopic redshifts and return a deduplicated catalog.

    Each source is queried independently — a failure in one does not
    prevent the others from running.

    Parameters
    ----------
    cluster : Cluster
        Must have coords set (ra, dec).
    sources : list of str, optional
        Which archives to query. Default: ["ned", "sdss", "desi"].
        SDSS is skipped if credentials are not provided.
    radius_arcmin : float
        Search radius in arcminutes.
    tolerance_deg : float
        Angular tolerance for deduplication.
    casjobs_user, casjobs_password : str, optional
        SDSS CasJobs credentials. If None, reads from env vars
        CASJOBS_USER / CASJOBS_PW. If still None, SDSS is skipped.

    Returns
    -------
    DataFrame[RA, Dec, z, sigma_z, spec_source]
        Deduplicated spectroscopic catalog. Priority: DESI > SDSS > NED.
    """
    if cluster.coords is None:
        raise ValueError("Cluster coordinates not set. Cannot query spectroscopy.")

    if sources is None:
        sources = ["ned", "sdss", "desi"]

    coord = cluster.coords
    cluster.ensure_directories()

    print(f"Spectroscopy: RA={coord.ra.deg:.5f}, Dec={coord.dec.deg:.5f}, "
          f"radius={radius_arcmin}', sources={sources}")

    # --- Resolve SDSS credentials ---
    user = casjobs_user or os.environ.get("CASJOBS_USER")
    password = casjobs_password or os.environ.get("CASJOBS_PW")

    # --- Query each source independently ---
    catalogs: dict[str, pd.DataFrame] = {}

    if "ned" in sources:
        catalogs["ned"] = _query_and_save(
            "NED", cluster.redshift_path,
            lambda: query_ned(coord, radius_arcmin),
        )

    if "sdss" in sources:
        if user and password:
            catalogs["sdss"] = _query_and_save(
                "SDSS", cluster.redshift_path,
                lambda: query_sdss(coord, radius_arcmin, user=user, password=password),
            )
        else:
            print("  SDSS skipped (no CasJobs credentials)")

    if "desi" in sources:
        catalogs["desi"] = _query_and_save(
            "DESI", cluster.redshift_path,
            lambda: query_desi(coord, radius_arcmin),
        )

    # --- Deduplicate: DESI > SDSS > NED ---
    combined = _deduplicate_catalogs(catalogs, tolerance_deg)

    # --- Save combined catalog ---
    output_path = os.path.join(cluster.redshift_path, "archival_z.csv")
    combined.to_csv(output_path, index=False)
    print(f"  Combined catalog: {len(combined)} sources → {output_path}")

    return combined


# ------------------------------------
# Query functions
# ------------------------------------

def query_ned(
    coord: SkyCoord,
    radius_arcmin: float,
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    initial_wait: int = DEFAULT_INITIAL_WAIT,
    timeout: int = DEFAULT_TIMEOUT,
) -> pd.DataFrame:
    """Query NED for spectroscopic galaxy redshifts around a coordinate."""
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
                "sigma_z": 0.0,  # NED does not provide redshift errors
                "spec_source": "NED",
            })
            return df.dropna(subset=["RA", "Dec", "z"]).reset_index(drop=True)

        except Exception as e:
            print(f"  NED attempt {attempt} failed: {e}")
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
    """Query SDSS DR18 via CasJobs REST API for galaxy redshifts."""
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

        # Filter to actual radius (SQL uses a box)
        coords = SkyCoord(df["RA"].to_numpy(), df["Dec"].to_numpy(), unit="deg")
        df = df[coords.separation(coord).arcmin < radius_arcmin].reset_index(drop=True)

        df["spec_source"] = "SDSS"
        return df[COLUMNS].reset_index(drop=True)

    except Exception as e:
        print(f"  SDSS query failed: {e}")
        return pd.DataFrame(columns=COLUMNS)


def query_desi(
    coord: SkyCoord,
    radius_arcmin: float,
) -> pd.DataFrame:
    """Query DESI DR1 via NOIRLab TAP for galaxy redshifts."""
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
        print(f"  DESI query failed: {e}")
        return pd.DataFrame(columns=COLUMNS)


# ------------------------------------
# Deduplication
# ------------------------------------

def filter_duplicates(
    df: pd.DataFrame,
    ref_coords: SkyCoord | None,
    tol_deg: float,
) -> pd.DataFrame:
    """Remove rows from df that are within tol_deg of any ref_coords."""
    if df.empty or ref_coords is None or len(ref_coords) == 0:
        return df.reset_index(drop=True)

    df_coords = SkyCoord(df["RA"].to_numpy(), df["Dec"].to_numpy(), unit="deg")
    _, sep2d, _ = df_coords.match_to_catalog_sky(ref_coords)
    return df[sep2d > (tol_deg * u.deg)].reset_index(drop=True)


def deduplicate_self(df: pd.DataFrame, tol_deg: float) -> pd.DataFrame:
    """Remove near-duplicate rows within a DataFrame. Keeps first occurrence."""
    if df.empty or len(df) == 1:
        return df.copy()

    coords = SkyCoord(df["RA"].to_numpy(), df["Dec"].to_numpy(), unit="deg")
    keep = np.ones(len(df), dtype=bool)

    for i in range(len(df) - 1):
        if not keep[i]:
            continue
        sep = coords[i].separation(coords[i + 1:]).deg
        dup_indices = np.where(sep < tol_deg)[0] + (i + 1)
        keep[dup_indices] = False

    return df[keep].reset_index(drop=True)


# ------------------------------------
# Internal helpers
# ------------------------------------

def _query_and_save(
    name: str,
    output_dir: str,
    query_fn,
) -> pd.DataFrame:
    """Run a query function, save the raw result, return the DataFrame."""
    try:
        df = query_fn()
        output_path = os.path.join(output_dir, f"{name.lower()}.csv")
        df.to_csv(output_path, index=False)
        print(f"  {name}: {len(df)} sources → {output_path}")
        return df
    except Exception as e:
        print(f"  {name}: query failed ({e})")
        return pd.DataFrame(columns=COLUMNS)


def _deduplicate_catalogs(
    catalogs: dict[str, pd.DataFrame],
    tol_deg: float,
) -> pd.DataFrame:
    """Deduplicate across catalogs with priority DESI > SDSS > NED."""
    priority = ["desi", "sdss", "ned"]
    cleaned: dict[str, pd.DataFrame] = {}

    # Self-deduplicate each catalog
    for name in priority:
        if name not in catalogs:
            continue
        df = catalogs[name]
        before = len(df)
        df = deduplicate_self(df, tol_deg)
        if before - len(df) > 0:
            print(f"  {name.upper()}: {before - len(df)} internal duplicates removed")
        cleaned[name] = df

    # Cross-deduplicate: remove lower-priority sources that match higher-priority
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

        # Update accumulated coords
        if not df.empty:
            new_coords = SkyCoord(df["RA"].to_numpy(), df["Dec"].to_numpy(), unit="deg")
            if accumulated_coords is None:
                accumulated_coords = new_coords
            else:
                accumulated_coords = SkyCoord(
                    np.concatenate([accumulated_coords.ra.deg, new_coords.ra.deg]),
                    np.concatenate([accumulated_coords.dec.deg, new_coords.dec.deg]),
                    unit="deg",
                )

    if not final_parts:
        return pd.DataFrame(columns=COLUMNS)

    return pd.concat(final_parts, ignore_index=True)


def _sdss_ra_wrap(ra_deg: float, radius_deg: float) -> str:
    """SQL WHERE clause for RA that handles the 0/360 wrap."""
    ra_min = ra_deg - radius_deg
    ra_max = ra_deg + radius_deg

    if 0.0 <= ra_min and ra_max <= 360.0:
        return f"s.ra BETWEEN {ra_min} AND {ra_max}"

    ra_min_wrapped = ra_min % 360.0
    ra_max_wrapped = ra_max % 360.0
    return f"(s.ra >= {ra_min_wrapped} OR s.ra <= {ra_max_wrapped})"
