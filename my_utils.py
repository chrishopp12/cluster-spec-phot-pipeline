#!/usr/bin/env python3
"""
my_utils.py

Cluster Analysis Utilities
---------------------------------------------------------

Utility functions for cluster data processing and analysis. Includes file management, coordinate resolution, data loading, 
and plotting utilities.

Requirements:
    - astropy
    - astroquery
    - pandas
    - numpy
    - matplotlib
    - scikit-image
    - scipy
    - cluster.py (local)
"""

from __future__ import annotations
import os
import re
import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from astropy.coordinates import Angle, SkyCoord
import astropy.units as u
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.wcs import WCS

from astroquery.ipac.ned import Ned
from astroquery.simbad import Simbad
from astroquery.hips2fits import hips2fits

from skimage.restoration import inpaint_biharmonic
from scipy.ndimage import uniform_filter
from scipy.stats import gaussian_kde

from astroquery.vizier import Vizier

if TYPE_CHECKING:
    from cluster import Cluster

# ------------------------------------
# Defaults/ Constants
# ------------------------------------
COSMO = FlatLambdaCDM(H0=70, Om0=0.3)


# ------------------------------------
# Miscellaneous Utilities
# ------------------------------------

def string_to_numeric(s: str) -> float | int | str:
    """
    Converts a string to a numeric type if possible.

    Parameters
    ----------
    s : str
        Input string.

    Returns
    -------
    float | int | str
        Converted numeric value or original string if conversion fails.
    """
    s = s.strip()
    if s == '':
        return s
    try:
        if '.' in s or 'e' in s.lower():
            return float(s)
        else:
            return int(s)
    except Exception:
        return s


def find_first_val(*vals: Any) -> Any:
    """
    Returns the first non-None value from the provided arguments.

    Parameters
    ----------
    *vals : Any
        Variable number of arguments.

    Returns
    -------
    Any
        The first non-None value, or None if all are None.
    """
    for v in vals:
        if v is not None:
            return v
    return None


def to_float_or_none(x: Any) -> float | None:
    """
    Convert to a finite float, else None.

    Handles: None, NaN/inf, numpy scalars, masked values, numeric strings.
    """
    if x is None:
        return None
    # numpy masked / astropy masked
    if hasattr(x, "mask") and bool(getattr(x, "mask", False)):
        return None
    try:
        v = float(x)
    except Exception:
        return None
    return v if np.isfinite(v) else None


def coerce_to_numeric(
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

# ------------------------------------
# File and Directory Utilities
# ------------------------------------

def make_directories(
        path: str | os.PathLike,
        cluster: str
    ) -> tuple[str, str]:
    """
    Create directories for photometry and redshift data.

    Parameters
    ----------
    path : str | os.PathLike
        Base path for the directories.
    cluster : str
        Cluster identifier.

    Returns
    -------
    photometry_dir : str
        Directory for photometry data.
    redshift_dir : str
        Directory for redshift data.
    """
    base = os.fspath(path)

    if not os.path.exists(base):
        raise ValueError(f"Base path {base} does not exist.")
    if not os.path.isdir(base):
        raise ValueError(f"Base path {base} is not a directory.")

    # Define paths
    base_photometry = os.path.join(base, "Photometry")
    base_redshift = os.path.join(base, "Redshifts")
    photometry_dir = os.path.join(base_photometry, f"{cluster}")
    redshift_dir = os.path.join(base_redshift, f"{cluster}")

    # Create necessary subdirectories
    os.makedirs(base_photometry, exist_ok=True)
    os.makedirs(base_redshift, exist_ok=True)
    os.makedirs(photometry_dir, exist_ok=True)
    os.makedirs(redshift_dir, exist_ok=True)

    return photometry_dir, redshift_dir


def read_json(path: Path) -> dict[str, Any]:
    """
    Reads a JSON file and returns its contents as a dictionary.

    Parameters
    ----------
    path : Path
        Path to the JSON file.

    Returns
    -------
    dict
        Contents of the JSON file as a dictionary.
    """
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError as e:
        raise FileNotFoundError(f"JSON file not found: {path}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON from file: {path}") from e
    
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON file to contain a dictionary, got {type(data)}")
    return data


def query_redmapper(
    coord: SkyCoord,
    *,
    radius_arcmin: float = 5.0,
    verbose: bool = True,
):
    """
    Query redMaPPer via VizieR near a coordinate and return results for the closest cluster.

    Notes
    -----
    - This is a *positional* query (no name matching).
    - If multiple clusters are returned, we select the closest on-sky.
      (In practice, this is almost always fine for small radii.)

    Parameters
    ----------
    coord : SkyCoord
        Target coordinate (typically the cluster center).
    radius_arcmin : float
        Search radius around `coord` in arcminutes.
    verbose : bool
        Print minimal diagnostics.

    Returns
    -------

    """

    # TODO: Allow name search

    if coord is None:
        print("No coordinate provided for redMaPPer query.")
        return None
    
    # TODO: Add support for DES redMaPPer catalog
    # VizieR SVA 1
    # catalog = "J/ApJS/224/1/cat_sva1"

    catalog = "J/ApJS/224/1/cat_dr8"

    cols = [
        "Name", "RAJ2000", "DEJ2000",
        "PCen0", "PCen1", "PCen2", "PCen3", "PCen4",
        "RA0deg", "DE0deg", "RA1deg", "DE1deg", "RA2deg", "DE2deg", "RA3deg", "DE3deg", "RA4deg", "DE4deg",
        "zlambda", "e_zlambda", "lambda", "e_lambda",
    ]

    viz = Vizier(columns=cols, row_limit=50)

    try:
        tabs = viz.query_region(coord, radius=radius_arcmin * u.arcmin, catalog=catalog)
    except Exception as e:
        if verbose:
            print(f"redMaPPer VizieR query failed: {e}")
        return None

    if not tabs or len(tabs[0]) == 0:
        if verbose:
            print("No redMaPPer clusters found in search radius.")
        return None

    t = tabs[0]

    # Choose the closest cluster by its center position.
    cl_coords = SkyCoord(t["RAJ2000"], t["DEJ2000"], unit=(u.deg, u.deg), frame="icrs")
    idx = int(coord.separation(cl_coords).argmin())
    redmapper_results = t[idx]

    if verbose:
        name = str(redmapper_results["Name"])
        sep_arcmin = float(coord.separation(cl_coords[idx]).to(u.arcmin).value)
        print(f"Found redMaPPer cluster '{name}' at separation {sep_arcmin:.2f} arcmin.")

    return redmapper_results


def get_redmapper_bcg_candidates(
    coord: SkyCoord,
    *,
    radius_arcmin: float = 5.0,
    verbose: bool = True,
) -> list[tuple[float, float, None, float | None]]:
    """
    Query redMaPPer via VizieR near a coordinate and return the closest cluster's 5 BCG candidates.

    Notes
    -----
    - This is a *positional* query (no name matching).
    - If multiple clusters are returned, we select the closest on-sky.
      (In practice, this is almost always fine for small radii.)

    Parameters
    ----------
    coord : SkyCoord
        Target coordinate (typically the cluster center).
    radius_arcmin : float
        Search radius around `coord` in arcminutes.
    verbose : bool
        Print minimal diagnostics.

    Returns
    -------
    bcgs : list of (RA_deg, Dec_deg, z=None, Pcen) tuples
         List of the 5 BCG candidates from the closest redMaPPer cluster,
         with their RA, Dec, and Pcen values.
    """

    if coord is None:
        print("No coordinate provided for redMaPPer query.")
        return []
    

    redmapper_results = query_redmapper(coord, radius_arcmin=radius_arcmin, verbose=verbose)
    if redmapper_results is None:
        print("No redMaPPer cluster found; cannot get BCG candidates.")
        return []


    bcgs: list[tuple[float, float, None, float | None]] = []
    for i in range(5):
        ra = to_float_or_none(redmapper_results[f"RA{i}deg"])
        dec = to_float_or_none(redmapper_results[f"DE{i}deg"])
        p = to_float_or_none(redmapper_results.get(f"PCen{i}"))

        bcgs.append((ra, dec, None, p))

    return bcgs


def get_redmapper_cluster_info(
    coord: SkyCoord,
    *,
    radius_arcmin: float = 5.0,
    verbose: bool = True,
) -> list[tuple[float, float, None, float | None]]:
    """
    Query redMaPPer via VizieR near a coordinate and return the closest cluster's info.

    Notes
    -----
    - This is a *positional* query (no name matching).
    - If multiple clusters are returned, we select the closest on-sky.
      (In practice, this is almost always fine for small radii.)

    Parameters
    ----------
    coord : SkyCoord
        Target coordinate (typically the cluster center).
    radius_arcmin : float
        Search radius around `coord` in arcminutes.
    verbose : bool
        Print minimal diagnostics.

    Returns
    -------
    cluster_info : dict with keys:
        - name: Cluster name (str)
        - zlambda: Cluster redshift (float or None)
        - e_zlambda: Uncertainty on zlambda (float or None)
        - lambda: Cluster richness (float or None)
        - e_lambda: Uncertainty on richness (float or None)
    """
    redmapper_results = query_redmapper(coord, radius_arcmin=radius_arcmin, verbose=verbose)

    if redmapper_results is None:
        print("No redMaPPer cluster found; cannot get BCG candidates.")
        return {
            "name": None,
            "zlambda": None,
            "e_zlambda": None,
            "lambda": None,
            "e_lambda": None,
        }

   
    cluster_info = {
        "name":  str(redmapper_results["Name"]),
        "zlambda": to_float_or_none(redmapper_results["zlambda"]),
        "e_zlambda": to_float_or_none(redmapper_results["e_zlambda"]),
        "lambda": to_float_or_none(redmapper_results["lambda"]),
        "e_lambda": to_float_or_none(redmapper_results["e_lambda"]),
    }

    return cluster_info



# ------------------------------------
# Cluster Name and Coordinate Utilities
# ------------------------------------

def simbad_coord_lookup(simbad_name: str) -> SkyCoord:
    """
    Attempts to resolve coordinates from SIMBAD by name.

    Parameters
    ----------
    simbad_name : str
        Name string for SIMBAD lookup.

    Returns
    -------
    SkyCoord
        Astropy SkyCoord object with resolved coordinates.

    Raises
    ------
    Exception
        If lookup fails.
    """

    try:
        print(f"Trying Simbad for {simbad_name}...")
        simbad = Simbad()
        simbad.TIMEOUT = 10
        simbad.add_votable_fields("coordinates")
        result = simbad.query_object(simbad_name)

        if result is None:
            raise ValueError(f"Simbad returned no results for {simbad_name}")
        
        else:
            if 'ra' in result.colnames and 'dec' in result.colnames:
                ra = result['ra'][0]
                dec = result['dec'][0]

                print(f"Found in SIMBAD: RA = {ra}, Dec = {dec}")
                return SkyCoord(ra=ra, dec=dec, unit='deg')
            
            elif 'RA' in result.colnames and 'DEC' in result.colnames:
                ra = result['RA'][0]
                dec = result['DEC'][0]
            
            else:
                print(f"No usable RA/DEC columns found in Simbad result for {simbad_name}")
                print(f"Columns returned: {result.colnames}")
                raise ValueError("No usable RA/DEC columns in Simbad result")
            
            print(f"Found in SIMBAD: RA = {ra}, Dec = {dec}")
            return SkyCoord(ra=ra, dec=dec, unit='deg')
            
    except Exception as e:
        print(f"SIMBAD lookup failed: {e}")
        raise


def get_cluster_id(identifier: str) -> str:
    """
    Maps observation ID or cluster short name to full RMJ name.

    Parameters
    ----------
    identifier : str
        Observation ID (e.g. '0881900801') or short cluster name (e.g. '1327').

    Returns
    -------
    str
        Full RMJ cluster name (e.g. 'RMJ132724.2+534656.5').

    Raises
    ------
    ValueError
        If the identifier is not recognized.
    """
    cluster_map = {
        '1234567890': 'RMJ213518.8+012527.0',
        '0881900801': 'RMJ000343.8+100123.8',
        '0881901001': 'RMJ080135.3+362807.5',
        '0881901201': 'RMJ132724.2+534656.5',
        '0901870201': 'RMJ092647.3+050004.0',
        '0881900301': 'RMJ121917.6+505432.8',
        '0901870901': 'RMJ082944.9+382754.4',
        '0881900501': 'RMJ163509.2+152951.5',
        '0881900701': 'RMJ125725.9+365429.4',
        '0922150101': 'RMJ232104.1+291134.5',
        '0922150301': 'RMJ104311.1+150151.9',
        '0922150401': 'RMJ010934.2+330301.0',
        '0922150601': 'RMJ021952.2+012952.2'
    }

    # Reverse map to allow lookup by short name
    short_name_map = {value[3:7]: value for value in cluster_map.values()}

    # Check if the identifier is an observation ID
    if identifier in cluster_map:
        return cluster_map[identifier]
    
    # Check if the identifier is a short name
    if identifier in short_name_map:
        return short_name_map[identifier]

    raise ValueError(f"Identifier {identifier} not found in the cluster map.")


def get_name(identifier: str) -> str:
    """
    Maps an identifier to a full RMJ cluster name.

    Parameters
    ----------
    identifier : str
        Observation ID, short cluster name, or full RMJ name.

    Returns
    -------
    str
        Full RMJ cluster name.

    Raises
    ------
    ValueError
        If the identifier is not recognized.
    """

    if identifier is None:
        raise ValueError("No identifier provided.")
    
    identifier = str(identifier).strip()
    print(f"Searching OBJ_ID catalog for {identifier}...")

    try:
        cluster_name = get_cluster_id(identifier)
        print(f"Identifier '{identifier}' mapped to cluster name '{cluster_name}'")
        return cluster_name

    except Exception as e:
        print(f"{e}")

    # Try the OBJ_ID catalog again with only cluster number
    cluster_name = identifier
    cluster_number = ''.join(re.findall(r'\d+', cluster_name))

    if cluster_number:
        try:
            print(f"Trying with cluster number '{cluster_number}'...")
            cluster_name = get_cluster_id(cluster_number)
            print(f"Cluster number '{cluster_number}' mapped to cluster name '{cluster_name}'")
            return cluster_name

        except Exception as e:
            print(f"{e}")

    print(f"Using provided identifier as cluster name: '{cluster_name}'")
    return identifier


def get_coordinates(identifier: str) -> SkyCoord:
    """
    Attempts to resolve a cluster name to coordinates using NED or SIMBAD.

    Parameters
    ----------
    identifier : str
        Cluster ID (XMM OBS_ID, 4-digit identifier, or custom), or cluster name. 

    Returns
    -------
    SkyCoord
        Astropy coordinate object for the cluster.

    Raises
    ------
    ValueError
        If coordinates cannot be found.
    """

    cluster_name = get_name(identifier)
    stripped_name = cluster_name.strip().replace(" ","")
    cluster_number = "".join(re.findall(r"\d+", cluster_name))

    print(f"Resolving coordinates for {cluster_name}...")
    # First try NED
    try:
        print(f"Trying NED for {cluster_name}...")
        result = Ned.query_object(cluster_name)

        if result is not None and len(result) > 0:
            ra = result['RA'][0]
            dec = result['DEC'][0]

            print(f"Found in NED: RA = {ra}, Dec = {dec}")
            return SkyCoord(ra=ra, dec=dec, unit='deg')
        
    except Exception as e:
        print(f"NED lookup failed: {e}")

    # Format for Simbad
    if stripped_name.lower().startswith('rmj'):
        simbad_name = f"[RRB2014] RM {stripped_name[2:]}"

    elif stripped_name.lower().startswith('a'):
        simbad_name = f"ACO {cluster_number}"

    else:
        print("Name does not match expected syntax, trying SIMBAD with raw name")
        simbad_name = cluster_name

    try:
        coord = simbad_coord_lookup(simbad_name)
        return coord
    
    except Exception as e:
        print(f"{e}")

    raise ValueError(f"Coordinates for {cluster_name} could not be found.")


def get_redshift(identifier: str, BCGs: list[tuple] | None = None) -> float:
    """
    Attempts to retrieve the redshift of a cluster from NED or Simbad, 
    falling back to BCG data if necessary.

    Parameters
    ----------
    identifier : str
        Observation ID, short cluster name, or full RMJ name.
    BCGs : list of tuples
        List of BCG coordinates and redshifts, where each tuple is (ra, dec, z).

    Returns
    -------
    float
        Redshift of the cluster.

    Raises
    ------
    ValueError
        If redshift cannot be found.
    """

    cluster_name = get_name(identifier)
    print(f"Retrieving redshift for {cluster_name}...")

    stripped_name = cluster_name.strip().replace(" ","")
    cluster_number = "".join(re.findall(r"\d+", cluster_name))

    # Try NED
    try:
        print(f"Trying NED for redshift of {cluster_name}...")
        ned_result = Ned.query_object(cluster_name)

        if ned_result is not None and len(ned_result) > 0:
            # Check if redshift is already part of the result
            if 'Redshift' in ned_result.colnames:
                z = ned_result['Redshift'][0]
                if not np.ma.is_masked(z):
                    print(f"Found in NED: z = {z}")
                    return float(z)
            else:
                # Try to pull redshift from redshift table
                try:
                    redshift_table = Ned.get_table(cluster_name, table='redshifts')
                    if redshift_table is not None and 'Redshift' in redshift_table.colnames:
                        z = redshift_table['Redshift'][0]
                        if not np.ma.is_masked(z):
                            print(f"Found in NED redshift table: z = {z}")
                            return float(z)
                except Exception as e_inner:
                    print(f"No redshift table found in NED: {e_inner}")
    except Exception as e:
        print(f"NED redshift lookup failed: {e}")


    # Try Simbad
    try:
        print(f"Trying Simbad for redshift of {cluster_name}...")

        # Format for Simbad
        if stripped_name.lower().startswith('rmj'):
            simbad_name = f"[RRB2014] RM {stripped_name[2:]}"

        elif stripped_name.lower().startswith('a'):
            simbad_name = f"ACO {cluster_number}"

        else:
            print("Name does not match expected syntax, trying SIMBAD with raw name")
            simbad_name = cluster_name

        
        custom_simbad = Simbad()
        custom_simbad.TIMEOUT = 10
        custom_simbad.add_votable_fields('rvz_redshift') 
        simbad_result = custom_simbad.query_object(simbad_name)

        if simbad_result is not None:
            # Check lowercase 'rvz_redshift'
            if 'rvz_redshift' in simbad_result.colnames:
                z = simbad_result['rvz_redshift'][0]
                if z is not None and not np.ma.is_masked(z):
                    print(f"Found in Simbad: z = {z}")
                    return float(z)
    except Exception as e:
        print(f"Simbad redshift lookup failed: {e}")

    # Fall back to BCGs if provided
    if BCGs is not None and len(BCGs) > 0:
        print(f"Falling back to BCG redshift for {cluster_name}...")
        first_bcg = BCGs[0]  # (ra, dec, z)
        if len(first_bcg) >= 3:
            z = first_bcg[2]
            print(f"Using BCG fallback: z = {z}")
            return float(z)

    raise ValueError(f"Could not find redshift for {cluster_name} in Simbad, NED, or BCG data.")


def make_skycoord(
    ra_deg: np.ndarray | list[float],
    dec_deg: np.ndarray | list[float],
) -> SkyCoord:
    """
    Create an ICRS SkyCoord from RA/Dec in degrees.

    Parameters
    ----------
    ra_deg : np.ndarray or list of float
        Right Ascension values in degrees.
    dec_deg : np.ndarray or list of float
        Declination values in degrees.

    Returns
    -------
    SkyCoord
        SkyCoord object with the provided RA and Dec in the ICRS frame.
    """
    return SkyCoord(
        ra=np.asarray(ra_deg, dtype=float) * u.deg,
        dec=np.asarray(dec_deg, dtype=float) * u.deg,
        frame="icrs",
    )

def skycoord_from_df(
    df: pd.DataFrame,
    *,
    ra_col: str = "RA",
    dec_col: str = "Dec",
) -> SkyCoord:
    """
    Create SkyCoord from a DataFrame with RA/Dec columns (degrees).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing RA and Dec columns.
    ra_col : str
        Name of the RA column (default: "RA").
    dec_col : str
        Name of the Dec column (default: "Dec").

    Returns
    -------
    SkyCoord
        SkyCoord object with coordinates from the specified DataFrame columns.
    """
    if ra_col not in df.columns or dec_col not in df.columns:
        raise KeyError(f"DataFrame must contain '{ra_col}' and '{dec_col}' columns.")
    return make_skycoord(df[ra_col].values, df[dec_col].values)


# ------------------------------------
# File Loading Utilities
# ------------------------------------

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


def load_bcg_catalog(cluster: "Cluster") -> dict[int, dict[str, Any]]:
    """
    Load BCGs.csv and return a dict keyed by bcg_id.

    Parameters
    ----------
    bcg_csv_path : str
        Path to 'BCGs.csv' (per-cluster file).

    Returns
    -------
    bcgs : dict
        bcgs[bcg_id] = {
            'bcg_id': int,
            'z': float,
            'z_err': float or None,
            'label': str or '',
        }
    """

    # TODO: This function is obsolete and should be deleted.

    df = pd.read_csv(cluster.bcg_file)
    if "bcg_id" not in df.columns or "z" not in df.columns:
        raise ValueError("BCGs.csv must have at least columns: 'bcg_id', 'z'.")

    if "z_err" not in df.columns:
        df["z_err"] = pd.Series([None] * len(df))
    if "label" not in df.columns:
        df["label"] = pd.Series([""] * len(df))

    bcgs: dict[int, dict[str, Any]] = {}
    for _, r in df.iterrows():
        bid = int(r["bcg_id"])
        z = float(r["z"])
        z_err = None if (pd.isna(r["z_err"])) else float(r["z_err"])
        bcgs[bid] = {
            "bcg_id": bid,
            "z": z,
            "z_err": z_err,
            "label": str(r.get("label", "")),
        }
    return bcgs


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


def pop_prefixed_kwargs(kwargs: dict[str, Any], prefix: str) -> dict[str, Any]:
    """
    Extracts and removes all keyword arguments from `kwargs` that start with the given `prefix`.

    Parameters
    ----------
    kwargs : dict
        Dictionary of keyword arguments.
    prefix : str
        Prefix to filter the keyword arguments.
    Returns
    -------
    dict[str, Any]
        A new dictionary containing only the keyword arguments that started with the specified prefix,
        with the prefix removed from their keys.
    """
    out = kwargs.pop(f'{prefix}_kwargs', {}).copy()
    for k in list(kwargs):
        if k.startswith(f'{prefix}_'):
            out[k[len(prefix) + 1:]] = kwargs.pop(k)
    return out


def str2bool(v: str | bool) -> bool:
    """
    Argparse-compatible string to boolean conversion.

    Parameters
    ----------
    v : str | bool
        Input string or boolean.

    Returns
    -------
    bool
        Converted boolean value.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
 

def get_optical_image(
        cluster: "Cluster",
        fov: float,
        ra_offset: float = 0.0,
        dec_offset: float = 0.0,
        check_validity: bool = True
    ) -> str | None:
    """Retrieves optical images from specified surveys (Legacy, PanSTAARS, more to come) and saves them as FITS files.
    
    Parameters
    ----------
    folder : str
        Path to the folder where the FITS file will be saved.
    cluster_coords : SkyCoord
        Astropy SkyCoord object with the cluster coordinates.
    fov : float
        Field of view in degrees for the image.
    ra_offset : float, optional
        Offset in degrees to apply to the Right Ascension of the cluster coordinates (default is 0).
    dec_offset : float, optional
        Offset in degrees to apply to the Declination of the cluster coordinates (default is 0).
    check_validity : bool, optional
        If True, checks if the retrieved FITS file is valid (default is True).

    Returns
    ---------
    fits_path : str
        Path to the saved FITS file containing the optical image.
    None
        If no valid image is retrieved from the surveys.
    """
    ra_offset_deg = ra_offset/ 60
    dec_offset_deg = dec_offset/ 60
    fov_deg = fov/ 60

    # Query prioritized surveys
    surveys = [
    {'name': 'PanSTARRS', 'hips': 'CDS/P/PanSTARRS/DR1/color-i-r-g'},
    {'name': 'Legacy Survey', 'hips': 'CDS/P/DESI-Legacy-Surveys/DR10/color'},
]

    fits_path = f"{cluster.photometry_path}/optical_image_{fov}_{ra_offset}_{dec_offset}.fits"
    if is_fits_valid(fits_path):
        return fits_path

    for survey in surveys:
        try:
            print(f"Querying {survey['name']}...")
            result = hips2fits.query(
                hips=survey['hips'],
                width=802,
                height=800,
                ra=cluster.coords.ra + ra_offset_deg*u.deg,
                dec=cluster.coords.dec + dec_offset_deg*u.deg,
                fov=Angle(fov_deg, 'deg'),
                projection='TAN',
                format='fits',
            )
            survey_path = f"{cluster.photometry_path}/{survey['name']}_optical_image_{fov}_{ra_offset}_{dec_offset}.fits"
            fits_path = f"{cluster.photometry_path}/optical_image_{fov}_{ra_offset}_{dec_offset}.fits"
            result.writeto(survey_path, overwrite=True)
            result.writeto(fits_path, overwrite=True)
            print(f"Image retrieved and saved: {survey_path}")
            
            if check_validity and not is_fits_valid(survey_path):
                print(f"{survey['name']} image appears empty, checking the next survey...")
                try:
                    os.remove(survey_path)  # Remove the empty file
                except FileNotFoundError:
                    pass
                try:
                    os.remove(fits_path)  # Remove the empty file
                except FileNotFoundError:
                    pass
                continue  # Try the next survey
            
            return fits_path
        except Exception as e:
            print(f"Failed to retrieve image from {survey['name']}: {e}")
    
    print("No data available from the prioritized surveys.")
    return None


def is_fits_valid(
        fits_path: str | os.PathLike,
        min_nonzero_fraction: float = 0.7,
        min_intensity_threshold: float = 1e-5,
        verbose: bool = False
    ) -> bool:
    """
    Checks if a FITS file contains meaningful, non-empty data.

    Parameters
    ----------

    fits_path : str or Path
        Path to the FITS file to be checked.
    min_nonzero_fraction : float, optional
        Minimum fraction of nonzero pixels required for the image to be considered valid (default is 0.5).
    min_intensity_threshold : float, optional
        Minimum intensity threshold for a pixel to be considered non-empty (default is 1e-5).

    Returns
    -------
    bool
        True if the FITS file is valid (contains sufficient nonzero pixels), False otherwise.

    """
    import warnings
    from astropy.utils.exceptions import AstropyWarning

    warnings.filterwarnings("ignore", category=AstropyWarning) # Unless you want a bright yellow message about the header


    try:
        with fits.open(fits_path, memmap=False) as hdul:
            image_data = hdul[0].data

            if image_data is None or getattr(image_data, "size", 0) == 0:
                print(f"FITS file {fits_path} is completely empty.")
                return False

            # Replace NaNs with zeros for processing
            image_data = np.nan_to_num(image_data)

            # Count the fraction of nonzero pixels
            nonzero_pixels = np.sum(image_data > min_intensity_threshold)
            total_pixels = image_data.size
            nonzero_fraction = nonzero_pixels / total_pixels

            if verbose:
                print(f"\nChecking FITS: {fits_path}")
                print(f"   - Nonzero pixel fraction: {nonzero_fraction:.2%} (min required: {min_nonzero_fraction:.2%})\n")

            # Reject images that are mostly empty
            if nonzero_fraction < min_nonzero_fraction:
                print(f"Image is mostly empty: Only {nonzero_fraction:.2%} nonzero pixels.\n")
                return False

            return True  # Image is considered useful

    except Exception as e:
        if verbose:
            print(f"Error checking FITS validity: {e}")
        return False


def emit_latex(
        latex_str: str,
        save_tex: bool = True,
        print_tex: bool = False,
        save_path: str | os.PathLike | None = None,
    ) -> None:
    """
    Print the LaTeX string and optionally save it to a .tex file.

    Parameters  
    ----------
    latex_str : str
        The LaTeX table text to print (and optionally save).
    save_tex : bool
        - None/False: don't write a file
        - True: write to `save_path`
    print_tex : bool
        If True, print the LaTeX string to the console.
    save_path : str | None | os.PathLike
        Fallback path used when save_tex is True.
    """
    if print_tex:
        print(latex_str)

    # Optionally save
    if save_tex:
        with open(save_path, "w") as f:
            f.write(latex_str)
        print(f"[saved] LaTeX written to {save_path}")


# ------------------------------------
# Plotting Utilities
# ------------------------------------

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


def finalize_figure(
        fig: plt.Figure,
        show_plots: bool = False,
        save_plots: bool = False,
        save_path: str | None | os.PathLike = None,
        filename: str | None = None
    ) -> None:
    """
    Save and/or show a figure, then close it so it can't pop up later.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to finalize.
    show_plots : bool
        If True, display the figure interactively.
    save_plots : bool
        If True, save the figure to `save_path`.
    save_path : str | None | os.PathLike
        Path to save the figure (PDF/PNG). Ignored if None or save_plots=False.
    filename : str or None
        Filename to use if save_path is a directory.

    Cases
    -----
    - save_plots and show_plots: save, show, then close
    - save_plots only:           save, then close
    - show_plots only:           show, then close
    - neither:                   do nothing (caller manages the figure)
    """

    save_file = resolve_save_path(save_plots, save_path, filename)
    if save_file:
        fig.savefig(save_file, dpi=450, bbox_inches="tight")

    if show_plots:
        try:
            fig.canvas.draw_idle()
        except Exception:
            pass
        plt.show()

    if save_plots or show_plots:
        plt.close(fig)


def resolve_save_path(
        save_plots: bool | None = None,
        save_path: str | None = None,
        filename: str | None = None
    ) -> str | None:
    """
    Determine the full save path for a file based on user preferences.

    Parameters
    ----------
    save_plots : bool
        Whether to save plots.
    save_path : str or None
        Directory or full path to save the file.
    filename : str or None
        Filename to use if save_path is a directory.

    Returns
    -------
    str or None
        Full path to save the file, or None if not saving.
    """
    if save_plots and save_path is not None:
        return os.path.join(save_path, filename) if os.path.isdir(save_path) else save_path
    return None


# ------------------------------------
# Unit Conversion and Cosmology Utilities
# ------------------------------------

def angular_sep(
        ra1: float,
        dec1: float,
        ra2: float,
        dec2: float
    ) -> float:
    """
    Computes angular separation in degrees using astropy's SkyCoord.

    Parameters
    ----------
    ra1, dec1 : float
        Coordinates of the first point (RA, Dec).
    ra2, dec2 : float
        Coordinates of the second point (RA, Dec).

    Returns
    -------
    float
        Angular separation in degrees.
    """
    c1 = SkyCoord(ra=ra1*u.deg, dec=dec1*u.deg)
    c2 = SkyCoord(ra=ra2*u.deg, dec=dec2*u.deg)
    sep = c1.separation(c2)

    return sep.deg


def redshift_to_comoving_distance(z: float | np.ndarray) -> float | np.ndarray:
    """
    Convert redshift to comoving distance using the Planck18 cosmology.
    
    Parameters
    ----------
    z : float | np.ndarray
        Redshift values
    
    Returns
    -------
    comoving_distance : float | np.ndarray
        Comoving distances in Mpc
    """
    comoving_distance = COSMO.comoving_distance(z).to(u.Mpc).value  # Convert to Mpc

    return comoving_distance


def redshift_to_proper_distance(z: float | np.ndarray) -> float | np.ndarray:
    """
    Convert redshift to proper distance at the time of emission using Planck18 cosmology.
    
    Parameters
    ----------
    z : float | np.ndarray
        Redshift values
    
    Returns
    -------
    D_A : float | np.ndarray
        Proper distances in Mpc
    """

    return COSMO.angular_diameter_distance(z).to(u.kpc).value  # Convert to kpc


def arcsec_to_pixel_std(
        fwhm_arcsec: float,
        wcs: WCS
    ) -> float:
    """
    Convert an FWHM in arcseconds to Gaussian stddev in pixels, using WCS.

    Parameters
    ----------
    fwhm_arcsec : float
        Desired smoothing FWHM in arcseconds.
    wcs : astropy.wcs.WCS
        WCS object for your image.

    Returns
    -------
    sigma_pixels : float
        Standard deviation in pixels for Gaussian kernel.
    """

    # Pixel scale in arcsec/pixel
    pixscale = np.abs(wcs.proj_plane_pixel_scales()[0]).to_value(u.arcsec)

    fwhm_pixels = fwhm_arcsec / pixscale
    sigma_pixels = float(fwhm_pixels / 2.355)  # Convert FWHM to sigma


    return sigma_pixels

# ------------------------------------
# Image Processing Utilities
# ------------------------------------

def fill_holes(
        img: np.ndarray,
        threshold_ratio: float = 0.1,
        filter_sizes: list[int] | None = None,
        plot: bool = False,
        exact: bool = False
    ):
    """
    Fill holes in an image based on a threshold ratio and optional filtering.

    Parameters
    ----------
    img : np.ndarray
        Input image array.
    threshold_ratio : float, optional
        Ratio to determine holes based on local mean [default: 0.1].
    filter_sizes : list[int] or None, optional
        Sizes of filters to apply for local mean calculation [default: None].
    plot : bool, optional
        Whether to plot intermediate results [default: False].
    exact : bool, optional
        If True, holes are exactly where img == 0 [default: False].

    Returns
    -------
    np.ndarray
        Image with holes filled.
    """
    if filter_sizes is None:
        filter_sizes = [5, 2]
    elif isinstance(filter_sizes, int):
        filter_sizes = [filter_sizes]

    working_img = np.array(img, copy=True)

    if exact:
        mask = img == 0
    else:
        local_mean = uniform_filter(img, size=filter_sizes[0], mode='mirror')
        mask = img < (threshold_ratio * local_mean)

    for filter_size in filter_sizes:
        
        local_mean = uniform_filter(working_img, size=filter_size, mode='mirror')
        adaptive_mask = working_img < (threshold_ratio * local_mean)
        img_filled = inpaint_biharmonic(working_img, adaptive_mask.astype(bool), channel_axis=None)
        working_img[adaptive_mask] = img_filled[adaptive_mask]
        working_img = np.nan_to_num(working_img, nan=0.0)

    final_img = np.array(img, copy=True)
    final_img[mask] = working_img[mask]
    final_img = np.nan_to_num(final_img, nan=0.0)

    return final_img


def smoothing(
        img: np.ndarray,
        kernel_std: float,
        filter_sizes: list[int] | None = None,
        fill: bool = True
    ) -> np.ndarray:
    """
    Smooth an image with a Gaussian kernel, optionally filling holes first.

    Parameters
    ----------
    img : np.ndarray
        Input image array.
    kernel_std : float
        Standard deviation for Gaussian kernel.
    filter_sizes : list[int] or None, optional
        Sizes of filters to use for hole filling [default: None].
    fill : bool, optional
        Whether to fill holes before smoothing [default: True].

    Returns
    -------
    np.ndarray
        Smoothed image.
    """

    if fill:
        filled_img = fill_holes(img, filter_sizes=filter_sizes)

    else:
        filled_img = np.nan_to_num(img, nan=0.0)

    # Smooth the result
    g = Gaussian2DKernel(x_stddev=kernel_std)
    smoothed_image = convolve(filled_img, g)

    return smoothed_image


def add_scalebar(
        ax: plt.Axes,
        wcs: WCS,
        redshift: float,
        scalebar_arcmin: float = 1.0,
        color: str = 'white',
        fontsize: float = 12,
        **kwargs
    ) -> None:
    """
    Adds a scale bar to a WCS axis, labeling both physical (kpc) and angular (arcmin) units.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to add scale bar to.
    wcs : astropy.wcs.WCS
        WCS for coordinate transforms.
    redshift : float
        Redshift for scale calculation.
    scalebar_arcmin : float, optional
        Length of scale bar in arcminutes (default: 1.0).
    color : str, optional
        Scale bar and text color (default: 'white').
    fontsize : float, optional
        Font size for scale bar labels (default: 12).
    **kwargs
        Additional keyword arguments; recognized:
          Scalebar kwargs:
            - scalebar_arcmin : float
            - scalebar_color : str
            - scalebar_fontsize : float
            - scalebar_* : any ax.text() kwarg
            - scalebar_kwargs : dict
          Plot kwargs:
            - scalebar_plot_color : str
            - scalebar_plot_lw : float
            - scalebar_plot_* : any ax.plot() kwarg
            - scalebar_plot_kwargs : dict

        These take precedence over the direct arguments if supplied.

    Returns
    -------
    None

    """

    # Namespaced kwarg overrides (scalebar_* takes precedence if supplied)
    scalebar_kwargs = pop_prefixed_kwargs(kwargs, 'scalebar')
    scalebar_arcmin = scalebar_kwargs.get('arcmin', scalebar_arcmin)
    color = scalebar_kwargs.get('color', color)
    fontsize = scalebar_kwargs.get('fontsize', fontsize)

    # Accept extra plot kwargs for the bar itself (e.g., lw, linestyle)
    plot_kwargs = scalebar_kwargs.get('plot_kwargs', {}).copy()
    plot_kwargs.update(pop_prefixed_kwargs(scalebar_kwargs, 'plot'))
    plot_color = plot_kwargs.pop('color', color)
    plot_lw = plot_kwargs.pop('lw', 3)

    # Convert scalebar length to degrees, radians, and kpc
    scalebar_deg = scalebar_arcmin / 60.0
    scalebar_radian = scalebar_arcmin * u.arcmin.to(u.rad)
    DA_kpc = redshift_to_proper_distance(redshift)
    scalebar_kpc = DA_kpc * scalebar_radian

    # Choose bar location
    start_pixel_x, start_pixel_y = 40, 40
    start_world = wcs.pixel_to_world(start_pixel_x, start_pixel_y)

    # Draw scale bar
    ax.plot(
        [start_world.ra.deg, start_world.ra.deg - scalebar_deg / np.cos(start_world.dec.radian)],
        [start_world.dec.deg, start_world.dec.deg],
        color=plot_color,
        lw=plot_lw,
        transform=ax.get_transform('icrs'),
        zorder=100,
        **plot_kwargs,
    )

    # Add physical label
    ax.text(
        start_world.ra.deg - scalebar_deg/2 / np.cos(start_world.dec.radian),
        start_world.dec.deg + scalebar_deg/20,
        f"{int(np.round(scalebar_kpc, -1))} kpc",
        color=color,
        ha='center',
        va='bottom',
        transform=ax.get_transform('icrs'),
        fontsize=fontsize,
        zorder=100,
        **scalebar_kwargs,
    )
    # Add angular label
    ax.text(
        start_world.ra.deg - scalebar_deg/2 / np.cos(start_world.dec.radian),
        start_world.dec.deg - scalebar_deg/20,
        f"{scalebar_arcmin:.0f}'",
        color=color,
        ha='center',
        va='top',
        transform=ax.get_transform('icrs'),
        fontsize=fontsize,
        zorder=100,
        **scalebar_kwargs,
    )


def add_xray_contours(
        ax: plt.Axes,
        xray_fits_file: str,
        optical_data: np.ndarray,
        wcs_optical: WCS,
        levels: tuple[float, float, float] = (0.5, 0.0, 12.0),
        psf: float = 8.0,
        color: str = 'tab:red',
        alpha: float = 1.0,
        linewidth: float = 1.2,
        cluster: "Cluster" | None = None,
        **kwargs
    ):

    """
    Adds X-ray surface brightness contours to a given axis, aligned to an optical WCS.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis on which to plot X-ray contours (should be WCS-aware).
    xray_fits_file : str
        Path to the X-ray FITS image file.
    optical_data : np.ndarray
        Optical image data array (for defining contours in the correct FOV).
    wcs_optical : astropy.wcs.WCS
        WCS object of the optical image (for alignment).
    levels : tuple, optional
        Requested X-ray contour levels (std from bottom, std from top, steps) (default: (0.5, 0.0, 12.0)).
    psf : float, optional
        Smoothing kernel FWHM in arcsec (default: 8.0).
    color : str, optional
        Contour color (default: 'tab:red').
    alpha : float, optional
        Contour transparency (default: 1.0).
    linewidth : float, optional
        Contour line width (default: 1.2).
    cluster : Cluster, optional
        Cluster object containing default contour levels and PSF.
    **kwargs
        Additional keyword arguments. Recognized namespaced overrides:
            - xray_levels : tuple
            - xray_psf : float
            - xray_color : str
            - xray_alpha : float
            - xray_linewidth : float
            - xray_* : any ax.contour() kwarg
            - xray_kwargs : dict

        If provided, these take precedence over the corresponding function arguments.
        Any additional kwargs are passed to `ax.contour()`.

    Returns
    -------
    contour_artist : QuadContourSet
        The matplotlib contour artist returned by `ax.contour()`.

    Notes
    -----
    - Uses Gaussian kernel smoothing based on PSF.
    - Namespaced kwargs enable flexible control in pipeline or multi-overlay figures.
    - Contour levels are calculated with respect to the optical FOV using `define_contours_fov`.
    """

    # Namespaced kwarg overrides (xray_* takes precedence if supplied)
    xray_kwargs = pop_prefixed_kwargs(kwargs, 'xray')
    user_levels = xray_kwargs.pop('levels', kwargs.get('levels', None))
    user_psf    = xray_kwargs.pop('psf', kwargs.get('psf', None))
    color = xray_kwargs.pop('color', color)
    alpha = xray_kwargs.pop('alpha', alpha)
    linewidth = xray_kwargs.pop('linewidth', linewidth)
    linewidth = xray_kwargs.pop('linewidths', linewidth)

    # Levels: user override > cluster > default
    if user_levels is not None and isinstance(user_levels, tuple) and len(user_levels) == 3:
        contour_levels = user_levels
    elif cluster is not None and hasattr(cluster, "contour_levels_tuple"):
        try:
            contour_levels = cluster.contour_levels_tuple
        except Exception as e:
            print(f"[WARNING] cluster.contour_levels_tuple is not a valid tuple: {e}")
            # WARNING: This could be a string, tuple, or anything
            contour_levels = getattr(cluster, "contour_levels", levels)
    else:
        contour_levels = levels

    # PSF: user override > cluster > default
    if user_psf is not None:
        contour_psf = user_psf
    elif cluster is not None and hasattr(cluster, "psf"):
        contour_psf = cluster.psf
    else:
        contour_psf = psf

    # Load X-ray data and WCS
    xray_data = fits.getdata(xray_fits_file)
    wcs_xray = WCS(fits.getheader(xray_fits_file), naxis=2)

    # Smooth X-ray data and define contour levels
    kernel_std = arcsec_to_pixel_std(contour_psf, wcs_xray)
    xray_smoothed = smoothing(xray_data, kernel_std)
    xray_levels = define_contours_fov( xray_smoothed, optical_data, wcs_optical, wcs_xray, contour_levels)

    # Plot contours
    print("X-ray contour levels:", xray_levels)
    contour_artist = ax.contour(
        xray_smoothed,
        transform=ax.get_transform(wcs_xray),
        levels=xray_levels,
        colors=color,
        alpha=alpha,
        linewidths=linewidth,
        **xray_kwargs
    )

    return contour_artist


def add_density_contours(
    ax: plt.Axes,
    photometric_file: str,
    bandwidth: float = 0.1,
    levels: int = 12,
    skip: int = 2,
    color: str = 'tab:blue',
    alpha: float = 1.0,
    linewidth: float = 1.2,
    cluster: "Cluster" | None = None,
    **kwargs: Any,
):
    """
    Adds photometric density contours to a given matplotlib axis using kernel density estimation (KDE).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis on which to plot density contours (should be WCS-aware).
    photometric_file : str
        Path to the photometric CSV file with source coordinates and luminosities.
    bandwidth : float, optional
        Bandwidth for the KDE smoothing (default: 0.1).
    levels : int, optional
        Number of density contour levels to compute (default: 12).
    skip : int, optional
        Number of lowest contour levels to skip (default: 2).
    color : str, optional
        Contour color (default: 'tab:blue').
    alpha : float, optional
        Contour transparency (default: 1.0).
    linewidth : float, optional
        Contour line width (default: 0.5).
    **kwargs
        Additional keyword arguments. Recognized namespaced overrides:
            - density_bandwidth : float
            - density_levels    : int
            - density_skip      : int
            - density_color     : str
            - density_alpha     : float
            - density_linewidth : float
            - density_*         : any ax.contour() kwarg
            - density_kwargs    : dict

        If provided, these take precedence over the corresponding function arguments.
        Any other kwargs are passed through to `ax.contour()`.

    Returns
    -------
    contour_artist : QuadContourSet
        The matplotlib contour artist returned by `ax.contour()`.

    Notes
    -----
    - This function supports both direct argument and kwargs-based overrides.
    - Uses KDE with weighted luminosities for adaptive density estimation.
    """


    # Namespaced kwarg overrides (density_* takes precedence if supplied)
    density_kwargs = pop_prefixed_kwargs(kwargs, 'density')
    user_bandwidth = density_kwargs.pop('bandwidth', kwargs.get('bandwidth', None))
    user_levels    = density_kwargs.pop('levels', kwargs.get('levels', None))
    user_skip      = density_kwargs.pop('skip', kwargs.get('skip', None))
    color          = density_kwargs.pop('color', color)
    alpha          = density_kwargs.pop('alpha', alpha)
    linewidth      = density_kwargs.pop('linewidth', linewidth)
    linewidth      = density_kwargs.pop('linewidths', linewidth)

    # Priority: user override > cluster > default
    if user_bandwidth is not None:
        contour_bandwidth = user_bandwidth
    elif cluster is not None and hasattr(cluster, "bandwidth"):
        contour_bandwidth = cluster.bandwidth
    else:
        contour_bandwidth = bandwidth

    if user_levels is not None and isinstance(user_levels, int):
        contour_levels = user_levels
    elif cluster is not None and hasattr(cluster, "phot_levels"):
        contour_levels = cluster.phot_levels
    else:
        contour_levels = levels

    if user_skip is not None and isinstance(user_skip, int):
        contour_skip = user_skip
    elif cluster is not None and hasattr(cluster, "phot_skip"):
        contour_skip = cluster.phot_skip
    else:
        contour_skip = skip

    # Load photometric source positions and weights
    RA_phot, Dec_phot, Lum_phot = load_photo_coords(photometric_file)
    ra = np.asarray(RA_phot)
    dec = np.asarray(Dec_phot)
    weights = np.asarray(Lum_phot)

    ra0, dec0 = float(np.median(ra)), float(np.median(dec))
    cosd0 = np.cos(np.deg2rad(dec0))

    # small-angle offsets (degrees)
    x = (ra - ra0) * cosd0
    y = (dec - dec0)

    # Kernel Density Estimation in RA/Dec
    xy = np.vstack([x, y])
    kde = gaussian_kde(xy, bw_method=contour_bandwidth, weights=weights)

    ra_grid = np.linspace(x.min(), x.max(), 300)
    dec_grid = np.linspace(y.min(), y.max(), 300)
    ra_mesh, dec_mesh = np.meshgrid(ra_grid, dec_grid)
    density = kde(np.vstack([ra_mesh.ravel(), dec_mesh.ravel()])).reshape(ra_mesh.shape)

    ra_mesh = ra_mesh / cosd0 + ra0
    dec_mesh = dec_mesh + dec0
    # Compute contour levels and plot
    density_levels = np.linspace(density.min(), density.max(), int(round(contour_levels)))[int(round(contour_skip)):]
    contour_artist = ax.contour(
        ra_mesh,
        dec_mesh,
        density,
        levels=density_levels,
        colors=color,
        alpha=alpha,
        linewidths=linewidth,
        transform=ax.get_transform('icrs'),
        **density_kwargs,
    )
    return contour_artist


def define_contours_fov(
        z: np.ndarray,
        optical_data: np.ndarray,
        wcs_optical: WCS,
        contour_wcs: WCS,
        levels: tuple[float, float, float],
        pedestal: float | None = None,
        logspace: bool = False,
        verbose: bool = False
    ) -> np.ndarray:
    """
    Defines contour levels based on the field of view (FOV) of the optical image.

    Parameters
    ----------
    z : np.ndarray
        X-ray data to define contours on.
    optical_data : np.ndarray
        Optical image data to determine the FOV.
    wcs_optical : WCS object
        WCS object for the optical image.
    contour_wcs : WCS object
        WCS object for the X-ray data.
    levels : tuple[float, float, float]
        Tuple containing (min_level, max_level, num_levels) for contour definition.
    pedestal : float | None, optional
        If provided, overrides the mean intensity for contour definition.
    logspace : bool, optional
        If True, uses logarithmic spacing for contour levels.
    verbose : bool, optional
        If True, prints detailed debug information.

    Returns
    ---------
    level : np.ndarray
        Array of contour levels defined within the optical FOV.

    """

    # Get optical image shape from the provided data
    shape = optical_data.shape

    if len(shape) == 3:
        # Channels will usually be 3 or 4 (RGB or RGBA)
        if shape[0] in (3, 4):
            # Raw image: (C, Y, X)
            nchan, ny_opt, nx_opt = shape
        elif shape[2] in (3, 4):
            # Transposed for display: (Y, X, C)
            ny_opt, nx_opt, nchan = shape
        else:
            # Unusual case: fallback, assume middle is Y, last is X
            _, ny_opt, nx_opt = shape
    elif len(shape) == 2:
        # Grayscale image: (Y, X)
        ny_opt, nx_opt = shape
        nchan = 1
    else:
        raise ValueError(f"Unexpected image shape: {shape}")

    if verbose:
        print(f"\nOptical image: ny={ny_opt}, nx={nx_opt}, channels={nchan}")

    # Find RA/Dec bounds of the optical image using WCS
    ra_corners: list[float] = []
    dec_corners: list[float] = []
    for x, y in [(0, 0), (0, ny_opt - 1), (nx_opt - 1, 0), (nx_opt - 1, ny_opt - 1)]:
        coord = wcs_optical.pixel_to_world(x, y)
        ra_corners.append(coord.ra.deg)
        dec_corners.append(coord.dec.deg)

    # Determine RA/Dec range
    ra_min, ra_max = min(ra_corners), max(ra_corners)
    dec_min, dec_max = min(dec_corners), max(dec_corners)

    # Convert X-ray pixels to world coordinates
    y_xray, x_xray = np.meshgrid(np.arange(z.shape[0]), np.arange(z.shape[1]), indexing='ij')
    world_coords = contour_wcs.pixel_to_world(x_xray, y_xray)

    # Mask X-ray pixels that are outside the optical FOV
    mask = (world_coords.ra.deg >= ra_min) & (world_coords.ra.deg <= ra_max) & \
           (world_coords.dec.deg >= dec_min) & (world_coords.dec.deg <= dec_max)

    # Extract only the X-ray data within the optical FOV
    z_fov = z[mask]

    if verbose:
        # Debug visualization (optional)
        plt.figure(figsize=(10, 10))
        plt.imshow(z, origin='lower', cmap='magma')
        plt.contour(mask, levels=[0.5], colors='cyan', linewidths=1)  # Overlay optical FOV
        plt.title("X-ray Data with Optical FOV Mask")
        plt.show()

    # Compute statistics only in this region
    if z_fov.size == 0:
        raise ValueError("No X-ray data falls within the optical field of view.")

    mean = float(z_fov.mean())
    std = float(z_fov.std())
    if verbose:
        print("\n-------- X-ray Contours --------")
        print(f"    Mean Intensity: {mean}")
        print(f"Standard Deviation: {std}")
        print(f"     Max Intensity: {z_fov.max()}")
        print(f"     Min Intensity: {z_fov.min()}")
    if pedestal:
        mean = pedestal  # Override mean if provided

    # Define contour levels
    vmin = float(z_fov.min() + levels[0] * std)
    vmax = float(z_fov.max() - levels[1] * std)

    if verbose:
        print(f"       Contour Min: {vmin}")
        print(f"       Contour Max: {vmax}")

    num_levels = int(round(levels[2]))

    if logspace:
        level = np.geomspace(vmin, vmax, num_levels)
    else:
        level = np.linspace(vmin, vmax, num_levels)

    return level


def overlay_bcg_markers(
    ax: plt.Axes,
    bcg_csv: str,
    background: str = "light",    # or "dark"
    zorder: int = 11,
    legend: bool = True,
    legend_loc: str = "upper left"
) -> plt.Axes:
    """
    Overlay BCG markers on a WCS-aware matplotlib axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot markers on.
    bcg_csv : str
        Path to the BCG CSV file.
    background : str
        "light" or "dark" (controls primary marker color).
    zorder : int
        Z-order for plotting.
    legend : bool
        Whether to add a legend.
    legend_loc : str
        Legend location.
    """
    # Load BCGs
    df = pd.read_csv(bcg_csv)
    bcg_colors = [
        'white',
        'tab:green', 
        'tab:purple',
        'tab:cyan',
        'tab:pink',
        'gold',
        'tab:orange',
        'tab:green',
        'tab:purple',
        'tab:pink'
    ]
    bcg_labels = ['A', 'B', 'C', 'D', 'E', 'F']
    
    # Optionally swap white/black for BCG 1 depending on background
    if background == "dark":
        bcg_colors[0] = "white"
    elif background == "light":
        bcg_colors[0] = "black"
    
    handles = []
    for i, row in df.iterrows():
        ra = float(row['RA'])
        dec = float(row['Dec'])
        z_bcg = row['z'] if not pd.isnull(row['z']) else None
        
        color = bcg_colors[i % len(bcg_colors)]
        if i < 5:
            label = f"BCG {i+1} (z = {'unknown' if z_bcg is None else f'{z_bcg:.3f}'})"
            marker = '*'
        else:
            label = f"BCG {bcg_labels[i-5]} (z = {'unknown' if z_bcg is None else f'{z_bcg:.3f}'})"
            marker = 'o'
        
        # Plot
        sc = ax.scatter(
            ra,
            dec,
            marker=marker,
            edgecolor=color,
            facecolor='none',
            s=200,
            zorder=zorder,
            transform=ax.get_transform('icrs'),
            label=label
        )
        # For manual legend
        if legend:
            handles.append(sc)
    
    if legend and handles:
        ax.legend(loc=legend_loc, fontsize=10, frameon=True)
    
    return ax

def plot_optical(
    optical_image_file,
    cluster=None,
    redshift=None,
    fig=None,
    ax=None,
    show_plots=True,
    save_plots=True,
    save_path=None,
    xray_fits_file=None,         
    photometric_file=None,       
    bcg_file = None,
    bcg_background = "light",
    legend_loc = "upper right",
    show_optical=True,
    show_legend=True,
    show_scalebar=True,
    **kwargs
):
    """
    Plot an optical image with WCS, axis labels, and a scale bar.

    Parameters
    ----------
    optical_image_file : str
        Path to FITS image.
    cluster : Cluster object, optional
        Cluster object to use for default redshift (if redshift not given).
    redshift : float, optional
        Redshift for scale bar. If None, uses cluster.redshift if cluster provided.
    fig, ax : matplotlib objects, optional
        Existing figure/axes. If None, creates new.
    show_plots : bool, optional
        If True, displays interactively.
    save_plots : bool, optional
        If True, saves figure to file.
    save_path : str or None
        If provided, saves figure to this path.
    xray_fits_file : str or None
        If provided, overlays X-ray contours from this FITS file.
    photometric_file : str or None
        If provided, overlays density contours from this photometric CSV file.
    bcg_file : str or None
        If provided, overlays BCG markers from this CSV file.
    bcg_background : str, optional
        "light" or "dark" - choose marker color scheme based on background.
    legend_loc : str, optional
        Location of legend (default: "upper right").
    show_optical : bool, optional
        If True, shows optical image; if False, shows blank background.
    **kwargs
        Additional options, including:
            - X-ray: xray_levels, xray_psf, xray_color, xray_alpha, xray_linewidth
            - Density: density_bandwidth, density_levels, density_skip, density_color, density_alpha, density_linewidth
            - Scalebar: scalebar_arcmin, scalebar_color, scalebar_fontsize

    Returns
    -------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        The figure and axis.
    """

    # optical_kwargs = pop_prefixed_kwargs(kwargs, 'optical')
    # show_scalebar = optical_kwargs.get('show_scalebar', True)

    # --- Load optical image and WCS ---
    with fits.open(optical_image_file) as hdul:
        optical_image_data = hdul[0].data
        wcs_optical = WCS(hdul[0].header, naxis=2)

    img = np.transpose(optical_image_data, (1, 2, 0)) if optical_image_data.ndim == 3 else optical_image_data

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': wcs_optical})

    # --- Show optical image or blank background ---
    if show_optical:
        bcg_background = "dark"
    else:
        bcg_background = "light"
        # Set blank background
        if img.dtype.kind == 'f':
            img[...] = 1.0
        else:
            img[...] = 255
        
    ax.imshow(img, origin='lower')

    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_ylim())
    ax.set_xlabel('R.A.')
    ax.set_ylabel('Decl.')
    ax.set_aspect('equal')

    # --- Add scale bar ---
    if redshift is None and cluster is not None:
        redshift = cluster.redshift
    if show_scalebar:
        add_scalebar(ax, wcs_optical, redshift, **kwargs)

    # --- Overlay X-ray contours ---
    if xray_fits_file is not None:
        add_xray_contours(
            ax=ax,
            xray_fits_file=xray_fits_file,
            optical_data=img, 
            wcs_optical=wcs_optical,
            cluster=cluster,
            **kwargs
        )
        
    # --- Overlay density contours ---
    if photometric_file is not None:
        add_density_contours(
            ax=ax,
            photometric_file=photometric_file,
            cluster=cluster,
            **kwargs
        )

    # --- Overlay BCG markers ---
    if bcg_file is not None:
        overlay_bcg_markers(ax, bcg_file, background=bcg_background)

    # --- Compose legend ---
    custom_handles = []
    if xray_fits_file is not None:
        custom_handles.append(Line2D([0], [0], color=kwargs.get('xray_color', 'tab:red'), lw=kwargs.get('xray_linewidth', 1.2), label='X-ray contours'))
    if photometric_file is not None:
        custom_handles.append(Line2D([0], [0], color=kwargs.get('density_color', 'tab:blue'), lw=kwargs.get('density_linewidth', 1.2), label='Density contours'))

    # Get handles/labels from actual artists (i.e., BCGs)
    handles, labels = ax.get_legend_handles_labels()

    # Combine: contours first, then BCGs
    final_handles = custom_handles + handles
    final_labels = [h.get_label() for h in custom_handles] + labels
    legend=ax.legend(final_handles, final_labels, loc=legend_loc, fontsize=10, frameon=True, framealpha=0.9, facecolor='white')
    legend.set_zorder(200)


    if not show_legend:
        ax.legend_.remove()

    if show_plots or save_plots:
        plt.tight_layout()
        finalize_figure(fig, save_path=save_path, save_plots=save_plots, show_plots=show_plots, filename="optical_image.pdf")



    if show_legend:
        return fig, ax
    else:
        return fig, ax, final_handles, final_labels

# ------------------------------------
# CMD Utilities
# ------------------------------------

def get_skycoord(df: pd.DataFrame) -> SkyCoord:
    """
    Convert DataFrame with RA/Dec columns to SkyCoord object.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'RA' and 'Dec' columns in degrees.

    Returns
    -------
    coords : SkyCoord
        SkyCoord object with the coordinates from the DataFrame.
    """

    # TODO: This function can be deleted and switched to skycoord_from_df() for consistency, but keeping it for now since it's used in some places and is more concise.    
    return SkyCoord(ra=df["RA"].values * u.deg, dec=df["Dec"].values * u.deg)


def get_color_mag_functions(
        color_type: str
    ) -> tuple[list[str], str, str, callable, callable]:
    """
    Returns required columns, labels, and functions for color and magnitude based on color_type.

    Parameters
    ----------
    color_type : str
        Color combination to use: "g-r", "r-i", or "g-i".

    Returns
    -------
    required_cols : list of str
        List of required magnitude columns for the chosen color_type.
    color_label : str
        Label for the color axis.
    mag_label : str
        Label for the magnitude axis.
    color_func : callable
        Function to compute color from a DataFrame.
    mag_func : callable
        Function to compute magnitude from a DataFrame.
    """


    if color_type == 'g-r' or color_type == 'gr':
        required_cols = ['gmag', 'rmag']
        color_label = 'g - r'
        mag_label = 'r'
        color_func = lambda df: df['gmag'] - df['rmag']  # noqa: E731
        mag_func = lambda df: df['rmag']  # noqa: E731
        return required_cols, color_label, mag_label, color_func, mag_func
    
    if color_type == 'r-i' or color_type == 'ri':
        required_cols = ['rmag', 'imag']
        color_label = 'r - i'
        mag_label = 'i'
        color_func = lambda df: df['rmag'] - df['imag']  # noqa: E731
        mag_func = lambda df: df['imag']  # noqa: E731
        return required_cols, color_label, mag_label, color_func, mag_func
    
    if color_type == 'g-i' or color_type == 'gi':
        required_cols = ['gmag', 'imag']
        color_label = 'g - i'
        mag_label = 'i'
        color_func = lambda df: df['gmag'] - df['imag'] # noqa: E731
        mag_func = lambda df: df['imag'] # noqa: E731
        return required_cols, color_label, mag_label, color_func, mag_func
    
    raise ValueError(f"Unknown color_type '{color_type}'. Supported types: 'g-r', 'gr', 'r-i', 'ri', 'g-i', 'gi'.")


def split_members_by_spec(
    members_df: pd.DataFrame,
    *,
    z_col: str = 'z',
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (spec_members, phot_members) from the final member catalog.
    
    Parameters
    ----------
    members_df : pd.DataFrame
        DataFrame containing the final member catalog.
    z_col : str, optional
        Name of the redshift column [default: Z_COL].

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Tuple containing (spec_members, phot_members).
    """
    spec_mask = members_df[z_col].notna()
    phot_mask = ~spec_mask

    spec_members = members_df.loc[spec_mask].copy()
    phot_members = members_df.loc[phot_mask].copy()
    return spec_members, phot_members


# def query_redmapper_bcg_candidates(
#     coord: SkyCoord,
#     *,
#     radius_arcmin: float = 5.0,
#     verbose: bool = True,
# ) -> list[tuple[float, float, float | None, float | None]]:
#     """
#     Query redMaPPer via VizieR near a given coordinate and return up to 5 BCG-center candidates.

#     Notes
#     -----
#     - This is a *positional* query (no name matching).
#     - If multiple clusters are returned, we select the closest on-sky.
#       (In practice, this is almost always fine for small radii.)

#     Parameters
#     ----------
#     coord : SkyCoord
#         Target coordinate (typically the cluster center).
#     radius_arcmin : float
#         Search radius around `coord`.
#     verbose : bool
#         Print basic diagnostics.

#     Returns
#     -------
#     list of tuple
#         Candidate list in (RA, Dec, z, P) format. z is None here; downstream
#         matching to your spectroscopic catalog can fill it in.
#     """
#     if coord is None:
#         return []

#     # TODO: Add support for DES redMaPPer catalog
#     # VizieR redMaPPer SDSS DR8 cluster catalog.
#     catalog = "J/ApJS/224/1/cat_dr8"

#     # Column names in this catalog (VizieR-style):
#     # - RAJ2000, DEJ2000 for cluster center
#     # - PCen0..PCen4 and RA0deg/DE0deg..RA4deg/DE4deg for centering candidates
#     cols = [
#         "Name", "RAJ2000", "DEJ2000",
#         "PCen0", "PCen1", "PCen2", "PCen3", "PCen4",
#         "RA0deg", "DE0deg", "RA1deg", "DE1deg", "RA2deg", "DE2deg", "RA3deg", "DE3deg", "RA4deg", "DE4deg",
#     ]

#     viz = Vizier(columns=cols)
#     viz.ROW_LIMIT = 50

#     try:
#         tabs = viz.query_region(coord, radius=radius_arcmin * u.arcmin, catalog=catalog)
#     except Exception as e:
#         if verbose:
#             print(f"redMaPPer VizieR query failed: {e}")
#         return []

#     if not tabs or len(tabs[0]) == 0:
#         if verbose:
#             print("No redMaPPer clusters found in search radius.")
#         return []

#     t = tabs[0]

#     # Choose the closest cluster (safe default; avoids a “richest vs closest” knob)
#     cl_coords = SkyCoord(t["RAJ2000"], t["DEJ2000"], unit=(u.deg, u.deg), frame="icrs")

#     idx = int(coord.separation(cl_coords).argmin())
#     row = t[idx]

#     if verbose:
#         sep = coord.separation(cl_coords[idx]).to(u.arcmin).value
#         name = str(row["Name"]) if "Name" in row.colnames else "unknown"
#         print(f"redMaPPer match: {name} (sep={sep:.2f} arcmin). Extracting PCen/RAi/DEi candidates...")

#     cands: list[tuple[float, float, float | None, float | None]] = []

#     for i in range(5):
#         ra_key = f"RA{i}deg"
#         de_key = f"DE{i}deg"
#         p_key = f"PCen{i}"

#         if ra_key not in row.colnames or de_key not in row.colnames:
#             continue

#         ra_i = row[ra_key]
#         de_i = row[de_key]
#         p_i = row[p_key] if p_key in row.colnames else np.nan

#         # Guard against masked/NaN
#         if ra_i is None or de_i is None:
#             continue

#         try:
#             ra_f = float(ra_i)
#             de_f = float(de_i)
#         except Exception:
#             continue

#         if not np.isfinite(ra_f) or not np.isfinite(de_f):
#             continue

#         p_f = None
#         try:
#             p_tmp = float(p_i)
#             p_f = p_tmp if np.isfinite(p_tmp) else None
#         except Exception:
#             p_f = None

#         cands.append((ra_f, de_f, None, p_f))

#     # Sort by probability (descending), with None at the end
#     cands.sort(key=lambda t: (-1.0 if t[3] is None else -t[3]))

#     if verbose and cands:
#         print(f"Recovered {len(cands)} redMaPPer BCG candidates.")

#     return cands
