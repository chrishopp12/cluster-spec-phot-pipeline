"""NED, SIMBAD, and redMaPPer queries for cluster resolution."""

from __future__ import annotations
import re

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.ipac.ned import Ned
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier

from cluster_pipeline.utils import to_float_or_none


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
        print(f"SIMBAD lookup failed ({type(e).__name__}): {e}")
        raise


def get_name(identifier: str) -> str:
    """Return the canonical cluster name for a given identifier.

    Parameters
    ----------
    identifier : str
        Cluster name or identifier string.

    Returns
    -------
    str
        The identifier string, stripped of whitespace.
    """
    if identifier is None:
        raise ValueError("No identifier provided.")
    return str(identifier).strip()


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
        print(f"NED lookup failed ({type(e).__name__}): {e}")

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
                    print(f"No redshift table found in NED ({type(e_inner).__name__}): {e_inner}")
    except Exception as e:
        print(f"NED redshift lookup failed ({type(e).__name__}): {e}")


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
        print(f"Simbad redshift lookup failed ({type(e).__name__}): {e}")

    # Fall back to BCGs if provided
    if BCGs is not None and len(BCGs) > 0:
        print(f"Falling back to BCG redshift for {cluster_name}...")
        first_bcg = BCGs[0]  # (ra, dec, z)
        if len(first_bcg) >= 3:
            z = first_bcg[2]
            print(f"Using BCG fallback: z = {z}")
            return float(z)

    raise ValueError(f"Could not find redshift for {cluster_name} in Simbad, NED, or BCG data.")


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
            print(f"redMaPPer VizieR query failed ({type(e).__name__}): {e}")
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

    # Print a nice table of cluster info
    cluster_info = get_redmapper_cluster_info(coord, radius_arcmin=radius_arcmin, verbose=verbose)
    if verbose:
        print(f"Cluster '{cluster_info['name']}': zlambda={cluster_info['zlambda']}, lambda={cluster_info['lambda']}")
        print("BCG candidates:")
        for i, (ra, dec, _, p) in enumerate(bcgs):
            print(f"  Candidate {i}: RA={ra}, Dec={dec}, Pcen={p}")

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
