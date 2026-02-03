#!/usr/bin/env python3
"""
run_cmd_pipeline.py

Cluster Analysis Master Pipeline
---------------------------------------------------------

This is the driver script for the photometric + spectroscopic portion of the
cluster pipeline. It orchestrates the following high-level steps:

1) Archival redshift querying + merging (archival_z_pipeline.py)
2) Archival photometry querying + merging (archival_phot_pipeline.py)
3) Catalog building / export (crossmatching, BCG matching, etc.) (make_catalogs_pipeline.py)
4) Color-magnitude (red sequence) fitting + member selection (color_magnitude_pipeline.py)
5) Standard plots (CMD, spatial, overlays) (color_magnitude_plotting.py)

Most of the science logic lives in the individual sub-pipelines. This script is
intended to remain lightweight and easy to run from the command line.

Requirements:
    - numpy
    - pandas
    - astropy
    - astroquery
    - scipy
    - matplotlib
    - my_utils (local)
    - cluster (local)
    - archival_z_pipeline (local)
    - archival_phot_pipeline (local)
    - make_catalogs_pipeline (local)
    - color_magnitude_pipeline (local)
    - color_magnitude_plotting (local)

Usage:
    python run_cmd_pipeline.py CLUSTER_ID [options] [cluster kwargs]

Examples:

Run full pipeline and save plots (default):

    python run_cmd_pipeline.py 1327 --show-plots False --save-plots True

Override z-range:

    python run_cmd_pipeline.py 1327 --zmin 0.30 --zmax 0.34

Provide CasJobs credentials via CLI or environment variables:

    export CASJOBS_USER="..."
    export CASJOBS_PW="..."
    python run_cmd_pipeline.py 1327


Options:
    --zmin             <float>   Minimum redshift for analysis. [default: z - 0.015]
    --zmax             <float>   Maximum redshift for analysis. [default: z + 0.015]
    --show-plots        <bool>    Show plots? [default: False]
    --save-plots        <bool>    Save plots? [default: True]
    --skip-redshifts    <bool>    Skip redshift pipeline? [default: False]
    --skip-photometry   <bool>    Skip photometry pipeline? [default: False]
    --skip-catalogs     <bool>    Skip catalog building pipeline? [default: False]
    --skip-cmd          <bool>    Skip color-magnitude pipeline? [default: False]
    --skip-plots        <bool>    Skip plotting utility? [default: False]

Cluster kwargs:
    --fov               <float>   Zoomed field of view in arcmin. [default: None]
    --fov-full          <float>   Full field of view in arcmin. [default: None]
    --ra-offset         <float>   RA offset in arcmin. [default: None]
    --dec-offset        <float>   Dec offset in arcmin. [default: None]
    --density-levels    <int>     Number of density contour levels. [default: None]
    --density-skip      <int>     Number of density levels to skip. [default: None]
    --density-bandwidth <float>   Bandwidth for density estimation. [default: None]

Downstream kwargs (JSON strings):
    --redshift-kwargs   <str>     JSON string of kwargs to pass to the redshift pipeline.
    --photometry-kwargs <str>     JSON string of kwargs to pass to the photometry pipeline.
    --catalog-kwargs    <str>     JSON string of kwargs to pass to the catalog building pipeline.
    --cmd-kwargs        <str>     JSON string of kwargs to pass to the color-magnitude pipeline.
    --plotting-kwargs   <str>     JSON string of kwargs to pass to the plotting utility.

"""
from __future__ import annotations

import os
import argparse
from getpass import getpass
import json

import numpy as np

from spec_phot_pipeline.archival_z_pipeline import run_redshift_pipeline
from spec_phot_pipeline.archival_phot_pipeline import run_photometry_pipeline
from spec_phot_pipeline.make_catalogs_pipeline import build_redshift_catalog
from spec_phot_pipeline.color_magnitude_pipeline import run_cmd_pipeline
from spec_phot_pipeline.color_magnitude_plotting import run_cluster_plots
from my_utils import str2bool
from cluster import Cluster


# ------------------------------------
# Defaults/ Constants
# ------------------------------------
DEFAULT_Z_PAD = 0.015

# ------------------------------------
# Helpers
# ------------------------------------

def _resolve_casjobs_credentials(
    user: str | None,
    password: str | None,
) -> tuple[str, str]:
    """Resolve CasJobs credentials from CLI args, env vars, or interactive prompt.
    
    Parameters
    ----------
    user : str | None
        CasJobs username from CLI arg or None.
    password : str | None
        CasJobs password from CLI arg or None.
        
    Returns
    -------
    tuple[str, str]
        Resolved CasJobs username and password.
    """
    resolved_user = user or os.environ.get("CASJOBS_USER")
    resolved_pw = password or os.environ.get("CASJOBS_PW")

    if not resolved_user:
        resolved_user = input("CasJobs username: ").strip()

    if not resolved_pw:
        resolved_pw = getpass("CasJobs password: ")

    return resolved_user, resolved_pw


def _resolve_z_range(
    cluster: Cluster,
    z_min: float | None,
    z_max: float | None,
) -> tuple[float, float]:
    """Choose the redshift bounds to use for downstream selection and plots.
    
    Parameters
    ----------
    cluster : Cluster
        Cluster object.
    z_min : float | None
        Minimum redshift from CLI arg or None.
    z_max : float | None
        Maximum redshift from CLI arg or None.
    """
    if z_min is not None:
        cluster.z_min = float(z_min)
    if z_max is not None:
        cluster.z_max = float(z_max)

    # Prefer already-populated values on the Cluster object.
    if cluster.z_min is None or (isinstance(cluster.z_min, float) and np.isnan(cluster.z_min)):
        cluster.z_min = float(cluster.redshift) - DEFAULT_Z_PAD

    if cluster.z_max is None or (isinstance(cluster.z_max, float) and np.isnan(cluster.z_max)):
        cluster.z_max = float(cluster.redshift) + DEFAULT_Z_PAD

    return float(cluster.z_min), float(cluster.z_max)


def _parse_json_kwargs(
        arg: str | None,
        flag_name: str
    ) -> dict:
    """Parse a JSON string from CLI into a dict of kwargs.

    Parameters
    ----------
    arg : str | None
        JSON string from CLI arg or None.
    flag_name : str
        Name of the CLI flag (for error messages).

    Returns
    -------
    dict
        Parsed kwargs dict.
    """

    if not arg:
        return {}
    try:
        parsed = json.loads(arg)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON string for {flag_name}") from e
    if not isinstance(parsed, dict):
        raise ValueError(f"{flag_name} must decode to a JSON object (dict), got {type(parsed).__name__}.")
    return parsed


# ------------------------------------
# Main Pipeline
# ------------------------------------

def run_full_pipeline(
        cluster: Cluster,
        *,
        z_min: float | None = None,
        z_max: float | None = None,
        casjobs_user: str | None = None,
        casjobs_password: str | None = None,
        manual_list: list[tuple[float, float]] | None = None,
        show_plots: bool = False,
        save_plots: bool = True,
        skip_redshifts: bool = False,
        skip_photometry: bool = False,
        skip_catalogs: bool = False,
        skip_cmd: bool = False,
        skip_plots: bool = False,
        redshift_kwargs: dict | None = None,
        photometry_kwargs: dict | None = None,
        catalog_kwargs: dict | None = None,
        cmd_kwargs: dict | None = None,
        plotting_kwargs: dict | None = None,
    ) -> None:
    """Run the full cluster analysis pipeline.

    Parameters
    ----------
    cluster : Cluster
        Cluster object.
    z_min : float | None, optional
        Minimum redshift for analysis. If None, defaults to cluster redshift - 0.015.
    z_max : float | None, optional
        Maximum redshift for analysis. If None, defaults to cluster redshift + 0.015.
    casjobs_user : str | None, optional
        CasJobs username. If None, will attempt to read from env var or prompt.
    casjobs_password : str | None, optional
        CasJobs password. If None, will attempt to read from env var or prompt.
    manual_list : list[tuple[float, float]] | None, optional
        List of manually specified BCG positions as (RA, DEC) tuples in degrees. If None, no manual BCGs are provided.
    show_plots : bool, optional
        Whether to show plots interactively. Default is False.
    save_plots : bool, optional
        Whether to save plots to disk. Default is True.
    skip_redshifts : bool, optional
        Whether to skip the archival redshift pipeline. Default is False.
    skip_photometry : bool, optional
        Whether to skip the archival photometry pipeline. Default is False.
    skip_catalogs : bool, optional
        Whether to skip the catalog building pipeline. Default is False.
    skip_cmd : bool, optional
        Whether to skip the color-magnitude pipeline. Default is False.
    skip_plots : bool, optional
        Whether to skip the plotting utility. Default is False.
    redshift_kwargs : dict | None, optional
        Additional kwargs to pass to the redshift pipeline.
    photometry_kwargs : dict | None, optional
        Additional kwargs to pass to the photometry pipeline.
    catalog_kwargs : dict | None, optional
        Additional kwargs to pass to the catalog building pipeline.
    cmd_kwargs : dict | None, optional
        Additional kwargs to pass to the color-magnitude pipeline.
    plotting_kwargs : dict | None, optional
        Additional kwargs to pass to the plotting utility.
    """
    print(f"Running full pipeline for cluster {cluster.identifier}...")

    # Normalize kwargs
    redshift_kwargs = dict(redshift_kwargs or {})
    photometry_kwargs = dict(photometry_kwargs or {})
    catalog_kwargs = dict(catalog_kwargs or {})
    cmd_kwargs = dict(cmd_kwargs or {})
    plotting_kwargs = dict(plotting_kwargs or {})


    # Verify folders exist
    os.makedirs(cluster.base_path, exist_ok=True)
    os.makedirs(cluster.redshift_path, exist_ok=True)
    os.makedirs(cluster.photometry_path, exist_ok=True)

    if manual_list and "manual_list" in catalog_kwargs:
        raise ValueError("Provide manual BCGs via either --bcg or --catalog-kwargs, not both.")
    
    z_min_resolved, z_max_resolved = _resolve_z_range(cluster, z_min, z_max)
    print(f"Using zmin = {z_min_resolved}, zmax = {z_max_resolved}")

    # Step 1: Redshifts

    if not skip_redshifts:
        print("\n--- Running archival redshift pipeline ---")
        casjobs_user_resolved, casjobs_password_resolved = _resolve_casjobs_credentials(
            casjobs_user,
            casjobs_password
        )
        run_redshift_pipeline(
            cluster=cluster,
            user=casjobs_user_resolved,
            password=casjobs_password_resolved,
            **redshift_kwargs,
        )
    else:
        print("\n--- Skipping archival redshift pipeline ---")


    # Step 2: Photometry
    if not skip_photometry:
        print("\n--- Running archival photometry pipeline ---")
        run_photometry_pipeline(
            cluster=cluster,
            **photometry_kwargs,
        )
    else:
        print("\n--- Skipping archival photometry pipeline ---")


    # Step 3: Catalogs
    if not skip_catalogs:
        print("\n--- Running catalog builder pipeline ---")
        build_redshift_catalog(
            cluster=cluster,
            manual_list=manual_list,
            **catalog_kwargs,
        )
    else:
        print("\n--- Skipping catalog builder pipeline ---")


    # Step 4: Color-magnitude selection
    if not skip_cmd:
        print("\n--- Running red sequence fitting pipeline ---")
        run_cmd_pipeline(
            cluster=cluster,
            plot_cmd_flag=True,
            plot_spatial_flag=True,
            show_plots=show_plots,
            save_plots=save_plots,
            **cmd_kwargs,
        )
    else:
        print("\n--- Skipping red sequence fitting pipeline ---")


    # Step 5: Plots
    if not skip_plots:
        print("\n--- Running plotting utility ---")
        run_cluster_plots(
            cluster=cluster,
            fov_size=cluster.fov,
            survey=cluster.survey,
            color_type=cluster.color_type,
            ra_offset=cluster.ra_offset,
            dec_offset=cluster.dec_offset,
            bandwidth=cluster.bandwidth,
            save_plots=save_plots,
            show_plots=show_plots,
            **plotting_kwargs,
        )
    else:
        print("\n--- Skipping plotting utility ---")

    print("\nPipeline complete!")

 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "cluster_id",
        type=str,
        help="Cluster ID (e.g. '1327') or full RMJ/Abell name."
    )
    parser.add_argument(
        "--zmin",
        type=float,
        default=None,
        help="Minimum redshift for analysis."
    )
    parser.add_argument(
        "--zmax",
        type=float,
        default=None,
        help="Maximum redshift for analysis."
    )
    parser.add_argument(
        "--casjobs-user",
        type=str,
        default=None,
        help="CasJobs username (or set env CASJOBS_USER)."
    )
    parser.add_argument(
        "--casjobs-password",
        type=str,
        default=None,
        help="CasJobs password (or set env CASJOBS_PW)."
    )
    parser.add_argument(
        "--fov",
        type=float,
        default=None,
        help="Zoomed field of view in arcmin."
    )
    parser.add_argument(
        "--fov-full",
        type=float,
        default=None,
        help="Full field of view in arcmin."
    )
    parser.add_argument(
        "--ra-offset",
        type=float,
        default=None,
        help="RA offset in arcmin (Positive looks to the right)."
    )
    parser.add_argument(
        "--dec-offset",
        type=float,
        default=None,
        help="Dec offset in arcmin (Positive looks up)."
    )
    parser.add_argument(
        "--density-levels",
        type=int,
        default=None,
        help="Number of density contour levels."
    )
    parser.add_argument(
        "--density-skip",
        type=int,
        default=None,
        help="Number of density levels to skip."
    )
    parser.add_argument(
        "--density-bandwidth",
        type=float,
        default=None,
        help="Bandwidth for density estimation."
    )
    parser.add_argument(
        "--save-plots",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Save plots? [default: True]"
    )
    parser.add_argument(
        "--show-plots",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Show plots? [default: False]"
    )
    parser.add_argument(
        "--manual-bcg",
        nargs=2,
        type=float,
        action="append",
        metavar=("RA", "DEC"),
        help="Manually specify a BCG position as RA DEC in degrees. Repeatable.",
    )


    # Skip flags
    parser.add_argument(
        "--skip-redshifts",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Skip archival redshift querying.")
    parser.add_argument(
        "--skip-photometry",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Skip archival photometry querying.")
    parser.add_argument(
        "--skip-catalogs",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Skip catalog building / export.")
    parser.add_argument(
        "--skip-cmd",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Skip red-sequence fitting and member selection.")
    parser.add_argument(
        "--skip-plots",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Skip the standard plotting stage.")
    
    # Downstream kwargs
    parser.add_argument(
        "--redshift-kwargs",
        type=str,
        default=None,
        help="JSON string of kwargs to pass to the redshift pipeline."
    )
    parser.add_argument(
        "--photometry-kwargs",
        type=str,
        default=None,
        help="JSON string of kwargs to pass to the photometry pipeline."
    )
    parser.add_argument(
        "--catalog-kwargs",
        type=str,
        default=None,
        help="JSON string of kwargs to pass to the catalog building pipeline."
    )
    parser.add_argument(
        "--cmd-kwargs",
        type=str,
        default=None,
        help="JSON string of kwargs to pass to the color-magnitude pipeline."
    )
    parser.add_argument(
        "--plotting-kwargs",
        type=str,
        default=None,
        help="JSON string of kwargs to pass to the plotting utility."
    )

    args = parser.parse_args()


    # -- Parse known CLI for cluster construction --
    cluster_kwargs = {}
    for key in (
        "fov",
        "fov_full",
        "ra_offset",
        "dec_offset",
        "density_levels",
        "density_skip",
        "density_bandwidth",
    ):
        val = getattr(args, key, None)
        if val is not None:
            cluster_kwargs[key.replace("-", "_")] = val

    manual_list = [tuple(x) for x in (args.manual_bcg or [])]  # list[tuple[float, float]]

    redshift_kwargs = _parse_json_kwargs(args.redshift_kwargs, "--redshift-kwargs")
    photometry_kwargs = _parse_json_kwargs(args.photometry_kwargs, "--photometry-kwargs")
    catalog_kwargs = _parse_json_kwargs(args.catalog_kwargs, "--catalog-kwargs")
    cmd_kwargs = _parse_json_kwargs(args.cmd_kwargs, "--cmd-kwargs")
    plotting_kwargs = _parse_json_kwargs(args.plotting_kwargs, "--plotting-kwargs")


    cluster = Cluster(args.cluster_id, **cluster_kwargs)
    cluster.populate()

    run_full_pipeline(
        cluster,
        z_min=args.zmin,
        z_max=args.zmax,
        casjobs_user=args.casjobs_user,
        casjobs_password=args.casjobs_password,
        manual_list=manual_list,
        show_plots=args.show_plots,
        save_plots=args.save_plots,
        skip_redshifts=args.skip_redshifts,
        skip_photometry=args.skip_photometry,
        skip_catalogs=args.skip_catalogs,
        skip_cmd=args.skip_cmd,
        skip_plots=args.skip_plots,
        redshift_kwargs=redshift_kwargs,
        photometry_kwargs=photometry_kwargs,
        catalog_kwargs=catalog_kwargs,
        cmd_kwargs=cmd_kwargs,
        plotting_kwargs=plotting_kwargs,
    )


if __name__ == "__main__":
    main()
