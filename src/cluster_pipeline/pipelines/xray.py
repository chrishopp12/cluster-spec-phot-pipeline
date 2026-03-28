#!/usr/bin/env python3
"""
xray.py

X-ray Pipeline Driver: Subcluster Analysis and Visualization
---------------------------------------------------------

Orchestrates the X-ray analysis pipeline: redshift processing, subcluster
member assignment, statistical analysis, and multi-panel figure generation.

Entry points:
  - ``run_xray(cluster, subclusters)`` — clean entry point with graceful
    skip if no X-ray data is available
  - ``analyze_cluster(...)`` — legacy orchestrator (v1 interface, being
    migrated to use Subcluster objects)

Requirements:
  - All cluster_pipeline subpackages

Notes:
  - X-ray processing is independent of Stages 1-4. It only needs
    X-ray FITS data and the cluster member catalogs.
  - If no X-ray FITS file exists, the pipeline skips X-ray image
    processing and contour generation but can still run subcluster
    analysis and redshift-based plots.
"""

from __future__ import annotations

import os
import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cluster_pipeline.models.cluster import Cluster
from cluster_pipeline.utils import str2bool, read_json, string_to_numeric, find_first_val, pop_prefixed_kwargs
from cluster_pipeline.io.catalogs import load_dataframes, load_bcg_catalog
from cluster_pipeline.subclusters.builder import build_subclusters
from cluster_pipeline.subclusters.assignment import (
    assign_subcluster_regions,
    assign_subcluster_members_multi,
    filter_members_by_config,
    make_member_catalogs,
    make_combined_groups,
)
from cluster_pipeline.subclusters.statistics import (
    velocity_dispersion,
    process_redshifts,
    make_stats_table,
    analyze_group,
    build_subcluster_summary,
    plot_stacked_velocity_histograms,
)
from cluster_pipeline.plotting.subclusters import (
    plot_subcluster_members_and_regions,
    plot_redshift_and_subclusters_figure,
    plot_stacked_redshift_histograms,
    plot_redshift_histogram_heatmap,
    plot_2panel_optical_contours,
    plot_3panel_optical_subclusters_figure,
    plot_combined_4panel_figure,
    plot_subcluster_regions_and_histograms,
)
from cluster_pipeline.plotting.xray import plot_redshift_overlay, make_plots


# ====================================================================
# Clean entry points
# ====================================================================

def run_xray_imaging(
    cluster: Cluster,
    *,
    save_plots: bool = True,
    show_plots: bool = False,
) -> None:
    """Run X-ray image processing and redshift analysis.

    Handles redshift processing (GMM fitting) and X-ray image visualization.
    Independent of subcluster analysis — does not need subclusters.

    Parameters
    ----------
    cluster : Cluster
        Cluster object with paths and metadata.
    save_plots : bool
        Save generated figures. [default: True]
    show_plots : bool
        Display figures interactively. [default: False]

    Notes
    -----
    - Redshift processing always runs (GMM fitting, velocity dispersions).
    - X-ray FITS file expected at ``cluster.xray_file``.
    - If no X-ray data, skips image processing gracefully.
    """
    has_xray = os.path.isfile(cluster.xray_file)

    # Redshift processing (independent of X-ray)
    print("\n--- Redshift processing ---")
    process_redshifts(cluster, save_plots=save_plots, show_plots=show_plots)

    # X-ray image processing (only if FITS exists)
    if has_xray:
        print(f"\n--- X-ray image processing ---")
        print(f"  X-ray data: {cluster.xray_file}")
        make_plots(cluster, save_plots=save_plots, show_plots=show_plots)
    else:
        print(f"\nNo X-ray data at {cluster.xray_file} — skipping image processing")


def run_subcluster_analysis(
    cluster: Cluster,
    subclusters: list | None = None,
    *,
    save_plots: bool = True,
    show_plots: bool = False,
) -> None:
    """Run subcluster member assignment, statistics, and plotting.

    Independent of X-ray image processing. Requires at least 2 subclusters
    for bisector-based region assignment.

    Parameters
    ----------
    cluster : Cluster
        Cluster object with paths and metadata.
    subclusters : list[Subcluster] or None
        Subcluster objects from Stage 5.
    save_plots : bool
        Save generated figures. [default: True]
    show_plots : bool
        Display figures interactively. [default: False]
    """
    # Subcluster analysis
    if subclusters is not None and len(subclusters) >= 2:
        print(f"\n--- Subcluster analysis ({len(subclusters)} subclusters) ---")
        cluster.ensure_directories()

        # Load member catalog and spectroscopic catalog
        members_path = os.path.join(cluster.members_path, "cluster_members.csv")
        spec_path = cluster.spec_file

        if not os.path.isfile(spec_path):
            print(f"  No combined_redshifts.csv found — skipping member assignment")
            return

        spec_df = pd.read_csv(spec_path)
        print(f"  Loaded {len(spec_df)} spectroscopic sources")

        # Load member catalog if available (for photometric members)
        if os.path.isfile(members_path):
            members_df = pd.read_csv(members_path)
            print(f"  Loaded {len(members_df)} cluster members")
        else:
            members_df = spec_df.copy()
            print(f"  No cluster_members.csv — using spec catalog only")

        # Stage 6: Assign regions + members
        print("\n  --- Assigning subcluster regions ---")
        bcg_regions = assign_subcluster_regions(subclusters, verbose=True)

        print("\n  --- Assigning spectroscopic members ---")
        spec_groups, bisectors = assign_subcluster_members_multi(subclusters, spec_df)
        spec_groups = filter_members_by_config(spec_groups, subclusters, spec=True)

        print("\n  --- Assigning photometric members ---")
        phot_groups, _ = assign_subcluster_members_multi(subclusters, members_df)
        phot_groups = filter_members_by_config(phot_groups, subclusters, spec=False)

        # Populate Subcluster objects with their members
        for i, sub in enumerate(subclusters):
            sub.spec_members = spec_groups[i]
            sub.phot_members = phot_groups[i]

        # Write member catalogs
        make_member_catalogs(cluster, spec_groups, phot_groups, subclusters)

        # Stage 7: Statistics
        print("\n  --- Subcluster statistics ---")
        group_stats = analyze_group(subclusters)
        for label, stats in group_stats.items():
            n = stats.get("n_spec", 0)
            sv = stats.get("sigma_v", 0)
            zm = stats.get("z_mean", 0)
            print(f"    Subcluster {label}: N_spec={n}, z_mean={zm:.4f}, sigma_v={sv:.1f} km/s")

        # Build combined groups (default: each subcluster is its own group)
        combined_configs = subclusters  # default: same as subclusters
        spec_groups_combined = spec_groups
        phot_groups_combined = phot_groups
        combined_indices = None

        # Check if any subclusters have group members defined
        has_groups = any(len(sub.group_members) > 1 for sub in subclusters)
        if has_groups:
            # Use make_combined_groups to merge grouped subclusters
            try:
                # Extract combine pairs from subcluster group info
                seen_groups = set()
                combine_pairs = []
                for sub in subclusters:
                    if len(sub.group_members) > 1 and sub.group_id not in seen_groups:
                        seen_groups.add(sub.group_id)
                        combine_pairs.append(list(sub.group_members))

                if combine_pairs:
                    spec_groups_combined, phot_groups_combined, combined_indices = \
                        make_combined_groups(cluster, spec_groups, phot_groups, subclusters,
                                            combine_groups=combine_pairs)
                    # Build combined configs (dominant subclusters only)
                    combined_configs = [sub for sub in subclusters if sub.is_dominant]
            except Exception as e:
                print(f"  Warning: combined group building failed ({e}), using individual subclusters")

        # Stage 9: Subcluster plots
        print("\n  --- Subcluster plots ---")
        save_path = cluster.image_path
        os.makedirs(save_path, exist_ok=True)

        # Common kwargs for plots that support combined groups
        combined_kw = dict(
            combined_configs=combined_configs,
            spec_groups_combined=spec_groups_combined,
            phot_groups_combined=phot_groups_combined,
            combined_indices=combined_indices,
        )

        try:
            plot_subcluster_members_and_regions(
                cluster=cluster,
                subclusters=subclusters,
                spec_groups=spec_groups_combined,
                phot_groups=phot_groups_combined,
                **combined_kw,
                save_plots=save_plots,
                show_plots=show_plots,
                save_path=save_path,
            )
            print("    Members + regions plot: OK")
        except Exception as e:
            print(f"    Members + regions plot FAILED: {e}")

        for layout in ("vertical", "horizontal"):
            try:
                plot_2panel_optical_contours(
                    cluster=cluster,
                    subclusters=subclusters,
                    layout=layout,
                    save_plots=save_plots,
                    show_plots=show_plots,
                    save_path=save_path,
                )
                print(f"    2-panel optical contours ({layout}): OK")
            except Exception as e:
                print(f"    2-panel optical contours ({layout}) FAILED: {e}")

        for layout in ("vertical", "horizontal"):
            try:
                plot_redshift_and_subclusters_figure(
                    cluster=cluster,
                    subclusters=subclusters,
                    spec_groups=spec_groups_combined,
                    phot_groups=phot_groups_combined,
                    layout=layout,
                    **combined_kw,
                    save_plots=save_plots,
                    show_plots=show_plots,
                    save_path=save_path,
                )
                print(f"    Redshift + subclusters ({layout}): OK")
            except Exception as e:
                print(f"    Redshift + subclusters ({layout}) FAILED: {e}")

        try:
            plot_stacked_redshift_histograms(
                cluster=cluster,
                subclusters=subclusters,
                spec_groups=spec_groups_combined,
                save_plots=save_plots,
                show_plots=show_plots,
                save_path=save_path,
            )
            print("    Stacked histograms: OK")
        except Exception as e:
            print(f"    Stacked histograms FAILED: {e}")

        for layout in ("vertical", "horizontal"):
            try:
                plot_3panel_optical_subclusters_figure(
                    cluster=cluster,
                    subclusters=subclusters,
                    spec_groups=spec_groups_combined,
                    phot_groups=phot_groups_combined,
                    layout=layout,
                    **combined_kw,
                    save_plots=save_plots,
                    show_plots=show_plots,
                    save_path=save_path,
                )
                print(f"    3-panel optical subclusters ({layout}): OK")
            except Exception as e:
                print(f"    3-panel optical subclusters ({layout}) FAILED: {e}")

        try:
            plot_combined_4panel_figure(
                cluster=cluster,
                subclusters=subclusters,
                spec_groups=spec_groups_combined,
                phot_groups=phot_groups_combined,
                **combined_kw,
                save_plots=save_plots,
                show_plots=show_plots,
                save_path=save_path,
            )
            print("    Combined 4-panel figure: OK")
        except Exception as e:
            print(f"    Combined 4-panel figure FAILED: {e}")

        try:
            plot_redshift_histogram_heatmap(
                cluster=cluster,
                save_plots=save_plots,
                show_plots=show_plots,
                save_path=save_path,
            )
            print("    Redshift histogram heatmap: OK")
        except Exception as e:
            print(f"    Redshift histogram heatmap FAILED: {e}")

        try:
            plot_subcluster_regions_and_histograms(
                cluster=cluster,
                subclusters=subclusters,
                spec_groups=spec_groups_combined,
                phot_groups=phot_groups_combined,
                **combined_kw,
                save_plots=save_plots,
                show_plots=show_plots,
                save_path=save_path,
            )
            print("    Regions + histograms: OK")
        except Exception as e:
            print(f"    Regions + histograms FAILED: {e}")

    elif subclusters is not None and len(subclusters) < 2:
        print(f"\n  Only {len(subclusters)} subcluster — need at least 2 for bisector analysis")


# ====================================================================
# analyze_cluster  (legacy orchestrator — being migrated)
# ====================================================================

def analyze_cluster(
        cluster,
        subcluster_configs,
        legend_loc="upper right",
        manual_list=None,
        combined_groups=None,
        show_plots=True,
        save_plots=False,
        save_path=None,
        plot_alt_regions=False,
        run_pipeline=False,
        verbose=False
    ):
    if save_path is None:
        save_path = cluster.image_path
    os.makedirs(save_path, exist_ok=True)

    # if run_pipeline:
    #     run_full_pipeline(cluster, manual_list=manual_list,save_plots=save_plots, show_plots=show_plots)
    process_redshifts(cluster, save_plots=save_plots, show_plots=show_plots)
    make_plots(cluster, save_plots=save_plots, show_plots=show_plots)

    spec_df, phot_df, bcg_df = load_dataframes(cluster)

    # Create first look at regions using dividing lines
    if verbose:
        _ = assign_subcluster_regions(subcluster_configs, plot=True)

    # Assign subcluster members based on dividing lines
    region_groups_spec, bisectors   = assign_subcluster_members_multi(subcluster_configs, spec_df)
    region_groups_phot, _ = assign_subcluster_members_multi(subcluster_configs, phot_df)

    # Filter members based on config (e.g., z_range, r_max)
    spec_groups = filter_members_by_config(region_groups_spec, subcluster_configs)
    phot_groups = filter_members_by_config(region_groups_phot, subcluster_configs, spec=False)

    print(f"\nSpectroscopic groups: {[len(g) for g in spec_groups]}")
    print(f"Photometric groups: {[len(g) for g in phot_groups]}")

    # Build member catalogs
    make_member_catalogs(cluster, spec_groups, phot_groups, subcluster_configs)
    build_subcluster_summary(cluster=cluster, configs=subcluster_configs, spec_groups=spec_groups, update_csv=True)

    z_data, vel_data = analyze_group(spec_groups)
    stats_table, stats_values = make_stats_table(
        z_data,
        bins=12,
        prefix="subcluster",
        make_plots=True,
        save_plots=True,
        show_plots=True,
        save_path="figures",   # or Path(...)
        ranges=None            # or [(z1_low, z1_high), (z2_low, z2_high), ...] if you want display ranges
)

    print(stats_table)

    # Combine groups if specified
    spec_groups_combined, phot_groups_combined, combined_indices = make_combined_groups(cluster, spec_groups, phot_groups, subcluster_configs, combined_groups)
    combined_configs = [cfg for cfg in subcluster_configs if cfg.get("is_dominant") == 1]

    z_data, vel_data = analyze_group(spec_groups_combined)
    stats_table, stats_values = make_stats_table(
        z_data,
        bins=12,
        prefix="subcluster",
        make_plots=True,
        save_plots=True,
        show_plots=True,
        save_path="figures",   # or Path(...)
        ranges=None            # or [(z1_low, z1_high), (z2_low, z2_high), ...] if you want display ranges
)

    print(stats_table)



    if plot_alt_regions:
        # Save file based on subclusters used.
        subclust_ids = [str(sub.bcg_id if hasattr(sub, 'bcg_id') else sub['bcg_id']) for sub in subcluster_configs]
        subclust_part = "_".join(subclust_ids)
        filename = f"{cluster.identifier}_subcluster_histograms_{subclust_part}.pdf"
        save_file = os.path.join(save_path, filename)
        plot_subcluster_regions_and_histograms(
                                        cluster=cluster,
                                        subcluster_configs=subcluster_configs,
                                        combined_configs=combined_configs,
                                        spec_groups=spec_groups,
                                        phot_groups=phot_groups,
                                        spec_groups_combined=spec_groups_combined,
                                        phot_groups_combined=phot_groups_combined,
                                        legend_loc=legend_loc,
                                        combined_indices=combined_indices,
                                        show_plots=show_plots,
                                        save_plots=True,
                                        save_path=save_file
        )
        return

    clean_id = cluster.identifier.replace(" ", "_")

    plot_subcluster_members_and_regions(
                                    cluster=cluster,
                                    subcluster_configs=subcluster_configs,
                                    combined_configs=combined_configs,
                                    spec_groups=spec_groups,
                                    spec_groups_combined=spec_groups_combined,
                                    phot_groups=phot_groups,
                                    phot_groups_combined=phot_groups_combined,
                                    legend_loc=legend_loc,
                                    combined_indices=combined_indices,
                                    show_plots=show_plots,
                                    save_plots=save_plots,
                                    save_path=os.path.join(cluster.image_path, f"{clean_id}_subcluster_map.pdf")
    )


    plot_redshift_and_subclusters_figure(
                                    cluster=cluster,
                                    subcluster_configs=subcluster_configs,
                                    combined_configs=combined_configs,
                                    spec_groups=spec_groups,
                                    spec_groups_combined=spec_groups_combined,
                                    phot_groups=phot_groups,
                                    phot_groups_combined=phot_groups_combined,
                                    legend_loc=legend_loc,
                                    combined_indices=combined_indices,
                                    legend_loc_bottom="upper right",
                                    show_plots=show_plots,
                                    save_plots=save_plots,
                                    save_path=os.path.join(cluster.image_path, f"{clean_id}_subcluster_z_scatter_vert.pdf"),
                                    layout='vertical'
    )

    plot_redshift_and_subclusters_figure(
                                    cluster=cluster,
                                    subcluster_configs=subcluster_configs,
                                    combined_configs=combined_configs,
                                    spec_groups=spec_groups,
                                    spec_groups_combined=spec_groups_combined,
                                    phot_groups=phot_groups,
                                    phot_groups_combined=phot_groups_combined,
                                    legend_loc=legend_loc,
                                    combined_indices=combined_indices,
                                    legend_loc_bottom="upper right",
                                    show_plots=show_plots,
                                    save_plots=save_plots,
                                    save_path=os.path.join(cluster.image_path, f"{clean_id}_subcluster_z_scatter_hor.pdf"),
                                    layout='horizontal'
    )

    plot_2panel_optical_contours(
                                    cluster=cluster,
                                    subcluster_configs=subcluster_configs,
                                    spec_groups=spec_groups,
                                    phot_groups=phot_groups,
                                    layout="vertical",
                                    legend_loc=legend_loc,
                                    show_plots=show_plots,
                                    save_plots=save_plots,
                                    save_path=os.path.join(cluster.image_path, f"{clean_id}_2panel_optical_contours_vert.pdf")
    )

    plot_2panel_optical_contours(
                                    cluster=cluster,
                                    subcluster_configs=subcluster_configs,
                                    spec_groups=spec_groups,
                                    phot_groups=phot_groups,
                                    layout="horizontal",
                                    legend_loc=legend_loc,
                                    show_plots=show_plots,
                                    save_plots=save_plots,
                                    save_path=os.path.join(cluster.image_path, f"{clean_id}_2panel_optical_contours_hor.pdf")
    )

    plot_redshift_histogram_heatmap(
                                    cluster=cluster,
                                    legend_loc=legend_loc,
                                    show_plots=show_plots,
                                    save_plots=save_plots,
                                    save_path=os.path.join(cluster.image_path, f"{clean_id}_redshifts_hist_hmap.pdf")
    )


    plot_stacked_redshift_histograms(
                                    cluster=cluster,
                                    subcluster_configs=combined_configs,
                                    spec_groups=spec_groups_combined,
                                    show_plots=show_plots,
                                    save_plots=save_plots,
                                    save_path=os.path.join(cluster.image_path, f"{clean_id}_stacked_redshifts.pdf")
    )

    plot_3panel_optical_subclusters_figure(
                                    cluster=cluster,
                                    subcluster_configs=subcluster_configs,
                                    combined_configs=combined_configs,
                                    spec_groups=spec_groups,
                                    spec_groups_combined=spec_groups_combined,
                                    phot_groups=phot_groups,
                                    phot_groups_combined=phot_groups_combined,
                                    combined_indices=combined_indices,
                                    show_plots=show_plots,
                                    save_plots=save_plots,
                                    save_path=os.path.join(cluster.image_path, f"{clean_id}_3panel_optical.pdf")
    )

    plot_combined_4panel_figure(
                                    cluster=cluster,
                                    subcluster_configs=subcluster_configs,
                                    combined_configs=combined_configs,
                                    spec_groups=spec_groups,
                                    spec_groups_combined=spec_groups_combined,
                                    phot_groups=phot_groups,
                                    phot_groups_combined=phot_groups_combined,
                                    combined_indices=combined_indices,
                                    legend_loc=legend_loc,
                                    show_plots=show_plots,
                                    save_plots=save_plots,
                                    save_path=os.path.join(cluster.image_path, f"{clean_id}_combined_4panel.pdf")
    )


# ============================================================
# CLI helpers and driver  (from run_xray_pipeline.py)
# ============================================================

# ------------------------------------
# Defaults/ Constants
# ------------------------------------
DEFAULT_FOV = 6.0  # arcmin
DEFAULT_FOV_FULL = 30.0  # arcmin
DEFAULT_PSF = 10.0  # arcsec
DEFAULT_RA_OFFSET = 0.0  # arcmin
DEFAULT_DEC_OFFSET = 0.0  # arcmin
DEFAULT_BANDWIDTH = 0.1
DEFAULT_PHOT_LEVELS = 10
DEFAULT_PHOT_SKIP = 1
DEFAULT_LEGEND_LOC = 'upper right'
DEFAULT_LEVELS = (1.0, 0.0, 12)  # (base level, min level, max level)

DEFAULT_CONFIG_FILE = 'cluster_configs.json'
DEFAULT_CONFIG_DIR = './../cluster_configs/'

DEFAULT_SAVE_PLOTS = True
DEFAULT_SHOW_PLOTS = False
DEFAULT_PLOT_ALT_REGIONS = False
DEFAULT_RUN_PIPELINE = False

# ------------------------------------
# Helpers
# ------------------------------------

def _parse_subcluster_kwargs(cli_args: list[str]) -> dict[str, Any]:
    """
    Parses unknown CLI args into a kwargs dict for build_subclusters. Accepts inputs in the form --{key}_{index} value or --{key}_{index}=value. Keys include, but aren't limited to: color, label, radius, z_range, group_z_range, etc.

    Common Examples:
        --color_4 tab:orange
        --label_2 Main
        --z_range_3 0.35 0.4
        --group_z_range_1_4=0.3,0.5

    Parameters:
    -----------
    cli_args : list[str]
        List of CLI arguments (typically from argparse's unknown args).

    Returns:
    --------
    dict[str, Any]
        Dictionary of parsed keyword arguments.

    """
    def _parse_two_floats(raw: str) -> tuple[float, float]:
        parts = [p for p in raw.replace(",", " ").split() if p]
        if len(parts) != 2:
            raise ValueError(f"Expected two values, got: {parts}")
        return (float(parts[0]), float(parts[1]))


    Z_RANGE_KEYS = ("z_range_", "group_z_range_")

    kw = {}
    i = 0
    while i < len(cli_args):
        arg = cli_args[i]
        if not arg.startswith("--"):
            i += 1
            continue

        key = arg.lstrip("-")
        val: Any = None
        # Support --key val or --key=val
        if "=" in key:
            key, raw_val = key.split("=", 1)
            if key.startswith(Z_RANGE_KEYS):
                val = _parse_two_floats(raw_val)
            else:
                val = string_to_numeric(raw_val)
            i += 1
        else:
            if key.startswith(Z_RANGE_KEYS):

                # Expect two subsequent values
                if i + 2 >= len(cli_args):
                    raise ValueError(f"Expected two values after {key}, got fewer.")
                val = _parse_two_floats(f"{cli_args[i + 1]} {cli_args[i + 2]}")

                i += 3
            else:
                if i + 1 >= len(cli_args):
                    raise ValueError(f"Expected a value after {key}, got none.")
                val = string_to_numeric(cli_args[i + 1])
                i += 2

        # Alias handling for bcg_label and label
        if key == "label":
            key = "bcg_label"
        elif key.startswith("label_"):
            key = key.replace("label_", "bcg_label_")


        kw[key] = val


    return kw


def _process_z_range(val: Any) -> tuple[float, float] | list[tuple[float, float]]:
    """
    Processes z_range input into standardized format.
    Accept either:
      - [zmin, zmax] or (zmin, zmax)  -> (zmin, zmax)
      - [[zmin,zmax], [zmin,zmax], ...] -> [(zmin,zmax), ...]  (order matches subclusters list)

    Raises ValueError for invalid shapes.

    Parameters:
    -----------
    val : Any
        Input z_range value.

    Returns:
    --------
    tuple[float, float] | list[tuple[float, float]]
        Processed z_range value(s).
    """
    if val is None:
        raise ValueError("z_range value is None.")

    # Case 1: a single pair
    if isinstance(val, (list, tuple)) and len(val) == 2 and not any(isinstance(x, (list, tuple, dict)) for x in val):
        return (float(val[0]), float(val[1]))

    # Case 2: list of pairs
    if isinstance(val, (list, tuple)):
        out: list[tuple[float, float]] = []
        for item in val:
            if not (isinstance(item, (list, tuple)) and len(item) == 2):
                raise ValueError(f"Expected each z_range entry to be a 2-sequence, got: {item!r}")
            out.append((float(item[0]), float(item[1])))
        if not out:
            raise ValueError("z_range list is empty.")
        return out

    raise ValueError(f"Unsupported z_range type: {type(val).__name__}")


def _resolve_config_path(
        cluster_id: str,
        config: str | None,
        config_dir: str | None
    ) -> Path | None:
    """
    Resolves the configuration file path based on provided arguments.

    Priority:
    1) Explicit config file path from CLI.
    2) <config_dir>/<cluster_id>.json or <config_dir>/<cluster_id>_config.json
    3) <config_dir>/cluster_config.json

    Parameters:
    -----------
    cluster_id : str
        The cluster identifier.
    config : str | None
        Explicit config file path from CLI.
    config_dir : str | None
        Config directory from CLI.

    Returns:
    --------
    Path | None
        Resolved config file path or None if not found.
    """
    if config is not None:
        return Path(config).expanduser().resolve()

    cfg_dir = Path(find_first_val(config_dir, DEFAULT_CONFIG_DIR))
    cfg_dir = (Path(__file__).parent / cfg_dir).resolve() if not cfg_dir.is_absolute() else cfg_dir
    possible_names = [
        f"{cluster_id}.json",
        f"{cluster_id}_config.json",
        DEFAULT_CONFIG_FILE
    ]
    for name in possible_names:
        candidate = cfg_dir / name
        if candidate.is_file():
            return candidate.resolve()

    return None


def _select_cluster_block(config: dict[str, Any], cluster_id: str) -> dict[str, Any]:
    """
    Support "global config file" patterns and return the effective block for cluster_id.

    Accepted shapes:
      A) { "<cluster_id>": { ... }, "<cluster_id2>": { ... } }
      B) { ... }  (single block)

    Parameters:
    -----------
    config : dict[str, Any]
        Loaded JSON config.
    cluster_id : str
        Cluster identifier.

    Returns:
    --------
    dict[str, Any]
        Effective config block for the specified cluster_id.
    """
    # A) keyed by cluster_id
    if cluster_id in config and isinstance(config[cluster_id], dict):
        return config[cluster_id]

    # B) treat as a single block
    return config


def _normalize_config_dict(config: dict[str, Any]) -> dict[str, Any]:
    """
    Normalizes subcluster config dict keys to use consistent naming conventions.
    Rules:
      - "label" -> "bcg_label"
      - "label_X" -> "bcg_label_X"
      - Convert "z_range" and "group_z_range" lists to tuples.

    Parameters:
    -----------
    config : dict[str, Any]
        Input subcluster config dict.

    Returns:
    --------
    dict[str, Any]
        Normalized subcluster config dict.
    """

    out_dict: dict[str, Any] = dict(config)
    # Remove private keys
    out_dict = {k: v for k, v in out_dict.items() if not k.startswith("_")}


    # Alias label to bcg_label
    if "label" in config:
        out_dict["bcg_label"] = out_dict.pop("label")

    # Individual labels
    for key in list(out_dict.keys()):
        if key.startswith("label_"):
            new_key = key.replace("label_", "bcg_label_")
            out_dict[new_key] = out_dict.pop(key)
    # z_range and group_z_range
    for key in ("z_range", "group_z_range"):
        if key in out_dict and isinstance(out_dict[key], list) and len(out_dict[key]) == 2:
            out_dict[key] = (float(out_dict[key][0]), float(out_dict[key][1]))

    return out_dict


def _parse_levels(val: Any) -> tuple[float, float, float]:
    """
    Parses contour levels from various input formats into a tuple of three floats.

    Acceptable formats:
    - Comma-separated string: "1.0,0,12"
    - List or tuple of numbers: [1.0, 0, 12] or (1.0, 0, 12)

    Parameters:
    -----------
    val : Any
        Input value representing contour levels.

    Returns:
    --------
    tuple[float, float, float]
        Parsed contour levels as a tuple.
    """
    if val is None:
        return DEFAULT_LEVELS

    if isinstance(val, str):
        parts = [float(x.strip()) for x in val.split(",") if x.strip() != ""]
        if len(parts) != 3:
            raise ValueError(f"Expected three contour levels in string, got: {parts}")
        return (float(parts[0]), float(parts[1]), float(parts[2]))

    elif isinstance(val, (list, tuple)):
        if len(val) != 3:
            raise ValueError(f"Expected three contour levels in list/tuple, got: {val}")
        return (float(val[0]), float(val[1]), float(val[2]))
    else:
        raise TypeError(f"Unsupported type for contour levels: {type(val)}")


def build_parser() -> argparse.ArgumentParser:
    """
    Builds the argument parser for the x-ray pipeline driver script.

    Returns:
    -----------
    argparse.ArgumentParser
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Analyze cluster substructures and build subcluster configs.")

    # -- JSON config --
    # All defaults are None to prevent accidental overrides
    parser.add_argument("--config", type=str, default=None, help=f"Path to JSON config file [default: {DEFAULT_CONFIG_FILE}].")
    parser.add_argument("--config-dir", type=str, default=None, help=f"Directory for config files [default: {DEFAULT_CONFIG_DIR}].")

    # -- Required arguments --
    parser.add_argument("cluster_id", type=str, help="Cluster ID (e.g. '1327') or full RMJ/Abell name.")
    parser.add_argument("--subclusters", nargs="+", type=int, default=None,
        help="List of BCG indices or subcluster IDs (e.g., 2 6 7).")

    # -- Variants --
    parser.add_argument("--variant", type=str, default=None,
    help="Optional config variant name (e.g. fiducial, alt_core).")


    # -- Cluster options --
    parser.add_argument("--z-min", type=float, default=None, help="Minimum redshift for analysis.")
    parser.add_argument("--z-max", type=float, default=None, help="Maximum redshift for analysis.")
    parser.add_argument("--fov", type=float, default=None, help="Zoomed field of view in arcmin.")
    parser.add_argument("--fov-full", type=float, default=None, help="Full field of view in arcmin.")
    parser.add_argument("--ra-offset", type=float, default=None, help="RA offset in arcmin (Positive looks to the right).")
    parser.add_argument("--dec-offset", type=float, default=None, help="Dec offset in arcmin (Positive looks up).")

    # -- Global subcluster options --
    parser.add_argument("--radius", type=float, default=None, help="Default search radius for all subclusters (Mpc).")
    parser.add_argument("--z-range", nargs=2, type=float, default=None, metavar=("ZMIN", "ZMAX"), help="Global z_range for subclusters.")
    parser.add_argument("--colors", nargs="+", default=None, help="List of colors for subclusters (e.g., tab:green gold tab:cyan).")
    parser.add_argument("--labels", nargs="+", default=None, help="List of labels for subclusters.")
    parser.add_argument("--combined-groups", nargs="+", type=int, default=None, help="List of subcluster groups to combine (e.g., --combined 1 4 --combined 2 3).")
    parser.add_argument(
        "--manual-bcg",
        nargs=2,
        type=float,
        action="append",
        metavar=("RA", "DEC"),
        help="Manually specify a BCG position as RA DEC in degrees. Repeatable.",
    )

    # -- X-ray/density/display options --
    parser.add_argument("--xray-levels", type=str, default=None, help="Comma-separated contour levels for X-ray (e.g. '0.5,0,12').")
    parser.add_argument("--psf", type=float, default=None, help="PSF for X-ray smoothing.")
    parser.add_argument("--density-levels", type=int, default=None, help="Number of density contour levels.")
    parser.add_argument("--density-skip", type=int, default=None, help="Number of density levels to skip.")
    parser.add_argument("--density-bandwidth", type=float, default=None, help="Bandwidth for density estimation.")

    parser.add_argument("--save-plots", type=str2bool, nargs="?", const=True, default=None, help="Save plots? [default: True]")
    parser.add_argument("--show-plots", type=str2bool, nargs="?", const=True, default=None, help="Show plots? [default: False]")
    parser.add_argument("--plot-alt-regions", type=str2bool, nargs="?", const=True, default=None, help="Plot alternative regions? [default: False]")
    parser.add_argument("--run-pipeline", type=str2bool, nargs="?", const=True, default=None, help="Run full pipeline (redshift processing, plots)? [default: False]")
    parser.add_argument("--legend-loc", type=str, default=None, help="Legend location on plots (e.g. 'upper right').")
    parser.add_argument("--verbose", type=str2bool, nargs="?", const=True, default=None, help="Verbose output? [default: True]")

    # -- Catch-all for extra per-subcluster kwargs (e.g. --color_4) --
    parser.add_argument("--subcluster-kwargs", nargs=argparse.REMAINDER, help="Extra per-subcluster args (e.g. --color_4 tab:orange --label_2 Main)")

    return parser

def main():
    """
    Main driver function for the x-ray pipeline.
    Parses arguments, constructs Cluster and subcluster configurations, and invokes analysis.
    """

    parser = build_parser()
    args, unknown = parser.parse_known_args()

    # -- Load config file if specified --
    config_path = _resolve_config_path(args.cluster_id, args.config, args.config_dir)
    config_raw: dict[str, Any] = {}
    if config_path is not None:
        print(f"Loading config from: {config_path}")
        config_raw = read_json(config_path)
    else:
        print("No config file found or specified; using CLI/default parameters.")

    # -- Extract proper cluster block --
    config = _select_cluster_block(config_raw, args.cluster_id) if config_raw else {}
    config = _normalize_config_dict(config)


    variant = args.variant
    variants = config.get("variants", {})

    if variant is not None:
        if variant not in variants:
            raise ValueError(f"Variant '{variant}' not found. Available: {list(variants)}")
        config = {**config, **variants[variant]}

    # -- Cluster kwargs --
    # Defaults, config, CLI (CLI takes precedence)
    fov = find_first_val(args.fov, config.get("fov"), DEFAULT_FOV)
    fov_full = find_first_val(args.fov_full, config.get("fov_full"), DEFAULT_FOV_FULL)
    ra_offset = find_first_val(args.ra_offset, config.get("ra_offset"), DEFAULT_RA_OFFSET)
    dec_offset = find_first_val(args.dec_offset, config.get("dec_offset"), DEFAULT_DEC_OFFSET)

    z_min = find_first_val(args.z_min, config.get("z_min"), config.get("zmin"), config.get("z-min"))
    z_max = find_first_val(args.z_max, config.get("z_max"), config.get("zmax"), config.get("z-max"))

    phot_levels = find_first_val(args.density_levels, config.get("density_levels"), config.get("phot_levels"), DEFAULT_PHOT_LEVELS)
    phot_skip = find_first_val(args.density_skip, config.get("density_skip"), config.get("phot_skip"), DEFAULT_PHOT_SKIP)
    bandwidth = find_first_val(args.density_bandwidth, config.get("density_bandwidth"), config.get("bandwidth"), DEFAULT_BANDWIDTH)

    psf = find_first_val(args.psf, config.get("psf"), DEFAULT_PSF)
    x_ray_levels_raw = find_first_val(args.xray_levels, config.get("xray_levels"), config.get("xray-levels"))
    contour_levels = _parse_levels(x_ray_levels_raw)

    cluster_kwargs: dict[str, Any] = {
        "fov": float(fov),
        "fov_full": float(fov_full),
        "ra_offset": float(ra_offset),
        "dec_offset": float(dec_offset),
        "psf": float(psf),
        "bandwidth": float(bandwidth),
        "phot_levels": int(phot_levels),
        "phot_skip": int(phot_skip),
        "contour_levels": contour_levels,
    }
    if z_min is not None:
        cluster_kwargs["z_min"] = float(z_min)
    if z_max is not None:
        cluster_kwargs["z_max"] = float(z_max)

    # -- Build the Cluster object --
    cluster = Cluster(args.cluster_id, **cluster_kwargs)
    cluster.populate(verbose=True)

    # -- Subcluster kwargs --
    subclusters = find_first_val(args.subclusters, config.get("subclusters"))
    if subclusters is None:
        raise ValueError("Subclusters must be specified via --subclusters or config file.")
    subclusters = [int(x) for x in subclusters]
    manual_list = [tuple(x) for x in (args.manual_bcg or [])]  # list[tuple[float, float]]

    # Global subcluster options
    radius = find_first_val(args.radius, config.get("radius"))
    z_range = find_first_val(args.z_range, config.get("z_range"), config.get("z-range"))
    colors = find_first_val(args.colors, config.get("colors"), config.get("color"))
    labels = find_first_val(args.labels, config.get("labels"), config.get("bcg_labels"), config.get("bcg_label"))

    subcluster_kwargs: dict[str, Any] = {}
    if labels is not None:
        if isinstance(labels, str):
            raise ValueError('Config key "labels" must be a list of strings, not a single string.')
        subcluster_kwargs["bcg_label"] = list(labels)
    if radius is not None:
        subcluster_kwargs["radius"] = float(radius)
    if z_range is not None:
        # Accept [zmin, zmax] or (zmin, zmax) or list of such pairs
        subcluster_kwargs["z_range"] = _process_z_range(z_range)
    if colors is not None:
        subcluster_kwargs["color"] = list(colors)
    if labels is not None:
        subcluster_kwargs["bcg_label"] = list(labels)

    # Subcluster groups
    combined = find_first_val(
        args.combined_groups,
        config.get("combined"),
        config.get("combined_groups"),
    )

    if combined is not None:
        subcluster_kwargs["combined"] = combined

    # Parse extra subcluster-keyword pairs from unknown or --subcluster-kwargs
    extra_kwargs: dict[str, Any] = {}

    if args.subcluster_kwargs is not None:
        extra_kwargs = _parse_subcluster_kwargs(args.subcluster_kwargs)
    elif unknown:
        extra_kwargs = _parse_subcluster_kwargs(unknown)

    # CLI overrides
    cli_remainder = args.subcluster_kwargs if args.subcluster_kwargs is not None else unknown
    if cli_remainder:
        extra_kwargs.update(_parse_subcluster_kwargs(cli_remainder))

    # Build subclusters
    subcluster_configs = build_subclusters(
        subclusters=subclusters,
        cluster=cluster,
        **subcluster_kwargs,
        **extra_kwargs
    )

    # -- Pipeline/ display options --
    save_plots = find_first_val(
        args.save_plots,
        config.get("save_plots"),
        config.get("save-plots"),
        DEFAULT_SAVE_PLOTS
    )
    show_plots = find_first_val(
        args.show_plots,
        config.get("show_plots"),
        config.get("show-plots"),
        DEFAULT_SHOW_PLOTS
    )
    plot_alt_regions = find_first_val(
        args.plot_alt_regions,
        config.get("plot_alt_regions"),
        config.get("plot-alt-regions"),
        DEFAULT_PLOT_ALT_REGIONS
    )
    run_pipeline = find_first_val(
        args.run_pipeline,
        config.get("run_pipeline"),
        config.get("run-pipeline"),
        DEFAULT_RUN_PIPELINE
    )
    legend_loc = find_first_val(
        config.get("legend_loc"),
        config.get("legend-loc"),
        DEFAULT_LEGEND_LOC
    )

    # -- Run analysis --
    analyze_cluster(
        cluster=cluster,
        subcluster_configs=subcluster_configs,
        manual_list=manual_list,
        legend_loc=legend_loc,
        combined_groups=combined,
        save_plots=save_plots,
        show_plots=show_plots,
        plot_alt_regions=plot_alt_regions,
        run_pipeline=run_pipeline
    )

if __name__ == "__main__":
    main()
