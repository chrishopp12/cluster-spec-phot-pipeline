#!/usr/bin/env python3
"""
xray.py

X-ray Pipeline Driver: Subcluster Analysis and Visualization
---------------------------------------------------------

Orchestrates the X-ray analysis pipeline: redshift processing, subcluster
member assignment, statistical analysis, and multi-panel figure generation.

Entry points:
  - ``run_xray_imaging(cluster)`` — X-ray processing + redshift analysis
  - ``run_subcluster_analysis(cluster, subclusters)`` — member assignment,
    statistics, and multi-panel figure generation

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

import numpy as np
import pandas as pd

from cluster_pipeline.io.catalogs import load_dataframes
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
    verbose: bool = True,
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
    verbose : bool
        Print detailed progress messages. [default: True]

    Notes
    -----
    - Redshift processing always runs (GMM fitting, velocity dispersions).
    - X-ray FITS file expected at ``cluster.xray_file``.
    - If no X-ray data, skips image processing gracefully.
    """
    has_xray = os.path.isfile(cluster.xray_file)

    # Redshift processing (independent of X-ray)
    print("\n--- Redshift processing ---")
    process_redshifts(cluster, save_plots=save_plots, show_plots=show_plots, verbose=verbose)

    # X-ray image processing (only if FITS exists)
    if has_xray:
        print(f"\n--- X-ray image processing ---")
        if verbose:
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
    verbose: bool = True,
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
    verbose : bool
        Print detailed progress messages. [default: True]
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
        if verbose:
            print(f"  Loaded {len(spec_df)} spectroscopic sources")

        # Load member catalog if available (for photometric members)
        if os.path.isfile(members_path):
            members_df = pd.read_csv(members_path)
            if verbose:
                print(f"  Loaded {len(members_df)} cluster members")
        else:
            members_df = spec_df.copy()
            print(f"  No cluster_members.csv — using spec catalog only")

        # Stage 6: Assign regions + members
        if verbose:
            print("\n  --- Assigning subcluster regions ---")
        bcg_regions = assign_subcluster_regions(subclusters, verbose=verbose)

        if verbose:
            print("\n  --- Assigning spectroscopic members ---")
        spec_groups, bisectors = assign_subcluster_members_multi(subclusters, spec_df)
        spec_groups = filter_members_by_config(spec_groups, subclusters, spec=True, verbose=verbose)

        if verbose:
            print("\n  --- Assigning photometric members ---")
        phot_groups, _ = assign_subcluster_members_multi(subclusters, members_df)
        phot_groups = filter_members_by_config(phot_groups, subclusters, spec=False, verbose=verbose)

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

        # Subcluster summary table
        try:
            build_subcluster_summary(cluster, subclusters)
            print("    Subcluster summary table: OK")
        except Exception as e:
            print(f"    Subcluster summary table FAILED ({type(e).__name__}): {e}")

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
                print(f"  Warning: combined group building failed ({type(e).__name__}: {e}), using individual subclusters")

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
            print(f"    Members + regions plot FAILED ({type(e).__name__}): {e}")

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
                print(f"    2-panel optical contours ({layout}) FAILED ({type(e).__name__}): {e}")

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
                print(f"    Redshift + subclusters ({layout}) FAILED ({type(e).__name__}): {e}")

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
            print(f"    Stacked histograms FAILED ({type(e).__name__}): {e}")

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
                print(f"    3-panel optical subclusters ({layout}) FAILED ({type(e).__name__}): {e}")

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
            print(f"    Combined 4-panel figure FAILED ({type(e).__name__}): {e}")

        try:
            plot_redshift_histogram_heatmap(
                cluster=cluster,
                save_plots=save_plots,
                show_plots=show_plots,
                save_path=save_path,
            )
            print("    Redshift histogram heatmap: OK")
        except Exception as e:
            print(f"    Redshift histogram heatmap FAILED ({type(e).__name__}): {e}")

        try:
            # Filename includes subcluster IDs so different runs don't overwrite
            sub_ids = "_".join(str(sub.bcg_id) for sub in subclusters)
            hist_file = os.path.join(save_path, f"subcluster_histograms_{sub_ids}.pdf")
            plot_subcluster_regions_and_histograms(
                cluster=cluster,
                subclusters=subclusters,
                spec_groups=spec_groups_combined,
                phot_groups=phot_groups_combined,
                **combined_kw,
                save_plots=save_plots,
                show_plots=show_plots,
                save_path=hist_file,
            )
            print("    Regions + histograms: OK")
        except Exception as e:
            print(f"    Regions + histograms FAILED ({type(e).__name__}): {e}")

    elif subclusters is not None and len(subclusters) < 2:
        print(f"\n  Only {len(subclusters)} subcluster — need at least 2 for bisector analysis")
