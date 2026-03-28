"""Multi-panel subcluster visualization: members, regions, histograms, and publication figures.

Functions
---------
plot_subcluster_members_and_regions
    Scatter members and draw bisector arcs on an optical WCS image.
plot_redshift_and_subclusters_figure
    Two-panel: redshift overlay + subcluster regions.
plot_stacked_redshift_histograms
    Vertically stacked redshift/velocity histograms per subcluster.
plot_redshift_histogram_heatmap
    Optical + colormap redshift scatter with histogram inset.
plot_2panel_optical_contours
    Plain optical vs. optical + contour overlays.
plot_3panel_optical_subclusters_figure
    Three-panel: optical, contours, subcluster regions.
plot_combined_4panel_figure
    2x2 publication figure combining contour, redshift, and subcluster panels.
plot_subcluster_regions_and_histograms
    Stacked histograms above a spatial regions panel.
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Any, TYPE_CHECKING

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import ConnectionPatch
from matplotlib.ticker import FuncFormatter, MaxNLocator, MultipleLocator, FormatStrFormatter
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
from astropy.coordinates import SkyCoord

from cluster_pipeline.plotting.common import finalize_figure, add_scalebar, overlay_bcg_markers
from cluster_pipeline.plotting.optical import plot_optical, add_xray_contours, add_density_contours
from cluster_pipeline.plotting.arcs import (
    draw_great_circle_segment,
    plot_bcg_region_arcs,
    find_segment_circle_crossings,
    add_region_fill_clipped_to_signature,
)
from cluster_pipeline.utils.cosmology import redshift_to_proper_distance
from cluster_pipeline.utils.coordinates import make_skycoord
from cluster_pipeline.utils import pop_prefixed_kwargs

if TYPE_CHECKING:
    from cluster_pipeline.models.subcluster import Subcluster


def _get_optical(cluster):
    """Get optical image path, using io.images if needed."""
    from cluster_pipeline.io.images import get_optical_image
    return get_optical_image(cluster, cluster.fov)


# -- Figures and Plots --
def plot_subcluster_members_and_regions(
    cluster,
    subclusters,
    spec_groups,
    combined_configs=None,
    spec_groups_combined=None,
    phot_groups=None,
    phot_groups_combined=None,
    fig=None,
    ax=None,
    legend_loc='upper right',
    combined_indices=None,
    show_legend=True,
    show_plots=True,
    save_plots=False,
    save_path=None,
    **kwargs
):
    """
    Plot subcluster member galaxies and bisector-defined region arcs on a WCS optical image.

    Parameters
    ----------
    cluster : Cluster
        Cluster object (must provide .get_optical_image(), .xray_file, .phot_file(), .bcg_file).
    subclusters : list[Subcluster]
        Subcluster objects defining regions, colors, and member data.
    spec_groups : list of pd.DataFrame
        Each DataFrame contains spectroscopic members for a subcluster.
    phot_groups : list of pd.DataFrame, optional
        Each DataFrame contains photometric members for a subcluster (if available).
    fig : matplotlib.figure.Figure, optional
        Existing figure to plot on. Created if None.
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on. Created if None.
    legend_loc : str, optional
        Legend location [default: 'upper right'].
    show_legend : bool, optional
        Whether to display the legend [default: True].
    show_plots : bool, optional
        Whether to display the plot with plt.show() [default: True].
    save_plots : bool, optional
        Whether to save the plot [default: False].
    save_path : str or None, optional
        Path or directory for saving figure if save_plots is True.
    **kwargs :
        Passed to region arc plotting and scatter plotting (e.g., linewidths).

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes.Axes
        Axes with plotted subclusters and regions.
    """
    # get_bisectors / build_bcg_signatures not yet extracted into the new package
    from cluster_pipeline.subclusters.geometry import get_bisectors, build_bcg_signatures

    subcluster_kwargs = pop_prefixed_kwargs(kwargs, 'subcluster')

    if combined_configs is not None:
        colors = [sub.color for sub in combined_configs]
    else:
        colors = [sub.color for sub in subclusters]

    if spec_groups_combined is not None:
        spec_groups = spec_groups_combined
    if phot_groups_combined is not None:
        phot_groups = phot_groups_combined

    optical_image_file = _get_optical(cluster)

    # Load WCS
    with fits.open(optical_image_file) as hdul:
        wcs_optical = WCS(hdul[0].header, naxis=2)

    # Set up fig/ax
    if ax is None:
        fig = plt.figure(figsize=(7, 7)) if fig is None else fig
        ax = fig.add_subplot(111, projection=wcs_optical)

    fig, ax = plot_optical(
        optical_image_file,
        cluster=cluster,
        fig=fig, ax=ax,
        show_plots=False,
        save_plots=False,
        xray_fits_file=cluster.xray_file,
        photometric_file=cluster.get_phot_file(),
        bcg_file=cluster.bcg_file,
        **subcluster_kwargs
    )

    # Subcluster member points
    scatter_handles = []
    labels = []
    for i, spec in enumerate(spec_groups):
        color = colors[i] if colors[i] is not None else f"C{i}"
        sc1 = ax.scatter(spec['RA'], spec['Dec'], s=12,
                         edgecolors=color, facecolors=color, linewidths=1.2, zorder=11,
                         transform=ax.get_transform('icrs'))
        scatter_handles.append(sc1)
        labels.append(f"Subcluster {i+1}")
        if phot_groups is not None:
            phot = phot_groups[i]
            ax.scatter(phot['RA'], phot['Dec'], s=12,
                        edgecolors=color, facecolors=color, linewidths=1.2, zorder=11,
                        transform=ax.get_transform('icrs'))

    # spec_df = pd.read_csv(cluster.spec_file)
    # # Ensure numeric dtypes (coerce bad entries to NaN so they won't match)
    # for col in ["RA", "Dec", "z"]:
    #     spec_df[col] = pd.to_numeric(spec_df[col], errors="coerce")

    # mask_1 = spec_df["z"].between(0.480, 0.495, inclusive="both")
    # mask_2 = spec_df["z"].between(0.493, 0.510, inclusive="both")

    # print("mask_1 counts:", mask_1.sum(), "mask_2 counts:", mask_2.sum())

    # ax.scatter(spec_df.loc[mask_1, 'RA'], spec_df.loc[mask_1, 'Dec'], s=18,
    #                      edgecolors='tab:orange', facecolors='tab:orange', linewidths=1.2, zorder=11,
    #                      transform=ax.get_transform('icrs'))
    # ax.scatter(spec_df.loc[mask_2, 'RA'], spec_df.loc[mask_2, 'Dec'], s=10,
    #                      edgecolors='tab:green', facecolors='tab:green', linewidths=1.2, zorder=12,
    #                      transform=ax.get_transform('icrs'))
    # ax.scatter(spec_df["RA"], spec_df["Dec"], s=8,
    #                      edgecolors='lightgray', facecolors='none', linewidths=0.8, zorder=12,
    #                      transform=ax.get_transform('icrs'))

    # RA_phot, Dec_phot, Lum_phot = load_photo_coords(cluster.phot_file())
    # ax.scatter(RA_phot, Dec_phot, s=12, marker='x',
    #                      edgecolors='red', facecolors='red', linewidths=1.2, zorder=11,
    #                      transform=ax.get_transform('icrs'))

    # Subcluster region arcs
    bisectors = get_bisectors(subclusters)
    bcg_signatures = build_bcg_signatures(subclusters, bisectors)
    plot_bcg_region_arcs(
        ax,
        subclusters,
        bisectors,
        bcg_signatures,
        combined_indices=combined_indices,
        transform=ax.get_transform('icrs'),
        **kwargs
    )

    if show_legend:
        legend = ax.legend(handles=scatter_handles, labels=labels, loc=legend_loc, fontsize=12, frameon=True)
        legend.set_zorder(20)

    if save_plots and save_path is not None:
        plt.tight_layout()
        save_file = os.path.join(save_path, "subcluster_regions.pdf") if os.path.isdir(save_path) else save_path
        fig.savefig(save_file, bbox_inches='tight')
    if show_plots:
        plt.tight_layout()
        plt.show()

    return fig, ax

def plot_redshift_and_subclusters_figure(
    cluster,
    subclusters,
    spec_groups,
    phot_groups,
    combined_indices=None,
    combined_configs=None,
    spec_groups_combined=None,
    phot_groups_combined=None,
    fig=None, ax1=None, ax2=None,
    legend_loc='upper right',
    legend_loc_top=None,
    legend_loc_bottom=None,
    show_plots=True,
    save_plots=False,
    save_path=None,
    layout='vertical',   # 'vertical' (default) or 'horizontal'
    **kwargs
):
    """
    Plot a two-panel figure showing:
        1. Redshift-selected cluster members overlaid on the optical image (with X-ray and density contours).
        2. Subcluster assignments, with member galaxies and bisector-defined region arcs.

    Optionally arranges panels horizontally or vertically.
    Parameters
    ----------
    cluster : Cluster
        Cluster object (must have .spec_file, .xray_file, .phot_file(), .bcg_file, etc.)
    subclusters : list[Subcluster]
        Subcluster objects defining regions, colors, and member data.
    spec_groups, phot_groups : list of pd.DataFrame
        List of DataFrames with cluster member galaxies for each subcluster (matched in order to subclusters).
    fig, ax1, ax2 : optional
        Optionally provide pre-existing Figure and Axes.
    legend_loc : str
        Location for the subcluster legend on the lower panel.
    show_plots, save_plots : bool
        Whether to display or save the plot.
    save_path : str or None
        Path to save the figure (if save_plots is True).
    **plot_kwargs :
        Any extra keyword arguments are passed to downstream plotting functions, including:
            - density_contour options
            - xray_contour options
            - colorbar options
            - region arc options
            - etc.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax1, ax2 : matplotlib.axes.Axes
        The axes for the two panels.
    sc : PathCollection
        The redshift scatter artist from ax1 (for colorbar).

    """
    # plot_redshift_overlay not yet extracted into the new package
    from cluster_pipeline.plotting.xray import plot_redshift_overlay

    if legend_loc_bottom is None:
        legend_loc_bottom = legend_loc
    if legend_loc_top is None:
        legend_loc_top = legend_loc

    optical_image_file = _get_optical(cluster)

    # -- Load optical images and WCS --
    with fits.open(optical_image_file) as hdul:
        wcs_optical = WCS(hdul[0].header, naxis=2)


    colorbar = False
    # -- Setup figure and axes --
    if ax1 is None and ax2 is None:
        if layout == 'horizontal':
            fig = plt.figure(figsize=(16, 8))
            gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.01)
            ax2 = fig.add_subplot(gs[0], projection=wcs_optical)  # left: subclusters
            ax1 = fig.add_subplot(gs[1], projection=wcs_optical, sharey=ax2)  # right: redshift overlay
        else:
            fig = plt.figure(figsize=(8, 12))
            gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.08)
            ax1 = fig.add_subplot(gs[0], projection=wcs_optical)  # top: redshift overlay
            ax2 = fig.add_subplot(gs[1], projection=wcs_optical, sharex=ax1)
        colorbar = True

    # ---------------- Top Panel: Optical image with scalebar and legend ----------------

    fig, ax1, sc = plot_redshift_overlay(
                                        optical_image_file=optical_image_file,
                                        spectroscopy_file=cluster.spec_file,
                                        cluster=cluster,
                                        ax=ax1, fig=fig,
                                        z_low=cluster.z_min, z_high=cluster.z_max,
                                        xray_fits_file=cluster.xray_file,
                                        photometric_file=cluster.get_phot_file(),
                                        bcg_file=cluster.bcg_file,
                                        show_plots=False,
                                        save_plots=False,
                                        save_path=None,
                                        colorbar=False,
                                        density_linewidth = 0.5,
                                        xray_linewidth = 0.5,
                                        legend_loc=legend_loc_top,
                                        **kwargs
)


    if layout == 'horizontal':
        ax1.tick_params(axis='y', labelleft=False)
    else:
        ax1.tick_params(axis='x', labelbottom=False)

    # -- Colorbar placement --
    if colorbar:
        pos = ax1.get_position()
        if layout == 'horizontal':
            # Colorbar to right of ax1 (right panel)
            cbar_ax = fig.add_axes([pos.x1, pos.y0 + 0.02, 0.025, pos.height + 0.02])
        else:
            # Colorbar to right of ax1 (top panel)
            cbar_ax = fig.add_axes([pos.x1 + 0.05, pos.y0 + 0.022, 0.04, pos.height + 0.058])
        cbar = fig.colorbar(sc, cax=cbar_ax)
        cbar.set_label("Redshift (z)")
        cbar.locator = MultipleLocator(0.005)
        cbar.formatter = FormatStrFormatter('%.3f')
        cbar.update_ticks()


    # ----- Bottom Panel: Subcluster members and regions -----
    fig, ax2 = plot_subcluster_members_and_regions(
                                    cluster=cluster,
                                    subclusters=subclusters,
                                    combined_configs=combined_configs,
                                    spec_groups=spec_groups,
                                    phot_groups=phot_groups,
                                    spec_groups_combined=spec_groups_combined,
                                    phot_groups_combined=phot_groups_combined,
                                    combined_indices=combined_indices,
                                    fig=fig, ax=ax2,
                                    legend_loc="upper right",
                                    show_plots=False,
                                    save_plots=False,
                                    save_path=os.path.join(cluster.image_path, "subcluster_map.pdf"),
                                    **kwargs
    )


    if layout == 'horizontal':
        fig.subplots_adjust(wspace=0.04, left=0.09, right=0.88, top=0.96, bottom=0.09)
    else:
        fig.subplots_adjust(hspace=0.08, top=0.96, bottom=0.07, left=0.08, right=0.95)


    if save_path is not None and save_plots:
        save_file = os.path.join(save_path, "subcluster_z_scatter.pdf") if os.path.isdir(save_path) else save_path
        fig.savefig(save_file, bbox_inches='tight')
    if show_plots:
        plt.show()

    return fig, ax1, ax2, sc

def plot_stacked_redshift_histograms(
    cluster,
    subclusters,
    spec_groups,
    bins=24,
    combined_color='darkorange',
    show_plots=True,
    save_plots=False,
    save_path=None
):
    """
    Plot vertically stacked redshift histograms for each subcluster, with velocity dispersion annotations
    and a combined redshift histogram for all subclusters, plus a zoomed-in histogram of all redshifts
    in a broader range. All histograms share consistent axes for direct comparison.

    Also outputs a LaTeX table of velocity dispersions for each subcluster and a stacked velocity histogram.

    Parameters
    ----------
    cluster : Cluster
        Cluster object with necessary metadata (spec_file, z_min, z_max, etc).
    subclusters : list[Subcluster]
        Subcluster objects (color, label, and primary_bcg.z used for annotations).
    spec_groups : list of pd.DataFrame
        Each DataFrame contains spectroscopic members for a subcluster (must include 'z').
    bins : int, optional
        Number of bins for histograms [default: 24].
    combined_color : str, optional
        Color for combined histogram [default: 'darkorange'].
    show_plots : bool, optional
        If True, display the figure [default: True].
    save_plots : bool, optional
        If True, save the figure as PDF [default: False].
    save_path : str, optional
        Path or directory for saving figure. If a directory, will be saved as "stacked_redshifts.pdf".

    Returns
    -------
    None
        Displays and/or saves the plot.
    """
    # analyze_group / plot_stacked_velocity_histograms not yet in the new package
    from cluster_pipeline.subclusters.statistics import analyze_group
    from cluster_pipeline.subclusters.statistics import plot_stacked_velocity_histograms

    # Load spectroscopic data
    spec_df = pd.read_csv(cluster.spec_file)
    z = spec_df['z'].values

    # Prepare data per subcluster
    colors = [sub.color for sub in subclusters]
    z_data, vel_data = analyze_group(spec_groups)

    # Filter redshifts
    z_low = 0.0
    z_high = 1.5
    cluster_low = cluster.z_min
    cluster_high = cluster.z_max

    mask = (z_low < z) & (z < z_high)
    z_subset = z[mask].reshape(-1, 1)
    z_inset_mask = (z > cluster_low) & (z < cluster_high)
    z_inset = z[z_inset_mask]
    print(f"Total galaxies in cluster range ({cluster_low:.3f} to {cluster_high:.3f}): {len(z_inset)}")

    # Format figure
    n_subclusters = len(z_data)
    all_z = np.concatenate(z_data)
    zmin, zmax = all_z.min(), all_z.max()

    hist_bins = np.linspace(zmin, zmax, bins)

    # Get max count for each histogram
    y_max_list = []
    for z_vals in z_data:
        counts, _ = np.histogram(z_vals, bins=hist_bins)
        y_max_list.append(counts.max())

    y_max_list = [2 if x == 1 else x for x in y_max_list] # Prevent 1-member panels from collapsing

    # Reverse the order to plot Subcluster 1 on top
    z_data = z_data[::-1]
    vel_data = vel_data[::-1]
    y_max_list = y_max_list[::-1]
    colors = colors[::-1]
    subclusters = subclusters[::-1]

    # For the combined plot
    combined_counts, _ = np.histogram(all_z, bins=hist_bins)
    y_max_combined = combined_counts.max()

    ratio = y_max_list / y_max_combined
    ratio = ratio.tolist()
    height_ratios = ratio + [1.0, 1.0]

    # Figure size scales with number of subclusters
    fig_height = 5 + 5 * n_subclusters / 3


    fig = plt.figure(figsize=(6, fig_height), constrained_layout=True)
    gs = fig.add_gridspec(n_subclusters + 2, 1, height_ratios=height_ratios)
    axes = [fig.add_subplot(gs[i]) for i in range(n_subclusters + 2)]



    # Subcluster histograms
    for i, (z_vals, (_, z_mean, sigma_v)) in enumerate(zip(z_data, vel_data)):
        color = colors[i]
        if color == 'white':
            color = 'gainsboro'
        z_bcg = subclusters[i].primary_bcg.z if subclusters[i].primary_bcg is not None else None

        ax = axes[i]
        ax.hist(z_vals, bins=hist_bins, color=color, edgecolor='black', alpha=1.0)
        ax.set_ylim(0, y_max_list[i] * 1.1)
        ax.set_xticks([])
        ax.tick_params(labelbottom=False)
        # Use consistent tick spacing, e.g., every 4.     #Probably a param option
        # ax.yaxis.set_major_locator(MultipleLocator(4))
        ax.set_xlim(hist_bins[0]-0.002, hist_bins[-1]+0.002)


        # Hide the '0' label, but keep the tick
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: '' if x == 0 else f'{int(x)}'))

        # Plot mean redshift line
        ax.axvline(z_mean, color='black', linestyle='-', lw=1.6, label=f'$\\bar{{z}}$= {z_mean:.5f}')

        # Plot BCG redshift line if available
        if z_bcg is not None:
            ax.axvline(z_bcg, color='tab:red', linestyle='-', lw=1.6, label=f'BCG {subclusters[i].label}')

        ax.yaxis.set_major_locator(MaxNLocator(integer=True))


        ax.annotate(
                f"Subcluster {n_subclusters - i}\nN = {len(z_vals)}\n$\\sigma_v$ = {sigma_v:.0f} km/s",
                xy=(0, 1), xycoords='axes fraction',  # top-left corner of Axes
                xytext=(6, -6), textcoords='offset points',  # shift right and down
                ha='left', va='top',
                fontsize=10,
                bbox=dict(facecolor='white', edgecolor='gray', alpha=0.7)
                )


        ax.legend(loc='upper right', fontsize=10, frameon=True, framealpha=0.8)

    fig.get_layout_engine().set(hspace=0.0, h_pad =0.0)

    # Combined histogram
    ax_comb = axes[-2]
    ax_comb.hist(all_z, bins=hist_bins, color=combined_color, edgecolor='black', alpha = 1.0)

    ax_comb.set_ylim(0, y_max_combined * 1.1)

    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    ax_comb.xaxis.set_major_locator(MultipleLocator(0.005))
    ax_comb.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax_comb.tick_params(axis='x', which='major', pad=6)
    ax_comb.set_xlabel(' ',fontsize=6)

    colors = colors[::-1]
    subclusters = subclusters[::-1]


    for i in range(n_subclusters):
        sub = subclusters[i]
        z_bcg = sub.primary_bcg.z if sub.primary_bcg is not None else None
        color = colors[i]
        if color == 'white':
            color = 'gainsboro'
        if z_bcg is not None:
            ax_comb.axvline(z_bcg, color=color, linestyle='-', lw=1.6, label=f'BCG {sub.label}')


    ax_comb.annotate(
                f"All Subclusters\nN = {len(all_z)}",
                xy=(0, 1), xycoords='axes fraction',  # top-left corner of Axes
                xytext=(6, -6), textcoords='offset points',  # shift right and down
                ha='left', va='top',
                fontsize=10,
                bbox=dict(facecolor='white', edgecolor='gray', alpha=0.7)
                )


    ax_comb.set_xlim(hist_bins[0]-0.002, hist_bins[-1]+0.002)
    ax_comb.legend(loc='upper right', fontsize=10, frameon=True, framealpha=0.8)

    fig.supylabel("Galaxy Counts", fontsize=16)
    axes[0].set_title(" ", fontsize=6)

    ax_all_z = axes[-1]

    # Histogram of redshifts in zoom region
    ax_all_z.hist(z_subset, bins=80, density=False, alpha=0.6, color='b', edgecolor='black')
    ax_all_z.set_xlim(z_low, z_high)
    ax_all_z.set_xlabel('Redshift (z)')

    # Inset for cluster members
    axins = inset_axes(ax_all_z, width=1.5, height=1.5, loc='upper right', borderpad=1.5)

    axins.hist(z_inset, bins=12, color='darkorange', edgecolor='black')
    axins.set_title("Cluster Members", fontsize=10)
    axins.tick_params(labelsize=8)
    axins.set_xlabel('z', fontsize=9)
    axins.set_ylabel('Counts', fontsize=9)
    axins.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.3f}"))
    axins.yaxis.set_major_locator(MaxNLocator(integer=True))

    fig.canvas.draw() # Need to draw before making connection lines

    # Connection lines
    main_xlim = (cluster_low, cluster_high)
    main_ylim = ax_all_z.get_ylim()
    inset_xlim = axins.get_xlim()
    inset_ylim = axins.get_ylim()


    con1 = ConnectionPatch(xyA=(main_xlim[0], main_ylim[0]), coordsA=ax_all_z.transData,
                        xyB=(inset_xlim[0], inset_ylim[0]), coordsB=axins.transData,
                        color='gray', linewidth=1,zorder=3)
    con2 = ConnectionPatch(xyA=(main_xlim[1], main_ylim[0]), coordsA=ax_all_z.transData,
                        xyB=(inset_xlim[1], inset_ylim[0]), coordsB=axins.transData,
                        color='gray', linewidth=1,zorder=3)

    for con in [con1, con2]:
        ax_all_z.add_artist(con)


    if save_plots and save_path is not None:
        save_file = os.path.join(save_path, "stacked_redshifts.pdf") if os.path.isdir(save_path) else save_path
        fig.savefig(save_file, bbox_inches='tight')
    if show_plots:
        plt.show()

    plot_stacked_velocity_histograms(vel_data,color=colors,show_plots=show_plots, save_plots=save_plots)

def plot_redshift_histogram_heatmap(cluster, legend_loc="lower right", fig=None, ax1=None, ax2=None, legend_loc_bottom=None, show_plots=True, save_plots=False, save_path=None, cmap="viridis", **kwargs):
    # plot_redshift_overlay not yet extracted into the new package
    from cluster_pipeline.plotting.xray import plot_redshift_overlay

    if legend_loc_bottom is None:
        legend_loc_bottom = legend_loc

    optical_image_file = _get_optical(cluster)

    # -- Load optical images and WCS --
    with fits.open(optical_image_file) as hdul:
        wcs_optical = WCS(hdul[0].header, naxis=2)


    colorbar = False
    # -- Setup figure and axes --
    if ax1 is None and ax2 is None:
        fig = plt.figure(figsize=(6, 9))
        gs = fig.add_gridspec(2, 1, height_ratios=[0.52, 1], hspace=0.32)
        ax1 = fig.add_subplot(gs[1], projection=wcs_optical)  # bottom: redshift overlay
        ax2 = fig.add_subplot(gs[0]) # top: histogram
        # fig.subplots_adjust(top=0.94, bottom=0.07, left=0.11, right=0.97)
        # fig.subplots_adjust(top=0.96, bottom=0.1, left=0.15, right=0.97)

        colorbar = True

    # ---------------- Bottom Panel: Optical image with scalebar and legend ----------------

    fig, ax1, handles, labels, sc = plot_redshift_overlay(
                                        optical_image_file=optical_image_file,
                                        spectroscopy_file=cluster.spec_file,
                                        cluster=cluster,
                                        ax=ax1, fig=fig,
                                        z_low=cluster.z_min, z_high=cluster.z_max,
                                        xray_fits_file=cluster.xray_file,
                                        photometric_file=cluster.get_phot_file(),
                                        bcg_file=cluster.bcg_file,
                                        scatter_cmap=cmap,
                                        show_legend=False,
                                        show_plots=False,
                                        save_plots=False,
                                        save_path=None,
                                        colorbar=False,
                                        density_linewidth = 0.8,
                                        xray_linewidth = 0.8,
                                        legend_loc=legend_loc_bottom,
    )


    ax1.set_aspect('equal')
    legend = ax1.legend(handles=handles, labels=labels, loc=legend_loc, fontsize=10, frameon=True)
    legend.set_zorder(200)

    fig.subplots_adjust(top=0.98, bottom=0.1, left=0.2, right=0.94)
    # -- Colorbar placement --
    if colorbar:
        pos = ax1.get_position()
        # Colorbar above ax1 (bottom panel)
        cbar_ax = fig.add_axes([
            pos.x0,              # left edge
            pos.y1 + 0.04,       # slightly above top edge
            pos.width,           # width of axes
            0.015                # thickness of cbar
        ])
        cbar = fig.colorbar(sc, cax=cbar_ax, orientation='horizontal')

        cbar.locator = MultipleLocator(0.005)
        cbar.formatter = FormatStrFormatter('%.3f')
        cbar.update_ticks()

    # Load spectroscopic data
    spec_df = pd.read_csv(cluster.spec_file)
    z = spec_df['z'].values

    # Filter redshifts
    z_low = 0.0
    z_high = 1.5
    cluster_low = cluster.z_min
    cluster_high = cluster.z_max

    mask = (z_low < z) & (z < z_high)
    z_subset = z[mask].reshape(-1, 1)
    z_inset_mask = (z > cluster_low) & (z < cluster_high)
    z_inset = z[z_inset_mask]
    print(f"Total galaxies in cluster range ({cluster_low:.3f} to {cluster_high:.3f}): {len(z_inset)}")


    # Colormap for inset histogram
    cmap = plt.get_cmap(cmap)

    zmin_cluster = np.min(z_inset)
    zmax_cluster = np.max(z_inset)

    # Normalize all to the histogram range
    norm_hist = mpl.colors.Normalize(vmin=zmin_cluster, vmax=zmax_cluster)
    color_location=zmin_cluster + (zmax_cluster - zmin_cluster)/4
    hist_color = cmap(norm_hist(color_location))


    # Full histogram of redshifts
    ax2.hist(z_subset, bins=80, density=False, alpha=1, color=hist_color, edgecolor='black')

    ax2.set_xlim(z_low, z_high)
    ax2.set_xlabel('Redshift (z)', labelpad=8)
    ax2.set_ylabel('Galaxy Counts', labelpad=36)


    pos = ax1.get_position()
    ax2.set_position([pos.x0, ax2.get_position().y0, pos.width, ax2.get_position().height])

    # Inset for cluster members
    axins = inset_axes(ax2, width=1.5, height=1.5, loc='upper right', borderpad=1.5)

    # Plot the histogram and get patches
    h = axins.hist(z_inset, bins=12, edgecolor='black')

    # Color each bar
    bin_centers = 0.5 * (h[1][:-1] + h[1][1:])
    colors = cmap(norm_hist(bin_centers))
    for patch, color in zip(h[2], colors):
        patch.set_facecolor(color)

    axins.set_title("Cluster Members", fontsize=10)
    axins.tick_params(labelsize=8)
    axins.set_xlabel('z', fontsize=9)
    axins.set_ylabel('Counts', fontsize=9)
    axins.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.3f}"))
    axins.yaxis.set_major_locator(MaxNLocator(integer=True))

    fig.canvas.draw() # Need to draw before making connection lines

    # Connection lines
    main_xlim = (cluster_low, cluster_high)
    main_ylim = ax2.get_ylim()
    inset_xlim = axins.get_xlim()
    inset_ylim = axins.get_ylim()


    con1 = ConnectionPatch(xyA=(main_xlim[0], main_ylim[0]), coordsA=ax2.transData,
                        xyB=(inset_xlim[0], inset_ylim[0]), coordsB=axins.transData,
                        color='gray', linewidth=1,zorder=3)
    con2 = ConnectionPatch(xyA=(main_xlim[1], main_ylim[0]), coordsA=ax2.transData,
                        xyB=(inset_xlim[1], inset_ylim[0]), coordsB=axins.transData,
                        color='gray', linewidth=1,zorder=3)

    for con in [con1, con2]:
        ax2.add_artist(con)


    if save_plots and save_path is not None:
        save_file = os.path.join(save_path, "redshifts_hist_hmap.pdf") if os.path.isdir(save_path) else save_path
        fig.savefig(save_file, dpi=450)
    if show_plots:
        plt.show()


def plot_2panel_optical_contours(
        cluster,
        subclusters,
        fig=None, ax1=None, ax2=None,
        legend_loc='upper right',
        show_plots=True,
        save_plots=False,
        save_path=None,
        layout='vertical',
        **kwargs
):
    """
    Plot a two-panel figure showing:
        1. Optical image, with no overlays.
        2. Optical image with xray and density contours.

    Optionally arranges panels horizontally or vertically.
    Parameters
    ----------
    cluster : Cluster
        Cluster object (must have .spec_file, .xray_file, .phot_file(), .bcg_file, etc.)
    subclusters : list[Subcluster]
        Subcluster objects (reserved for future use).
    fig, ax1, ax2 : optional
        Optionally provide pre-existing Figure and Axes.
    legend_loc : str
        Location for the subcluster legend on the upper/ left panel.
    show_plots, save_plots : bool
        Whether to display or save the plot.
    save_path : str or None
        Path to save the figure (if save_plots is True).
    **plot_kwargs :
        Any extra keyword arguments are passed to downstream plotting functions, including:
            - density_contour options
            - xray_contour options
            - colorbar options
            - region arc options
            - etc.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax1, ax2 : matplotlib.axes.Axes
        The axes for the two panels.
    """


    optical_image_file = _get_optical(cluster)

    # -- Load optical images and WCS --
    with fits.open(optical_image_file) as hdul:
        wcs_optical = WCS(hdul[0].header, naxis=2)

    # -- Setup figure and axes --
    if ax1 is None and ax2 is None:
        if layout == 'horizontal':
            fig = plt.figure(figsize=(14, 8))
            gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.01)
            ax1 = fig.add_subplot(gs[0], projection=wcs_optical)  # left: optical
            ax2 = fig.add_subplot(gs[1], projection=wcs_optical, sharey=ax1)  # right: contours
        else:
            fig = plt.figure(figsize=(6, 10.5))
            gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.02)
            ax1 = fig.add_subplot(gs[0], projection=wcs_optical)  # top: optical
            ax2 = fig.add_subplot(gs[1], projection=wcs_optical, sharex=ax1)  # bottom: contours

    # ---------------- Top (Left) Panel: Optical image with scalebar and legend ----------------

    fig, ax1 = plot_optical(
        optical_image_file,
        cluster=cluster,
        fig=fig, ax=ax1,
        show_plots=False,
        save_plots=False,
        legend_loc=legend_loc,
        **kwargs
    )


    if layout == 'horizontal':
        ax2.tick_params(axis='y', labelleft=False)
    else:
        ax1.tick_params(axis='x', labelbottom=False)



    # ----- Bottom (Right) Panel: Optical image with contours -----

    fig, ax2, handles, labels = plot_optical(
        optical_image_file,
        cluster=cluster,
        fig=fig, ax=ax2,
        show_plots=False,
        save_plots=False,
        xray_fits_file=cluster.xray_file,
        photometric_file=cluster.get_phot_file(),
        bcg_file=cluster.bcg_file,
        show_legend=False,
        show_scalebar=False,
        **kwargs
    )
    legend = ax1.legend(handles=handles, labels=labels, loc=legend_loc, fontsize=10, frameon=True)
    legend.set_zorder(20)

    if layout == 'horizontal':
        fig.subplots_adjust(wspace=0.04, left=0.09, right=0.92, top=0.96, bottom=0.09)
    else:
        fig.subplots_adjust(hspace=0.08, top=0.98, bottom=0.1, left=0.2, right=0.94)


    # plt.tight_layout()
    if save_path is not None and save_plots:
        save_file = os.path.join(save_path, "subcluster_z_scatter.pdf") if os.path.isdir(save_path) else save_path
        fig.savefig(save_file, dpi=450)
    if show_plots:
        plt.show()

    return fig, ax1, ax2

def plot_3panel_optical_subclusters_figure(
    cluster,
    subclusters,
    spec_groups,
    phot_groups,
    combined_configs=None,
    spec_groups_combined=None,
    phot_groups_combined=None,
    combined_indices=None,
    fig=None,
    axes=None,
    layout='vertical',
    show_legend=True,
    legend_loc='upper right',
    show_plots=True,
    save_plots=False,
    save_path=None,
    **kwargs
):
    """
    Plot a 3-panel figure (vertical or horizontal) for cluster images, using custom plot_optical overlays.

    Panels:
        1. Plain optical image (with optional overlays if provided via kwargs)
        2. Optical image + overlays (returns handles/labels for legend)
        3. Optical image + overlays (for regions, contours, etc)

    Parameters
    ----------
    cluster : Cluster
        Cluster object (must provide .get_optical_image(), .xray_file, .phot_file(), .bcg_file).
    subclusters : list[Subcluster]
        Subcluster objects defining regions, colors, and member data.
    fig : matplotlib.figure.Figure, optional
        Existing Figure object to use (otherwise created).
    axes : list of Axes or None
        Existing list of axes [ax1, ax2, ax3] (otherwise created).
    layout : str, optional
        'vertical' (default) or 'horizontal' for 3-panel arrangement.
    show_legend : bool, optional
        Whether to show legend on panel 2 [default: True].
    legend_loc : str, optional
        Legend location for panel 2 [default: 'upper right'].
    show_plots : bool, optional
        Whether to display the plot [default: True].
    save_plots : bool, optional
        Whether to save the plot [default: False].
    save_path : str or None, optional
        Path or directory to save the figure if save_plots is True.
    **kwargs :
        Passed to all plot_optical calls (xray, bcg, contour, etc).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    axes : list of matplotlib.axes.Axes
        List of axes [ax1, ax2, ax3].
    handles, labels : list, list
        Scatter handles/labels from panel 2 for legend building.
    """


    optical_image_file = _get_optical(cluster)

    # -- Load WCS for axis projection
    with fits.open(optical_image_file) as hdul:
        wcs_optical = WCS(hdul[0].header, naxis=2)

    # Figure/axes setup
    if axes is not None and len(axes) == 3:
        ax1, ax2, ax3 = axes
    else:
        if layout == 'horizontal':
            fig = plt.figure(figsize=(21, 7)) if fig is None else fig
            gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.07)
            ax1 = fig.add_subplot(gs[0], projection=wcs_optical)
            ax2 = fig.add_subplot(gs[1], projection=wcs_optical, sharey=ax1)
            ax3 = fig.add_subplot(gs[2], projection=wcs_optical, sharey=ax1)
        else:
            fig = plt.figure(figsize=(7, 18)) if fig is None else fig
            gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1], hspace=0.01)
            ax1 = fig.add_subplot(gs[0], projection=wcs_optical)
            ax2 = fig.add_subplot(gs[1], projection=wcs_optical, sharex=ax1)
            ax3 = fig.add_subplot(gs[2], projection=wcs_optical, sharex=ax1)
    axes = [ax1, ax2, ax3]

    # Panel 1: Plain (or minimally decorated) optical image
    fig, ax1 = plot_optical(
        optical_image_file,
        cluster=cluster,
        fig=fig, ax=ax1,
        show_plots=False,
        save_plots=False,
        **kwargs
    )

    # Panel 2: Optical + overlays, return handles/labels
    fig, ax2, handles, labels = plot_optical(
        optical_image_file,
        cluster=cluster,
        fig=fig, ax=ax2,
        show_plots=False,
        save_plots=False,
        xray_fits_file=cluster.xray_file,
        photometric_file=cluster.get_phot_file(),
        bcg_file=cluster.bcg_file,
        show_legend=False,
        show_scalebar=False,
        **kwargs
    )

    if show_legend:
        leg = ax1.legend(handles=handles, labels=labels, loc=legend_loc, fontsize=10, frameon=True)
        leg.set_zorder(20)

    # Panel 3: Subcluster members and regions
    fig, ax3 = plot_subcluster_members_and_regions(
                                cluster=cluster,
                                subclusters=subclusters,
                                combined_configs=combined_configs,
                                spec_groups=spec_groups,
                                spec_groups_combined=spec_groups_combined,
                                phot_groups=phot_groups,
                                phot_groups_combined=phot_groups_combined,
                                combined_indices=combined_indices,
                                fig=fig, ax=ax3,
                                legend_loc="upper right",
                                show_plots=False,
                                save_plots=False,
                                save_path=None,
                                subcluster_show_scalebar=False,
                                **kwargs
    )
    if layout == 'horizontal':
        ax2.tick_params(axis='y', labelleft=False)
        ax3.tick_params(axis='y', labelleft=False)
    else:
        ax1.tick_params(axis='x', labelbottom=False)
        ax2.tick_params(axis='x', labelbottom=False)

    # Adjust figure spacing
    if layout == 'horizontal':
        fig.subplots_adjust(wspace=0.01, hspace=0.01, top=0.95, bottom=0.1, left=0.08, right=0.98)
    else:
        fig.subplots_adjust(hspace=0.08, top=0.98, bottom=0.07, left=0.1, right=0.99)

    if save_plots and save_path is not None:
        save_file = os.path.join(save_path, "cluster_3panel.pdf") if os.path.isdir(save_path) else save_path
        fig.savefig(save_file, bbox_inches='tight')
    if show_plots:
        plt.show()

    return fig, axes, handles, labels

def plot_combined_4panel_figure(
                        cluster,
                        subclusters,
                        spec_groups,
                        phot_groups,
                        combined_configs=None,
                        spec_groups_combined=None,
                        phot_groups_combined=None,
                        combined_indices=None,
                        legend_loc="upper left",
                        show_plots=True,
                        save_plots=False,
                        save_path=None
):
    """
    Plot a 2x2 figure with subcluster overlays, density/X-ray contours, and redshift information.

    The layout is as follows:
        - Top left:  Optical image with legend
        - Bottom left: Optical image with bcgs and contours
        - Top right:  Redshift-selected member heatmap
        - Bottom right: Subcluster region assignments and bisector arcs


    Parameters
    ----------
    cluster : Cluster
        Cluster object with necessary file references and metadata.
    subclusters : list[Subcluster]
        Subcluster objects defining regions, colors, and member data.
    spec_groups : list of pd.DataFrame
        List of spectroscopic member DataFrames for each subcluster.
    phot_groups : list of pd.DataFrame
        List of photometric member DataFrames for each subcluster.
    legend_loc : str, optional
        Legend location for overlays [default: 'upper left'].
    show_plots : bool, optional
        Whether to display the figure [default: True].
    save_plots : bool, optional
        Whether to save the figure to file [default: False].
    save_path : str or None, optional
        Path or directory to save the figure if save_plots is True.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created Figure object.

    Notes
    -----
    - Panels are created using `plot_2panel_optical_contours` and `plot_redshift_and_subclusters_figure`.
    """
    optical_image_file = _get_optical(cluster)

    # Load WCS
    with fits.open(optical_image_file) as hdul:
        wcs_optical = WCS(hdul[0].header, naxis=2)

    fig = plt.figure(figsize=(13.4, 12))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1], wspace=0.0, hspace=0.04)

    ax1 = fig.add_subplot(gs[0, 0], projection=wcs_optical)
    ax2 = fig.add_subplot(gs[1, 0], projection=wcs_optical)
    ax3 = fig.add_subplot(gs[0, 1], projection=wcs_optical)
    ax4 = fig.add_subplot(gs[1, 1], projection=wcs_optical)

    # LEFT COLUMN
    plot_2panel_optical_contours(
                                cluster=cluster,
                                subclusters=subclusters,
                                fig=fig, ax1=ax1, ax2=ax2,
                                legend_loc=legend_loc,
                                layout="vertical",
                                show_plots=False,
                                save_plots=False,
                                save_path=None
    )

    # RIGHT COLUMN
    _,_,_, sc= plot_redshift_and_subclusters_figure(
                                cluster=cluster,
                                subclusters=subclusters,
                                combined_configs=combined_configs,
                                spec_groups=spec_groups,
                                phot_groups=phot_groups,
                                spec_groups_combined=spec_groups_combined,
                                phot_groups_combined=phot_groups_combined,
                                combined_indices=combined_indices,
                                fig=fig, ax1=ax3, ax2=ax4,
                                legend_loc="lower left",
                                legend_loc_bottom="upper right",
                                show_plots=False,
                                save_plots=False,
                                save_path=None,
                                layout='vertical'
    )


    ax3.set_ylabel('')
    ax3.tick_params(axis='y', labelleft=False, pad=0)
    ax3.legend_.remove()

    ax4.set_ylabel('')
    ax4.tick_params(axis='y', labelleft=False, pad=0)


    fig.subplots_adjust(left=0.085, right=0.90)


    # After scatter plotted in ax3
    pos = ax3.get_position()
    cbar_ax = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.015, pos.height])
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label("Redshift (z)")
    cbar.locator = MultipleLocator(0.005)
    cbar.formatter = FormatStrFormatter('%.3f')
    cbar.update_ticks()


    if save_plots and save_path is not None:
        save_file = os.path.join(save_path, "combined_4panel.pdf") if os.path.isdir(save_path) else save_path
        fig.savefig(save_file, bbox_inches='tight')
    if show_plots:
        plt.show()
    return fig

def plot_subcluster_regions_and_histograms(
    cluster,
    subclusters,
    spec_groups,
    phot_groups,
    spec_groups_combined=None,
    phot_groups_combined=None,
    combined_configs=None,
    combined_indices=None,
    legend_loc="upper right",
    show_plots=True,
    save_plots=False,
    save_path=None,
    **kwargs
):
    """
    Plot a stacked figure with subcluster redshift histograms and a spatial regions panel.

    This function creates a figure with N histogram panels (one per subcluster) showing the redshift
    distribution for each region, followed by a bottom panel that overlays subcluster member galaxies,
    region boundaries, and photometric members on the optical image of the cluster.

    - Histogram panel heights are scaled dynamically by galaxy count.


    Parameters
    ----------
    cluster : Cluster
        Cluster object with necessary metadata (spec_file, z_min, z_max, etc).
    subclusters : list[Subcluster]
        Subcluster objects (color, region_center, primary_bcg.z, display_label used for annotations).
    spec_groups : list of pd.DataFrame
        Each DataFrame contains spectroscopic members for a subcluster.
    phot_groups : list of pd.DataFrame
        Each DataFrame contains photometric members for a subcluster.
    legend_loc : str, optional
        Legend location for the spatial region panel [default: 'upper right'].
    show_plots : bool, optional
        Whether to display the plot [default: True].
    save_plots : bool, optional
        Whether to save the plot to file [default: False].
    save_path : str or None, optional
        Path or directory to save figure if save_plots is True.
    **kwargs :
        Additional arguments passed to region/member overlay plotting.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created Figure object.
    """
    # analyze_group not yet extracted into the new package
    from cluster_pipeline.subclusters.statistics import analyze_group



    # Prepare data per subcluster
    if combined_configs is not None:
        colors = [sub.color for sub in combined_configs]
        z_group = spec_groups_combined
        print("Using combined subcluster configurations for colors and data.")
    else:
        colors = [sub.color for sub in subclusters]
        z_group = spec_groups

    print(f"Colors: {colors}")





    z_data, vel_data = analyze_group(z_group)


    # outputs['tex_path'] and outputs['pairs_csv'] have the written files


    n_subclusters = len(z_data)
    print(f"Number of subclusters: {n_subclusters}")
    print(f"Subclusters: {subclusters}")
    hist_bins = np.linspace(min(np.concatenate(z_data)), max(np.concatenate(z_data)), 24)
    y_max_list = [np.histogram(zs, bins=hist_bins)[0].max() for zs in z_data]
    y_max_list = [2 if val == 1 else val for val in y_max_list]
    ratio = (np.array(y_max_list) * n_subclusters) / (sum(y_max_list) * 6)
    height_ratios = ratio.tolist() + [1]  # image panel taller

    fig = plt.figure(figsize=(6, 6.75 + n_subclusters))
    gs = GridSpec(n_subclusters + 1, 1, height_ratios=height_ratios, hspace=0.0)

    ax_list = [fig.add_subplot(gs[i]) for i in range(0, n_subclusters)]
    ax_img = fig.add_subplot(gs[-1], projection=WCS(fits.getheader(_get_optical(cluster), 0), naxis=2))



        # ---------------- Top Panels: Histograms ----------------
    hist_bins = np.linspace(min(np.concatenate(z_data)), max(np.concatenate(z_data)), 24)
    group_labels = []
    for i, ax in enumerate(ax_list):
        if subclusters[i].display_label in group_labels:
            j = i+1
        else:
            j = i
            group_labels.append(subclusters[i].display_label)
        zs = z_data[i]
        z_mean = vel_data[i][1]
        sigma_v = vel_data[i][2]
        color = colors[i] if colors[i] != 'white' else 'gainsboro'
        z_bcg = subclusters[j].primary_bcg.z if subclusters[j].primary_bcg is not None else None
        bcg_label = subclusters[j].display_label

        ax.hist(zs, bins=hist_bins, color=color, edgecolor='black')
        ax.axvline(z_mean, color='black', lw=1.5, linestyle='-', label=f"$\\bar{{z}}$ = {z_mean:.4f}")
        if z_bcg is not None:
            ax.axvline(z_bcg, color='tab:red', lw=1.5, linestyle='-', label=f"BCG {bcg_label}")
        ax.set_xlim(hist_bins[0]-0.002, hist_bins[-1]+0.002)
        ax.set_ylim(0, y_max_list[i]*1.1)
        ax.legend(fontsize=9, loc='upper right')
        ax.yaxis.set_major_locator(MultipleLocator(2))
        # ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax.annotate(f"Subcluster {i+1}\nN = {len(zs)}\n$\\sigma_v$ = {round(sigma_v, -1):.0f} km/s",
                    xy=(0, 1), xycoords='axes fraction', xytext=(6, -6),
                    textcoords='offset points', ha='left', va='top',
                    fontsize=9, bbox=dict(facecolor='white', edgecolor='gray', alpha=0.7))
        if i < n_subclusters - 1:
            ax.tick_params(labelbottom=False)
            ax.set_xticks([])
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: '' if x == 0 else f'{int(x)}'))
            ax.tick_params(axis='x', which='major', pad=0)
        else:
            ax.set_xlabel("Redshift (z)")


    # ---------------- Top Panel: Contours + Regions ----------------
    plot_subcluster_members_and_regions(
                                    cluster=cluster,
                                    subclusters=subclusters,
                                    combined_configs=combined_configs,
                                    spec_groups=spec_groups,
                                    spec_groups_combined=spec_groups_combined,
                                    phot_groups=phot_groups,
                                    phot_groups_combined=phot_groups_combined,
                                    fig=fig, ax=ax_img,
                                    legend_loc=legend_loc,
                                    combined_indices=combined_indices,
                                    show_plots=False,
                                    save_plots=False,
                                    save_path=os.path.join(cluster.image_path, "subcluster_map.pdf"),
                                    **kwargs
    )


    ax_img.set_xlabel('')
    ax_img.set_xticklabels([])
    ax_img.set_ylabel('Decl.')
    ax_img.set_xlabel('R.A.')
    ax_img.set_aspect('equal')
    ax_img.set_title(" ", fontsize=6)

    fig.subplots_adjust(top=0.96, bottom=0.02, left=0.2, right=0.94)
    # Get position of top and bottom histogram axes in figure coordinates
    bbox_top = ax_list[0].get_position()
    bbox_bot = ax_list[-1].get_position()

    # Compute center between top and bottom axes
    y_c = (bbox_top.y1 + bbox_bot.y0) / 2
    x_l = bbox_top.x0 - 0.16

    fig.text(x_l, y_c, "Galaxy Counts", va='center', ha='center', rotation='vertical')



    if save_plots and save_path is not None:
        save_file = os.path.join(save_path, "subcluster_histograms.pdf") if os.path.isdir(save_path) else save_path
        fig.savefig(save_file, dpi=450)
    if show_plots:
        plt.show()
    return fig
