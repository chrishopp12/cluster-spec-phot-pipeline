#!/usr/bin/env python3
"""
cmd.py

Color-Magnitude Diagram Plotting
---------------------------------------------------------
Generates CMD-related plots for galaxy cluster analysis:

    - ``plot_color_magnitude``   -- single CMD scatter with optional spec-z overlay
    - ``plot_all_color_magnitude`` -- 2x3 grid (PanSTARRS + Legacy, three colors)
    - ``plot_cmd``               -- CMD with red-sequence fit line and member overlay
    - ``plot_spatial``           -- RA/Dec spatial map of photometric + spec-z members
    - ``plot_density_contours``  -- luminosity-weighted KDE density map
    - ``run_cluster_plots``      -- convenience wrapper that runs the full plot suite

Supported surveys: Legacy, PanSTARRS
Supported color bands: g-r, r-i, g-i
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

from astropy.wcs import WCS

from scipy.stats import gaussian_kde

from cluster_pipeline.utils import get_color_mag_functions, split_members_by_spec
from cluster_pipeline.utils.coordinates import get_skycoord
from cluster_pipeline.io.catalogs import get_redseq_filename
from cluster_pipeline.plotting.common import finalize_figure

if TYPE_CHECKING:
    from cluster_pipeline.models.cluster import Cluster


# ------------------------------------
# Column name constants
# ------------------------------------
RA_COL = "RA"
DEC_COL = "Dec"
Z_COL = "z"


# ------------------------------------
# Diagnostic CMD Plots
# ------------------------------------

def plot_color_magnitude(
    phot_df: pd.DataFrame,
    color_col: str,
    mag_col: str,
    title: str | None = None,
    ax: plt.Axes | None = None,
    plot_legend: bool = False,
    color: str = "gray",
    ylim: tuple[float, float] = (-4, 4),
    xlim: tuple[float, float] = (10, 27.5),
    spec_df: pd.DataFrame | None = None,
    z_min: float | None = None,
    z_max: float | None = None,
    overlay_label: str = "Spec-z members",
) -> None:
    """
    Plot a single color-magnitude diagram with optional spectroscopic overlay.

    Scatters color vs magnitude from a photometry DataFrame.  If *spec_df*
    is provided, confirmed spectroscopic members are overlaid (optionally
    filtered to *z_min* -- *z_max*).

    Parameters
    ----------
    phot_df : pd.DataFrame
        Photometric catalog containing *color_col* and *mag_col*.
    color_col : str
        Column name for color (e.g. ``"g_r"``).
    mag_col : str
        Column name for magnitude (e.g. ``"rmag"``).
    title : str, optional
        Subplot title.
    ax : matplotlib.axes.Axes, optional
        Axis to draw on.  If None, a new figure is created.
    plot_legend : bool
        Show legend. [default: False]
    color : str
        Color for photometric points. [default: "gray"]
    ylim, xlim : tuple of float
        Axis limits for color (y) and magnitude (x).
    spec_df : pd.DataFrame, optional
        Spectroscopic catalog to overlay.
    z_min, z_max : float, optional
        If both provided, filter *spec_df* to this redshift range.
    overlay_label : str
        Legend label for the spectroscopic overlay. [default: "Spec-z members"]
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    # Main photometric scatter
    valid = phot_df[[color_col, mag_col]].dropna()
    ax.scatter(
        valid[mag_col], valid[color_col],
        s=5, color=color, label="All photometric sources",
    )

    # Optional spectroscopic overlay
    if spec_df is not None:
        spec_valid = spec_df.copy()
        if z_min is not None and z_max is not None and Z_COL in spec_valid.columns:
            spec_valid = spec_valid[
                (spec_valid[Z_COL] >= z_min) & (spec_valid[Z_COL] <= z_max)
            ]
        spec_valid = spec_valid.dropna(subset=[color_col, mag_col])
        if not spec_valid.empty:
            ax.scatter(
                spec_valid[mag_col], spec_valid[color_col],
                s=20, color="crimson", marker="o",
                label=overlay_label, edgecolor="black",
            )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_ylabel(color_col.replace("_", " - "))
    ax.set_xlabel(f"{mag_col[0]} magnitude")

    if title:
        ax.set_title(title)
    if plot_legend:
        ax.legend()


def plot_all_color_magnitude(
    cluster: Cluster,
    panstarrs_df: pd.DataFrame,
    legacy_df: pd.DataFrame,
    panstarrs_spec: pd.DataFrame | None = None,
    legacy_spec: pd.DataFrame | None = None,
    z_min: float | None = None,
    z_max: float | None = None,
    show_plots: bool = True,
    save_plots: bool = False,
    save_path: str | None = None,
) -> None:
    """
    Plot a 2x3 grid of color-magnitude diagrams for PanSTARRS and Legacy.

    Row 0: PanSTARRS  (g-r vs r, r-i vs i, g-i vs i)
    Row 1: Legacy     (g-r vs r, r-i vs i, g-i vs i)

    Optionally overlays spectroscopically confirmed members, filtered to
    a redshift range if *z_min* / *z_max* are supplied.

    Parameters
    ----------
    cluster : Cluster
        Cluster object (uses ``z_min`` / ``z_max`` as fallback).
    panstarrs_df : pd.DataFrame
        PanSTARRS photometric catalog (needs g_r, r_i, g_i, rmag, imag).
    legacy_df : pd.DataFrame
        Legacy photometric catalog (same column requirements).
    panstarrs_spec, legacy_spec : pd.DataFrame, optional
        Spectroscopic match catalogs to overlay on the respective rows.
    z_min, z_max : float, optional
        Redshift bounds for filtering overlaid spec-z members.
        Falls back to ``cluster.z_min`` / ``cluster.z_max``.
    show_plots, save_plots : bool
        Plot display/save controls.
    save_path : str or None
        Directory for saved plot.
    """
    # Fall back to cluster redshift bounds
    z_min = z_min if z_min is not None else cluster.z_min
    z_max = z_max if z_max is not None else cluster.z_max

    fig, axs = plt.subplots(2, 3, figsize=(16, 12), sharex=False, sharey=False)

    # -- PanSTARRS row --
    plot_color_magnitude(
        panstarrs_df, "g_r", "rmag", "PanSTARRS: g-r vs r",
        ax=axs[0, 0], plot_legend=True,
        spec_df=panstarrs_spec, z_min=z_min, z_max=z_max,
    )
    plot_color_magnitude(
        panstarrs_df, "r_i", "imag", "PanSTARRS: r-i vs i",
        ax=axs[0, 1],
        spec_df=panstarrs_spec, z_min=z_min, z_max=z_max,
    )
    plot_color_magnitude(
        panstarrs_df, "g_i", "imag", "PanSTARRS: g-i vs i",
        ax=axs[0, 2],
        spec_df=panstarrs_spec, z_min=z_min, z_max=z_max,
    )

    # -- Legacy row --
    plot_color_magnitude(
        legacy_df, "g_r", "rmag", "Legacy: g-r vs r",
        ax=axs[1, 0],
        spec_df=legacy_spec, z_min=z_min, z_max=z_max,
    )
    plot_color_magnitude(
        legacy_df, "r_i", "imag", "Legacy: r-i vs i",
        ax=axs[1, 1],
        spec_df=legacy_spec, z_min=z_min, z_max=z_max,
    )
    plot_color_magnitude(
        legacy_df, "g_i", "imag", "Legacy: g-i vs i",
        ax=axs[1, 2],
        spec_df=legacy_spec, z_min=z_min, z_max=z_max,
    )

    fig.tight_layout()

    finalize_figure(
        fig,
        show_plots=show_plots,
        save_plots=save_plots,
        save_path=save_path,
        filename="CMDs.pdf",
    )


# ------------------------------------
# Red-Sequence Plotting Functions
# ------------------------------------

def plot_spatial(
    member_df: pd.DataFrame,
    phot_df: pd.DataFrame | None = None,
    *,
    zmin: float | None = None,
    zmax: float | None = None,
    survey: str = "Legacy",
    color_type: str = "g-r",
    save_name: str = "",
    with_cmap: bool = False,
    show_plots: bool = True,
    save_plots: bool = False,
    save_path: str | None = None,
) -> None:
    """
    Spatial plot of photometric and spectroscopic red-sequence members.

    Parameters
    ----------
    member_df : pd.DataFrame
        Red-sequence member catalog (must contain RA, Dec, and z columns).
    phot_df : pd.DataFrame or None
        Full photometric catalog for background scatter. If None, only
        member sources are shown.
    zmin, zmax : float or None
        Redshift range for colorbar labeling.
    survey, color_type : str
        Labels for plot title.
    save_name : str
        Suffix for saved filename.
    with_cmap : bool
        Color spec-z members by redshift. [default: False]
    show_plots, save_plots : bool
        Plot display/save controls.
    save_path : str or None
        Directory for saved plot.
    """
    spec_member_df, phot_member_df = split_members_by_spec(member_df)

    spec_member_coords = get_skycoord(spec_member_df)
    phot_member_coords = get_skycoord(phot_member_df)

    if phot_df is None:
        phot_coords = phot_member_coords
        phot_color = "lightgray"
    else:
        phot_coords = get_skycoord(phot_df)
        phot_color = "crimson"

    # Coordinate extent for WCS
    ra_max = phot_coords.ra.deg.max()
    ra_min = phot_coords.ra.deg.min()
    dec_max = phot_coords.dec.deg.max()
    dec_min = phot_coords.dec.deg.min()

    # Define WCS
    nx, ny = 1000, 1000
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [nx / 2, ny / 2]
    wcs.wcs.cdelt = [-(ra_max - ra_min) / nx, (dec_max - dec_min) / ny]
    wcs.wcs.crval = [(ra_max + ra_min) / 2, (dec_max + dec_min) / 2]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    # WCS-aware figure
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection=wcs)

    if phot_df is not None:
        ax.scatter(
            phot_coords.ra.deg, phot_coords.dec.deg,
            transform=ax.get_transform("world"),
            c="lightgray", s=3, label="All Sources",
        )

    ax.scatter(
        phot_member_coords.ra.deg, phot_member_coords.dec.deg,
        transform=ax.get_transform("world"),
        c=phot_color, s=5, label="Photometric Members",
    )

    if with_cmap:
        sc = ax.scatter(
            spec_member_coords.ra.deg, spec_member_coords.dec.deg,
            transform=ax.get_transform("world"),
            c=spec_member_df[Z_COL], cmap="viridis",
            s=20, label="Spec-z Members", edgecolors="black",
        )
        cbar = fig.colorbar(sc, ax=ax, label="Redshift", shrink=0.72)
        if zmin is not None and zmax is not None:
            cbar.locator = MultipleLocator((zmax - zmin) / 5)
            cbar.formatter = FormatStrFormatter("%.3f")
        else:
            cbar.locator = MultipleLocator(0.5)
            cbar.formatter = FormatStrFormatter("%.2f")
        cbar.update_ticks()
    else:
        ax.scatter(
            spec_member_coords.ra.deg, spec_member_coords.dec.deg,
            transform=ax.get_transform("world"),
            c="blue", s=20, label="Spec-z Members",
            marker="o", edgecolors="black",
        )

    ax.coords[0].set_axislabel("R.A.")
    ax.coords[1].set_axislabel("Decl.")
    ax.set_title(f"{survey} {color_type} Red Sequence Candidates")
    ax.legend(loc="upper left")
    ax.set_box_aspect(1)

    plt.tight_layout()

    finalize_figure(
        fig,
        show_plots=show_plots,
        save_plots=save_plots,
        save_path=save_path,
        filename=f"{survey}_{color_type}_{save_name}_spatial.pdf",
    )


def plot_cmd(
    member_df: pd.DataFrame,
    phot_df: pd.DataFrame,
    a: float,
    b: float,
    *,
    survey: str = "Legacy",
    color_type: str = "g-r",
    save_name: str = "",
    ylim: tuple[float, float] = (-4.0, 4.0),
    xlim: tuple[float, float] = (15.0, 27.5),
    show_plots: bool = True,
    save_plots: bool = False,
    with_cmap: bool = False,
    save_path: str | None = None,
) -> None:
    """
    Color-magnitude diagram with red-sequence fit line and member overlay.

    Parameters
    ----------
    member_df : pd.DataFrame
        Red-sequence member catalog.
    phot_df : pd.DataFrame
        Full photometric catalog (background scatter).
    a, b : float
        Slope and intercept of the red-sequence linear fit.
    survey, color_type : str
        Labels for plot title and filename.
    save_name : str
        Suffix for saved filename.
    ylim, xlim : tuple of float
        Axis limits for color (y) and magnitude (x).
    show_plots, save_plots : bool
        Plot display/save controls.
    with_cmap : bool
        Color spec-z members by redshift. [default: False]
    save_path : str or None
        Directory for saved plot.
    """
    required_cols, color_label, mag_label, color_func, mag_func = (
        get_color_mag_functions(color_type)
    )

    spec_member_df, phot_member_df = split_members_by_spec(member_df)

    mag_all = mag_func(phot_df)
    color_all = color_func(phot_df)

    mag_spec = mag_func(spec_member_df)
    color_spec = color_func(spec_member_df)

    mag_phot = mag_func(phot_member_df)
    color_phot = color_func(phot_member_df)

    mag_range = np.linspace(mag_all.min(), mag_all.max(), 100)

    # CMD figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # All photometric sources
    ax.scatter(mag_all, color_all, c="lightgray", s=5, label="All photometric sources")

    # Photometric members
    if not phot_member_df.empty:
        ax.scatter(
            mag_phot, color_phot,
            c="crimson", s=10, label="Red sequence candidates",
        )

    # Spectroscopic members
    if not spec_member_df.empty:
        if with_cmap:
            sc = ax.scatter(
                mag_spec, color_spec,
                c=spec_member_df[Z_COL], cmap="viridis",
                s=25, edgecolor="k", linewidth=0.3,
                label="Spec-z cluster members",
            )
            cbar = fig.colorbar(sc, ax=ax, label="Redshift")
            zmin = spec_member_df[Z_COL].min()
            zmax = spec_member_df[Z_COL].max()
            cbar.locator = MultipleLocator((zmax - zmin) / 5)
            cbar.formatter = FormatStrFormatter("%.3f")
            cbar.update_ticks()
        else:
            ax.scatter(
                mag_spec, color_spec,
                c="blue", s=20, label="Spec-z cluster members",
                marker="o", edgecolors="black",
            )

    ax.plot(mag_range, a * mag_range + b, "k--", label="Red sequence fit")

    ax.set_ylabel(color_label)
    ax.set_xlabel(f"{mag_label} magnitude")
    ax.set_title(f"{survey} {color_type} CMD with Red Sequence Fit")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.legend()
    plt.tight_layout()

    finalize_figure(
        fig,
        show_plots=show_plots,
        save_plots=save_plots,
        save_path=save_path,
        filename=f"{survey}_{color_type}_{save_name}_cmd.pdf",
    )


# ------------------------------------
# Density Contours
# ------------------------------------

def plot_density_contours(
    phot_df: pd.DataFrame,
    *,
    wcs: WCS | None = None,
    levels: int = 30,
    bandwidth: float = 0.2,
    zmin: float | None = None,
    zmax: float | None = None,
    base_color: str = "black",
    color_type: str = "g-r",
    lum_weight: str = "lum_weight_r",
    survey: str = "Legacy",
    fill: bool = True,
    save_plots: bool = False,
    show_plots: bool = False,
    save_path: str | None = None,
) -> None:
    """
    Luminosity-weighted 2D KDE density contour plot in RA-Dec space.

    Parameters
    ----------
    phot_df : pd.DataFrame
        Member catalog with RA, Dec, and luminosity weight columns.
    wcs : WCS, optional
        WCS for the projection. If None, one is constructed from the data.
    levels : int
        Number of contour levels. [default: 30]
    bandwidth : float
        KDE bandwidth (larger = smoother). [default: 0.2]
    zmin, zmax : float, optional
        Redshift range for title annotation.
    base_color : str
        Contour line color when *fill* is False. [default: "black"]
    color_type : str
        Color index label. [default: "g-r"]
    lum_weight : str
        Name of luminosity weight column. [default: "lum_weight_r"]
    survey : str
        Survey label. [default: "Legacy"]
    fill : bool
        Filled contours (True) or lines only. [default: True]
    save_plots, show_plots : bool
        Plot save/display controls.
    save_path : str or None
        Directory for saved plot.
    """
    weights = phot_df[lum_weight].values
    ra = phot_df[RA_COL].values
    dec = phot_df[DEC_COL].values

    ra = np.asarray(ra)
    dec = np.asarray(dec)
    weights = np.asarray(weights) if weights is not None else None

    # Mask invalid entries
    valid_mask = np.isfinite(ra) & np.isfinite(dec)
    if weights is not None:
        valid_mask &= np.isfinite(weights)
        weights = weights[valid_mask]
    ra = ra[valid_mask]
    dec = dec[valid_mask]

    npix = 300
    if wcs is None:
        nx, ny = npix, npix
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [nx / 2, ny / 2]
        wcs.wcs.cdelt = [
            -(ra.max() - ra.min()) / nx,
            (dec.max() - dec.min()) / ny,
        ]
        wcs.wcs.crval = [(ra.max() + ra.min()) / 2, (dec.max() + dec.min()) / 2]
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    x, y = wcs.wcs_world2pix(ra, dec, 0)
    xy = np.vstack([x, y])

    # KDE
    kde = gaussian_kde(xy, bw_method=bandwidth, weights=weights)

    # Evaluation grid
    x_grid = np.linspace(x.min(), x.max(), npix)
    y_grid = np.linspace(y.min(), y.max(), npix)
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
    density = kde(np.vstack([x_mesh.ravel(), y_mesh.ravel()])).reshape(x_mesh.shape)

    # WCS-aware figure
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection=wcs)

    if fill:
        ax.contourf(x_mesh, y_mesh, density, levels=levels, cmap="inferno")
    else:
        ax.contour(x_mesh, y_mesh, density, levels=levels, colors=base_color)

    title = f"{survey} {color_type} Density Map"
    if zmin is not None and zmax is not None:
        title += f" ({zmin:.3f} < z < {zmax:.3f})"
    ax.set_title(title)

    ax.set_xlabel("R.A.")
    ax.set_ylabel("Decl.")
    ax.set_box_aspect(1)

    plt.tight_layout()

    finalize_figure(
        fig,
        show_plots=show_plots,
        save_plots=save_plots,
        save_path=save_path,
        filename=f"{survey}_{color_type}_lum_density.pdf",
    )


# ------------------------------------
# Full Plot Suite
# ------------------------------------

def run_cluster_plots(
    cluster: Cluster,
    fov_size: float,
    survey: str,
    color_type: str,
    *,
    ra_offset: float = 0,
    dec_offset: float = 0,
    bandwidth: float = 0.1,
    show_plots: bool = True,
    save_plots: bool = True,
) -> None:
    """
    Generate the full suite of spatial, CMD, and density plots for one survey/color.

    Parameters
    ----------
    cluster : Cluster
        Cluster object with paths and metadata.
    fov_size : float
        Field of view in arcmin for optical image retrieval.
    survey : str
        ``"Legacy"`` or ``"PanSTARRS"``.
    color_type : str
        One of ``"g-r"``, ``"r-i"``, ``"g-i"``.
    ra_offset, dec_offset : float
        RA/Dec offsets in arcmin. [default: 0.0]
    bandwidth : float
        KDE bandwidth for density plots. [default: 0.1]
    show_plots, save_plots : bool
        Plot display/save controls.
    """
    from cluster_pipeline.catalog.redsequence import fit_red_sequence
    from cluster_pipeline.io.images import get_optical_image
    from cluster_pipeline.plotting.optical import plot_optical

    img_dir = os.path.join(cluster.photometry_path, "Images")
    os.makedirs(img_dir, exist_ok=True)

    # Load photometric + matched catalogs based on survey
    survey_lower = survey.lower()
    phot_file = os.path.join(cluster.photometry_path, f"photometry_{survey_lower}.csv")
    matched_file = os.path.join(cluster.photometry_path, f"{survey_lower}_matched.csv")

    if not os.path.isfile(phot_file):
        print(f"  Photometry file not found: {phot_file} — skipping CMD plots")
        return
    if not os.path.isfile(matched_file):
        print(f"  Matched file not found: {matched_file} — skipping CMD plots")
        return

    phot_df = pd.read_csv(phot_file)
    matched_df = pd.read_csv(matched_file)

    # Member catalog (deduplicated, has z column + member_type)
    members_file = os.path.join(cluster.members_path, "cluster_members.csv")
    if not os.path.isfile(members_file):
        # Fall back to redseq file
        members_file = get_redseq_filename(cluster.members_path, survey, color_type)
    if not os.path.isfile(members_file):
        members_file = get_redseq_filename(cluster.photometry_path, survey, color_type)
    if not os.path.isfile(members_file):
        print(f"  Member catalog not found — skipping CMD plots for {survey} {color_type}")
        return
    redseq_df = pd.read_csv(members_file)

    # Luminosity weight — prefer i-band for r-i and g-i colors, fall back to r-band
    if color_type.replace(" ", "").replace("-", "") in ("gr",):
        lum_weight = "lum_weight_r"
    elif "lum_weight_i" in redseq_df.columns and redseq_df["lum_weight_i"].notna().any():
        lum_weight = "lum_weight_i"
    else:
        lum_weight = "lum_weight_r"

    a, b = fit_red_sequence(
        matched_df,
        color_type=color_type,
        zmin=cluster.z_min,
        zmax=cluster.z_max,
    )

    # 1. Spatial: all sources + members
    plot_spatial(
        member_df=redseq_df,
        phot_df=phot_df,
        zmin=cluster.z_min,
        zmax=cluster.z_max,
        survey=survey,
        color_type=color_type,
        save_name="All",
        show_plots=show_plots,
        save_plots=save_plots,
        save_path=img_dir,
        with_cmap=True,
    )

    # 2. Spatial: members only
    plot_spatial(
        member_df=redseq_df,
        zmin=cluster.z_min,
        zmax=cluster.z_max,
        survey=survey,
        color_type=color_type,
        save_name="Members",
        show_plots=show_plots,
        save_plots=save_plots,
        save_path=img_dir,
        with_cmap=False,
    )

    # 3. Spatial: members only with cmap
    plot_spatial(
        member_df=redseq_df,
        zmin=cluster.z_min,
        zmax=cluster.z_max,
        survey=survey,
        color_type=color_type,
        save_name="Members_cmap",
        show_plots=show_plots,
        save_plots=save_plots,
        save_path=img_dir,
        with_cmap=True,
    )

    # 4. CMD
    plot_cmd(
        member_df=redseq_df,
        phot_df=phot_df,
        a=a,
        b=b,
        survey=survey,
        color_type=color_type,
        show_plots=show_plots,
        save_plots=save_plots,
        with_cmap=False,
        save_path=img_dir,
    )

    # 5. CMD with cmap
    plot_cmd(
        member_df=redseq_df,
        phot_df=phot_df,
        a=a,
        b=b,
        survey=survey,
        color_type=color_type,
        save_name="cmap",
        show_plots=show_plots,
        save_plots=save_plots,
        with_cmap=True,
        save_path=img_dir,
    )

    # 6. Density heatmap
    plot_density_contours(
        redseq_df,
        bandwidth=bandwidth,
        color_type=color_type,
        lum_weight=lum_weight,
        survey=survey,
        show_plots=show_plots,
        save_plots=save_plots,
        save_path=img_dir,
    )

    # 7. Density contours over optical image
    optical_image_path = get_optical_image(
        cluster, fov=fov_size, ra_offset=ra_offset, dec_offset=dec_offset,
    )

    phot_file = cluster.get_phot_file()
    bcg_file = cluster.bcg_file

    plot_optical(
        optical_image_path,
        cluster=cluster,
        show_plots=show_plots,
        save_plots=save_plots,
        save_path=img_dir,
        photometric_file=phot_file,
        bcg_file=bcg_file,
    )
