#!/usr/bin/env python3
"""
color_magnitude_plotting.py

Cluster Red Sequence Plotting Utility
---------------------------------------------------------

This script generates a set of standard plots for galaxy clusters using
archival photometric and redshift data. Plots include:

    - Spatial overlays of photometric and spectroscopic sources
    - Color-magnitude diagrams (CMDs) with spectroscopic overlay
    - Density heatmap
    - Density contours over optical images

Supported surveys: Legacy, PanSTARRS
Supported color bands: g-r, r-i, g-i

Requirements:
    - astropy
    - astroquery
    - scipy
    - numpy
    - pandas
    - matplotlib
    - cluster.py (local)
    - my_utils.py (local)


Usage:
    python color_magnitude_plotting.py CLUSTER_ID [options]

Example:
    python color_magnitude_plotting.py "Abell 2355" --zmin 0.215 --zmax 0.245
        --survey Legacy --color g-r --fov 8 --save-plots True

Options:

    --zmin        <float>    Minimum cluster redshift                  [required]
    --zmax        <float>    Maximum cluster redshift                  [required]
    --survey      <str>      Survey to use: Legacy or PanSTARRS        [required]
    --color       <str>      Color band: g-r, r-i, or g-i              [required]
    --fov         <float>    Field of view (arcmin) for images         [default: 6]
    --base-path   <str>      Science base path                         [default: cwd]
    --ra-offset   <float>    RA offset in arcmin                       [default: 0.0]
    --dec-offset  <float>    Dec offset in arcmin                      [default: 0.0]
    --bandwidth   <float>    KDE bandwidth for density contours        [default: 0.1]
    --save-plots  <bool>     Save plots to file                        [default: True]



Notes:
    - "CLUSTER_ID" positional arg can be input as a cluster identifier mapped to a cluster name (eg. '1219' or '0881900301') or the
    full RMJ/Abell name (possibly others).
"""
from __future__ import annotations

import os
import argparse

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

from astropy.wcs import WCS

from scipy.stats import gaussian_kde

from cluster import Cluster
from my_utils import str2bool, get_redseq_filename, finalize_figure, get_color_mag_functions, split_members_by_spec, get_skycoord, plot_optical



plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14


# ------------------------------------
# Defaults/ Constants
# ------------------------------------
DEFAULT_MAG = 'rmag'

RA_COL = 'RA'
DEC_COL = 'Dec'
Z_COL = 'z'
SIGMA_Z_COL = 'sigma_z'
SPEC_SOURCE_COL = 'spec_source'
PHOT_SOURCE_COL = 'phot_source'


# ------------------------------------
# Plotting Functions
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
        save_path: str | None = None
    ) -> None:
    """
    Generate a spatial plot of photometric and spectroscopic members.

    Parameters
    ----------
    member_df: pd.DataFrame
        DataFrame containing the final member catalog.
    phot_df : pd.DataFrame | None
        DataFrame containing RA and Dec of all photometric sources.
    zmin, zmax : float or None
        Optional redshift range for colorbar labeling.
    survey, color_type : str
        Plot labeling. [default: "Legacy", "g-r"]
    save_name : str
        Base name for saving plots. [default: ""]
    with_cmap : bool
        If True, color spectroscopic members by redshift. [default: False]
    show_plots, save_plots : bool
        Plot controls passed through to ``finalize_figure``.
    save_path : str or None, optional
        Directory (or full file path) for plot saving.

    Returns
    -------
    None
        The function displays or saves the generated spatial plot.
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

    # Extract min/max values as floats (not Quantity)
    ra_max = phot_coords.ra.deg.max()
    ra_min = phot_coords.ra.deg.min()
    dec_max = phot_coords.dec.deg.max()
    dec_min = phot_coords.dec.deg.min()

    # Define WCS
    nx = 1000
    ny = 1000
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [nx / 2, ny / 2]
    wcs.wcs.cdelt = [-(ra_max - ra_min) / nx, (dec_max - dec_min) / ny] 
    wcs.wcs.crval = [(ra_max + ra_min) / 2, (dec_max + dec_min) / 2]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    # WCS-aware plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection=wcs)

    if phot_df is not None:
        ax.scatter(phot_coords.ra.deg, phot_coords.dec.deg, transform=ax.get_transform('world'),
                c="lightgray", s=3, label="All Sources")
    ax.scatter(phot_member_coords.ra.deg, phot_member_coords.dec.deg, transform=ax.get_transform('world'),
            c=phot_color, s=5, label="Photometric Members")
    
    if with_cmap:
        sc = ax.scatter(spec_member_coords.ra.deg, spec_member_coords.dec.deg, transform=ax.get_transform('world'),
            c=spec_member_df[Z_COL], cmap="viridis", s=20, label="Spec-z Members", edgecolors="black")
        cbar = fig.colorbar(sc, ax=ax, label="Redshift", shrink=0.72)
        if zmin is not None and zmax is not None:
            cbar.locator = MultipleLocator((zmax - zmin) / 5)
            cbar.formatter = FormatStrFormatter('%.3f')
        else:
            cbar.locator = MultipleLocator(0.5)
            cbar.formatter = FormatStrFormatter('%.2f')
        cbar.update_ticks()

    else:
        ax.scatter(spec_member_coords.ra.deg, spec_member_coords.dec.deg, transform=ax.get_transform('world'),
                c="blue", s=20, label="Spec-z Members", marker="o", edgecolors="black")

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
        filename=f"{survey}_{color_type}_{save_name}_spatial.pdf"
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
    Plot the color-magnitude diagram with fit, photometric and spec-z members.

    Parameters
    ----------
    photometry_dir : str
        Path to directory containing photometric csvs. Builds subdir /Images to save figs.
    mag, color : np.ndarray
        Arrays of magnitudes and colors (photometric).
    a, b : float
        Slope and intercept of red sequence fit.
    mag_label, color_label : str
        Axis labels. [default: "r", "g - r"]
    survey, color_type : str
        Plot labeling. [default: "PanSTARRS, "g-r"]
    xlim, ylim : tuple of float
        X-axis (mag), Y-axis (color) plot limits [default: (10, 27.5), (-4,4)]
    phot_clean : pd.DataFrame, optional
        DataFrame with 'on_red_sequence' column.
    color_spec, mag_spec : np.ndarray, optional
        Spectroscopic source colors and mags.
    save_flag: bool
        If True, saves plots to photometry_dir/Images/*.csv [default: True]
    save_name: str
        Suffix for saved plot filenames.
    Returns
    -------
    None
        The function displays or saves the generated plot.
    """

    required_cols, color_label, mag_label, color_func, mag_func = get_color_mag_functions(color_type)

    spec_member_df, phot_member_df = split_members_by_spec(member_df)

    mag_all = mag_func(phot_df)
    color_all = color_func(phot_df)

    mag_spec = mag_func(spec_member_df)
    color_spec = color_func(spec_member_df)

    mag_phot = mag_func(phot_member_df)
    color_phot = color_func(phot_member_df)

    mag_range = np.linspace(mag_all.min(), mag_all.max(), 100)
    print(f"Mag range for fit line: {mag_range.min():.2f} to {mag_range.max():.2f}")


    # CMD Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # All photometric
    ax.scatter(mag_all, color_all, c="lightgray", s=5, label="All photometric sources")

    # Photometric members
    if phot_member_df is not None:
        ax.scatter(mag_phot, color_phot,
                c="crimson", s=10, label="Red sequence candidates")

    # Spectroscopic members
    if spec_member_df is not None:
        if with_cmap:
            sc = ax.scatter(mag_spec, color_spec,
                    c=spec_member_df[Z_COL], cmap="viridis",
                    s=25, edgecolor="k", linewidth=0.3, label="Spec-z cluster members")
            cbar = fig.colorbar(sc, ax=ax, label="Redshift")
            zmin = spec_member_df[Z_COL].min()
            zmax = spec_member_df[Z_COL].max()
            cbar.locator = MultipleLocator((zmax - zmin) / 5)
            cbar.formatter = FormatStrFormatter('%.3f')
            cbar.update_ticks()
        else:
            ax.scatter(mag_spec, color_spec, c="blue", s=20, label="Spec-z cluster members", marker="o", edgecolors="black")
    

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
        filename=f"{survey}_{color_type}_{save_name}_cmd.pdf"
    )


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
        save_path: str | None = None
    ):
    """
    Create a 2D density contour plot of RA-Dec distribution using WCS-based axis formatting.

    Parameters
    ----------
    phot_df : DataFrame
        Input data containing RA, Dec, and luminosity weights.
    wcs : WCS, optional
        World Coordinate System for the plot. [default: None]
    levels : int, optional
        Number of contour levels to draw. [default: 30]
    bandwidth : float, optional
        Bandwidth for Gaussian KDE smoothing (larger = smoother). [default: 0.2]
    zmin, zmax : float, optional
        Redshift range for title (if applicable). [default: None]
    base_color : str, optional
        Contour line color. [default: "black"]
    fill : bool, optional
        Whether to fill contours (True) or show lines only. [default: True]
    color_type : str, optional
        Color index to use (e.g., "g-r", "r-i"). [default: "g-r"]
    lum_weight : str, optional
        Luminosity weight column name. [default: "lum_weight_r"]
    survey : str, optional
        Survey name for labeling. [default: "Legacy"]
    save_plots : bool, optional
        Whether to save the plot. [default: False]
    show_plots : bool, optional
        Whether to display the plot. [default: False]
    save_path : str or None, optional
        Directory to save plots. [default: None]
    """
    weights = phot_df[f"{lum_weight}"].values
    ra = phot_df[RA_COL].values
    dec = phot_df[DEC_COL].values

    # Prepare coordinate and weight arrays
    ra = np.asarray(ra)
    dec = np.asarray(dec)
    weights = np.asarray(weights) if weights is not None else None

    # Start with validity of ra and dec
    valid_mask = np.isfinite(ra) & np.isfinite(dec)

    # If weights are provided, update the mask to also require valid weights
    if weights is not None:
        valid_mask &= np.isfinite(weights)
        weights = weights[valid_mask]

    ra = ra[valid_mask]
    dec = dec[valid_mask]

    npix = 300
    if wcs is None:
    # Construct dummy WCS from extent of data
        nx, ny = npix, npix
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [nx / 2, ny / 2]
        wcs.wcs.cdelt = [-(ra.max() - ra.min()) / nx, (dec.max() - dec.min()) / ny]  # RA decreasing to right
        wcs.wcs.crval = [(ra.max() + ra.min()) / 2, (dec.max() + dec.min()) / 2]
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    x, y = wcs.wcs_world2pix(ra, dec, 0)
    xy = np.vstack([x, y])

    # KDE
    kde = gaussian_kde(xy, bw_method=bandwidth, weights=weights)

    # Grid to evaluate KDE
    x_grid = np.linspace(x.min(), x.max(), npix)
    y_grid = np.linspace(y.min(), y.max(), npix)
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
    density = kde(np.vstack([x_mesh.ravel(), y_mesh.ravel()])).reshape(x_mesh.shape)


    # WCS-aware plot
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
        filename=f"{survey}_{color_type}_lum_density.pdf"
    )


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
    Generates spatial, CMD, and density plots for a cluster and survey/color.

    Parameters
    ----------
    cluster : Cluster
        Cluster object.
    fov_size : float
        Field of view in arcmin for optical image retrieval.
    survey : str
        "Legacy" or "PanSTARRS".
    color_type : str
        One of "g-r", "r-i", "g-i".
    ra_offset, dec_offset : float, optional
        Optional RA/Dec offsets in arcmin. [default: 0.0]
    bandwidth : float, optional
        KDE bandwidth for density plots. [default: 0.1]
    show_plots : bool, optional
        If True, displays plots interactively. [default: True]
    save_plots : bool, optional
        If True, saves plots to disk. [default: True]
    """
    from spec_phot_pipeline.color_magnitude_pipeline import fit_red_sequence

    img_dir = os.path.join(cluster.photometry_path, "Images")
    os.makedirs(img_dir, exist_ok=True)
    
    # File selection helpers
    if survey.lower() == "legacy":
        phot_df = pd.read_csv(f"{cluster.photometry_path}/photometry_legacy.csv")
        matched_df = pd.read_csv(f"{cluster.photometry_path}/legacy_matched.csv")
    elif survey.lower() == "panstarrs":
        phot_df = pd.read_csv(f"{cluster.photometry_path}/photometry_PanSTARRS.csv")
        matched_df = pd.read_csv(f"{cluster.photometry_path}/panstarrs_matched.csv")
    else:
        raise ValueError(f"Unknown survey: {survey}")

    redseq_file = get_redseq_filename(cluster.photometry_path, survey, color_type)
    print(f"Loading red sequence members from {redseq_file}")
    redseq_df = pd.read_csv(redseq_file)

    # --- File names ---
    phot_file = cluster.get_phot_file()
    bcg_file = cluster.bcg_file

    # Label for photometric luminosity weight
    if color_type.replace(" ", "") in ["g-r", "gr"]:
        lum_weight = "lum_weight_r"
    else:
        lum_weight = "lum_weight_i"

    a, b = fit_red_sequence(
        matched_df,
        color_type=color_type,
        zmin=cluster.z_min,
        zmax=cluster.z_max,
    )

    # Plotting
    # 1. All phot, all spec
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

    # 2. Cluster members only
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

    # 3. Cluster members only with cmap
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
    optical_image_path = cluster.get_optical_image(fov=fov_size, ra_offset=ra_offset, dec_offset=dec_offset)

    plot_optical(
        optical_image_path,
        cluster=cluster,
        show_plots=show_plots,
        save_plots=save_plots,
        save_path=img_dir,
        photometric_file=phot_file,
        bcg_file=bcg_file,
    )


def main():
    parser = argparse.ArgumentParser(description="Cluster red sequence plotting utility")
    parser.add_argument(
        "cluster_id",
        type=str,
        help="Cluster ID (e.g. '1327') or full RMJ/Abell name."
    )
    parser.add_argument(
        "--zmin",
        type=float,
        required=True,
        help="Minimum cluster redshift"
    )
    parser.add_argument(
        "--zmax",
        type=float,
        required=True,
        help="Maximum cluster redshift"
    )
    parser.add_argument(
        "--survey",
        type=str,
        choices=["Legacy", "PanSTARRS"],
        required=True,
        help="Survey to use"
    )
    parser.add_argument(
        "--color",
        type=str,
        choices=["g-r", "r-i", "g-i"],
        required=True,
        help="Color band to use"
    )
    parser.add_argument(
        "--fov",
        type=float,
        default=6,
        help="FOV (arcmin) for image retrieval [default: 6]"
    )
    parser.add_argument(
        "--base-path",
        type=str,
        default=os.getcwd(),
        help="Science base path (default: current dir)"
    )
    parser.add_argument(
        "--ra-offset",
        type=float,
        default=0.0,
        help="RA offset in arcmin [default: 0]"
    )
    parser.add_argument(
        "--dec-offset",
        type=float,
        default=0.0,
        help="Dec offset in arcmin [default: 0]"
    )
    parser.add_argument(
        "--bandwidth",
        type=float,
        default=0.1,
        help="KDE bandwidth for density contours [default: 0.1]"
    )
    parser.add_argument(
        '--show-plots',
        type=str2bool,
        nargs='?',
        const=True,
        default=True,
        help="Whether to display plots interactively. [default: True]. Accepts true/false, yes/no, y/n, t/f, 1/0."
    )
    parser.add_argument(
        '--save-plots',
        type=str2bool,
        nargs='?',
        const=True,
        default=True,
        help="Whether to save plots to file. [default: True]. Accepts true/false, yes/no, y/n, t/f, 1/0."
    )
    args = parser.parse_args()

    cluster = Cluster(args.cluster_id, base_path=args.base_path)
    if args.zmin is not None:
        cluster.z_min = args.zmin
    if args.zmax is not None:
        cluster.z_max = args.zmax
    if cluster.z_min is None:
        cluster.z_min = cluster.redshift - 0.015
    if cluster.z_max is None:
        cluster.z_max = cluster.redshift + 0.015

    run_cluster_plots(
        cluster=cluster,
        fov_size=args.fov,
        survey=args.survey,
        color_type=args.color,
        ra_offset=args.ra_offset,
        dec_offset=args.dec_offset,
        bandwidth=args.bandwidth,
        save_plots=args.save_plots,
        show_plots=args.show_plots
    )


if __name__ == "__main__":
    main()
