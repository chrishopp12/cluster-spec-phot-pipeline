#!/usr/bin/env python3
"""
xray.py

X-ray Image Visualization and Multi-Wavelength Overlays
---------------------------------------------------------

Generates X-ray surface brightness images, optical+X-ray composites, GGM
(Gaussian Gradient Magnitude) edge-detection overlays, and redshift scatter
overlays.  The core plotting function ``plot_xray()`` has complex branching
to accept data from several input combinations (raw arrays, FITS paths,
Cluster objects) and optionally reprojects X-ray data onto an optical WCS.

Key functions:
  - plot_xray()              Single X-ray image with optional optical background
  - make_xray_plots()        Suite of raw/filled/smoothed/reprojected images
                              plus 3x3 and 1x3 summary panels
  - plot_redshift_overlay()  Spectroscopic galaxies color-coded by redshift on
                              optical or blank background with contours
  - make_plots()             Top-level orchestrator that drives all X-ray,
                              optical, contour, and redshift overlay plots

Data products:
  - Xray/Images/*.pdf        Individual and multi-panel X-ray figures
  - Photometry/Images/*.pdf  Optical and contour overlay figures
  - Redshifts/Images/*.pdf   Redshift scatter overlay

Requirements:
  - astropy, reproject, matplotlib, numpy, pandas

Notes:
  - plot_xray() resolves the X-ray array and WCS from whichever combination
    of (img, xray_image_path, xray_fits_folder, cluster) is provided;
    review the branching carefully before modifying.
  - make_plots() calls io.images.get_optical_image() at runtime to download
    or locate the optical FITS cutout for overlay and reprojection.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator, FormatStrFormatter
from matplotlib.collections import LineCollection

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from reproject import reproject_interp, reproject_exact
from scipy.ndimage import gaussian_filter, gaussian_gradient_magnitude

from cluster_pipeline.utils.coordinates import arcsec_to_pixel_std
from cluster_pipeline.utils import pop_prefixed_kwargs
from cluster_pipeline.constants import DEFAULT_XRAY_FILENAME, DEFAULT_PSF_ARCSEC, DEFAULT_CMAP_XRAY
from cluster_pipeline.xray.image import fill_holes, smoothing
from cluster_pipeline.plotting.common import finalize_figure
from cluster_pipeline.plotting.optical import plot_optical
from cluster_pipeline.xray.analysis import profile_between_points


def plot_xray(
        cluster=None,
        img=None,
        optical_img=None,
        wcs_xray=None,
        wcs_optical=None,
        xray_fits_folder=None,
        xray_image_path=None,
        optical_image_file=None,
        fig=None,
        ax=None,
        suffix=None,
        show_plots=True,
        save_plots=True,
        save_path=None):
    """
    Plot X-ray image, optionally with optical background overlay.

    Parameters
    ----------
    cluster : Cluster object, optional
        Cluster object to use for default paths (if img/xray_image_path not given).
    img : ndarray, optional
        X-ray data array.
    optical_img : ndarray, optional
        Optical data array to overlay (optional).
    wcs_xray : astropy.wcs.WCS, optional
        WCS object for X-ray image.
    wcs_optical : astropy.wcs.WCS, optional
        WCS object for optical image.
    xray_fits_folder : str, optional
        Folder containing standard X-ray file (if img/xray_image_path not given).
    xray_image_path : str, optional
        Path to X-ray FITS file.
    optical_image_file : str, optional
        Path to optical FITS file.
    fig, ax : matplotlib objects, optional
        Figure and axis (created if not provided).
    suffix : str, optional
        Extra string to append to output filename.
    show_plots : bool, optional
        Whether to display the figure.
    save_plots : bool, optional
        Whether to save the figure.
    save_path : str, optional
        Path to save the figure (if None, uses cluster path).

    Returns
    -------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        Figure and axis used.
    """


    # Load X-ray image and WCS if not provided
    if img is not None:
        xray_data = img
        if wcs_xray is not None:
            pass  # already provided
        elif xray_image_path is not None:
            wcs_xray = WCS(fits.getheader(xray_image_path), naxis=2)
        elif xray_fits_folder is not None:
            xray_image_path = os.path.join(xray_fits_folder, DEFAULT_XRAY_FILENAME)
            wcs_xray = WCS(fits.getheader(xray_image_path), naxis=2)
        elif cluster is not None:
            xray_image_path = os.path.join(cluster.xray_path, DEFAULT_XRAY_FILENAME)
            if not os.path.exists(xray_image_path):
                print(f"X-ray image not found at {xray_image_path}, proceeding without WCS.")
                wcs_xray = None
            else:
                wcs_xray = WCS(fits.getheader(xray_image_path), naxis=2)
        else:
            wcs_xray = None
    elif xray_image_path is not None:
        xray_data = fits.getdata(xray_image_path)
        wcs_xray = WCS(fits.getheader(xray_image_path), naxis=2)
    elif xray_fits_folder is not None:
        xray_image_path = os.path.join(xray_fits_folder, DEFAULT_XRAY_FILENAME)
        if not os.path.exists(xray_image_path):
            raise FileNotFoundError(f"X-ray image not found at {xray_image_path}")
        xray_data = fits.getdata(xray_image_path)
        wcs_xray = WCS(fits.getheader(xray_image_path), naxis=2)
    else:
        # Attempt to use Cluster object as default if provided
        if cluster is not None:
            xray_image_path = os.path.join(cluster.xray_path, DEFAULT_XRAY_FILENAME)
            if not os.path.exists(xray_image_path):
                raise FileNotFoundError(
                    f"X-ray image not found at {xray_image_path}.\n"
                    f"Either supply a valid 'img'/'xray_image_path' or check your cluster pipeline."
                )
            xray_data = fits.getdata(xray_image_path)
            wcs_xray = WCS(fits.getheader(xray_image_path), naxis=2)
        else:
            raise ValueError(
                "Must provide either (img and wcs), or xray_image_path, or folder, or a cluster object with valid redshift_path."
            )


    # Load optical overlay if requested
    if optical_img is not None and wcs_optical is not None:
        optical_data = np.transpose(optical_img, (1, 2, 0)) if optical_img.ndim == 3 else optical_img
    elif optical_image_file is not None:
        with fits.open(optical_image_file) as hdul:
            optical_data = hdul[0].data
            wcs_optical = WCS(hdul[0].header, naxis=2)
            if optical_data.ndim == 3:
                optical_data = np.transpose(optical_data, (1, 2, 0))
    else:
        optical_data = None

    # Choose axis/projection
    if ax is None:
        if wcs_xray is not None and wcs_optical is not None:
            ax = (fig or plt.figure(figsize=(8, 8))).add_subplot(111, projection=wcs_optical)
        elif wcs_xray is not None:
            ax = (fig or plt.figure(figsize=(8, 8))).add_subplot(111, projection=wcs_xray)
        else:
            ax = (fig or plt.figure(figsize=(8, 8))).add_subplot(111)
    if fig is None:
        fig = ax.figure

    # Optical background first
    if optical_data is not None:
        ax.imshow(optical_data, origin='lower', alpha=0.3)
        ax.set_xlim(ax.get_xlim())
        ax.set_ylim(ax.get_ylim())

    # Main X-ray image
    if wcs_xray is not None:
        ax.imshow(
            xray_data,
            origin='lower',
            cmap=DEFAULT_CMAP_XRAY,
            transform=ax.get_transform(wcs_xray),
            vmin=np.percentile(xray_data, 5),
            vmax=np.percentile(xray_data, 99.95)
        )
    else:
        ax.imshow(
            xray_data,
            origin='lower',
            cmap=DEFAULT_CMAP_XRAY,
            vmin=np.percentile(xray_data, 5),
            vmax=np.percentile(xray_data, 99.95)
        )

    ax.set_xlabel("R.A.")
    ax.set_ylabel("Decl.")
    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_ylim())

    plt.tight_layout()

    # Determine folder and filename
    if save_path is not None:
        xray_folder = save_path
    elif cluster is not None:
        xray_folder = os.path.join(cluster.xray_path, "Images")
        os.makedirs(xray_folder, exist_ok=True)

    if cluster is not None and hasattr(cluster, "identifier"):
        fname = f'{cluster.identifier}_xray'
    else:
        fname = "xray_image"

    if optical_data is not None:
        fname += '_optical'
    if suffix is not None:
        fname += f'_{suffix}'

    finalize_figure(fig, save_path=save_path, save_plots=save_plots, show_plots=show_plots, filename=f"{fname}.pdf")

    return fig, ax

def make_xray_plots(cluster, optical_image_file, show_plots=False, save_plots=True, save_path=None, xray_fits_file=None):
    """
    Generates a suite of X-ray images and overlays for a given cluster, including:
    - Raw, filled, and smoothed images
    - Scaled images with optical background
    - Reprojected images onto the optical frame
    - Summary panels in various grid layouts (3x3 and 1x3)

    Parameters
    ----------
    cluster : Cluster object
        The cluster whose data should be visualized.
    optical_image_file : str
        Path to the optical FITS image to use for overlays and reprojection.
    show_plots : bool, optional
        Whether to display figures interactively (default: False).
    save_plots : bool, optional
        Whether to save figures (default: True).
    save_path : str or None
        Directory or filename to save output figures. If a directory, standard names are used.
    xray_fits_file : str or None
        Path to the X-ray FITS file. If None, uses default from cluster object.
    """

    # -- Load X-ray data --
    if xray_fits_file is None:
        xray_fits_file = os.path.join(cluster.xray_path, DEFAULT_XRAY_FILENAME)
    raw_xray= fits.getdata(xray_fits_file)
    wcs_xray = WCS(fits.getheader(xray_fits_file), naxis=2)

    # -- Load optical data -- (only for WCS and reprojection)
    with fits.open(optical_image_file) as hdul:
        optical_image_data = hdul[0].data
        wcs_optical = WCS(hdul[0].header, naxis=2)


    # -- Process X-ray images --
    psf_arcsec = cluster.psf if hasattr(cluster, 'psf') and cluster.psf is not None else DEFAULT_PSF_ARCSEC
    kernel_std_xray = arcsec_to_pixel_std(psf_arcsec, wcs_xray)

    filled_raw = fill_holes(raw_xray, exact=True)
    smoothed_raw = smoothing(filled_raw, kernel_std_xray)

    # Reprojected onto optical frame
    reprojected_raw, _ = reproject_interp((raw_xray, wcs_xray), wcs_optical, shape_out=optical_image_data.shape[1:])
    reprojected_filled, _ = reproject_interp((filled_raw, wcs_xray), wcs_optical, shape_out=optical_image_data.shape[1:])
    kernel_std_optical = arcsec_to_pixel_std(psf_arcsec, wcs_optical)
    reprojected_smoothed = smoothing(reprojected_filled, kernel_std_optical)

    # --- Persist smoothed X-ray images as FITS ---
    try:
        fits.writeto(
            os.path.join(cluster.xray_path, "smoothed_raw.fits"),
            smoothed_raw.astype("float32"), wcs_xray.to_header(), overwrite=True,
        )
        if np.any(np.isfinite(reprojected_smoothed)):
            fits.writeto(
                os.path.join(cluster.xray_path, "smoothed_reprojected.fits"),
                reprojected_smoothed.astype("float32"), wcs_optical.to_header(),
                overwrite=True,
            )
    except Exception as e:
        print(f"    Smoothed X-ray FITS write FAILED: {type(e).__name__}: {e}")


    # -- Plot single images --
    # Full images
    plot_xray(cluster = cluster, img = raw_xray, show_plots = show_plots, save_plots=save_plots, suffix='raw')
    plot_xray(cluster = cluster, img = filled_raw, show_plots = show_plots, save_plots=save_plots, suffix='filled_raw')
    plot_xray(cluster = cluster, img = smoothed_raw, show_plots = show_plots, save_plots=save_plots, suffix='smoothed_raw')

    # Scaled images
    plot_xray(cluster = cluster, img = raw_xray, optical_image_file=optical_image_file, show_plots = show_plots, save_plots=save_plots, suffix='raw')
    plot_xray(cluster = cluster, img = filled_raw, optical_image_file=optical_image_file, show_plots = show_plots, save_plots=save_plots, suffix='filled_raw')
    plot_xray(cluster = cluster, img = smoothed_raw, optical_image_file=optical_image_file, show_plots = show_plots, save_plots=save_plots, suffix='smoothed_raw')

    # Reprojected
    plot_xray(cluster = cluster, img = reprojected_raw, wcs_xray = wcs_optical, optical_image_file=optical_image_file, show_plots = show_plots, save_plots=save_plots, suffix='reprojected_raw')
    plot_xray(cluster = cluster, img = reprojected_filled, wcs_xray = wcs_optical, optical_image_file=optical_image_file, show_plots = show_plots, save_plots=save_plots, suffix='reprojected_filled')
    plot_xray(cluster = cluster, img = reprojected_smoothed, wcs_xray = wcs_optical, optical_image_file=optical_image_file, show_plots = show_plots, save_plots=save_plots, suffix='reprojected_smoothed')


    # -- Plot summary panels --
    fig = plt.figure(figsize=(12, 12))
    axs = np.empty((3, 3), dtype=object)

    axs[0, 0] = fig.add_subplot(3, 3, 1, projection=wcs_xray)
    axs[0, 1] = fig.add_subplot(3, 3, 2, projection=wcs_xray)
    axs[0, 2] = fig.add_subplot(3, 3, 3, projection=wcs_xray)

    axs[1, 0] = fig.add_subplot(3, 3, 4, projection=wcs_optical)
    axs[1, 1] = fig.add_subplot(3, 3, 5, projection=wcs_optical)
    axs[1, 2] = fig.add_subplot(3, 3, 6, projection=wcs_optical)

    axs[2, 0] = fig.add_subplot(3, 3, 7, projection=wcs_optical)
    axs[2, 1] = fig.add_subplot(3, 3, 8, projection=wcs_optical)
    axs[2, 2] = fig.add_subplot(3, 3, 9, projection=wcs_optical)


    # Full images
    plot_xray(cluster = cluster, img = raw_xray, show_plots = False, save_plots = False, ax=axs[0, 0], fig=fig, suffix='raw')
    plot_xray(cluster = cluster, img = filled_raw, show_plots = False, save_plots = False, ax=axs[0, 1], fig=fig, suffix='filled_raw')
    plot_xray(cluster = cluster, img = smoothed_raw, show_plots = False, save_plots = False, ax=axs[0, 2], fig=fig, suffix='smoothed_raw')

    # Scaled images
    plot_xray(cluster = cluster, img = raw_xray, optical_image_file=optical_image_file, show_plots = False, save_plots = False, ax=axs[1, 0], fig=fig, suffix='raw')
    plot_xray(cluster = cluster, img = filled_raw, optical_image_file=optical_image_file, show_plots = False, save_plots = False, ax=axs[1, 1], fig=fig, suffix='filled_raw')
    plot_xray(cluster = cluster, img = smoothed_raw, optical_image_file=optical_image_file, show_plots = False, save_plots = False, ax=axs[1, 2], fig=fig, suffix='smoothed_raw')

    # Reprojected
    plot_xray(cluster = cluster, img = reprojected_raw, wcs_xray = wcs_optical, optical_image_file=optical_image_file, show_plots = False, save_plots = False, ax=axs[2, 0], fig=fig, suffix='reprojected_raw')
    plot_xray(cluster = cluster, img = reprojected_filled, wcs_xray = wcs_optical, optical_image_file=optical_image_file, show_plots = False, save_plots = False, ax=axs[2, 1], fig=fig, suffix='reprojected_filled')
    plot_xray(cluster = cluster, img = reprojected_smoothed, wcs_xray = wcs_optical, optical_image_file=optical_image_file, show_plots = False, save_plots = False, ax=axs[2, 2], fig=fig, suffix='reprojected_smoothed')

    plt.tight_layout()
    finalize_figure(fig, save_path=save_path, save_plots=save_plots, show_plots=show_plots, filename="xray_comparison.pdf")

    # -- 1x3 Panels --

    # Full images
    fig = plt.figure(figsize=(18, 6))
    axs = np.empty((1, 3), dtype=object)

    axs[0, 0] = fig.add_subplot(1, 3, 1, projection=wcs_xray)
    axs[0, 1] = fig.add_subplot(1, 3, 2, projection=wcs_xray)
    axs[0, 2] = fig.add_subplot(1, 3, 3, projection=wcs_xray)

    plot_xray(cluster = cluster, img = raw_xray, show_plots = False, save_plots = False, ax=axs[0, 0], fig=fig, suffix='raw')
    plot_xray(cluster = cluster, img = filled_raw, show_plots = False, save_plots = False, ax=axs[0, 1], fig=fig, suffix='filled_raw')
    plot_xray(cluster = cluster, img = smoothed_raw, show_plots = False, save_plots = False, ax=axs[0, 2], fig=fig, suffix='smoothed_raw')

    plt.tight_layout()
    finalize_figure(fig, save_path=save_path, save_plots=save_plots, show_plots=show_plots, filename="xray_full.pdf")


    # Scaled images
    fig = plt.figure(figsize=(18, 6))
    axs = np.empty((1, 3), dtype=object)

    axs[0, 0] = fig.add_subplot(1, 3, 1, projection=wcs_optical)
    axs[0, 1] = fig.add_subplot(1, 3, 2, projection=wcs_optical)
    axs[0, 2] = fig.add_subplot(1, 3, 3, projection=wcs_optical)

    plot_xray(cluster = cluster, img = raw_xray, optical_image_file=optical_image_file, show_plots = False, save_plots = False, ax=axs[0, 0], fig=fig, suffix='raw')
    plot_xray(cluster = cluster, img = filled_raw, optical_image_file=optical_image_file, show_plots = False, save_plots = False, ax=axs[0, 1], fig=fig, suffix='filled_raw')
    plot_xray(cluster = cluster, img = smoothed_raw, optical_image_file=optical_image_file, show_plots = False, save_plots = False, ax=axs[0, 2], fig=fig, suffix='smoothed_raw')

    plt.tight_layout()
    finalize_figure(fig, save_path=save_path, save_plots=save_plots, show_plots=show_plots, filename="xray_scaled.pdf")


    # Reprojected images
    fig = plt.figure(figsize=(18, 6))
    axs = np.empty((1, 3), dtype=object)

    axs[0, 0] = fig.add_subplot(1, 3, 1, projection=wcs_optical)
    axs[0, 1] = fig.add_subplot(1, 3, 2, projection=wcs_optical)
    axs[0, 2] = fig.add_subplot(1, 3, 3, projection=wcs_optical)

    plot_xray(cluster = cluster, img = reprojected_raw, wcs_xray = wcs_optical, optical_image_file=optical_image_file, show_plots = False, save_plots = False, ax=axs[0, 0], fig=fig, suffix='reprojected_raw')
    plot_xray(cluster = cluster, img = reprojected_filled, wcs_xray = wcs_optical, optical_image_file=optical_image_file, show_plots = False, save_plots = False, ax=axs[0, 1], fig=fig, suffix='reprojected_filled')
    plot_xray(cluster = cluster, img = reprojected_smoothed, wcs_xray = wcs_optical, optical_image_file=optical_image_file, show_plots = False, save_plots = False, ax=axs[0, 2], fig=fig, suffix='reprojected_smoothed')

    plt.tight_layout()
    finalize_figure(fig, save_path=save_path, save_plots=save_plots, show_plots=show_plots, filename="xray_reprojected.pdf")


def plot_redshift_overlay(
    optical_image_file,
    spectroscopy_file,
    ax=None, fig=None,
    cluster=None,
    z_low=None, z_high=None,
    bcg_file=None,
    xray_fits_file=None,
    photometric_file=None,
    colorbar=True,
    show_legend=True,
    show_plots=True,
    save_plots=True,
    save_path=None,
    scatter_kwargs=None,
    **kwargs
):

    """
    Plots galaxies with color-coded redshift, optionally overlays contours/BCGs on optical/blank background.

    Parameters
    ----------
    optical_image_file : str
        Path to FITS optical image.
    spectroscopy_file : str
        CSV file of spectroscopic sources (must contain 'RA', 'Dec', 'z').
    ax, fig : matplotlib objects, optional
        Existing axis/figure.
    cluster : Cluster object, optional
        Cluster object (for z_min, z_max, redshift).
    z_low, z_high : float, optional
        Redshift limits for cluster member selection. If None, uses cluster.z_min/z_max.
    bcg_file, xray_fits_file, photometric_file : str, optional
        File paths for overlays, passed directly to plot_optical.
    colorbar : bool, optional
        Whether to show a colorbar.
    show_plots, save_plots : bool, optional
        Whether to show or save the plot.
    save_path : str or None
        Output file or directory for saving.
    scatter_kwargs : dict, optional
        Passed to ax.scatter() for additional marker options.
    **kwargs
        Additional plotting settings for overlays and scatter (see below).

    Other Parameters
    ----------------
    scatter kwargs: scatter_cmap, scatter_size, scatter_alpha, scatter_zorder : optional
        Marker settings for scatter plot.
    density_levels, xray_levels, legend_loc, etc. : optional
        Passed to plot_optical.
    """

    if z_low is None or np.isnan(z_low):
        z_low = cluster.z_min
    if z_high is None or np.isnan(z_high):
        z_high = cluster.z_max

    if show_legend:

        fig, ax = plot_optical(
            optical_image_file=optical_image_file,
            cluster=cluster,
            fig=fig, ax=ax,
            show_plots=False,
            save_plots=False,
            show_optical=False,
            xray_fits_file=xray_fits_file,
            photometric_file=photometric_file,
            bcg_file=bcg_file,
            **kwargs
        )

    else:

        fig, ax, handles, labels = plot_optical(
            optical_image_file=optical_image_file,
            cluster=cluster,
            fig=fig, ax=ax,
            show_plots=False,
            save_plots=False,
            show_optical=False,
            xray_fits_file=xray_fits_file,
            photometric_file=photometric_file,
            bcg_file=bcg_file,
            show_legend=False,
            **kwargs
        )



    # -- Read spectroscopy data --
    df_spec = pd.read_csv(spectroscopy_file)
    ra = df_spec['RA'].values
    dec = df_spec['Dec'].values
    z = df_spec['z'].values
    # -- Select cluster members --
    mask_members = (z > z_low) & (z < z_high)

    # -- Namespaced scatter kwargs
    scatter_kwargs = pop_prefixed_kwargs(kwargs, 'scatter')

    # -- Scatter galaxies by redshift in [z_low, z_high] --
    sc_kwargs = dict(
                c=z[mask_members],
                cmap=scatter_kwargs.pop('cmap', 'viridis'),
                s=scatter_kwargs.get('size', 20),
                alpha=scatter_kwargs.get('alpha', 0.9),
                zorder=scatter_kwargs.get('zorder', 22),
                transform=ax.get_transform('icrs'),
                label=f"Cluster Members: {z_low:.3f} < z < {z_high:.3f}",
                **scatter_kwargs
    )
    sc = ax.scatter(ra[mask_members], dec[mask_members], **sc_kwargs)

    # -- Colorbar --
    if colorbar:
        cbar = fig.colorbar(sc, ax=ax, pad=0.01, shrink=0.76)
        cbar.set_label("Redshift (z)")
        cbar.locator = MaxNLocator(nbins=5)
        cbar.formatter = FormatStrFormatter('%.3f')
        cbar.update_ticks()

    if save_plots or show_plots:
        plt.tight_layout()
        finalize_figure(fig, save_path=save_path, save_plots=save_plots, show_plots=show_plots, filename="redshift_overlay.pdf")

    if show_legend:
        if colorbar:
            return fig, ax
        else:
            return fig, ax, sc
    else:
        if colorbar:
            return fig, ax, handles, labels
        else:
            return fig, ax, handles, labels, sc


# -- Main driver function to make all plots --

def make_plots(
    cluster,
    save_plots=True,
    show_plots=True,
    fov=None,          # can override, else cluster.fov
    fov_full=None,     # can override, else cluster.fov_full
    z_min=None,        # can override, else cluster.z_min
    z_max=None,        # can override, else cluster.z_max
    **kwargs
):
    # --- Use cluster attributes unless overridden ---
    fov = fov if fov is not None else cluster.fov
    fov_full = fov_full if fov_full is not None else cluster.fov_full

    cluster.z_min = z_min if z_min is not None else getattr(cluster, "z_min", None)
    cluster.z_max = z_max if z_max is not None else getattr(cluster, "z_max", None)
    if cluster.z_min is None or (hasattr(np, "isnan") and np.isnan(cluster.z_min)):
        cluster.z_min = float(cluster.redshift) - 0.015
    if cluster.z_max is None or (hasattr(np, "isnan") and np.isnan(cluster.z_max)):
        cluster.z_max = float(cluster.redshift) + 0.015

    # --- Paths (all from Cluster) ---
    phot_images_path = os.path.join(cluster.photometry_path, "Images")
    spec_images_path = os.path.join(cluster.redshift_path, "Images")
    xray_images_path = os.path.join(cluster.xray_path, "Images")
    os.makedirs(phot_images_path, exist_ok=True)
    os.makedirs(spec_images_path, exist_ok=True)
    os.makedirs(xray_images_path, exist_ok=True)

    # --- File names ---
    xray_file = cluster.xray_file
    phot_file = cluster.get_phot_file()
    bcg_file = cluster.bcg_file
    spec_file = cluster.spec_file

    # --- Optical image files ---
    from cluster_pipeline.io.images import get_optical_image
    optical_kwargs = {k: v for k, v in kwargs.items() if k in ('ra_offset', 'dec_offset')}
    optical_image_file_full = get_optical_image(cluster, fov_full, **optical_kwargs)
    optical_image_file = get_optical_image(cluster, fov, **optical_kwargs)

    # --- Output pdf filenames ---
    optical_image_pdf_full = os.path.join(phot_images_path, "optical_image_full.pdf") if save_plots else None
    optical_image_pdf = os.path.join(phot_images_path, "optical_image.pdf") if save_plots else None
    contour_image_file = os.path.join(phot_images_path, "contours.pdf") if save_plots else None
    redshift_image_file = os.path.join(spec_images_path, "redshift_overlay.pdf") if save_plots else None
    gauss_image_file = os.path.join(xray_images_path, "gaussian_gradient.pdf") if save_plots else None

    # --- Plots ---


    # Gaussian grad/ unsharp masking
    gaussian_grad_magnitude(
        cluster=cluster,
        xray_file=xray_file,
        optical_image_file=optical_image_file,
        show_plots=show_plots,
        save_plots=save_plots,
        save_path=gauss_image_file
    )



    # Optical, full fov
    plot_optical(
        optical_image_file_full, cluster=cluster, show_legend=False,
        show_plots=show_plots, save_plots=save_plots, save_path=optical_image_pdf_full
    )

    # Optical, zoomed fov
    plot_optical(
        optical_image_file, cluster=cluster, show_legend=False,
        show_plots=show_plots, save_plots=save_plots, save_path=optical_image_pdf
    )

    # X-ray
    make_xray_plots(
        cluster=cluster,
        optical_image_file=optical_image_file,
        show_plots=show_plots,
        save_plots=save_plots,
        save_path=xray_images_path
    )

    # Optical, contours
    plot_optical(
        optical_image_file, cluster=cluster, show_plots=show_plots, save_plots=save_plots,
        save_path=contour_image_file, xray_fits_file=xray_file, photometric_file=phot_file, bcg_file=bcg_file
    )

    # Redshift heatmap overlay
    plot_redshift_overlay(
        optical_image_file=optical_image_file,
        spectroscopy_file=spec_file,
        cluster=cluster,
        z_low=cluster.z_min, z_high=cluster.z_max,
        bcg_file=bcg_file,
        xray_fits_file=xray_file,
        photometric_file=phot_file,
        show_plots=show_plots,
        save_plots=save_plots,
        save_path=redshift_image_file,
        density_linewidth=0.5,
        xray_linewidth=0.5
    )


def gaussian_grad_magnitude(
    cluster,
    xray_file,
    optical_image_file,
    psf_arcsec=None,
    with_raw=False,
    show_plots=True,
    save_plots=True,
    save_path=None,
):
    """
    Show smoothed X-ray surface brightness, gradient magnitude, and unsharp masking.

    Parameters
    ----------
    optical_image_file : str
        Path to FITS optical image.
    xray_file : str
        Path to FITS X-ray image.
    psf_arcsec : float, optional
        PSF size in arcseconds for Gaussian smoothing.
    show_plots : bool, optional
        Whether to display plot interactively.
    save_plots : bool, optional
        Whether to save the plot.
    save_path : str, optional
        Directory or file to save plot. If directory, uses standard filename.
    """


    # --- Load data ---
    with fits.open(optical_image_file) as hdul:
        optical_image_data = hdul[0].data
        wcs_optical = WCS(hdul[0].header, naxis=2)
        optical_data = np.transpose(optical_image_data, (1, 2, 0)) if optical_image_data.ndim == 3 else optical_image_data

    xray_data = fits.getdata(xray_file)
    wcs_xray = WCS(fits.getheader(xray_file), naxis=2)

    pixscale_xray = wcs_xray.proj_plane_pixel_scales()[0] * 3600  # arcsec/pixel
    pixscale_optical = wcs_optical.proj_plane_pixel_scales()[0]   # arcsec/pixel

    # --- Processing ---
    psf_arcsec = cluster.psf if hasattr(cluster, 'psf') and cluster.psf is not None else 8.0
    kernel_std_xray = arcsec_to_pixel_std(psf_arcsec, wcs_xray)
    xray_data = fill_holes(xray_data, exact=True)

    xray_reprojected, _ = reproject_exact(
        (xray_data, wcs_xray), wcs_optical, shape_out=optical_image_data.shape[1:])

    kernel_std_optical = arcsec_to_pixel_std(psf_arcsec, wcs_optical)
    xray_smoothed = gaussian_filter(xray_reprojected, sigma=kernel_std_optical)

    gradient_magnitude = gaussian_gradient_magnitude(xray_smoothed, sigma=kernel_std_optical)
    gradient_magnitude = gradient_magnitude / pixscale_optical.value

    fine = gaussian_filter(xray_reprojected, sigma=kernel_std_optical)
    broad = gaussian_filter(xray_reprojected, sigma=5 * kernel_std_optical)
    unsharp = fine - broad

    # Example: define endpoints (RA, Dec) in degrees
    # p1 = (34.975, 1.48333)
    # p2 = (34.96731, 1.49783)
    p1 = (35, 1.49)
    p2 = (34.95, 1.49167)


    # --- Full-width horizontal cut at fixed Dec ---
    ny_opt, nx_opt = optical_data.shape[:2]
    dec_cut_deg = wcs_optical.pixel_to_world(nx_opt/2, ny_opt/2).dec.deg -0.4/60

    # Pixel row in optical WCS at this Dec
    ra_mid_opt = wcs_optical.pixel_to_world(nx_opt/2, ny_opt/2).ra.deg
    _, y_opt = wcs_optical.world_to_pixel(SkyCoord(ra_mid_opt * u.deg, dec_cut_deg * u.deg))
    if not np.isfinite(y_opt) or y_opt < 0 or y_opt > (ny_opt - 1):
        raise ValueError("Dec of the cut is outside the optical FOV.")

    # World coords of optical left/right edges at that Dec
    sky_left  = wcs_optical.pixel_to_world(0.0,         y_opt)
    sky_right = wcs_optical.pixel_to_world(nx_opt - 1., y_opt)
    p_left_world  = (float(sky_left.ra.deg),  float(sky_left.dec.deg))
    p_right_world = (float(sky_right.ra.deg), float(sky_right.dec.deg))


    if with_raw:
        # Profile over optical span, sampled from native X-ray image + WCS
        s_h, sb_h, xh, yh = profile_between_points(
            image2d=xray_data,                # native X-ray image
            wcs=wcs_xray,                     # native X-ray WCS
            p1=p_left_world, p2=p_right_world,
            n_samples=nx_opt,                 # match optical width
            width_pix=1,                      # width in *X-ray* pixels
            aggregate="mean",
        )
    else:
        # Profile along the full width at that Dec
        s_h, sb_h, xh, yh = profile_between_points(
            image2d=xray_smoothed, wcs=wcs_optical,
            p1=p_left_world, p2=p_right_world,
            n_samples=nx_opt, width_pix=1, aggregate="mean",
        )



    bg = "#0e0e0e8b"
    fig = plt.figure(figsize=(7, 12))
    gs = fig.add_gridspec(4, 1, height_ratios=[0.25, 1, 1, 1], hspace=0.04)

    if with_raw:
        # Panel 1: Smoothed X-ray
        ax1 = fig.add_subplot(gs[1], projection=wcs_optical)
        ax1.imshow(optical_data, origin='lower', alpha=0.3)
        ax1.set_xlim(ax1.get_xlim())
        ax1.set_ylim(ax1.get_ylim())
        im1 = ax1.imshow(xray_data, cmap='magma', origin='lower',
                transform=ax1.get_transform(wcs_xray),
                vmin=np.percentile(xray_data, 5),
                vmax=np.percentile(xray_data, 99.95))
        ax1.plot(xh, yh, color="white", linestyle='--', lw=0.7, alpha=0.5, transform=ax1.get_transform(wcs_xray))

    else:
        # Panel 1: Smoothed X-ray
        ax1 = fig.add_subplot(gs[1], projection=wcs_optical)
        ax1.imshow(optical_data, origin='lower', alpha=0.3)
        ax1.set_xlim(ax1.get_xlim())
        ax1.set_ylim(ax1.get_ylim())
        im1 = ax1.imshow(xray_smoothed, cmap='magma', origin='lower',
                vmin=np.percentile(xray_smoothed, 5),
                vmax=np.percentile(xray_smoothed, 99.95))
        ax1.plot(xh, yh, color="white", linestyle='--', lw=0.7, alpha=0.5, transform=ax1.get_transform(wcs_optical))

    ax1.set_ylabel('Decl.')
    ax1.tick_params(axis='x', top=False, labelbottom=False)


    axp = fig.add_subplot(gs[0])
    pts = np.column_stack([s_h, sb_h])                     # (N, 2)
    segs = np.stack([pts[:-1], pts[1:]], axis=1)

    lc = LineCollection(segs, cmap=im1.get_cmap(), norm=im1.norm)
    lc.set_array(0.5*(sb_h[:-1] + sb_h[1:]))
    lc.set_linewidth(1.6)
    axp.add_collection(lc)
    bg = "#0e0e0e8b"
    axp.set_facecolor(bg)
    axp.set_xlim(s_h.min(), s_h.max())
    ymin, ymax = np.min(sb_h), np.max(sb_h)
    pad = 0.05*(ymax - ymin)
    axp.set_ylim(ymin - pad, ymax + pad)

    axp.tick_params(axis='x', top=False, bottom=False, labelbottom=False, labeltop=False)
    axp.tick_params(axis='y', left=False, right=False, labelleft=False, labelright=False)



    # Panel 2: Gradient magnitude
    ax2 = fig.add_subplot(gs[2], projection=wcs_optical, sharey=ax1)
    im2 = ax2.imshow(gradient_magnitude, origin='lower', cmap='viridis')
    ax2.set_ylabel('Decl.')
    ax2.tick_params(axis='x', labelbottom=False)

    # Panel 3: Unsharp masking
    ax3 = fig.add_subplot(gs[3], projection=wcs_optical, sharey=ax1)
    im3 = ax3.imshow(unsharp, origin='lower', cmap='RdBu_r',
        vmin=-np.percentile(np.abs(unsharp), 99),
        vmax=+np.percentile(np.abs(unsharp), 99))
    ax3.set_xlabel('R.A.')
    ax3.set_ylabel('Decl.')

    fig.subplots_adjust(top=0.96, bottom=0.05, left=0.01, right=0.92)
    plt.draw()  # or: fig.canvas.draw()

    pos1 = ax1.get_position()  # final box of the top WCS axis (after aspect)

    # Match widths (keep each axis's current vertical placement/height)
    for a in (ax2, ax3):
        p = a.get_position()
        a.set_position([pos1.x0, p.y0, pos1.width, p.height])
    p = axp.get_position()
    axp.set_position([pos1.x0, pos1.y1, pos1.width, p.height])


    # --- Colorbars ---
    plt.draw()  # positions must be final before we read them

    # 1) Combined SB + profile colorbar (to the right of ax1, spanning ax1+axp)
    pos1 = ax1.get_position()
    posp = axp.get_position()
    y0 = pos1.y0
    h  = (posp.y0 + posp.height) - pos1.y0
    cax_sb = fig.add_axes([pos1.x1 + 0.02, y0, 0.02, h])

    cbar_sb = fig.colorbar(im1, cax=cax_sb)
    cbar_sb.set_label("Surface Brightness\n [cts s$^{-1}$ deg$^{-2}$]", labelpad=8)
    cbar_sb.ax.yaxis.set_label_coords(7.8, 0.5)

    # 2) Individual colorbars for the two lower panels
    for im, ax, label in zip(
        [im2, im3],
        [ax2, ax3],
        ["Gradient Magnitude\n [cts s$^{-1}$ deg$^{-3}$]",
        "Unsharp Brightness\n [cts s$^{-1}$ deg$^{-2}$]"]
    ):
        pos = ax.get_position()
        cax = fig.add_axes([pos.x1 + 0.02, pos.y0, 0.02, pos.height])
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(label, labelpad=8)
        cbar.ax.yaxis.set_label_coords(7.8, 0.5)

    # --- Save/show ---
    finalize_figure(fig, save_path=save_path, save_plots=save_plots, show_plots=show_plots, filename="gauss.pdf")
