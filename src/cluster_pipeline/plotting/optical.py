#!/usr/bin/env python3
"""
optical.py

Optical Image Plotting with Contour Overlays
---------------------------------------------------------

Displays WCS-projected optical images and composites X-ray surface
brightness contours, photometric galaxy density contours, radio continuum
contours, BCG markers, and scale bars on top of them.  Used directly by
nearly every multi-panel figure in the pipeline.

Key functions:
  - plot_optical()           WCS optical image with optional contour and BCG
                              overlays, scale bar, and legend
  - add_xray_contours()      Smooth and contour an X-ray FITS image aligned
                              to the optical WCS
  - add_density_contours()   KDE-based luminosity-weighted galaxy density
                              contours from a photometric catalog
  - add_radio_contours()     Geometric-spaced radio continuum contours from a
                              radio FITS image, aligned to the optical WCS
  - define_contours_fov()    Compute contour levels restricted to the optical
                              field of view (avoids edge artifacts)

Requirements:
  - astropy, scipy, matplotlib, numpy

Notes:
  - All overlay functions accept namespaced **kwargs (e.g., xray_color,
    density_bandwidth) so callers can tune appearance without separate calls.
  - Contour level and PSF defaults cascade: explicit kwarg > Cluster
    attribute > constants.py default.
  - plot_optical() returns (fig, ax) when show_legend=True, or
    (fig, ax, handles, labels) when show_legend=False, to support
    downstream legend composition.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.lines import Line2D
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter

from cluster_pipeline.utils import pop_prefixed_kwargs
from cluster_pipeline.utils.coordinates import arcsec_to_pixel_std
from cluster_pipeline.xray.image import smoothing
from cluster_pipeline.radio.image import (
    load_radio_image,
    mask_compact_sources,
    radio_contour_levels,
    robust_rms,
)
from cluster_pipeline.io.catalogs import load_photo_coords
from cluster_pipeline.plotting.common import (
    finalize_figure,
    add_scalebar,
    overlay_bcg_markers,
    evaluate_kde_grid,
)
from cluster_pipeline.constants import (
    DEFAULT_PSF_ARCSEC,
    DEFAULT_CONTOUR_LEVELS,
    DEFAULT_BANDWIDTH,
    DEFAULT_KDE_GRID_SIZE,
    DEFAULT_LEGEND_LOC,
    DEFAULT_RADIO_COLOR,
    DEFAULT_RADIO_N_LEVELS,
    DEFAULT_RADIO_START_SIGMA,
    DEFAULT_RADIO_CONTOUR_STEP,
    DEFAULT_RADIO_SMOOTH_PIX,
    DEFAULT_RADIO_LINEWIDTH,
    DEFAULT_RADIO_OUTLINE,
    DEFAULT_RADIO_OUTLINE_COLOR,
    DEFAULT_RADIO_OUTLINE_EXTRA,
)

if TYPE_CHECKING:
    from cluster_pipeline.models.cluster import Cluster


def add_xray_contours(
        ax: plt.Axes,
        xray_fits_file: str,
        optical_data: np.ndarray,
        wcs_optical: WCS,
        levels: tuple[float, float, float] = DEFAULT_CONTOUR_LEVELS,
        psf: float = DEFAULT_PSF_ARCSEC,
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
    elif cluster is not None and hasattr(cluster, "contour_levels"):
        cl = cluster.contour_levels
        if isinstance(cl, (tuple, list)) and len(cl) == 3:
            contour_levels = tuple(cl)
        else:
            print(f"[WARNING] cluster.contour_levels is not a valid 3-tuple: {cl!r}")
            contour_levels = levels
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
    bandwidth: float = DEFAULT_BANDWIDTH,
    levels: int = 12,
    skip: int = 2,
    color: str = 'tab:blue',
    alpha: float = 1.0,
    linewidth: float = 1.2,
    cluster: "Cluster" | None = None,
    weighted: bool = True,
    **kwargs,
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
    weighted : bool, optional
        Weight each source by ``lum_weight_r`` when computing the KDE (default: True)
    **kwargs
        Additional keyword arguments. Recognized namespaced overrides:
            - density_bandwidth : float
            - density_levels    : int
            - density_skip      : int
            - density_color     : str
            - density_alpha     : float
            - density_linewidth : float
            - density_weighted  : bool
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
    if 'weighted' in density_kwargs:
        weighted = bool(density_kwargs.pop('weighted'))

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
    weights = np.asarray(Lum_phot) if weighted else None

    ra0, dec0 = float(np.median(ra)), float(np.median(dec))
    cosd0 = np.cos(np.deg2rad(dec0))

    # small-angle offsets (degrees)
    x = (ra - ra0) * cosd0
    y = (dec - dec0)

    # Kernel Density Estimation in RA/Dec
    ra_mesh, dec_mesh, density = evaluate_kde_grid(x, y, weights, contour_bandwidth, DEFAULT_KDE_GRID_SIZE)

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


def add_radio_contours(
        ax: plt.Axes,
        radio_fits_file: str,
        color: str = DEFAULT_RADIO_COLOR,
        n_levels: int = DEFAULT_RADIO_N_LEVELS,
        start_sigma: float = DEFAULT_RADIO_START_SIGMA,
        contour_step: float = DEFAULT_RADIO_CONTOUR_STEP,
        smooth_pix: float = DEFAULT_RADIO_SMOOTH_PIX,
        linewidth: float = DEFAULT_RADIO_LINEWIDTH,
        outline: bool = DEFAULT_RADIO_OUTLINE,
        outline_color: str = DEFAULT_RADIO_OUTLINE_COLOR,
        outline_extra: float = DEFAULT_RADIO_OUTLINE_EXTRA,
        mask_catalog: str | None = None,
        cluster: "Cluster" | None = None,
        **kwargs,
    ):
    """
    Adds geometric-spaced radio continuum contours to an axis, aligned to an optical WCS.

    Contour levels follow ``start_sigma * sigma * contour_step ** k`` for
    ``k = 0 .. n_levels - 1``, where ``sigma`` is the robust (MAD-based) noise
    of the radio map. The default sqrt(2) step gives the canonical
    (3, 6, 12, 24, ...) sigma ladder.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis on which to plot radio contours (should be WCS-aware).
    radio_fits_file : str
        Path to the radio FITS image.
    color : str, optional
        Contour color (default: 'white').
    n_levels : int, optional
        Number of contour levels (default: 12).
    start_sigma : float, optional
        Lowest contour in units of sigma above the noise (default: 4.0).
    contour_step : float, optional
        Geometric ratio between successive levels (default: sqrt 2).
    smooth_pix : float, optional
        Gaussian smoothing kernel in pixels applied before contouring; 0
        disables smoothing (default: 0.0).
    linewidth : float, optional
        Contour line width in points (default: 0.6).
    outline : bool, optional
        Draw a contrasting halo stroke behind each contour (default: True).
    outline_color : str, optional
        Halo stroke color (default: 'black').
    outline_extra : float, optional
        Extra line width (points) for the halo beyond ``linewidth``; the halo
        is ``outline_extra / 2`` points per side (default: 1.6).
    mask_catalog : str or None, optional
        Compact-source catalog to NaN-mask before contouring. If None and the
        cluster enables masking, ``cluster.radio_catalog_file`` is used.
    cluster : Cluster, optional
        Cluster object providing default radio parameters.
    **kwargs
        Recognized namespaced overrides take precedence over the corresponding
        function arguments:
            - radio_color, radio_n_levels, radio_start_sigma, radio_contour_step
            - radio_smooth_pix, radio_linewidth
            - radio_outline, radio_outline_color, radio_outline_extra
            - radio_* : any ax.contour() kwarg
            - radio_kwargs : dict

    Returns
    -------
    contour_artist : QuadContourSet or None
        The matplotlib contour artist, or None if the noise could not be
        estimated (degenerate map).

    Notes
    -----
    - Radio parameters cascade: explicit ``radio_*`` kwarg > Cluster attribute
      > constants.py default, mirroring ``add_xray_contours``.
    - ``*SUB`` (point-source-subtracted) maps need no masking; enable masking
      only for un-subtracted cutouts via ``cluster.radio_mask_compact``.
    """

    # Namespaced kwarg overrides (radio_* takes precedence if supplied)
    radio_kwargs = pop_prefixed_kwargs(kwargs, 'radio')

    def _resolve(key, cluster_attr, current):
        # radio_<key> kwarg > cluster.<cluster_attr> > current (function default)
        if key in radio_kwargs:
            return radio_kwargs.pop(key)
        if cluster is not None and getattr(cluster, cluster_attr, None) is not None:
            return getattr(cluster, cluster_attr)
        return current

    color        = _resolve('color', 'radio_color', color)
    n_levels     = _resolve('n_levels', 'radio_n_levels', n_levels)
    start_sigma  = _resolve('start_sigma', 'radio_start_sigma', start_sigma)
    contour_step = _resolve('contour_step', 'radio_contour_step', contour_step)
    smooth_pix   = _resolve('smooth_pix', 'radio_smooth_pix', smooth_pix)
    linewidth    = _resolve('linewidth', 'radio_linewidth', linewidth)
    linewidth    = radio_kwargs.pop('linewidths', linewidth)
    outline       = radio_kwargs.pop('outline', outline)
    outline_color = radio_kwargs.pop('outline_color', outline_color)
    outline_extra = radio_kwargs.pop('outline_extra', outline_extra)

    # Compact-source masking: explicit catalog arg > cluster config
    if mask_catalog is None and cluster is not None and getattr(cluster, 'radio_mask_compact', False):
        mask_catalog = getattr(cluster, 'radio_catalog_file', None)

    # Load radio map and celestial WCS, then optionally mask and smooth
    data, wcs_radio = load_radio_image(radio_fits_file)
    if mask_catalog is not None:
        data, n_masked = mask_compact_sources(data, wcs_radio, mask_catalog)
        print(f"  masked {n_masked} compact radio sources")
    if smooth_pix and smooth_pix > 0:
        data = gaussian_filter(data, smooth_pix)

    sigma = robust_rms(data)
    if not np.isfinite(sigma) or sigma <= 0:
        print(f"[WARNING] could not estimate radio RMS for {radio_fits_file}; skipping radio overlay")
        return None

    levels = radio_contour_levels(sigma, start_sigma, n_levels, contour_step)
    contour_artist = ax.contour(
        data,
        levels=levels,
        transform=ax.get_transform(wcs_radio),
        colors=color,
        linewidths=linewidth,
        alpha=0.95,
        **radio_kwargs,
    )

    if outline:
        # Stroke a wider band of `outline_color` beneath each contour, leaving a
        # thin halo (outline_extra/2 per side) that preserves contrast where
        # light radio contours cross other contours.
        effects = [
            path_effects.Stroke(linewidth=linewidth + outline_extra, foreground=outline_color),
            path_effects.Normal(),
        ]
        try:
            contour_artist.set_path_effects(effects)
        except AttributeError:
            for c in contour_artist.collections:
                c.set_path_effects(effects)

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

    # Compute statistics only in this region
    if z_fov.size == 0:
        raise ValueError("No X-ray data falls within the optical field of view.")

    exposure_mask = np.isfinite(z) & (z > 0)
    z_valid = z[exposure_mask]
    if z_valid.size == 0:
        raise ValueError("No valid (finite, non-zero) X-ray exposure pixels.")
    med = float(np.median(z_valid))
    mad = float(np.median(np.abs(z_valid - med)))
    std = 1.4826 * mad
    pmin = float(np.percentile(z_valid, 1.0))

    z_fov_valid = z_fov[np.isfinite(z_fov) & (z_fov > 0)]
    if z_fov_valid.size == 0:
        raise ValueError(
            "No valid X-ray pixels inside the optical FoV — "
            "is the cluster center inside the X-ray exposure?"
        )
    pmax = float(z_fov_valid.max())

    if verbose:
        print("\n-------- X-ray Contours --------")
        print(f"  Exposure pixels: {z_valid.size} / {z.size} "
              f"({100*z_valid.size/z.size:.1f}%)")
        print(f"           Median: {med:.4e}")
        print(f"        sigma (1.48·MAD): {std:.4e}")
        print(f"  pmin (1.0 pct, exposure): {pmin:.4e}")
        print(f"  pmax (max in FoV):        {pmax:.4e}")
    if pedestal:
        # pedestal override
        pmin = pedestal

    # Define contour levels
    vmin = float(pmin + levels[0] * std)
    vmax = float(pmax - levels[1] * std)

    if verbose:
        print(f"       Contour Min: {vmin}")
        print(f"       Contour Max: {vmax}")

    num_levels = int(round(levels[2]))

    if logspace:
        level = np.geomspace(vmin, vmax, num_levels)
    else:
        level = np.linspace(vmin, vmax, num_levels)

    return level


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
    radio_fits_file=None,
    bcg_file = None,
    bcg_background = "light",
    legend_loc = None,
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
    radio_fits_file : str or None
        If provided, overlays radio continuum contours from this FITS file.
    bcg_file : str or None
        If provided, overlays BCG markers from this CSV file.
    bcg_background : str, optional
        "light" or "dark" - choose marker color scheme based on background.
    legend_loc : str or None, optional
        Legend corner. If None, uses ``cluster.legend_loc`` (set by the cluster's
        position in the field, shared across optical/X-ray/radio figures),
        falling back to the package default ("upper right").
    show_optical : bool, optional
        If True, shows optical image; if False, shows blank background.
    **kwargs
        Additional options, including:
            - X-ray: xray_levels, xray_psf, xray_color, xray_alpha, xray_linewidth
            - Density: density_bandwidth, density_levels, density_skip, density_color, density_alpha, density_linewidth, density_weighted
            - Radio: radio_start_sigma, radio_n_levels, radio_contour_step, radio_smooth_pix, radio_color, radio_linewidth, radio_outline
            - Scalebar: scalebar_arcmin, scalebar_color, scalebar_fontsize

    Returns
    -------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        The figure and axis.
    """

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

    # --- Overlay radio contours ---
    if radio_fits_file is not None:
        add_radio_contours(
            ax=ax,
            radio_fits_file=radio_fits_file,
            cluster=cluster,
            **kwargs
        )

    # --- Overlay BCG markers ---
    if bcg_file is not None:
        overlay_bcg_markers(ax, bcg_file, background=bcg_background)

    # --- Compose legend ---
    # Legend location: explicit arg > cluster.legend_loc > package default
    if legend_loc is None:
        legend_loc = getattr(cluster, "legend_loc", None) or DEFAULT_LEGEND_LOC

    custom_handles = []
    if xray_fits_file is not None:
        custom_handles.append(Line2D([0], [0], color=kwargs.get('xray_color', 'tab:red'), lw=kwargs.get('xray_linewidth', 1.2), label='X-ray contours'))
    if photometric_file is not None:
        custom_handles.append(Line2D([0], [0], color=kwargs.get('density_color', 'tab:blue'), lw=kwargs.get('density_linewidth', 1.2), label='Density contours'))
    if radio_fits_file is not None:
        radio_legend_color = kwargs.get('radio_color') or getattr(cluster, 'radio_color', None) or DEFAULT_RADIO_COLOR
        radio_legend_lw = kwargs.get('radio_linewidth') or getattr(cluster, 'radio_linewidth', None) or DEFAULT_RADIO_LINEWIDTH
        custom_handles.append(Line2D([0], [0], color=radio_legend_color, lw=radio_legend_lw, label='Radio contours'))

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
