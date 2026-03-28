"""Optical image plotting with X-ray and density contour overlays."""

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from astropy.io import fits
from astropy.wcs import WCS
from scipy.stats import gaussian_kde

from cluster_pipeline.utils import pop_prefixed_kwargs
from cluster_pipeline.utils.coordinates import arcsec_to_pixel_std
from cluster_pipeline.xray.image import smoothing
from cluster_pipeline.io.catalogs import load_photo_coords
from cluster_pipeline.plotting.common import (
    finalize_figure,
    add_scalebar,
    overlay_bcg_markers,
)

if TYPE_CHECKING:
    from cluster_pipeline.models.cluster import Cluster


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
