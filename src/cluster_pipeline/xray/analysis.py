"""X-ray analysis: surface brightness profiles, gradient magnitude, unsharp masking."""

from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.collections import LineCollection

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.ndimage import map_coordinates, gaussian_filter, gaussian_gradient_magnitude
from reproject import reproject_interp, reproject_adaptive, reproject_exact

from cluster_pipeline.utils.coordinates import arcsec_to_pixel_std
from cluster_pipeline.xray.image import fill_holes, smoothing
from cluster_pipeline.plotting.common import finalize_figure, add_scalebar
from cluster_pipeline.plotting.optical import add_xray_contours


def profile_between_points(
    image2d,
    wcs,
    p1,
    p2,
    n_samples=512,
    width_pix=1,
    aggregate="mean"
):
    """
    Sample a surface-brightness profile between two points on an image.

    Parameters
    ----------
    image2d : np.ndarray (ny, nx)
        2D image array to sample (e.g., smoothed X-ray map).
    wcs : astropy.wcs.WCS
        WCS for `image2d` (2D WCS).
    p1, p2 : tuple or SkyCoord
        Endpoints of the profile. If tuple of length 2:
           - Interpreted as (RA_deg, Dec_deg) if values look like sky coords
             (i.e., |RA|<=360 and |Dec|<=90).
           - Otherwise interpreted as pixel (x, y).
        SkyCoord is also accepted (assumed ICRS).
    n_samples : int, optional
        Number of samples along the line.
    width_pix : int, optional
        Odd integer number of pixels to average across perpendicular to the
        line (1 means a single-pixel line).
    aggregate : {"mean","median"}, optional
        How to combine the perpendicular samples.

    Returns
    -------
    s_arcsec : np.ndarray (n_samples,)
        Cumulative distance along the line in arcsec (starts at 0).
    values : np.ndarray (n_samples,)
        Sampled (and optionally averaged) surface brightness.
    x_line, y_line : np.ndarray
        Pixel coordinates of the sampled centerline (for plotting).
    """
    # --- Resolve endpoints to pixel coordinates ---
    def _as_pix(pt):
        if isinstance(pt, SkyCoord):
            x, y = wcs.world_to_pixel(pt)
            return float(x), float(y)
        if isinstance(pt, (tuple, list, np.ndarray)) and len(pt) == 2:
            a, b = float(pt[0]), float(pt[1])
            if -1.0 <= b <= 91.0 and -1.0 <= a <= 361.0:  # looks like (RA, Dec)
                x, y = wcs.world_to_pixel(SkyCoord(a*u.deg, b*u.deg))
                return float(x), float(y)
            return a, b  # assume pixel coords
        raise ValueError("p1/p2 must be (RA,Dec), (x,y), or SkyCoord")

    def _pixscale_arcsec_per_pix(wcs):
        """
        Return mean pixel scale in arcsec/pixel as a float.
        Handles: ndarray Quantity, list/tuple of Quantities, or plain floats.
        If unitless, assume degrees-per-pixel.
        """
        s = wcs.proj_plane_pixel_scales()
        try:
            q = u.Quantity(s)                          # normalize to a Quantity array
        except Exception:
            q = np.asarray(s) * u.deg                  # plain floats -> assume deg/pix

        if q.unit is u.dimensionless_unscaled:
            q = q * u.deg                              # unitless -> assume deg/pix

        return np.mean(q.to(u.arcsec)).value           # -> float arcsec/pix

    x1, y1 = _as_pix(p1)
    x2, y2 = _as_pix(p2)

    # Line samples in pixel space
    x_line = np.linspace(x1, x2, n_samples)
    y_line = np.linspace(y1, y2, n_samples)

    # Perpendicular offsets (for width averaging)
    width_pix = int(width_pix)
    if width_pix < 1:
        width_pix = 1
    if width_pix % 2 == 0:
        width_pix += 1
    offsets = np.arange(-(width_pix//2), width_pix//2 + 1, dtype=float)

    # Unit perpendicular vector
    dx = x2 - x1
    dy = y2 - y1
    L = np.hypot(dx, dy) + 1e-12
    px = -dy / L
    py =  dx / L

    # Sample with map_coordinates (y first, then x)
    samples = []
    for o in offsets:
        xs = x_line + o * px
        ys = y_line + o * py
        vals = map_coordinates(image2d, [ys, xs], order=1, mode="nearest")
        samples.append(vals)
    stack = np.vstack(samples)

    if aggregate == "median":
        values = np.median(stack, axis=0)
    else:
        values = np.mean(stack, axis=0)

    # Distance axis (arcsec)
    pixscale_arcsec = _pixscale_arcsec_per_pix(wcs)
    s_pix = np.linspace(0.0, L, n_samples)
    s_arcsec = s_pix * pixscale_arcsec

    return s_arcsec, values, x_line, y_line


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

    # plot_xray_3d(xray_smoothed, wcs_optical, step=4)

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



    # ny, nx = xray_data.shape
    # # ny, nx = xray_smoothed.shape
    # dec_cut_deg = float(p1[1])

    # # Pixel row for that Dec
    # # ra_mid = wcs_optical.pixel_to_world(nx/2, ny/2).ra.deg
    # # _, y0 = wcs_optical.world_to_pixel(SkyCoord(ra_mid * u.deg, dec_cut_deg * u.deg))

    # ra_mid = wcs_xray.pixel_to_world(nx/2, ny/2).ra.deg
    # _, y0 = wcs_xray.world_to_pixel(SkyCoord(ra_mid * u.deg, dec_cut_deg * u.deg))

    # # Left/right pixel endpoints across the whole field
    # p_left  = (0, y0)
    # p_right = (nx - 1, y0)

    # # Profile along the full width at that Dec
    # s_h, sb_h, xh, yh = profile_between_points(
    #     image2d=xray_data, wcs=wcs_xray,
    #     p1=p_left, p2=p_right,
    #     n_samples=nx, width_pix=1, aggregate="mean",
    # )

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
    if save_plots and save_path is not None:
        if os.path.isdir(save_path):
            fname = f"gauss.pdf"
            full_path = os.path.join(save_path, fname)
        else:
            full_path = save_path
        fig.savefig(full_path, bbox_inches='tight')
        print(f"Saved figure: {full_path}")
    if show_plots:
        plt.show()
