#!/usr/bin/env python3
"""
analysis.py

X-ray Analysis: Surface Brightness Profiles
---------------------------------------------------------

Higher-level X-ray analysis routines that operate on FITS image arrays:
  - Surface brightness profile extraction between two sky/pixel positions

Independent of the optical pipeline stages (1-7); operates on an X-ray
image array plus its WCS. (The GGM / unsharp-masking diagnostic panels
live in ``plotting/xray.py`` as ``gaussian_grad_magnitude``.)

Requirements:
  - numpy, scipy, astropy

Notes:
  - ``profile_between_points`` supports both sky coordinates and pixel positions.
"""

from __future__ import annotations

import numpy as np

from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.ndimage import map_coordinates


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
