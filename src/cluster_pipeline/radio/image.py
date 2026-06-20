#!/usr/bin/env python3
"""
image.py

Stage 10: Radio Image Processing — Loading, Noise, and Contour Levels
---------------------------------------------------------

Low-level utilities for radio FITS data: reading and axis-squeezing,
robust noise estimation, optional compact-source masking, and geometric
contour-level construction. These functions operate on arrays (and the
FITS path needed to recover a 2-axis WCS); plotting is handled upstream
by ``plotting/optical.py:add_radio_contours``.

Data products:
  - None (operates on in-memory arrays; figure output handled by callers).

Requirements:
  - numpy, pandas, scipy, astropy

Notes:
  - ``load_radio_image`` squeezes the degenerate (Stokes, frequency) axes
    carried by interferometric maps and returns a celestial 2-axis WCS so
    contours register on a WCSAxes built from the optical image.
  - ``robust_rms`` uses the median absolute deviation, which is insensitive
    to bright real emission and to the negative bowls left by point-source
    subtraction — far more reliable than ``np.std`` for radio maps.
  - ``mask_compact_sources`` expects an LoTSS-DR2-style source catalog
    (RAJ2000, DEJ2000, Maj, SCode columns).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS

from cluster_pipeline.constants import (
    DEFAULT_RADIO_MASK_RADIUS_ARCSEC,
    DEFAULT_RADIO_MASK_SIZE_MULTIPLIER,
)


def load_radio_image(radio_fits: str | Path) -> tuple[np.ndarray, WCS]:
    """Open a radio FITS and return its 2D data array and celestial WCS.

    Interferometric maps are commonly stored as (Stokes, frequency, y, x);
    the degenerate leading axes are squeezed and a 2-axis sky WCS is
    extracted for overlay on an optical WCSAxes.

    Parameters
    ----------
    radio_fits : str or Path
        Path to the radio FITS image.

    Returns
    -------
    data : np.ndarray
        2D radio image as float.
    wcs : astropy.wcs.WCS
        Celestial (2-axis) WCS for the image.
    """
    with fits.open(radio_fits) as hdul:
        data = np.squeeze(hdul[0].data)
        wcs = WCS(hdul[0].header).celestial
    return data.astype(float), wcs


def robust_rms(data: np.ndarray) -> float:
    """Estimate the noise RMS via the median absolute deviation.

    Returns ``1.4826 * MAD``, the MAD-to-sigma scaling for a Gaussian.

    Parameters
    ----------
    data : np.ndarray
        Radio image array. Non-finite pixels are ignored.

    Returns
    -------
    float
        Robust noise estimate, or NaN if no finite pixels are present.
    """
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return np.nan
    med = np.median(finite)
    mad = np.median(np.abs(finite - med))
    return float(1.4826 * mad)


def mask_compact_sources(
    data: np.ndarray,
    wcs: WCS,
    catalog_csv: str | Path,
    extra_radius_arcsec: float = DEFAULT_RADIO_MASK_RADIUS_ARCSEC,
    size_multiplier: float = DEFAULT_RADIO_MASK_SIZE_MULTIPLIER,
    only_scode: tuple[str, ...] = ("S",),
) -> tuple[np.ndarray, int]:
    """NaN-mask pixels near each catalogued compact source.

    The mask radius per source is ``max(extra_radius_arcsec,
    size_multiplier * Maj)``. Only sources whose ``SCode`` is in
    ``only_scode`` (default single-Gaussian ``"S"``) are masked, so
    multi-Gaussian complexes that may be partial diffuse detections are
    left intact.

    Parameters
    ----------
    data : np.ndarray
        Radio image array.
    wcs : astropy.wcs.WCS
        Celestial WCS for the image.
    catalog_csv : str or Path
        LoTSS-DR2-style source catalog (RAJ2000, DEJ2000, Maj, SCode).
    extra_radius_arcsec : float, optional
        Floor mask radius in arcsec.
    size_multiplier : float, optional
        Multiplier applied to each source's major axis when larger than
        the floor.
    only_scode : tuple of str, optional
        Source-code values eligible for masking.

    Returns
    -------
    masked : np.ndarray
        Copy of ``data`` with masked pixels set to NaN.
    n_masked : int
        Number of sources actually masked.
    """
    df = pd.read_csv(catalog_csv)
    if "SCode" in df.columns:
        df = df[df["SCode"].isin(only_scode)]
    if len(df) == 0:
        return data, 0

    ra = df["RAJ2000"].to_numpy()
    dec = df["DEJ2000"].to_numpy()
    maj = df["Maj"].to_numpy() if "Maj" in df.columns else np.zeros(len(df))
    rad_arcsec = np.maximum(extra_radius_arcsec, size_multiplier * maj)

    x_src, y_src = wcs.all_world2pix(ra, dec, 0)
    try:
        pix_scale_deg = np.mean(np.abs(np.diag(wcs.pixel_scale_matrix)))
    except Exception:
        pix_scale_deg = abs(wcs.wcs.cdelt[0])
    pix_scale_arcsec = pix_scale_deg * 3600.0

    ny, nx = data.shape
    yy, xx = np.indices((ny, nx))

    masked = data.copy()
    n_masked = 0
    for xs, ys, r_arcsec in zip(x_src, y_src, rad_arcsec):
        if not (np.isfinite(xs) and np.isfinite(ys)):
            continue
        r_pix = r_arcsec / pix_scale_arcsec
        if xs + r_pix < 0 or xs - r_pix > nx or ys + r_pix < 0 or ys - r_pix > ny:
            continue
        d2 = (xx - xs) ** 2 + (yy - ys) ** 2
        m = d2 <= r_pix ** 2
        if m.any():
            masked[m] = np.nan
            n_masked += 1
    return masked, n_masked


def radio_contour_levels(
    sigma: float,
    start_sigma: float,
    n_levels: int,
    contour_step: float,
) -> np.ndarray:
    """Build a geometric ladder of radio contour levels.

    Levels are ``start_sigma * sigma * contour_step ** k`` for
    ``k = 0 .. n_levels - 1``. The default ``contour_step`` of sqrt(2)
    gives the canonical doubling-every-two-levels ladder
    (e.g. 3, 6, 12, 24 sigma).

    Parameters
    ----------
    sigma : float
        Noise RMS (e.g. from ``robust_rms``).
    start_sigma : float
        Lowest contour in units of ``sigma``.
    n_levels : int
        Number of contour levels.
    contour_step : float
        Geometric ratio between successive levels.

    Returns
    -------
    np.ndarray
        Contour levels in image units, ascending.
    """
    return start_sigma * sigma * contour_step ** np.arange(int(round(n_levels)))
