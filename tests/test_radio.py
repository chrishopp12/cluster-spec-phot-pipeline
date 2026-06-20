"""Tests for the radio image utilities in cluster_pipeline.radio.image.

These exercise the array-level helpers (noise, contour ladder, compact-source
masking, FITS axis-squeezing) with no external data or network access.
"""
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS

from cluster_pipeline.radio.image import (
    load_radio_image,
    robust_rms,
    radio_contour_levels,
    mask_compact_sources,
)
from cluster_pipeline.constants import DEFAULT_RADIO_CONTOUR_STEP


def _tan_wcs(ra0=150.0, dec0=30.0, npix=100, span_deg=0.1):
    """A simple 2-axis TAN WCS spanning span_deg across npix pixels."""
    w = WCS(naxis=2)
    w.wcs.crpix = [npix / 2, npix / 2]
    w.wcs.cdelt = [-span_deg / npix, span_deg / npix]
    w.wcs.crval = [ra0, dec0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return w


# ------------------------------------------------------------------
# robust_rms
# ------------------------------------------------------------------

def test_robust_rms_recovers_sigma():
    """MAD-based RMS recovers the input Gaussian sigma and ignores non-finite."""
    rng = np.random.default_rng(0)
    data = rng.normal(0.0, 2.0, (300, 300))
    sigma = robust_rms(data)
    assert abs(sigma - 2.0) < 0.1, sigma

    # Determinism: same input -> same output
    assert robust_rms(data) == sigma


def test_robust_rms_insensitive_to_bright_outliers():
    """A handful of bright pixels barely move the MAD estimate."""
    rng = np.random.default_rng(1)
    data = rng.normal(0.0, 1.0, (200, 200))
    clean = robust_rms(data)
    data[:5, :5] = 1e4  # injected bright source
    assert abs(robust_rms(data) - clean) < 0.05


def test_robust_rms_all_nan_is_nan():
    """An all-non-finite map yields NaN rather than raising."""
    assert np.isnan(robust_rms(np.full((10, 10), np.nan)))


# ------------------------------------------------------------------
# radio_contour_levels
# ------------------------------------------------------------------

def test_radio_contour_levels_geometric_ladder():
    """Levels are start_sigma*sigma at the base and step geometrically."""
    sigma = 0.5
    levels = radio_contour_levels(sigma, start_sigma=3.0, n_levels=8,
                                  contour_step=DEFAULT_RADIO_CONTOUR_STEP)
    assert len(levels) == 8
    assert abs(levels[0] - 3.0 * sigma) < 1e-12
    ratios = levels[1:] / levels[:-1]
    assert np.allclose(ratios, DEFAULT_RADIO_CONTOUR_STEP)


def test_radio_contour_levels_custom_step():
    """A finer step (1.2) is honored for low-dynamic-range maps."""
    levels = radio_contour_levels(1.0, start_sigma=2.5, n_levels=5, contour_step=1.2)
    assert np.allclose(levels[1:] / levels[:-1], 1.2)
    assert len(levels) == 5


# ------------------------------------------------------------------
# mask_compact_sources
# ------------------------------------------------------------------

def test_mask_compact_sources_masks_single_gaussian(tmp_path):
    """A single-Gaussian (SCode 'S') source NaN-masks a disk around its center."""
    npix = 100
    wcs = _tan_wcs(npix=npix)
    data = np.ones((npix, npix))

    center = wcs.pixel_to_world(npix / 2, npix / 2)
    cat = pd.DataFrame({
        "RAJ2000": [center.ra.deg],
        "DEJ2000": [center.dec.deg],
        "Maj": [0.0],
        "SCode": ["S"],
    })
    cat_csv = tmp_path / "cat.csv"
    cat.to_csv(cat_csv, index=False)

    masked, n = mask_compact_sources(data, wcs, cat_csv, extra_radius_arcsec=30.0)
    assert n == 1
    assert np.isnan(masked[npix // 2, npix // 2])   # center masked
    assert not np.isnan(masked[0, 0])               # far corner untouched
    assert np.isnan(data).sum() == 0                # input not mutated


def test_mask_compact_sources_skips_multigaussian(tmp_path):
    """Only SCode in only_scode is masked; 'M' complexes are left intact."""
    npix = 100
    wcs = _tan_wcs(npix=npix)
    data = np.ones((npix, npix))
    center = wcs.pixel_to_world(npix / 2, npix / 2)
    cat = pd.DataFrame({
        "RAJ2000": [center.ra.deg],
        "DEJ2000": [center.dec.deg],
        "Maj": [0.0],
        "SCode": ["M"],
    })
    cat_csv = tmp_path / "cat.csv"
    cat.to_csv(cat_csv, index=False)

    masked, n = mask_compact_sources(data, wcs, cat_csv)
    assert n == 0
    assert not np.isnan(masked).any()


# ------------------------------------------------------------------
# load_radio_image
# ------------------------------------------------------------------

def test_load_radio_image_squeezes_degenerate_axes(tmp_path):
    """A (Stokes, freq, y, x) cube loads as a 2D array with a 2-axis WCS."""
    w = WCS(naxis=4)
    w.wcs.ctype = ["RA---SIN", "DEC--SIN", "FREQ", "STOKES"]
    w.wcs.crpix = [25, 25, 1, 1]
    w.wcs.cdelt = [-0.001, 0.001, 1e6, 1]
    w.wcs.crval = [150.0, 30.0, 1.4e9, 1]

    data = np.ones((1, 1, 50, 50), dtype="float32")
    fits_path = tmp_path / "radio.fits"
    fits.PrimaryHDU(data=data, header=w.to_header()).writeto(fits_path)

    out, wcs_out = load_radio_image(fits_path)
    assert out.shape == (50, 50)
    assert out.dtype == np.float64       # cast to float
    assert wcs_out.naxis == 2            # celestial only
