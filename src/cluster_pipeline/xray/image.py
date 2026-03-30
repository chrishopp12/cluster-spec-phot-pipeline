#!/usr/bin/env python3
"""
image.py

X-ray Image Processing: Hole-Filling and Smoothing
---------------------------------------------------------

Low-level image processing utilities for X-ray FITS data. Handles
hole-filling via biharmonic inpainting and Gaussian smoothing.

These functions operate on raw numpy arrays — they do not load FITS
files or reference the Cluster class. FITS I/O is handled upstream
by the orchestration layer.

Requirements:
  - numpy, scipy, scikit-image, astropy

Notes:
  - ``fill_holes`` uses iterative biharmonic inpainting with adaptive
    thresholds to fill gaps in X-ray images (e.g., chip gaps, masked regions).
  - ``smoothing`` applies a Gaussian kernel after optional hole-filling.
  - Both functions handle NaN values by replacing with 0 before processing.
"""

from __future__ import annotations

import numpy as np
from astropy.convolution import Gaussian2DKernel, convolve
from scipy.ndimage import uniform_filter
from skimage.restoration import inpaint_biharmonic


def fill_holes(
    img: np.ndarray,
    threshold_ratio: float = 0.1,
    filter_sizes: list[int] | None = None,
    plot: bool = False,
    exact: bool = False,
) -> np.ndarray:
    """Fill holes in an image via iterative biharmonic inpainting.

    Identifies pixels that fall below a fraction of the local mean
    (or exactly zero if ``exact=True``) and fills them using
    biharmonic interpolation.

    Parameters
    ----------
    img : np.ndarray
        Input 2D image array.
    threshold_ratio : float, optional
        Pixels below ``threshold_ratio * local_mean`` are treated as holes.
        [default: 0.1]
    filter_sizes : list[int] or None, optional
        Uniform filter sizes for computing local means. Multiple sizes
        trigger iterative passes (coarse then fine). [default: [5, 2]]
    plot : bool, optional
        If True, display intermediate diagnostic plots. [default: False]
    exact : bool, optional
        If True, only pixels exactly equal to 0 are treated as holes.
        Overrides ``threshold_ratio``. [default: False]

    Returns
    -------
    np.ndarray
        Image with holes filled. Same shape as input.

    Notes
    -----
    - Uses ``skimage.restoration.inpaint_biharmonic`` for interpolation.
    - NaN values are replaced with 0 before processing.
    """
    if filter_sizes is None:
        filter_sizes = [5, 2]
    elif isinstance(filter_sizes, int):
        filter_sizes = [filter_sizes]

    working_img = np.array(img, copy=True)

    if exact:
        mask = img == 0
    else:
        local_mean = uniform_filter(img, size=filter_sizes[0], mode='mirror')
        mask = img < (threshold_ratio * local_mean)

    for filter_size in filter_sizes:
        local_mean = uniform_filter(working_img, size=filter_size, mode='mirror')
        adaptive_mask = working_img < (threshold_ratio * local_mean)
        img_filled = inpaint_biharmonic(working_img, adaptive_mask.astype(bool), channel_axis=None)
        working_img[adaptive_mask] = img_filled[adaptive_mask]
        working_img = np.nan_to_num(working_img, nan=0.0)

    final_img = np.array(img, copy=True)
    final_img[mask] = working_img[mask]
    final_img = np.nan_to_num(final_img, nan=0.0)

    return final_img


def smoothing(
    img: np.ndarray,
    kernel_std: float,
    filter_sizes: list[int] | None = None,
    fill: bool = True,
) -> np.ndarray:
    """Smooth an image with a Gaussian kernel, optionally filling holes first.

    Parameters
    ----------
    img : np.ndarray
        Input 2D image array.
    kernel_std : float
        Standard deviation for the Gaussian kernel (in pixels).
    filter_sizes : list[int] or None, optional
        Filter sizes for hole-filling (passed to ``fill_holes``).
        [default: None, uses fill_holes default of [5, 2]]
    fill : bool, optional
        If True, fill holes before smoothing. [default: True]

    Returns
    -------
    np.ndarray
        Smoothed image. Same shape as input.

    Notes
    -----
    - Uses ``astropy.convolution.Gaussian2DKernel`` and ``convolve``.
    - If ``fill=False``, NaN values are replaced with 0 before smoothing.
    """
    if fill:
        filled_img = fill_holes(img, filter_sizes=filter_sizes)
    else:
        filled_img = np.nan_to_num(img, nan=0.0)

    g = Gaussian2DKernel(x_stddev=kernel_std)
    return convolve(filled_img, g)
