"""X-ray image loading, hole-filling, and smoothing."""

import numpy as np
from astropy.convolution import Gaussian2DKernel, convolve
from scipy.ndimage import uniform_filter
from skimage.restoration import inpaint_biharmonic


def fill_holes(
        img: np.ndarray,
        threshold_ratio: float = 0.1,
        filter_sizes: list[int] | None = None,
        plot: bool = False,
        exact: bool = False
    ):
    """
    Fill holes in an image based on a threshold ratio and optional filtering.

    Parameters
    ----------
    img : np.ndarray
        Input image array.
    threshold_ratio : float, optional
        Ratio to determine holes based on local mean [default: 0.1].
    filter_sizes : list[int] or None, optional
        Sizes of filters to apply for local mean calculation [default: None].
    plot : bool, optional
        Whether to plot intermediate results [default: False].
    exact : bool, optional
        If True, holes are exactly where img == 0 [default: False].

    Returns
    -------
    np.ndarray
        Image with holes filled.
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
        fill: bool = True
    ) -> np.ndarray:
    """
    Smooth an image with a Gaussian kernel, optionally filling holes first.

    Parameters
    ----------
    img : np.ndarray
        Input image array.
    kernel_std : float
        Standard deviation for Gaussian kernel.
    filter_sizes : list[int] or None, optional
        Sizes of filters to use for hole filling [default: None].
    fill : bool, optional
        Whether to fill holes before smoothing [default: True].

    Returns
    -------
    np.ndarray
        Smoothed image.
    """

    if fill:
        filled_img = fill_holes(img, filter_sizes=filter_sizes)

    else:
        filled_img = np.nan_to_num(img, nan=0.0)

    # Smooth the result
    g = Gaussian2DKernel(x_stddev=kernel_std)
    smoothed_image = convolve(filled_img, g)

    return smoothed_image
