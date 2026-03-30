#!/usr/bin/env python3
"""
images.py

Optical Image Retrieval and Caching
---------------------------------------------------------

Queries HiPS survey servers for optical FITS images centered on a cluster
position, with optional RA/Dec offsets. Surveys are tried in priority order
(PanSTARRS DR1, then Legacy DR10); the first to return a valid image wins.
Results are cached to the cluster's Photometry/ directory so repeat calls
skip the download.

Data products:
  - Photometry/{cluster_id}_optical.fits   Cached optical cutout

Requirements:
  - astropy, astroquery (hips2fits), numpy

Notes:
  - FOV and offsets are specified in arcminutes.
  - is_fits_valid() rejects images that are mostly empty (>50% zero pixels).
  - Image dimensions default to DEFAULT_IMAGE_PIXELS from constants.py.
"""

from __future__ import annotations
import os
from typing import TYPE_CHECKING

import numpy as np
import astropy.units as u
from astropy.coordinates import Angle
from astropy.io import fits
from astroquery.hips2fits import hips2fits

from cluster_pipeline.constants import DEFAULT_IMAGE_PIXELS

if TYPE_CHECKING:
    from cluster_pipeline.models.cluster import Cluster


def get_optical_image(
        cluster: "Cluster",
        fov: float,
        ra_offset: float = 0.0,
        dec_offset: float = 0.0,
        check_validity: bool = True
    ) -> str | None:
    """Retrieve an optical image from HiPS surveys and save as FITS.

    Queries PanSTARRS DR1 and Legacy DR10 in priority order. The first
    survey to return a valid (mostly non-empty) image wins. Results are
    cached: if the FITS file already exists and is valid, it is returned
    immediately.

    Parameters
    ----------
    cluster : Cluster
        Cluster object providing coordinates and ``photometry_path``.
    fov : float
        Field of view in arcminutes (converted to degrees internally).
    ra_offset : float, optional
        RA offset in arcminutes (default 0).
    dec_offset : float, optional
        Dec offset in arcminutes (default 0).
    check_validity : bool, optional
        If True, reject images that are mostly empty (default True).

    Returns
    -------
    str or None
        Path to the saved FITS file, or None if no valid image was found.
    """
    ra_offset_deg = ra_offset/ 60
    dec_offset_deg = dec_offset/ 60
    fov_deg = fov/ 60

    # Query prioritized surveys
    surveys = [
    {'name': 'PanSTARRS', 'hips': 'CDS/P/PanSTARRS/DR1/color-i-r-g'},
    {'name': 'Legacy Survey', 'hips': 'CDS/P/DESI-Legacy-Surveys/DR10/color'},
]

    img_name = f"optical_image_{fov}_{ra_offset}_{dec_offset}.fits"
    fits_path = os.path.join(cluster.photometry_path, img_name)
    if is_fits_valid(fits_path):
        return fits_path

    for survey in surveys:
        try:
            print(f"Querying {survey['name']}...")
            result = hips2fits.query(
                hips=survey['hips'],
                width=DEFAULT_IMAGE_PIXELS[0],
                height=DEFAULT_IMAGE_PIXELS[1],
                ra=cluster.coords.ra + ra_offset_deg*u.deg,
                dec=cluster.coords.dec + dec_offset_deg*u.deg,
                fov=Angle(fov_deg, 'deg'),
                projection='TAN',
                format='fits',
            )
            survey_path = os.path.join(cluster.photometry_path, f"{survey['name']}_{img_name}")
            fits_path = os.path.join(cluster.photometry_path, img_name)
            result.writeto(survey_path, overwrite=True)
            result.writeto(fits_path, overwrite=True)
            print(f"Image retrieved and saved: {survey_path}")

            if check_validity and not is_fits_valid(survey_path):
                print(f"{survey['name']} image appears empty, checking the next survey...")
                try:
                    os.remove(survey_path)  # Remove the empty file
                except FileNotFoundError:
                    pass
                try:
                    os.remove(fits_path)  # Remove the empty file
                except FileNotFoundError:
                    pass
                continue  # Try the next survey

            return fits_path
        except Exception as e:
            print(f"Failed to retrieve image from {survey['name']}: {e}")

    print("No data available from the prioritized surveys.")
    return None


def is_fits_valid(
        fits_path: str | os.PathLike,
        min_nonzero_fraction: float = 0.7,
        min_intensity_threshold: float = 1e-5,
        verbose: bool = False
    ) -> bool:
    """
    Checks if a FITS file contains meaningful, non-empty data.

    Parameters
    ----------

    fits_path : str or Path
        Path to the FITS file to be checked.
    min_nonzero_fraction : float, optional
        Minimum fraction of nonzero pixels required for the image to be considered valid (default is 0.5).
    min_intensity_threshold : float, optional
        Minimum intensity threshold for a pixel to be considered non-empty (default is 1e-5).

    Returns
    -------
    bool
        True if the FITS file is valid (contains sufficient nonzero pixels), False otherwise.

    """
    import warnings
    from astropy.utils.exceptions import AstropyWarning

    warnings.filterwarnings("ignore", category=AstropyWarning) # Unless you want a bright yellow message about the header


    try:
        with fits.open(fits_path, memmap=False) as hdul:
            image_data = hdul[0].data

            if image_data is None or getattr(image_data, "size", 0) == 0:
                print(f"FITS file {fits_path} is completely empty.")
                return False

            # Replace NaNs with zeros for processing
            image_data = np.nan_to_num(image_data)

            # Count the fraction of nonzero pixels
            nonzero_pixels = np.sum(image_data > min_intensity_threshold)
            total_pixels = image_data.size
            nonzero_fraction = nonzero_pixels / total_pixels

            if verbose:
                print(f"\nChecking FITS: {fits_path}")
                print(f"   - Nonzero pixel fraction: {nonzero_fraction:.2%} (min required: {min_nonzero_fraction:.2%})\n")

            # Reject images that are mostly empty
            if nonzero_fraction < min_nonzero_fraction:
                print(f"Image is mostly empty: Only {nonzero_fraction:.2%} nonzero pixels.\n")
                return False

            return True  # Image is considered useful

    except Exception as e:
        if verbose:
            print(f"Error checking FITS validity: {e}")
        return False
