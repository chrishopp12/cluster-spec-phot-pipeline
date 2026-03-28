"""FITS I/O and optical image retrieval/caching."""

from __future__ import annotations
import os
from typing import TYPE_CHECKING

import numpy as np
import astropy.units as u
from astropy.coordinates import Angle
from astropy.io import fits
from astroquery.hips2fits import hips2fits

if TYPE_CHECKING:
    from cluster_pipeline.models.cluster import Cluster


def get_optical_image(
        cluster: "Cluster",
        fov: float,
        ra_offset: float = 0.0,
        dec_offset: float = 0.0,
        check_validity: bool = True
    ) -> str | None:
    """Retrieves optical images from specified surveys (Legacy, PanSTAARS, more to come) and saves them as FITS files.

    Parameters
    ----------
    folder : str
        Path to the folder where the FITS file will be saved.
    cluster_coords : SkyCoord
        Astropy SkyCoord object with the cluster coordinates.
    fov : float
        Field of view in degrees for the image.
    ra_offset : float, optional
        Offset in degrees to apply to the Right Ascension of the cluster coordinates (default is 0).
    dec_offset : float, optional
        Offset in degrees to apply to the Declination of the cluster coordinates (default is 0).
    check_validity : bool, optional
        If True, checks if the retrieved FITS file is valid (default is True).

    Returns
    ---------
    fits_path : str
        Path to the saved FITS file containing the optical image.
    None
        If no valid image is retrieved from the surveys.
    """
    ra_offset_deg = ra_offset/ 60
    dec_offset_deg = dec_offset/ 60
    fov_deg = fov/ 60

    # Query prioritized surveys
    surveys = [
    {'name': 'PanSTARRS', 'hips': 'CDS/P/PanSTARRS/DR1/color-i-r-g'},
    {'name': 'Legacy Survey', 'hips': 'CDS/P/DESI-Legacy-Surveys/DR10/color'},
]

    fits_path = f"{cluster.photometry_path}/optical_image_{fov}_{ra_offset}_{dec_offset}.fits"
    if is_fits_valid(fits_path):
        return fits_path

    for survey in surveys:
        try:
            print(f"Querying {survey['name']}...")
            result = hips2fits.query(
                hips=survey['hips'],
                width=802,
                height=800,
                ra=cluster.coords.ra + ra_offset_deg*u.deg,
                dec=cluster.coords.dec + dec_offset_deg*u.deg,
                fov=Angle(fov_deg, 'deg'),
                projection='TAN',
                format='fits',
            )
            survey_path = f"{cluster.photometry_path}/{survey['name']}_optical_image_{fov}_{ra_offset}_{dec_offset}.fits"
            fits_path = f"{cluster.photometry_path}/optical_image_{fov}_{ra_offset}_{dec_offset}.fits"
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
