"""SkyCoord helpers and angular utilities."""

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.wcs import WCS


def make_skycoord(
    ra_deg: np.ndarray | list[float],
    dec_deg: np.ndarray | list[float],
) -> SkyCoord:
    """
    Create an ICRS SkyCoord from RA/Dec in degrees.

    Parameters
    ----------
    ra_deg : np.ndarray or list of float
        Right Ascension values in degrees.
    dec_deg : np.ndarray or list of float
        Declination values in degrees.

    Returns
    -------
    SkyCoord
        SkyCoord object with the provided RA and Dec in the ICRS frame.
    """
    return SkyCoord(
        ra=np.asarray(ra_deg, dtype=float) * u.deg,
        dec=np.asarray(dec_deg, dtype=float) * u.deg,
        frame="icrs",
    )


def skycoord_from_df(
    df: pd.DataFrame,
    *,
    ra_col: str = "RA",
    dec_col: str = "Dec",
) -> SkyCoord:
    """
    Create SkyCoord from a DataFrame with RA/Dec columns (degrees).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing RA and Dec columns.
    ra_col : str
        Name of the RA column (default: "RA").
    dec_col : str
        Name of the Dec column (default: "Dec").

    Returns
    -------
    SkyCoord
        SkyCoord object with coordinates from the specified DataFrame columns.
    """
    if ra_col not in df.columns or dec_col not in df.columns:
        raise KeyError(f"DataFrame must contain '{ra_col}' and '{dec_col}' columns.")
    return make_skycoord(df[ra_col].values, df[dec_col].values)



def angular_sep(
        ra1: float,
        dec1: float,
        ra2: float,
        dec2: float
    ) -> float:
    """
    Computes angular separation in degrees using astropy's SkyCoord.

    Parameters
    ----------
    ra1, dec1 : float
        Coordinates of the first point (RA, Dec).
    ra2, dec2 : float
        Coordinates of the second point (RA, Dec).

    Returns
    -------
    float
        Angular separation in degrees.
    """
    c1 = SkyCoord(ra=ra1*u.deg, dec=dec1*u.deg)
    c2 = SkyCoord(ra=ra2*u.deg, dec=dec2*u.deg)
    sep = c1.separation(c2)

    return sep.deg


def arcsec_to_pixel_std(
        fwhm_arcsec: float,
        wcs: WCS
    ) -> float:
    """
    Convert an FWHM in arcseconds to Gaussian stddev in pixels, using WCS.

    Parameters
    ----------
    fwhm_arcsec : float
        Desired smoothing FWHM in arcseconds.
    wcs : astropy.wcs.WCS
        WCS object for your image.

    Returns
    -------
    sigma_pixels : float
        Standard deviation in pixels for Gaussian kernel.
    """

    # Pixel scale in arcsec/pixel
    pixscale = np.abs(wcs.proj_plane_pixel_scales()[0]).to_value(u.arcsec)

    fwhm_pixels = fwhm_arcsec / pixscale
    sigma_pixels = float(fwhm_pixels / 2.355)  # Convert FWHM to sigma


    return sigma_pixels
