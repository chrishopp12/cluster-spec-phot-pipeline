"""Cosmological distance calculations."""

import numpy as np
import astropy.units as u

from cluster_pipeline.constants import COSMO


def redshift_to_comoving_distance(z: float | np.ndarray) -> float | np.ndarray:
    """
    Convert redshift to comoving distance using the Planck18 cosmology.

    Parameters
    ----------
    z : float | np.ndarray
        Redshift values

    Returns
    -------
    comoving_distance : float | np.ndarray
        Comoving distances in Mpc
    """
    comoving_distance = COSMO.comoving_distance(z).to(u.Mpc).value  # Convert to Mpc

    return comoving_distance


def redshift_to_proper_distance(z: float | np.ndarray) -> float | np.ndarray:
    """
    Convert redshift to proper distance at the time of emission using Planck18 cosmology.

    Parameters
    ----------
    z : float | np.ndarray
        Redshift values

    Returns
    -------
    D_A : float | np.ndarray
        Proper distances in Mpc
    """

    return COSMO.angular_diameter_distance(z).to(u.kpc).value  # Convert to kpc
