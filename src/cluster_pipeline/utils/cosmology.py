#!/usr/bin/env python3
"""
cosmology.py

Cosmological Distance Calculations
---------------------------------------------------------

Thin wrappers around astropy's cosmology module for the two distance
measures used in the pipeline: comoving distance (Mpc) for large-scale
structure and angular-diameter distance (kpc/arcsec) for on-sky
projections at the cluster redshift.

Key functions:
  - redshift_to_comoving_distance()   Comoving distance in Mpc
  - redshift_to_proper_distance()     Angular diameter distance in kpc/arcsec

Requirements:
  - astropy, numpy

Notes:
  - Uses FlatLambdaCDM(H0=70, Om0=0.3) defined as COSMO in constants.py.
  - Returns bare floats/arrays (astropy units stripped on output).
"""

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
        Angular diameter distances in kpc.
    """

    return COSMO.angular_diameter_distance(z).to(u.kpc).value
