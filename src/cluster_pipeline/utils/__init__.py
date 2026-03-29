"""Small generic utilities used across the package."""

from __future__ import annotations
import argparse
import os
from typing import Any

import numpy as np
import pandas as pd


def string_to_numeric(s: str) -> float | int | str:
    """
    Converts a string to a numeric type if possible.

    Parameters
    ----------
    s : str
        Input string.

    Returns
    -------
    float | int | str
        Converted numeric value or original string if conversion fails.
    """
    s = s.strip()
    if s == '':
        return s
    try:
        if '.' in s or 'e' in s.lower():
            return float(s)
        else:
            return int(s)
    except Exception:
        return s


def find_first_val(*vals: Any) -> Any:
    """
    Returns the first non-None value from the provided arguments.

    Parameters
    ----------
    *vals : Any
        Variable number of arguments.

    Returns
    -------
    Any
        The first non-None value, or None if all are None.
    """
    for v in vals:
        if v is not None:
            return v
    return None


def to_float_or_none(x: Any) -> float | None:
    """
    Convert to a finite float, else None.

    Handles: None, NaN/inf, numpy scalars, masked values, numeric strings.
    """
    if x is None:
        return None
    # numpy masked / astropy masked
    if hasattr(x, "mask") and bool(getattr(x, "mask", False)):
        return None
    try:
        v = float(x)
    except Exception:
        return None
    return v if np.isfinite(v) else None


def coerce_to_numeric(
        df: pd.DataFrame,
        columns: list[str] | tuple[str, ...]
    ) -> pd.DataFrame:
    """
    Coerces specified columns of a DataFrame to numeric, forcing errors to NaN.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list[str] | tuple[str, ...]
        List of column names to coerce.

    Returns
    -------
    out : pd.DataFrame
        Modified DataFrame with specified columns coerced to numeric.
    """
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors='coerce')
    return out



def pop_prefixed_kwargs(kwargs: dict[str, Any], prefix: str) -> dict[str, Any]:
    """
    Extracts and removes all keyword arguments from `kwargs` that start with the given `prefix`.

    Parameters
    ----------
    kwargs : dict
        Dictionary of keyword arguments.
    prefix : str
        Prefix to filter the keyword arguments.
    Returns
    -------
    dict[str, Any]
        A new dictionary containing only the keyword arguments that started with the specified prefix,
        with the prefix removed from their keys.
    """
    out = kwargs.pop(f'{prefix}_kwargs', {}).copy()
    for k in list(kwargs):
        if k.startswith(f'{prefix}_'):
            out[k[len(prefix) + 1:]] = kwargs.pop(k)
    return out


def str2bool(v: str | bool) -> bool:
    """
    Argparse-compatible string to boolean conversion.

    Parameters
    ----------
    v : str | bool
        Input string or boolean.

    Returns
    -------
    bool
        Converted boolean value.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_color_mag_functions(
        color_type: str
    ) -> tuple[list[str], str, str, callable, callable]:
    """
    Returns required columns, labels, and functions for color and magnitude based on color_type.

    Parameters
    ----------
    color_type : str
        Color combination to use: "g-r", "r-i", or "g-i".

    Returns
    -------
    required_cols : list of str
        List of required magnitude columns for the chosen color_type.
    color_label : str
        Label for the color axis.
    mag_label : str
        Label for the magnitude axis.
    color_func : callable
        Function to compute color from a DataFrame.
    mag_func : callable
        Function to compute magnitude from a DataFrame.
    """


    if color_type == 'g-r' or color_type == 'gr':
        required_cols = ['gmag', 'rmag']
        color_label = 'g - r'
        mag_label = 'r'
        color_func = lambda df: df['gmag'] - df['rmag']  # noqa: E731
        mag_func = lambda df: df['rmag']  # noqa: E731
        return required_cols, color_label, mag_label, color_func, mag_func

    if color_type == 'r-i' or color_type == 'ri':
        required_cols = ['rmag', 'imag']
        color_label = 'r - i'
        mag_label = 'i'
        color_func = lambda df: df['rmag'] - df['imag']  # noqa: E731
        mag_func = lambda df: df['imag']  # noqa: E731
        return required_cols, color_label, mag_label, color_func, mag_func

    if color_type == 'g-i' or color_type == 'gi':
        required_cols = ['gmag', 'imag']
        color_label = 'g - i'
        mag_label = 'i'
        color_func = lambda df: df['gmag'] - df['imag'] # noqa: E731
        mag_func = lambda df: df['imag'] # noqa: E731
        return required_cols, color_label, mag_label, color_func, mag_func

    raise ValueError(f"Unknown color_type '{color_type}'. Supported types: 'g-r', 'gr', 'r-i', 'ri', 'g-i', 'gi'.")


def split_members_by_spec(
    members_df: pd.DataFrame,
    *,
    z_col: str = 'z',
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (spec_members, phot_members) from the final member catalog.

    Parameters
    ----------
    members_df : pd.DataFrame
        DataFrame containing the final member catalog.
    z_col : str, optional
        Name of the redshift column [default: Z_COL].

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Tuple containing (spec_members, phot_members).
    """
    spec_mask = members_df[z_col].notna()
    phot_mask = ~spec_mask

    spec_members = members_df.loc[spec_mask].copy()
    phot_members = members_df.loc[phot_mask].copy()
    return spec_members, phot_members
