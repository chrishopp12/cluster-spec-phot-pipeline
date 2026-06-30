#!/usr/bin/env python3
"""
common.py

Shared Plotting Utilities
---------------------------------------------------------

Provides the foundational helpers that all other plotting modules depend on:
style configuration, figure persistence, simple-WCS and KDE-grid construction,
angular/physical scale bars, and BCG marker overlays.  These are intentionally
kept independent of any specific plot type so they can be composed freely.

Key functions:
  - setup_plot_style()       Set publication rcParams (call from CLI entry
                              points, not at import time)
  - finalize_figure()        Save figure to PDF via atomic temp-file staging,
                              optionally display, then close
  - resolve_save_path()      Turn (save_path, filename) into a full path
  - make_simple_wcs()        Build a simple TAN WCS spanning an RA/Dec box
  - evaluate_kde_grid()      Gaussian-KDE a weighted 2D point set onto a mesh
  - add_scalebar()           Draw a dual-label scale bar (kpc + arcmin) on a
                              WCS axis using the cluster redshift
  - overlay_bcg_markers()    Plot BCG positions as star/circle markers with a
                              color palette and legend

Requirements:
  - astropy, matplotlib, numpy, pandas, scipy

Notes:
  - finalize_figure() writes to a local temp directory first, then moves
    the file into place, avoiding partial-write issues on network mounts.
  - BCG marker colors follow the _BCG_COLORS palette; the first BCG swaps
    between black/white depending on the background argument.
  - overlay_bcg_markers() accepts either a list of BCG objects or a CSV
    path string for backward compatibility.
"""

from __future__ import annotations

import os
import tempfile
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.wcs import WCS
from scipy.stats import gaussian_kde

from cluster_pipeline.utils import pop_prefixed_kwargs
from cluster_pipeline.utils.coordinates import make_skycoord
from cluster_pipeline.utils.cosmology import redshift_to_proper_distance
from cluster_pipeline.constants import DEFAULT_DPI, DEFAULT_MARKER_SIZE_BCG

if TYPE_CHECKING:
    from cluster_pipeline.models.bcg import BCG


def setup_plot_style():
    """Set default matplotlib rcParams for publication figures.

    Call from CLI entry points, not at import time.
    """
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14


def resolve_save_path(
        save_plots: bool | None = None,
        save_path: str | None = None,
        filename: str | None = None
    ) -> str | None:
    """
    Determine the full save path for a file based on user preferences.

    Parameters
    ----------
    save_plots : bool
        Whether to save plots.
    save_path : str or None
        Directory or full path to save the file.
    filename : str or None
        Filename to use if save_path is a directory.

    Returns
    -------
    str or None
        Full path to save the file, or None if not saving.
    """
    if save_plots and save_path is not None:
        return os.path.join(save_path, filename) if os.path.isdir(save_path) else save_path
    return None


def finalize_figure(
    fig: plt.Figure,
    show_plots: bool = False,
    save_plots: bool = False,
    save_path: str | None | os.PathLike = None,
    filename: str | None = None,
    bbox_inches: str | None = "tight",
) -> None:
    # bbox_inches: defaults to "tight" (crops whitespace) for every figure. Pass
    # None for figures whose composition is already hand-tuned AND that carry a
    # colorbar placed in figure/divider coordinates: the tight-bbox crop remaps
    # absolute positions in the PDF backend and misrenders the colorbar QuadMesh
    # (the gradient lands in the wrong corner while the cax frame stays put).
    save_file = resolve_save_path(save_plots, save_path, filename)

    if save_file:
        save_file = Path(save_file)
        save_file.parent.mkdir(parents=True, exist_ok=True)

        # Stage to a guaranteed-local temp dir, then move into final location.
        with tempfile.TemporaryDirectory() as td:
            staged = Path(td) / save_file.name

            # Avoid Type3 embedding path; keep PDF output consistent
            with mpl.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}):
                fig.savefig(staged, dpi=DEFAULT_DPI, bbox_inches=bbox_inches)

            shutil.move(str(staged), str(save_file))

    if show_plots:
        try:
            fig.canvas.draw_idle()
        except Exception:
            pass
        plt.show()

    if save_plots or show_plots:
        plt.close(fig)


def make_simple_wcs(ra_min, ra_max, dec_min, dec_max, npix):
    """Build a simple TAN (gnomonic) WCS spanning an RA/Dec box on an npix x npix grid.

    Parameters
    ----------
    ra_min, ra_max, dec_min, dec_max : float
        Bounding box of the field, in degrees.
    npix : int
        Number of pixels per side.

    Returns
    -------
    astropy.wcs.WCS
        A 2D WCS centered on the box.
    """
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [npix / 2, npix / 2]
    wcs.wcs.cdelt = [-(ra_max - ra_min) / npix, (dec_max - dec_min) / npix]
    wcs.wcs.crval = [(ra_max + ra_min) / 2, (dec_max + dec_min) / 2]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return wcs


def evaluate_kde_grid(x, y, weights, bandwidth, npix):
    """Gaussian-KDE a 2D point set onto a regular npix x npix mesh.

    The caller supplies ``x``/``y`` already in the desired coordinate space
    (WCS pixels, tangent-plane offsets, ...); this evaluates the KDE over a grid
    spanning the data extent. Callers handle any coordinate transform themselves.

    Parameters
    ----------
    x, y : ndarray
        Point coordinates in a common 2D space.
    weights : ndarray or None
        Per-point KDE weights.
    bandwidth : float or str
        ``bw_method`` passed to ``scipy.stats.gaussian_kde``.
    npix : int
        Grid resolution per side.

    Returns
    -------
    x_mesh, y_mesh, density : ndarray
        Meshgrid coordinates and evaluated density, each shape (npix, npix).
    """
    kde = gaussian_kde(np.vstack([x, y]), bw_method=bandwidth, weights=weights)
    x_grid = np.linspace(x.min(), x.max(), npix)
    y_grid = np.linspace(y.min(), y.max(), npix)
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
    density = kde(np.vstack([x_mesh.ravel(), y_mesh.ravel()])).reshape(x_mesh.shape)
    return x_mesh, y_mesh, density


def add_scalebar(
        ax: plt.Axes,
        wcs,
        redshift: float,
        scalebar_arcmin: float = 1.0,
        color: str = 'white',
        fontsize: float = 12,
        **kwargs
    ) -> None:
    """
    Adds a scale bar to a WCS axis, labeling both physical (kpc) and angular (arcmin) units.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to add scale bar to.
    wcs : astropy.wcs.WCS
        WCS for coordinate transforms.
    redshift : float
        Redshift for scale calculation.
    scalebar_arcmin : float, optional
        Length of scale bar in arcminutes (default: 1.0).
    color : str, optional
        Scale bar and text color (default: 'white').
    fontsize : float, optional
        Font size for scale bar labels (default: 12).
    **kwargs
        Additional keyword arguments; recognized:
          Scalebar kwargs:
            - scalebar_arcmin : float
            - scalebar_color : str
            - scalebar_fontsize : float
            - scalebar_* : any ax.text() kwarg
            - scalebar_kwargs : dict
          Plot kwargs:
            - scalebar_plot_color : str
            - scalebar_plot_lw : float
            - scalebar_plot_* : any ax.plot() kwarg
            - scalebar_plot_kwargs : dict

        These take precedence over the direct arguments if supplied.

    Returns
    -------
    None

    """

    # Namespaced kwarg overrides (scalebar_* takes precedence if supplied)
    scalebar_kwargs = pop_prefixed_kwargs(kwargs, 'scalebar')
    scalebar_arcmin = scalebar_kwargs.get('arcmin', scalebar_arcmin)
    color = scalebar_kwargs.get('color', color)
    fontsize = scalebar_kwargs.get('fontsize', fontsize)

    # Accept extra plot kwargs for the bar itself (e.g., lw, linestyle)
    plot_kwargs = scalebar_kwargs.get('plot_kwargs', {}).copy()
    plot_kwargs.update(pop_prefixed_kwargs(scalebar_kwargs, 'plot'))
    plot_color = plot_kwargs.pop('color', color)
    plot_lw = plot_kwargs.pop('lw', 3)

    # Convert scalebar length to degrees, radians, and kpc
    scalebar_deg = scalebar_arcmin / 60.0
    scalebar_radian = scalebar_arcmin * u.arcmin.to(u.rad)
    DA_kpc = redshift_to_proper_distance(redshift)
    scalebar_kpc = DA_kpc * scalebar_radian

    # Choose bar location
    start_pixel_x, start_pixel_y = 40, 40
    start_world = wcs.pixel_to_world(start_pixel_x, start_pixel_y)

    # Draw scale bar
    ax.plot(
        [start_world.ra.deg, start_world.ra.deg - scalebar_deg / np.cos(start_world.dec.radian)],
        [start_world.dec.deg, start_world.dec.deg],
        color=plot_color,
        lw=plot_lw,
        transform=ax.get_transform('icrs'),
        zorder=100,
        **plot_kwargs,
    )

    # Add physical label
    ax.text(
        start_world.ra.deg - scalebar_deg/2 / np.cos(start_world.dec.radian),
        start_world.dec.deg + scalebar_deg/20,
        f"{int(np.round(scalebar_kpc, -1))} kpc",
        color=color,
        ha='center',
        va='bottom',
        transform=ax.get_transform('icrs'),
        fontsize=fontsize,
        zorder=100,
        **scalebar_kwargs,
    )
    # Add angular label
    ax.text(
        start_world.ra.deg - scalebar_deg/2 / np.cos(start_world.dec.radian),
        start_world.dec.deg - scalebar_deg/20,
        f"{scalebar_arcmin:.0f}'",
        color=color,
        ha='center',
        va='top',
        transform=ax.get_transform('icrs'),
        fontsize=fontsize,
        zorder=100,
        **scalebar_kwargs,
    )


# --------------------------------------------------------------------------
# Default BCG marker palette (indexed by position in the BCG list)
# --------------------------------------------------------------------------
_BCG_COLORS = [
    'white',        # swapped to black on light backgrounds
    'tab:green',
    'tab:purple',
    'tab:cyan',
    'tab:pink',
    'gold',
    'tab:orange',
    'tab:green',
    'tab:purple',
    'tab:pink',
]


def _bcgs_from_csv(path: str) -> list[BCG]:
    """Read a BCGs.csv file and return a list of BCG objects.

    This keeps backward compatibility for callers that still pass a file path.
    """
    from cluster_pipeline.models.bcg import BCG

    df = pd.read_csv(path)
    bcgs: list[BCG] = []
    for i, row in df.iterrows():
        bcg = BCG.from_dataframe_row(row, bcg_id=i + 1)
        # Assign a default label if the CSV didn't provide one
        if not bcg.label or bcg.label == str(bcg.bcg_id):
            if i < 5:
                bcg.label = str(i + 1)
            else:
                bcg.label = chr(ord("A") + i - 5)  # "A", "B", "C", ...
        bcgs.append(bcg)
    return bcgs


def overlay_bcg_markers(
    ax: plt.Axes,
    bcgs: list[BCG] | str,
    background: str = "light",
    zorder: int = 11,
    legend: bool = True,
    legend_loc: str = "upper left",
) -> plt.Axes:
    """Overlay BCG markers on a WCS-aware matplotlib axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot markers on (should be WCS-aware).
    bcgs : list[BCG] or str
        BCG objects to plot, or path to a BCGs.csv file for backward
        compatibility.
    background : str
        ``"light"`` or ``"dark"`` -- controls the primary marker color for
        BCG 1 (black on light backgrounds, white on dark). [default: "light"]
    zorder : int
        Z-order for the marker layer. [default: 11]
    legend : bool
        Whether to add a legend. [default: True]
    legend_loc : str
        Legend location string passed to ``ax.legend``. [default: "upper left"]

    Returns
    -------
    matplotlib.axes.Axes
        The input axis, for optional chaining.
    """
    # Accept either a list of BCG objects or a CSV path (backward compat)
    if isinstance(bcgs, (str, os.PathLike)):
        bcgs = _bcgs_from_csv(str(bcgs))

    handles = []
    for i, bcg in enumerate(bcgs):
        # --- color ---
        # Allow BCG objects to carry an explicit color (future-proofing);
        # fall back to the default palette.
        color = getattr(bcg, "color", None) or _BCG_COLORS[i % len(_BCG_COLORS)]

        # Swap the first BCG's color based on background
        if i == 0:
            color = "black" if background == "light" else "white"

        # --- marker style ---
        # Stars for the first 5 (bcg_id 1-5), circles for the rest
        marker = "*" if bcg.bcg_id <= 5 else "o"

        # --- label ---
        z_str = "unknown" if bcg.z is None else f"{bcg.z:.3f}"
        if bcg.label:
            display_label = f"BCG {bcg.label} (z = {z_str})"
        else:
            display_label = f"BCG {bcg.bcg_id} (z = {z_str})"

        # --- plot ---
        coord = make_skycoord([bcg.ra], [bcg.dec])
        sc = ax.scatter(
            coord.ra.deg,
            coord.dec.deg,
            marker=marker,
            edgecolor=color,
            facecolor="none",
            s=DEFAULT_MARKER_SIZE_BCG,
            zorder=zorder,
            transform=ax.get_transform("icrs"),
            label=display_label,
        )
        if legend:
            handles.append(sc)

    if legend and handles:
        ax.legend(loc=legend_loc, fontsize=10, frameon=True)

    return ax
