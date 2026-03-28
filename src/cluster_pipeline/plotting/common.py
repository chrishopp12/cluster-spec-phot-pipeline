"""Shared plotting utilities: figure saving, scale bars, BCG markers."""

from __future__ import annotations
import os
import tempfile
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import astropy.units as u

from cluster_pipeline.utils import pop_prefixed_kwargs
from cluster_pipeline.utils.cosmology import redshift_to_proper_distance


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
) -> None:
    save_file = resolve_save_path(save_plots, save_path, filename)

    if save_file:
        save_file = Path(save_file)
        save_file.parent.mkdir(parents=True, exist_ok=True)

        # Stage to a guaranteed-local temp dir, then move into final location.
        with tempfile.TemporaryDirectory() as td:
            staged = Path(td) / save_file.name

            # Avoid Type3 embedding path; keep PDF output consistent
            with mpl.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}):
                fig.savefig(staged, dpi=450, bbox_inches="tight")

            shutil.move(str(staged), str(save_file))

    if show_plots:
        try:
            fig.canvas.draw_idle()
        except Exception:
            pass
        plt.show()

    if save_plots or show_plots:
        plt.close(fig)


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


def overlay_bcg_markers(
    ax: plt.Axes,
    bcg_csv: str,
    background: str = "light",    # or "dark"
    zorder: int = 11,
    legend: bool = True,
    legend_loc: str = "upper left"
) -> plt.Axes:
    """
    Overlay BCG markers on a WCS-aware matplotlib axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot markers on.
    bcg_csv : str
        Path to the BCG CSV file.
    background : str
        "light" or "dark" (controls primary marker color).
    zorder : int
        Z-order for plotting.
    legend : bool
        Whether to add a legend.
    legend_loc : str
        Legend location.
    """
    # Load BCGs
    df = pd.read_csv(bcg_csv)
    bcg_colors = [
        'white',
        'tab:green',
        'tab:purple',
        'tab:cyan',
        'tab:pink',
        'gold',
        'tab:orange',
        'tab:green',
        'tab:purple',
        'tab:pink'
    ]
    bcg_labels = ['A', 'B', 'C', 'D', 'E', 'F']

    # Optionally swap white/black for BCG 1 depending on background
    if background == "dark":
        bcg_colors[0] = "white"
    elif background == "light":
        bcg_colors[0] = "black"

    handles = []
    for i, row in df.iterrows():
        ra = float(row['RA'])
        dec = float(row['Dec'])
        z_bcg = row['z'] if not pd.isnull(row['z']) else None

        color = bcg_colors[i % len(bcg_colors)]
        if i < 5:
            label = f"BCG {i+1} (z = {'unknown' if z_bcg is None else f'{z_bcg:.3f}'})"
            marker = '*'
        else:
            label = f"BCG {bcg_labels[i-5]} (z = {'unknown' if z_bcg is None else f'{z_bcg:.3f}'})"
            marker = 'o'

        # Plot
        sc = ax.scatter(
            ra,
            dec,
            marker=marker,
            edgecolor=color,
            facecolor='none',
            s=200,
            zorder=zorder,
            transform=ax.get_transform('icrs'),
            label=label
        )
        # For manual legend
        if legend:
            handles.append(sc)

    if legend and handles:
        ax.legend(loc=legend_loc, fontsize=10, frameon=True)

    return ax
