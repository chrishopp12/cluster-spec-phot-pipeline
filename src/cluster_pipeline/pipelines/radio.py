#!/usr/bin/env python3
"""
radio.py

Radio Pipeline Driver: Continuum Contour Overlays
---------------------------------------------------------

Orchestrates the radio overlay stage: composites radio continuum contours
onto the optical + X-ray + galaxy-density + BCG figure produced by
``plotting/optical.py:plot_optical``. The radio FITS is user-provided (like
the X-ray image) and located via ``cluster.radio_file``.

Entry point:
  - ``run_radio_imaging(cluster)`` — single-panel radio overlay figure

Data products:
  - Radio/Images/{cluster_id}_radio_overlay.pdf

Requirements:
  - All cluster_pipeline subpackages

Notes:
  - Radio processing is independent of Stages 1-8. It needs a radio FITS
    file and, for the underlying layers, the optical / X-ray / member catalogs.
  - If no radio FITS is configured or present, the stage is skipped.
  - The overlay FoV is ``cluster.radio_fov`` when set (radio relics often sit
    outside the default optical FoV), else ``cluster.fov``.
"""

from __future__ import annotations

import os

from cluster_pipeline.models.cluster import Cluster
from cluster_pipeline.io.images import get_optical_image
from cluster_pipeline.plotting.optical import plot_optical


def run_radio_imaging(
    cluster: Cluster,
    *,
    save_plots: bool = True,
    show_plots: bool = False,
    verbose: bool = True,
) -> None:
    """Composite radio continuum contours onto the optical overlay figure.

    Parameters
    ----------
    cluster : Cluster
        Cluster object with paths, metadata, and radio_* parameters.
    save_plots : bool
        Save the generated figure. [default: True]
    show_plots : bool
        Display the figure interactively. [default: False]
    verbose : bool
        Print detailed progress messages. [default: True]

    Notes
    -----
    - Radio FITS expected at ``cluster.radio_file`` (set via the ``radio.filename``
      config key, relative to ``Radio/``).
    - X-ray, density, and BCG layers are included when their inputs exist, so
      the stage degrades gracefully if run before the optical pipeline.
    """
    radio_file = cluster.radio_file
    if not radio_file or not os.path.isfile(radio_file):
        print(f"\nNo radio data at {radio_file} — skipping radio overlay")
        return

    if verbose:
        print(f"  Radio data: {radio_file}")

    # Radio relics often extend beyond the default optical FoV
    fov = cluster.radio_fov if cluster.radio_fov else cluster.fov
    optical_file = get_optical_image(cluster, fov)
    if optical_file is None:
        print("  No optical image available — skipping radio overlay")
        return

    radio_images_path = os.path.join(cluster.radio_path, "Images")
    os.makedirs(radio_images_path, exist_ok=True)
    slug = cluster.identifier.replace(" ", "_")
    out_pdf = os.path.join(radio_images_path, f"{slug}_radio_overlay.pdf")

    # Underlying layers are included only when their inputs are present
    xray_file = cluster.xray_file if os.path.isfile(cluster.xray_file) else None
    phot_file = cluster.get_phot_file()
    phot_file = phot_file if phot_file and os.path.isfile(phot_file) else None
    bcg_file = cluster.bcg_file if os.path.isfile(cluster.bcg_file) else None

    plot_optical(
        str(optical_file),
        cluster=cluster,
        xray_fits_file=xray_file,
        photometric_file=phot_file,
        radio_fits_file=radio_file,
        bcg_file=bcg_file,
        show_plots=show_plots,
        save_plots=save_plots,
        save_path=out_pdf,
    )
    print(f"  Radio overlay: {out_pdf}")
