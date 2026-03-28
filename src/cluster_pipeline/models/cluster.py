"""Galaxy cluster metadata and path management.

The Cluster class is a lean data container — it holds identity,
coordinates, analysis parameters, and computes file paths. It does
NOT query external services or read/write files. For that, use
``cluster_init()`` to construct a fully populated Cluster from
config.yaml, clusters.csv, and/or NED/SIMBAD.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

from cluster_pipeline.constants import (
    DEFAULT_FOV_ARCMIN,
    DEFAULT_FOV_FULL_ARCMIN,
    DEFAULT_RA_OFFSET_ARCMIN,
    DEFAULT_DEC_OFFSET_ARCMIN,
    DEFAULT_SURVEY,
    DEFAULT_COLOR_TYPE,
    DEFAULT_PHOT_LEVELS,
    DEFAULT_PHOT_SKIP,
    DEFAULT_CONTOUR_LEVELS,
    DEFAULT_PSF_ARCSEC,
    DEFAULT_BANDWIDTH,
    DEFAULT_Z_PAD,
    DEFAULT_XRAY_FILENAME,
)

# Base path for cluster data directories
_BASE_PATH_ENV_VAR = "XSORTER_CLUSTER_BASE_PATH"
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_DEFAULT_CLUSTER_PATH = _REPO_ROOT.parent / "Clusters"
DEFAULT_BASE_PATH = Path(
    os.environ.get(_BASE_PATH_ENV_VAR, _DEFAULT_CLUSTER_PATH)
).expanduser().resolve()


def _clean_folder_name(identifier: str) -> str:
    """Collapse whitespace into underscores for folder names."""
    return "_".join(str(identifier).strip().split())


@dataclass
class Cluster:
    """Galaxy cluster metadata, analysis parameters, and path management.

    Create directly with known values, or use ``cluster_init()`` to
    populate from config files and external queries.
    """

    # --- Identity ---
    identifier: str
    base_path: Path | str = field(default_factory=lambda: DEFAULT_BASE_PATH)
    name: str | None = None
    ra: float | None = None
    dec: float | None = None
    redshift: float | None = None
    redshift_err: float | None = None
    z_min: float | None = None
    z_max: float | None = None
    richness: float | None = None
    richness_err: float | None = None

    # --- Analysis parameters (from config.yaml or defaults) ---
    fov: float = DEFAULT_FOV_ARCMIN
    fov_full: float = DEFAULT_FOV_FULL_ARCMIN
    ra_offset: float = DEFAULT_RA_OFFSET_ARCMIN
    dec_offset: float = DEFAULT_DEC_OFFSET_ARCMIN
    survey: str = DEFAULT_SURVEY
    color_type: str = DEFAULT_COLOR_TYPE
    psf: float = DEFAULT_PSF_ARCSEC
    bandwidth: float = DEFAULT_BANDWIDTH
    phot_levels: int = DEFAULT_PHOT_LEVELS
    phot_skip: int = DEFAULT_PHOT_SKIP
    contour_levels: tuple[float, float, float] = DEFAULT_CONTOUR_LEVELS

    # --- Cached ---
    _coords: SkyCoord | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.identifier = str(self.identifier).strip()
        self.base_path = Path(self.base_path)

    # ------------------------------------------------------------------
    # Paths (all computed, no _init_paths needed)
    # ------------------------------------------------------------------

    @property
    def cluster_path(self) -> str:
        return str(self.base_path / _clean_folder_name(self.identifier))

    @property
    def redshift_path(self) -> str:
        return os.path.join(self.cluster_path, "Redshifts")

    @property
    def photometry_path(self) -> str:
        return os.path.join(self.cluster_path, "Photometry")

    @property
    def xray_path(self) -> str:
        return os.path.join(self.cluster_path, "Xray")

    @property
    def image_path(self) -> str:
        return os.path.join(self.cluster_path, "Images")

    @property
    def members_path(self) -> str:
        return os.path.join(self.cluster_path, "Members")

    @property
    def tables_path(self) -> str:
        return os.path.join(self.cluster_path, "Tables")

    @property
    def config_path(self) -> str:
        return os.path.join(self.cluster_path, "config.yaml")

    # --- Derived file paths ---

    @property
    def xray_file(self) -> str:
        return os.path.join(self.xray_path, DEFAULT_XRAY_FILENAME)

    @property
    def spec_file(self) -> str:
        return os.path.join(self.redshift_path, "combined_redshifts.csv")

    @property
    def bcg_file(self) -> str:
        return os.path.join(self.cluster_path, "BCGs.csv")

    # ------------------------------------------------------------------
    # Coordinates
    # ------------------------------------------------------------------

    @property
    def coords(self) -> SkyCoord | None:
        """SkyCoord for the cluster center, or None if RA/Dec not set."""
        if self._coords is not None:
            return self._coords
        if self.ra is None or self.dec is None:
            return None
        try:
            self._coords = SkyCoord(ra=self.ra * u.deg, dec=self.dec * u.deg)
        except Exception:
            return None
        return self._coords

    @property
    def coords_str(self) -> str:
        if self.coords is None:
            return "RA: None, Dec: None"
        return f"RA: {self.coords.ra.deg:.6f}, Dec: {self.coords.dec.deg:.6f}"

    # ------------------------------------------------------------------
    # Redshift range
    # ------------------------------------------------------------------

    def resolve_z_range(
        self,
        z_min: float | None = None,
        z_max: float | None = None,
    ) -> tuple[float, float]:
        """Return (z_min, z_max), filling from cluster redshift if needed.

        CLI overrides > stored values > padded from redshift.
        """
        zlo = z_min if z_min is not None else self.z_min
        zhi = z_max if z_max is not None else self.z_max

        if zlo is None or zhi is None:
            if self.redshift is None:
                raise ValueError(
                    "Cannot resolve z-range: no redshift, z_min, or z_max available."
                )
            if zlo is None:
                zlo = self.redshift - DEFAULT_Z_PAD
            if zhi is None:
                zhi = self.redshift + DEFAULT_Z_PAD

        self.z_min = float(zlo)
        self.z_max = float(zhi)
        return self.z_min, self.z_max

    # ------------------------------------------------------------------
    # Photometry path helper
    # ------------------------------------------------------------------

    def get_phot_file(self, survey: str | None = None, color_type: str | None = None) -> str:
        """Return path to the red-sequence member catalog for a survey/color."""
        from cluster_pipeline.io.catalogs import get_redseq_filename
        s = survey or self.survey
        c = color_type or self.color_type
        return get_redseq_filename(self.members_path, s, c)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        ra = f"{self.ra:.6f}" if self.ra is not None else "None"
        dec = f"{self.dec:.6f}" if self.dec is not None else "None"
        return (
            f"\n{'='*40}\n"
            f"  Cluster: {self.identifier}\n"
            f"     Name: {self.name}\n"
            f" Redshift: {self.redshift}\n"
            f"  z range: [{self.z_min}, {self.z_max}]\n"
            f"   Coords: ({ra}, {dec})\n"
            f"{'='*40}\n"
        )

    # ------------------------------------------------------------------
    # Directory creation
    # ------------------------------------------------------------------

    def ensure_directories(self) -> None:
        """Create the cluster directory tree if it doesn't exist."""
        for path in [
            self.cluster_path,
            self.redshift_path,
            self.photometry_path,
            self.xray_path,
            self.image_path,
            self.members_path,
            self.tables_path,
        ]:
            os.makedirs(path, exist_ok=True)
