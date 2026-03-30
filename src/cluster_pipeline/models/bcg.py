"""Brightest Cluster Galaxy (BCG) data class.

Lightweight container for BCG candidates. There are typically 5-10 per
cluster, so object overhead is negligible compared to the thousands of
galaxies stored in DataFrames.
"""

from __future__ import annotations

from dataclasses import dataclass

from astropy.coordinates import SkyCoord
import astropy.units as u


@dataclass
class BCG:
    """A single BCG candidate.

    Attributes
    ----------
    bcg_id : int
        Identifier (typically 1-based priority from redMaPPer).
    ra : float
        Right ascension in degrees.
    dec : float
        Declination in degrees.
    z : float or None
        Spectroscopic redshift, if known.
    sigma_z : float or None
        Redshift uncertainty.
    rank : int or None
        redMaPPer centering rank (1 = most likely center).
    probability : float or None
        redMaPPer centering probability (PCen).
    label : str
        Display label for plots (e.g., "1", "A", "Main").
    """

    bcg_id: int
    ra: float
    dec: float
    z: float | None = None
    sigma_z: float | None = None
    rank: int | None = None
    probability: float | None = None
    label: str = ""

    @property
    def coord(self) -> SkyCoord:
        """SkyCoord for this BCG's position."""
        return SkyCoord(ra=self.ra * u.deg, dec=self.dec * u.deg, frame="icrs")

    @property
    def has_redshift(self) -> bool:
        """True if a spectroscopic redshift is available."""
        return self.z is not None

    def __repr__(self) -> str:
        z_str = f"{self.z:.4f}" if self.z is not None else "None"
        return (
            f"BCG(id={self.bcg_id}, label='{self.label}', "
            f"ra={self.ra:.5f}, dec={self.dec:.5f}, z={z_str})"
        )

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, bcg_id: int, cfg: dict) -> BCG:
        """Create a BCG from a config dict (as stored in YAML).

        Parameters
        ----------
        bcg_id : int
            The BCG identifier.
        cfg : dict
            Keys: ra, dec, and optionally z, sigma_z, rank, prob, label.
        """
        return cls(
            bcg_id=bcg_id,
            ra=cfg["ra"],
            dec=cfg["dec"],
            z=cfg.get("z"),
            sigma_z=cfg.get("sigma_z"),
            rank=cfg.get("rank"),
            probability=cfg.get("prob", cfg.get("probability")),
            label=cfg.get("label", str(bcg_id)),
        )

    @classmethod
    def from_dataframe_row(cls, row, bcg_id: int | None = None) -> BCG:
        """Create a BCG from a pandas Series (one row of BCGs.csv).

        Parameters
        ----------
        row : pd.Series
            Must have 'RA' and 'Dec'. Optional: 'z', 'sigma_z',
            'BCG_priority', 'BCG_probability'.
        bcg_id : int, optional
            Override for bcg_id. If None, uses row['BCG_priority'].
        """
        import pandas as pd

        bid = bcg_id if bcg_id is not None else int(row.get("BCG_priority", 0))

        def _safe(val):
            if val is None or (isinstance(val, float) and pd.isna(val)):
                return None
            return val

        return cls(
            bcg_id=bid,
            ra=float(row["RA"]),
            dec=float(row["Dec"]),
            z=_safe(row.get("z")),
            sigma_z=_safe(row.get("sigma_z")),
            rank=_safe(row.get("BCG_priority")),
            probability=_safe(row.get("BCG_probability")),
            label=str(row.get("label", bid)),
        )

    def to_config(self) -> dict:
        """Serialize to a dict suitable for YAML config storage."""
        d: dict = {"ra": self.ra, "dec": self.dec}
        if self.z is not None:
            d["z"] = self.z
        if self.sigma_z is not None:
            d["sigma_z"] = self.sigma_z
        if self.rank is not None:
            d["rank"] = self.rank
        if self.probability is not None:
            d["prob"] = self.probability
        if self.label and self.label != str(self.bcg_id):
            d["label"] = self.label
        return d
