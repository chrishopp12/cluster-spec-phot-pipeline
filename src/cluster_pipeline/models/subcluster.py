"""Subcluster data class.

A Subcluster captures the structural relationships between BCGs, the
spatial region assigned by bisector geometry, and the member galaxies
(stored as DataFrames for efficient bulk operations).

Key design decision: BCGs can play different roles in a subcluster.

  - **primary_bcg**: Defines the subcluster's identity — its color,
    label, and statistics come from this BCG.
  - **border_bcg**: Defines the bisector boundary.  Often the same as
    primary_bcg, but differs when subclusters are grouped (e.g., BCGs
    1+5 form one subcluster, but BCG 5 is closer to the boundary with
    BCG 2, so the bisector uses BCG 5's position while the subcluster's
    identity comes from BCG 1).
  - **member_bcgs**: All BCGs that belong to this subcluster.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from astropy.coordinates import SkyCoord

from cluster_pipeline.models.bcg import BCG
from cluster_pipeline.models.region import Region


@dataclass
class Subcluster:
    """A subcluster within a galaxy cluster.

    Attributes
    ----------
    bcg_id : int
        Identifier of the primary BCG (defines this subcluster).
    primary_bcg : BCG
        The BCG that defines this subcluster's identity (color, label, stats).
    border_bcg : BCG or None
        The BCG whose position defines the bisector boundary.  If None,
        defaults to primary_bcg.
    member_bcgs : list[BCG]
        All BCGs belonging to this subcluster (includes primary and border).

    label : str
        Display label (e.g., "1", "Main", "A").
    color : str
        Matplotlib color for plots.
    radius_mpc : float
        Search radius in Mpc.
    z_range : tuple[float, float]
        Redshift range (z_min, z_max) for member selection.

    group_id : str or None
        Group identifier when multiple subclusters are combined.
    group_members : tuple[int, ...]
        BCG IDs of all subclusters in this group.
    is_dominant : bool
        Whether this is the dominant subcluster in its group
        (i.e., the one whose label/color/stats represent the group).
    group_label : str or None
        Override label for the combined group.
    group_color : str or None
        Override color for the combined group.

    spec_members : DataFrame or None
        Spectroscopic member galaxies (populated during analysis).
    phot_members : DataFrame or None
        Photometric member galaxies (populated during analysis).
    region : Region or None
        Bisector-defined sky region (populated during analysis).
    """

    # Identity
    bcg_id: int
    primary_bcg: BCG
    border_bcg: BCG | None = None
    member_bcgs: list[BCG] = field(default_factory=list)

    # Display / analysis parameters
    label: str = ""
    color: str = "white"
    radius_mpc: float = 2.5
    z_range: tuple[float, float] = (0.0, 1.0)

    # Group info (for combined subclusters)
    group_id: str | None = None
    group_members: tuple[int, ...] = ()
    is_dominant: bool = True
    group_label: str | None = None
    group_color: str | None = None

    # Populated during analysis (not serialized to config)
    spec_members: pd.DataFrame | None = field(default=None, repr=False)
    phot_members: pd.DataFrame | None = field(default=None, repr=False)
    region: Region | None = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def region_center(self) -> SkyCoord:
        """Position used for bisector computation.

        Uses the border BCG if set, otherwise the primary BCG.
        """
        bcg = self.border_bcg if self.border_bcg is not None else self.primary_bcg
        return bcg.coord

    @property
    def display_center(self) -> SkyCoord:
        """Position used for labels and markers on plots."""
        return self.primary_bcg.coord

    @property
    def display_label(self) -> str:
        """Label for display — group label if grouped, else own label."""
        if self.group_label is not None:
            return self.group_label
        return self.label

    @property
    def display_color(self) -> str:
        """Color for display — group color if grouped, else own color."""
        if self.group_color is not None:
            return self.group_color
        return self.color

    @property
    def n_spec(self) -> int:
        """Number of spectroscopic members, or 0 if not yet assigned."""
        return len(self.spec_members) if self.spec_members is not None else 0

    @property
    def n_phot(self) -> int:
        """Number of photometric members, or 0 if not yet assigned."""
        return len(self.phot_members) if self.phot_members is not None else 0

    def __repr__(self) -> str:
        border_str = f", border={self.border_bcg.bcg_id}" if self.border_bcg else ""
        group_str = f", group={self.group_members}" if self.group_members else ""
        return (
            f"Subcluster(bcg_id={self.bcg_id}, label='{self.label}', "
            f"color='{self.color}'{border_str}{group_str}, "
            f"n_spec={self.n_spec}, n_phot={self.n_phot})"
        )

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        cfg: dict[str, Any],
        bcgs: dict[int, BCG],
    ) -> Subcluster:
        """Create a Subcluster from a config dict and a BCG lookup.

        Parameters
        ----------
        cfg : dict
            One entry from the ``subclusters`` list in config.yaml.
            Required keys: ``bcg_id``.
            Optional: ``label``, ``color``, ``radius_mpc``, ``z_range``,
            ``border_bcg_id``, ``group_id``, ``group_members``, etc.
        bcgs : dict[int, BCG]
            Lookup of all BCGs by bcg_id.
        """
        bcg_id = cfg["bcg_id"]
        primary = bcgs[bcg_id]

        border_id = cfg.get("border_bcg_id")
        border = bcgs.get(border_id) if border_id is not None else None

        # Member BCGs: listed explicitly, or just the primary
        member_ids = cfg.get("member_bcg_ids", [bcg_id])
        members = [bcgs[mid] for mid in member_ids if mid in bcgs]

        z_range = cfg.get("z_range", (0.0, 1.0))
        if isinstance(z_range, list):
            z_range = tuple(z_range)

        group_members = cfg.get("group_members", ())
        if isinstance(group_members, list):
            group_members = tuple(group_members)

        return cls(
            bcg_id=bcg_id,
            primary_bcg=primary,
            border_bcg=border,
            member_bcgs=members,
            label=cfg.get("label", str(bcg_id)),
            color=cfg.get("color", "white"),
            radius_mpc=cfg.get("radius_mpc", 2.5),
            z_range=z_range,
            group_id=cfg.get("group_id"),
            group_members=group_members,
            is_dominant=cfg.get("is_dominant", True),
            group_label=cfg.get("group_label"),
            group_color=cfg.get("group_color"),
        )

    def to_config(self) -> dict[str, Any]:
        """Serialize to a dict suitable for YAML config storage."""
        d: dict[str, Any] = {
            "bcg_id": self.bcg_id,
            "label": self.label,
            "color": self.color,
            "radius_mpc": self.radius_mpc,
            "z_range": list(self.z_range),
        }
        if self.border_bcg is not None and self.border_bcg.bcg_id != self.bcg_id:
            d["border_bcg_id"] = self.border_bcg.bcg_id
        if len(self.member_bcgs) > 1:
            d["member_bcg_ids"] = [b.bcg_id for b in self.member_bcgs]
        if self.group_id is not None:
            d["group_id"] = self.group_id
        if self.group_members:
            d["group_members"] = list(self.group_members)
        if not self.is_dominant:
            d["is_dominant"] = False
        if self.group_label is not None:
            d["group_label"] = self.group_label
        if self.group_color is not None:
            d["group_color"] = self.group_color
        return d
