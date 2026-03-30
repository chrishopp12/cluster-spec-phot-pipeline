"""Bisector-based sky region for a subcluster.

A Region encapsulates the result of the bisector geometry computation
for one subcluster.  It stores the signature vector (which side of
each bisector this region is on) and the bounding arc segments that
define the region's boundary on the sky.

The actual geometry computations live in ``subclusters/geometry.py``.
This class is a data container for the results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Region:
    """Sky region defined by bisector geometry for one subcluster.

    Attributes
    ----------
    bcg_index : int
        Index into the subcluster list that this region belongs to.
    signature : list[int]
        Sign vector: +1, -1, or 0 for each bisector.  Two points with
        matching signatures belong to the same region.  A 0 means this
        region's BCG is not involved in that bisector pair.
    segments : list
        Bounding arc segments as (p1, p2) coordinate pairs.  These are
        the portions of bisector great-circles that form this region's
        boundary within the field of view.
    bisectors : list[dict]
        Reference to the full list of bisector dicts (shared across all
        regions).  Each dict has keys: ``pole``, ``mid``, ``pair``.
    """

    bcg_index: int
    signature: list[int] = field(default_factory=list)
    segments: list[Any] = field(default_factory=list)
    bisectors: list[dict] = field(default_factory=list)

    def matches(self, point_signature: list[int]) -> bool:
        """Check if a point's signature matches this region.

        A point matches if, for every bisector where both the region
        and the point have nonzero signs, the signs agree.

        Parameters
        ----------
        point_signature : list[int]
            Signature of the test point (same length as self.signature).

        Returns
        -------
        bool
            True if the point belongs to this region.
        """
        for s_region, s_point in zip(self.signature, point_signature):
            if s_region != 0 and s_point != 0 and s_region != s_point:
                return False
        return True

    def __repr__(self) -> str:
        sig_str = "".join("+" if s > 0 else "-" if s < 0 else "0" for s in self.signature)
        return (
            f"Region(bcg_index={self.bcg_index}, "
            f"signature=[{sig_str}], "
            f"n_segments={len(self.segments)})"
        )
