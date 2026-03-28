#!/usr/bin/env python3
"""
geometry.py

Bisector-Based Sky Region Geometry
---------------------------------------------------------
Computes great-circle bisectors between BCG pairs, partitions the sky
into unique regions via signature matching, and produces bounding arc
segments for visualization.

Functions accept either a list of Subcluster objects, a list of dicts
with a 'center' key, or a plain list of SkyCoord objects.  All geometry
math is identical regardless of input type.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

if TYPE_CHECKING:
    from cluster_pipeline.models.subcluster import Subcluster


# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------

def _get_centers(subclusters) -> list[SkyCoord]:
    """Extract center positions from Subcluster objects or dicts."""
    centers = []
    for sub in subclusters:
        if hasattr(sub, 'region_center'):
            centers.append(sub.region_center)
        elif isinstance(sub, dict) and 'center' in sub:
            centers.append(sub['center'])
        elif isinstance(sub, SkyCoord):
            centers.append(sub)
        else:
            raise TypeError(f"Cannot extract center from {type(sub)}")
    return centers


# ------------------------------------------------------------------
# Bisector computation
# ------------------------------------------------------------------

def get_bisectors(subclusters):
    """
    Compute all pairwise great-circle bisectors between centers (SkyCoord objects).

    Each bisector is the unique great circle that is **perpendicular** (at every point)
    to the arc connecting two BCGs (Brightest Cluster Galaxies) on the celestial sphere.

    For N centers, this function returns N(N-1)/2 bisectors. Each bisector divides the sky
    into two hemispheres: each BCG in the pair is assigned to the hemisphere containing itself.

    Parameters
    ----------
    subclusters : list[Subcluster] | list[dict] | list[SkyCoord]
        Subcluster objects, config dicts with a 'center' key, or bare SkyCoord positions.

    Returns
    -------
    bisectors : list of dict
        Each dictionary contains:
            - 'pair': tuple (i, j)
                Indices of the BCGs defining this bisector.
            - 'pole': ndarray (3,)
                Unit vector normal (the pole) of the bisector great circle.
            - 'mid': SkyCoord
                Spherical midpoint of the two BCGs (for plotting/anchoring).
            - 'c1_vec', 'c2_vec': ndarray (3,)
                Unit vectors for c1 and c2 (for debug/geometry checks).

    Notes
    -----
    - In 3D, any pair of points on the unit sphere defines a great circle arc.
    - The great-circle bisector is the unique great circle perpendicular to this arc.
      Its pole (the normal vector) is proportional to v1 - v2, where v1 and v2 are the 3D unit vectors of the BCGs.
    - The spherical midpoint is (v1 + v2) normalized (not an average in RA/Dec!).
    - This bisector passes through the midpoint and is perpendicular to the vector connecting the two BCGs.
    - For any test point (as a unit vector), the sign of the dot product with the pole tells you
      which side of the bisector the point lies on.

    """

    bisectors = []
    centers = _get_centers(subclusters)

    n = len(centers)
    for i in range(n):
        for j in range(i + 1, n):
            c1, c2 = centers[i], centers[j]
            v1 = c1.cartesian.xyz.value  # 3D unit vector for c1
            v2 = c2.cartesian.xyz.value  # 3D unit vector for c2

            # The bisector's pole (normal): v1 - v2
            pole = v1 - v2
            pole /= np.linalg.norm(pole)

            # Spherical midpoint (not RA/Dec arithmetic mean!)
            mid_vec = v1 + v2
            mid_vec /= np.linalg.norm(mid_vec)
            mid = SkyCoord(x=mid_vec[0], y=mid_vec[1], z=mid_vec[2],
                           representation_type='cartesian', frame='icrs')

            bisectors.append({
                'pair': (i, j),
                'pole': pole,
                'mid': mid,
                'c1_vec': v1,
                'c2_vec': v2,
            })
    return bisectors

def define_bbox(subclusters, margin_frac):
    """
    Compute a bounding box in RA/Dec that encloses all centers, with optional padding.

    Parameters
    ----------
    subclusters : list[Subcluster] | list[dict] | list[SkyCoord]
        Subcluster objects, config dicts with a 'center' key, or bare SkyCoord positions.
    margin_frac : float
        Padding factor. The margin added to each side is margin_frac times
        the largest extent (either RA or Dec) among the centers.

    Returns
    -------
    ra_min : float
        Minimum RA of the bounding box (degrees, with margin).
    ra_max : float
        Maximum RA of the bounding box (degrees, with margin).
    dec_min : float
        Minimum Dec of the bounding box (degrees, with margin).
    dec_max : float
        Maximum Dec of the bounding box (degrees, with margin).

    Notes
    -----
    - If margin_frac = 0, the bounding box tightly wraps the centers.
    - Padding is symmetric and based on the *maximum* span in RA or Dec,
      so the box is always square in angular size.
    """
    centers = _get_centers(subclusters)

    ra_vals = np.array([c.ra.deg for c in centers])
    dec_vals = np.array([c.dec.deg for c in centers])

    ra_range = np.ptp(ra_vals)
    dec_range = np.ptp(dec_vals)
    max_range = max(ra_range, dec_range)
    margin = margin_frac * max_range


    return ra_vals.min() - margin, ra_vals.max() + margin, dec_vals.min() - margin, dec_vals.max() + margin

def get_segments(bisectors, bbox_edges, segment_length=0.2, n_points=1000):
    """
    Generate great-circle arc segments representing the portion of each bisector
    that lies within a rectangular bounding box in RA/Dec.

    For each bisector (between BCGs), we compute a segment of the corresponding
    great circle (centered on the spherical midpoint between the two BCGs,
    with direction set by the bisector's pole) and find where it enters and exits
    the plot region. Only the portion inside the bounding box is kept.

    Parameters
    ----------
    bisectors : list of dict
        Output from get_bisectors. Each dict should contain 'mid' (SkyCoord, anchor),
        and 'pole' (3D unit vector, normal to the bisector).
    bbox_edges : list of 4 tuples
        List of 4 box edges, each as ((ra1, dec1), (ra2, dec2)), defining the plot region.
    segment_length : float, optional
        Angular length of the arc to sample (degrees, symmetric about the midpoint). Default 0.2.
    n_points : int, optional
        Number of sample points per arc (default 1000).

    Returns
    -------
    segments : list of tuple
        Each tuple is (pt1, pt2, pair), where pt1 and pt2 are (ra, dec) in degrees,
        and pair is the tuple of BCG indices defining the bisector.

    Notes
    -----
    - The segment is computed by parameterizing the great circle through the anchor point.
    - Only the endpoints of the portion that falls inside the bounding box are returned.
    - This does not guarantee perfect intersection with the box edges; for large
      segment_length you may want to increase n_points for accuracy.
    """
    segments = []
    for b in bisectors:
        # Anchor (spherical midpoint) and pole (normal vector)
        anchor_vec = b['mid'].cartesian.xyz.value    # 3D unit vector
        pole = b['pole']                             # Normal vector of the bisector
        tangent = np.cross(pole, anchor_vec)         # Direction along the great circle
        tangent /= np.linalg.norm(tangent)

        # Parameterize great circle: anchor * cos(theta) + tangent * sin(theta)
        thetas = np.linspace(-segment_length / 2, segment_length / 2, n_points) * np.pi / 180  # radians
        pts_xyz = (anchor_vec[:, None] * np.cos(thetas) +
                   tangent[:, None] * np.sin(thetas)).T

        pts_gc = SkyCoord(x=pts_xyz[:, 0], y=pts_xyz[:, 1], z=pts_xyz[:, 2],
                          representation_type='cartesian', frame='icrs')
        pts_gc_sph = pts_gc.represent_as('spherical')

        # Extract RA/Dec arrays
        ra = pts_gc_sph.lon.deg
        dec = pts_gc_sph.lat.deg

        # Bounding box edges in RA/Dec
        ra_min, ra_max = min(e[0][0] for e in bbox_edges), max(e[1][0] for e in bbox_edges)
        dec_min, dec_max = min(e[0][1] for e in bbox_edges), max(e[1][1] for e in bbox_edges)

        # Find indices inside the bounding box
        inside = (ra >= ra_min) & (ra <= ra_max) & (dec >= dec_min) & (dec <= dec_max)
        idxs = np.where(inside)[0]
        if len(idxs) >= 2:
            # Use first and last points inside as endpoints
            p1 = (ra[idxs[0]], dec[idxs[0]])
            p2 = (ra[idxs[-1]], dec[idxs[-1]])
            segments.append((p1, p2, b['pair']))
    return segments

def build_bcg_signatures(subclusters, bisectors):
    """
    For each BCG, compute its signature vector with respect to all bisectors.

    Each entry is:
        +1 : BCG is on the positive side of the bisector (as defined by bisector pole)
        -1 : BCG is on the negative side
         0 : BCG not involved in this bisector

    Parameters
    ----------
    subclusters : list[Subcluster] | list[dict] | list[SkyCoord]
        Subcluster objects, config dicts with a 'center' key, or bare SkyCoord positions.
    bisectors : list of dict
        Output from get_bisectors; each dict must have 'pair' (tuple of indices) and 'pole' (3D unit vector).
    Returns
    -------
    signatures : dict
        Dictionary mapping BCG index (int) to list of int (+1, -1, or 0).
        Each entry is a signature vector for that BCG, giving its location with respect
        to every bisector.

    Notes
    -----
    - For each BCG, the sign of the dot product with each bisector pole tells which
      hemisphere (side) of the bisector the BCG lies in.
    - These signature vectors are later used to assign sky regions and classify membership.

    """
    centers = _get_centers(subclusters)
    signatures = {}
    for k, bcg in enumerate(centers):
        v = bcg.cartesian.xyz.value  # 3D unit vector for this BCG
        sig = []
        for b in bisectors:
            # Only care about bisectors this BCG is part of
            if k in b['pair']:
                pole = b['pole']
                sign = np.sign(np.dot(v, pole))  # +1, -1, or 0
                sig.append(sign)
            else:
                sig.append(0)
        signatures[k] = sig
    return signatures

def build_point_signature(coord, bisectors, exclude_pair=None):
    """
    Compute the signature vector (+1, -1, or 0) for a point with respect to all bisectors.

    If exclude_pair is provided, the corresponding bisector will have signature 0.
    This is used for segments that lie *on* their own bisector.

    Parameters
    ----------
    coord : SkyCoord
        The point to evaluate (e.g., midpoint, galaxy position).
    bisectors : list of dict
        Output from get_bisectors.
    exclude_pair : tuple or None
        If set, the bisector with this pair gets signature 0.

    Returns
    -------
    sig : list of int
        Signature (+1, -1, or 0) for each bisector.
    """
    v = coord.cartesian.xyz.value
    sig = []
    for b in bisectors:
        if exclude_pair is not None and set(b['pair']) == set(exclude_pair):
            sig.append(0)
        else:
            pole = b['pole']
            sign = np.sign(np.dot(v, pole))
            sig.append(sign)
    return sig

def classify_segments(segments, bisectors, bcg_signatures, verbose=False):
    """
    Assigns each bisector segment (line between intersection points) to one or more BCG-defined regions
    using signature vectors, so each region is bounded by the relevant segments for that BCG.

    This routine partitions the sky into unique regions for each BCG, bounded by bisectors and
    defined by which "side" of each bisector the region's midpoint falls on, relative to the BCG's signature.

    Parameters
    ----------
    segments : list of tuple
        Output from get_segments. Each entry is (p1, p2, pair), where p1/p2 are (RA, Dec) segment endpoints,
        and 'pair' is the (i, j) index of the BCGs that define the bisector.
    bisectors : list of dict
        Output from get_bisectors. Each must contain 'pair' and 'pole'.
    bcg_signatures : dict
        Output from build_bisector_signatures. Keys are BCG indices; values are signature lists (+1, -1, 0).
    verbose : bool, optional
        If True, print detailed classification info.

    Returns
    -------
    regions : dict
        Dictionary mapping BCG index to list of line segments [(p1, p2), ...] bounding the region.

    Notes
    -----
    - Each segment is classified by computing the signature of its midpoint with respect to all bisectors.
    - The segment is assigned to each BCG whose nonzero signature elements match those of the segment.
    """


    regions = defaultdict(list)          # BCG index -> list of segment tuples
    bisector_points = defaultdict(list)  # bisector pair -> list of segment endpoints/intersections

    # Collect all segment endpoints for each bisector
    for p1, p2, pair in segments:
        bisector_points[pair].extend([p1, p2])

    # Find intersections between segments and add to bisector points
    for i, (a1, a2, pair1) in enumerate(segments):
        for j, (b1, b2, pair2) in enumerate(segments[i+1:], i+1):
            inter = segment_segment_intersection(a1, a2, b1, b2)
            if inter:
                bisector_points[pair1].append(inter)
                bisector_points[pair2].append(inter)

    # For each bisector, walk through ordered endpoints, form segments, and classify
    for pair, pts in bisector_points.items():
        pts = sorted(pts)
        for i in range(len(pts) - 1):
            (x1, y1), (x2, y2) = pts[i], pts[i+1]
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            segment_midpoint = SkyCoord(mx * u.deg, my * u.deg, frame='icrs')

            # Build signature for midpoint: for its own bisector, set to 0
            segment_sig = build_point_signature(segment_midpoint, bisectors, exclude_pair=pair)
            if verbose:
                print(f"Midpoint {mx:.2f},{my:.2f}: signature {segment_sig}")

            # Compare midpoint signature to each BCG's signature
            for bcg_idx, bcg_sig in bcg_signatures.items():
                if bcg_idx not in pair: # Only consider segments from bisectors involving this BCG
                    continue
                segment_sig = np.array(segment_sig)
                bcg_sig = np.array(bcg_sig)
                # Mask: both nonzero (both BCG and point are defined with respect to this bisector)
                if verbose:
                    print(f"  Comparing to BCG {bcg_idx} with signature {bcg_sig} to segment signature {segment_sig}")
                mask = (bcg_sig != 0) & (segment_sig != 0)
                if np.all(segment_sig[mask] == bcg_sig[mask]):
                    regions[bcg_idx].append(((x1, y1), (x2, y2)))

    return regions

def segment_segment_intersection(p1_start, p1_end, p2_start, p2_end):
    """
    Compute the intersection of two 2D line segments (if any).

    Parameters
    ----------
    p1_start, p1_end : tuple of float
        Endpoints of the first segment.
    p2_start, p2_end : tuple of float
        Endpoints of the second segment.

    Returns
    -------
    (x, y) : tuple of float
        Intersection point if the segments cross, else None.

    Notes
    -----
    - Handles parallel segments gracefully (returns None).
    - Segments that just touch at endpoints are considered intersecting.
    """
    return line_intersection_with_bounds(
        *segment_to_line(p1_start, p1_end),
        *segment_to_line(p2_start, p2_end),
        t_bounds=(0, 1), s_bounds=(0, 1)
    )

def line_intersection_with_bounds(p1, d1, p2, d2, t_bounds=None, s_bounds=(0, 1)):
    """
    Compute the intersection (if any) of two 2D lines, each in parametric form:
        Line 1: p1 + t * d1
        Line 2: p2 + s * d2

    Bounds on t and s allow this to handle infinite lines, rays, or segments.

    Parameters
    ----------
    p1 : tuple of float
        Origin of line 1.
    d1 : tuple of float
        Direction vector for line 1.
    p2 : tuple of float
        Origin of line 2.
    d2 : tuple of float
        Direction vector for line 2.
    t_bounds : tuple of float or None, optional
        Allowed interval for t (default: None, infinite line).
    s_bounds : tuple of float, optional
        Allowed interval for s (default: (0,1), segment only).

    Returns
    -------
    (x, y) : tuple of float
        Intersection point if within bounds, else None.

    Notes
    -----
    - If t_bounds is (0,1), both lines are segments.
    - If t_bounds is None, line 1 is treated as infinite.
    - Returns None if lines are parallel or intersection is outside bounds.
    """
    A = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]])
    b = np.array([p2[0] - p1[0], p2[1] - p1[1]])
    try:
        t, s = np.linalg.solve(A, b)
        t_valid = True if t_bounds is None else (t_bounds[0] <= t <= t_bounds[1])
        s_valid = s_bounds[0] <= s <= s_bounds[1]
        if t_valid and s_valid:
            return (p1[0] + t * d1[0], p1[1] + t * d1[1])
    except np.linalg.LinAlgError:
        # Lines are parallel or coincident: no unique intersection
        pass
    return None

def segment_to_line(p_start, p_end):
    """
    Convert a 2D line segment (defined by endpoints) to parametric form.

    Parameters
    ----------
    p_start : tuple of float
        Starting point (x, y) of the segment.
    p_end : tuple of float
        Ending point (x, y) of the segment.

    Returns
    -------
    p : tuple of float
        Origin point (x, y) for the parametric line.
    d : tuple of float
        Direction vector (dx, dy) from start to end.
    """
    return p_start, (p_end[0] - p_start[0], p_end[1] - p_start[1])
