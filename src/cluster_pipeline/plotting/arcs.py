#!/usr/bin/env python3
"""
arcs.py

Great-Circle Arc Drawing and Region Fill for Bisector Visualization
---------------------------------------------------------

Handles the geometry of drawing bisector boundary arcs on WCS axes and
filling subcluster regions with translucent color.  Arcs are computed via
spherical linear interpolation (slerp) and optionally clipped to a
circular field of view around each subcluster center.

Key functions:
  - draw_great_circle_segment()              Slerp-interpolated arc between
                                              two SkyCoords, with optional
                                              radius clipping
  - plot_bcg_region_arcs()                   Draw dashed arcs outlining each
                                              BCG's assigned region based on
                                              bisector signatures
  - find_segment_circle_crossings()          Locate where a great-circle
                                              segment crosses a circular
                                              boundary on the sky
  - add_region_fill_clipped_to_signature()   Fill the convex hull of a BCG's
                                              region polygon, including only
                                              bounding-box corners that match
                                              the bisector signature

Requirements:
  - astropy, scipy, matplotlib, numpy

Notes:
  - draw_great_circle_segment() always draws the shorter (<= 180 deg) arc.
  - Circle crossings use linear interpolation in separation space, which is
    accurate for the fine step sizes used in practice.
  - Region filling relies on ConvexHull; non-convex regions will be
    approximated by their convex envelope.
"""

from __future__ import annotations

import numpy as np
from astropy.coordinates import SkyCoord, CartesianRepresentation, Angle
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull

from cluster_pipeline.subclusters.geometry import (
    build_point_signature,
    segment_segment_intersection,
)


def find_segment_circle_crossings(
    segment_points,
    center,
    radius_arcmin,
    tolerance=1e-3,
):
    """
    Find intersection points between a great-circle segment and a circular boundary on the sky.

    For each pair of consecutive points in the segment, interpolates where the segment crosses
    the circle centered at `center` with radius `radius_arcmin`. Returns the intersection points
    as (angle, ra, dec) tuples, where angle is the position angle (deg) from the center.

    Parameters
    ----------
    segment_points : astropy.coordinates.SkyCoord
        Array of points tracing the segment (must be in same frame as center).
    center : astropy.coordinates.SkyCoord
        The center of the circle to check for crossings.
    radius_arcmin : float
        Radius of the circle in arcminutes.
    tolerance : float, optional, must uncomment grazing
        Minimum arcminute separation from the boundary to consider as a crossing (default: 1e-3).
        Minimum arcminute separation from the boundary to consider as a crossing (default: 1e-3).

    Returns
    -------
    intersections : list of tuple
        List of (angle_deg, ra_deg, dec_deg) for each intersection point.

    Notes
    -----
    - Interpolates linearly between points in separation space. This is a good approximation
      for small step sizes along the segment.
    - If a segment endpoint lies exactly on the circle, it is *not* included unless it is a true crossing.
    - Angle is measured in degrees east of north from the center (same as SkyCoord.position_angle).
    """
    sep = segment_points.separation(center).arcmin
    intersections = []

    for i in range(len(segment_points) - 1):
        s1, s2 = segment_points[i], segment_points[i + 1]
        sep1, sep2 = sep[i], sep[i + 1]

        # Only consider pairs that cross the boundary (i.e., sep1 and sep2 straddle radius)
        if (sep1 - radius_arcmin) * (sep2 - radius_arcmin) < 0:
            # Linear interpolation in separation space
            t = (radius_arcmin - sep1) / (sep2 - sep1)
            ra_interp = s1.ra.deg + t * (s2.ra.deg - s1.ra.deg)
            dec_interp = s1.dec.deg + t * (s2.dec.deg - s1.dec.deg)
            p = SkyCoord(ra=ra_interp * u.deg, dec=dec_interp * u.deg, frame=s1.frame)
            angle = center.position_angle(p).to_value(u.deg)
            intersections.append((angle, ra_interp, dec_interp))

        # # Optionally: check for "grazing" (endpoint nearly on circle) Be careful, this will find a grazing point
        # # before the actual crossing
        # elif abs(sep1 - radius_arcmin) < tolerance:
        #     angle = center.position_angle(s1).to_value(u.deg)
        #     intersections.append((angle, s1.ra.deg, s1.dec.deg))

    return intersections

def draw_great_circle_segment(
    ax,
    coord1,
    coord2,
    n_points=1000,
    arc_linestyle='--',
    arc_linewidth=1.5,
    arc_alpha=1.0,
    arc_color="magenta", #So you know you didn't set it right
    radius=None,
    center=None,
    transform=None,
    plot_segment=True,
    return_points=False,
    **kwargs
):
    """
    Draw a great-circle segment (arc) between two points on the celestial sphere.

    Interpolates points along the shortest path between `coord1` and `coord2` using
    spherical linear interpolation (slerp) in 3D Cartesian space. Optionally, restricts output to
    points within a given angular radius of a specified center.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matplotlib axis to plot on (must be WCS-aware if using transform).
    coord1, coord2 : astropy.coordinates.SkyCoord
        Endpoints of the segment.
    n_points : int, optional
        Number of points to interpolate along the arc (default: 100).
    radius : float or astropy.units.Quantity, optional
        If given, restricts output to points within this radius (arcmin) of `center`.
    center : astropy.coordinates.SkyCoord, optional
        Center for radius clipping (required if `radius` is given).
    transform : optional
        Matplotlib coordinate transform (e.g., ax.get_transform('icrs')).
    plot_segment : bool, optional
        If True (default), plots the segment on the axis.
    return_points : bool, optional
        If True, returns the SkyCoord array of interpolated points.
    **kwargs
        Additional keyword arguments passed to `ax.plot()`.

    Returns
    -------
    interp_coords : astropy.coordinates.SkyCoord or None
        Interpolated points along the segment (if `return_points` is True), otherwise None.

    Notes
    -----
    - The arc is always the *shorter* segment between coord1 and coord2 (<=180 deg).
    - If coord1 and coord2 are identical, returns None.
    - If radius/center are given, only points within `radius` of `center` are kept.
    - Plots in (ra, dec) degrees, assumes input coordinates are in degrees.
    """
    from cluster_pipeline.utils import pop_prefixed_kwargs

    arc_kwargs = pop_prefixed_kwargs(kwargs, 'arc')
    linestyle = arc_kwargs.get('linestyle', arc_linestyle)
    linewidth = arc_kwargs.get('linewidth', arc_linewidth)
    alpha = arc_kwargs.get('alpha', arc_alpha)

    # Convert to Cartesian
    xyz1 = coord1.cartesian.xyz.value
    xyz2 = coord2.cartesian.xyz.value

    # Normalize to unit vectors
    xyz1 /= np.linalg.norm(xyz1)
    xyz2 /= np.linalg.norm(xyz2)

    # Compute the angle between the vectors
    omega = np.arccos(np.clip(np.dot(xyz1, xyz2), -1.0, 1.0))
    if np.isclose(omega, 0.0):
        # Points are identical or extremely close; nothing to draw/return
        return None if return_points else None

    # Interpolate with spherical linear interpolation (slerp)
    t = np.linspace(0, 1, n_points)
    sin_omega = np.sin(omega)
    interp_xyz = (np.sin((1 - t) * omega)[:, None] * xyz1 + np.sin(t * omega)[:, None] * xyz2) / sin_omega

    # Convert back to SkyCoord
    interp_cartesian = CartesianRepresentation(interp_xyz.T * u.one)
    interp_coords = SkyCoord(interp_cartesian, frame=coord1.frame)


    if radius is not None:
        if center is None:
            raise ValueError("If 'radius' is given, 'center' must also be specified.")
        if not isinstance(radius, u.Quantity):
            radius = radius * u.arcmin
        sep = center.separation(interp_coords)
        mask = sep <= radius
        interp_coords = interp_coords[mask]

    # Plot if requested
    if plot_segment and len(interp_coords) > 0:
        ax.plot(interp_coords.ra.deg, interp_coords.dec.deg,
                transform=transform, linestyle=linestyle, linewidth=linewidth, alpha=alpha, color=arc_color, **arc_kwargs)

    if return_points:
        return interp_coords

def plot_bcg_region_arcs(
    ax,
    subcluster_configs,
    bisectors,
    bcg_signatures,
    transform=None,
    n_points=1000,
    arc_linestyle='--',
    arc_linewidth=1.5,
    arc_alpha=1.0,
    verbose=False,
    combined_indices=None,
    **kwargs
):
    """
    Plot dashed arcs outlining each BCG's assigned region, based on bisector signatures.
    The arcs are clipped to match only the visible boundary of the region.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis on which to plot.
    subcluster_configs : list of dict
        Subcluster configuration dictionaries.
    bisectors : list of dict
        Each dict should have 'pair', 'mid', and 'pole'.
    bcg_signatures : dict
        Maps BCG index to list of bisector signatures (+1, -1, 0).
    colors : list, optional
        List of colors for each region (defaults to tab10 colormap).
    transform : optional
        Coordinate transform (e.g., ax.get_transform('icrs')).
    n_points : int, optional
        Number of interpolation points for arc segments (default: 1000).
    arc_linestyle : str, optional
        Linestyle for region arcs (default: '--').
    arc_linewidth : float, optional
        Line width for region arcs (default: 1.5).
    arc_alpha : float, optional
        Alpha for region arcs (default: 1.0).
    verbose : bool, optional
        If True, prints extra debug output and diagnostic points.
    skip_pair : tuple, optional
        Pair of BCG indices to skip when plotting arcs for combined regions.
    **kwargs : dict
        Additional keyword arguments passed to plotting functions.

    Returns
    -------
    None

    Notes
    -----
    - Each BCG's region is outlined by arcs, clipped to intersections with other region boundaries.
    - The correct assignment of each arc segment is determined by the bisector signature logic.
    - This function does not return any data; it only plots.
    """
    from collections import defaultdict
    # assign_subcluster_regions lives in the old module until fully extracted
    from cluster_pipeline.subclusters.assignment import assign_subcluster_regions

    def _seg_key_exact(seg):
        """Order-independent exact key for a segment ((ra1,dec1),(ra2,dec2))."""
        return frozenset((seg[0], seg[1]))


    if combined_indices is None:
        colors = [sub.color if hasattr(sub, 'color') else sub.get('color') for sub in subcluster_configs]
    else:
        colors = [sub.display_color if hasattr(sub, 'display_color') else sub.get('group_color') for sub in subcluster_configs]

    centers = [sub.region_center if hasattr(sub, 'region_center') else sub['center'] for sub in subcluster_configs]
    radii = [sub.radius_mpc if hasattr(sub, 'radius_mpc') else sub['radius'] for sub in subcluster_configs]

    # Build region segment list for each BCG (segment = pair of (RA, Dec))
    bcg_regions = assign_subcluster_regions(subcluster_configs)
    arc_points_dict = {}
    bcg_intersections = defaultdict(list)

    # Working copy we'll use for plotting (original bcg_regions remains intact)
    cleaned_seglists = {i: list(seglist) for i, seglist in bcg_regions.items()}

    if combined_indices:
        for a, b in combined_indices:

            A = cleaned_seglists[a]
            B = cleaned_seglists[b]

            keys_A = {_seg_key_exact(s) for s in A}
            keys_B = {_seg_key_exact(s) for s in B}
            shared = keys_A & keys_B  # segments common to both regions

            if verbose:
                print(f"Removing {len(shared)} shared segment(s) between regions {a} and {b}")

            # drop the shared boundary from BOTH sides
            cleaned_seglists[a] = [s for s in A if _seg_key_exact(s) not in shared]
            cleaned_seglists[b] = [s for s in B if _seg_key_exact(s) not in shared]




    for i, seglist in bcg_regions.items():
        seglist_to_plot = cleaned_seglists.get(i, seglist)

        if verbose:
            print(f"Processing BCG {i} with {len(seglist_to_plot)} segments.")

        center = centers[i]
        radius = radii[i]
        for (x1, y1), (x2, y2) in seglist_to_plot:
            coord1 = SkyCoord(ra=x1 * u.deg, dec=y1 * u.deg)
            coord2 = SkyCoord(ra=x2 * u.deg, dec=y2 * u.deg)
            if coord1.separation(coord2) < 1e-2 * u.arcsec:
                if verbose:
                    print(f"Skipping degenerate segment for BCG {i}")
                continue


            # Plot segments within each BCG region
            draw_great_circle_segment(
                ax, coord1, coord2, n_points=n_points,
                arc_color=colors[i], arc_linestyle=arc_linestyle, arc_linewidth=arc_linewidth, arc_alpha=arc_alpha,
                transform=transform if transform else ax.get_transform('icrs'),
                radius=radius, center=center, plot_segment=True, return_points=False, **kwargs
            )

        all_points = []
        for (x1, y1), (x2, y2) in seglist:
            coord1 = SkyCoord(ra=x1 * u.deg, dec=y1 * u.deg)
            coord2 = SkyCoord(ra=x2 * u.deg, dec=y2 * u.deg)
            if coord1.separation(coord2) < 1e-2 * u.arcsec:
                if verbose:
                    print(f"Skipping degenerate segment for BCG {i}")
                continue

            # Get segments
            segment_points = draw_great_circle_segment(
                ax, coord1, coord2, n_points=n_points,
                arc_color=colors[i], arc_linestyle=arc_linestyle, arc_linewidth=arc_linewidth, arc_alpha=arc_alpha,
                transform=transform if transform else ax.get_transform('icrs'),
                plot_segment=False, return_points=True, **kwargs
            )
            if verbose:
                print(f"Segment {i}: {len(segment_points)} points")
            all_points.append(segment_points)
            arc_points_dict[i] = all_points

            # Find intersection points
            for segment in all_points:
                intersections = find_segment_circle_crossings(segment, center, radius)
                if verbose:
                    for angle, ra, dec in intersections:
                        ax.plot(ra, dec, marker='o', color='red', markersize=5,
                                transform=transform if transform else ax.get_transform('icrs'))
                bcg_intersections[i].extend(intersections)



    for i, center in enumerate(centers):
        sig_i = np.array(bcg_signatures[i])
        intersections = bcg_intersections[i]
        if len(intersections) < 2:
            continue

        # Sort by angle
        intersections.sort()
        angles, ra_vals, dec_vals = zip(*intersections)

        # Loop through arc segments
        for k in range(len(angles)):
            ang1, ang2 = angles[k], angles[(k+1)%len(angles)]

            # Determine midpoint for signature test
            mid_angle = (ang1 + ang2) / 2
            if ang2 < ang1:
                mid_angle = (ang1 + (ang2 + 360)) / 2 % 360

            test_point = center.directional_offset_by(Angle(mid_angle, unit=u.deg), Angle(radii[i], unit=u.arcmin))
            if verbose:
                ax.scatter(test_point.ra.deg, test_point.dec.deg, marker='x', color='red', s=20,
                        transform=transform if transform else ax.get_transform('icrs'))

            # Compute signature of test point
            test_sig = np.array(build_point_signature(test_point, bisectors))
            mask = sig_i != 0
            if verbose:
                print(f"Masked test sig: {test_sig[mask]}, expected sig: {sig_i[mask]}")
            if np.all(test_sig[mask] == sig_i[mask]):
                if verbose:
                    print(f"Drawing arc for BCG {i} from {ang1:.1f} to {ang2:.1f}")

                theta = np.linspace(ang1, ang2 + (360 if ang2 < ang1 else 0), 1000) % 360
                angle_array = Angle(theta, unit=u.deg)
                radius_array = Angle(radii[i], unit=u.arcmin)

                # Repeat the center to match the number of angles
                center_array = SkyCoord(
                    ra=np.full_like(angle_array.value, center.ra.deg) * u.deg,
                    dec=np.full_like(angle_array.value, center.dec.deg) * u.deg,
                    frame=center.frame.name
                )

                # Now safely call directional_offset_by for each angle
                arc = center_array.directional_offset_by(angle_array, radius_array)

                # Plot the arc
                ax.plot(arc.ra.deg, arc.dec.deg,
                        linestyle=arc_linestyle, color=colors[i], linewidth=arc_linewidth, alpha=arc_alpha,
                        transform=transform if transform else ax.get_transform('world'), **kwargs)

            else:
                if verbose:
                    print(f"Rejected arc for BCG {i} from {ang1:.1f} to {ang2:.1f} — test sig = {test_sig}, expected = {sig_i}")

def add_region_fill_clipped_to_signature(ax, region_segments, color, bcg_sig, bisectors, alpha=1.0):
    """
    Fill a polygonal sky region associated with a given BCG's bisector signature.

    This function gathers all unique segment endpoints belonging to a BCG's assigned region,
    includes bounding box corners that are inside the region (by signature logic),
    and fills the convex hull of all valid points as a patch on the plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis on which to plot the filled region.
    region_segments : list of tuple
        List of segment endpoints [(p1, p2), ...] for this region, where each point is (RA, Dec) in degrees.
    color : color-like
        Fill color for the polygon (matplotlib style).
    bcg_sig : list or np.ndarray
        Signature vector for the BCG's region, as returned by build_bisector_signatures.
    bisectors : list of dict
        Output from get_bisectors. Used for spherical signature test at corners.
    alpha : float, optional
        Transparency for the polygon fill (default 1.0 = opaque).

    Notes
    -----
    - This function ensures the region polygon includes only those bounding box corners
      that are "inside" the region as determined by the BCG signature.
    - Uses a convex hull to ensure a valid polygon for plotting.
    - Will print a warning if the convex hull creation fails (e.g., if too few points).
    """

    # Get plot boundaries
    ra_min, ra_max = ax.get_xlim()
    dec_min, dec_max = ax.get_ylim()

    # Define bounding box corners
    corners = [
        (ra_min, dec_min),
        (ra_min, dec_max),
        (ra_max, dec_max),
        (ra_max, dec_min),
    ]

    # Check which corners are inside the BCG's region by signature
    valid_corners = []
    for corner in corners:
        coord = SkyCoord(ra=corner[0] * u.deg, dec=corner[1] * u.deg, frame='icrs')
        v = coord.cartesian.xyz.value
        sig = []
        for b in bisectors:
            pole = b['pole']
            sign = np.sign(np.dot(v, pole))
            sig.append(sign)
        sig = np.array(sig)
        mask = (np.array(bcg_sig) != 0) & (sig != 0)
        if np.all(sig[mask] == np.array(bcg_sig)[mask]):
            valid_corners.append(corner)

    # Collect all points: endpoints + matched corners
    points = np.array([pt for seg in region_segments for pt in seg] + valid_corners)

    # Remove duplicates
    points = np.unique(points, axis=0)

    # Fill the polygon
    if len(points) >= 3:
        try:
            hull = ConvexHull(points)
            poly = Polygon(points[hull.vertices], closed=True, facecolor=color,
                           edgecolor='none', alpha=alpha)
            ax.add_patch(poly)
        except Exception as e:
            print(f"Polygon creation failed ({type(e).__name__}): {e}")
