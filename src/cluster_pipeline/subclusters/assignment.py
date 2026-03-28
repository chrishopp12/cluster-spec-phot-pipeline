"""Subcluster member assignment via bisector signature matching."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

from cluster_pipeline.subclusters.geometry import (
    get_bisectors,
    define_bbox,
    get_segments,
    build_bcg_signatures,
    build_point_signature,
    classify_segments,
)


def assign_subcluster_regions(subcluster_configs, margin=0.05, margin_frac=5.0, plot=False, verbose=False):
    """
    Assign spatial regions to each BCG (Brightest Cluster Galaxy) using all pairwise bisectors.

    This partitions the sky such that each BCG's region is bounded by the set of bisectors it is involved with.
    Returns a dictionary mapping each BCG to the list of region-defining line segments.
    Optionally displays a diagnostic plot of the regions and bounding box.

    Parameters
    ----------
    subcluster_configs : list of dict
        List of subcluster configuration dictionaries, each containing at least a 'center' key with a SkyCoord value.
    margin : float, optional
        Absolute padding (in degrees) to expand the bounding box. [default: 0.05]
    margin_frac : float, optional
        Fractional padding (as a multiple of the max RA/Dec extent) to expand the bounding box. [default: 5.0]
    plot : bool, optional
        If True, display a diagnostic plot of the assigned regions. [default: False]

    Returns
    -------
    bcg_regions : dict
        Mapping from BCG index to list of line segments [(p1, p2), ...] defining the region.
        Each (p1, p2) is a tuple of RA/Dec in degrees.

    Notes
    -----
    - The region assigned to each BCG is the intersection of all areas on the sky where the BCG
      is on the "correct" side of each relevant bisector.

    Raises
    ------
    ValueError if fewer than two centers are provided.
    """

    centers = [sub['center'] for sub in subcluster_configs]
    colors = [sub['color'] for sub in subcluster_configs]
    if len(centers) < 2:
        raise ValueError("At least two BCG centers are required to assign subcluster regions.")


    # --- Compute all pairwise bisectors ---
    bisectors = get_bisectors(subcluster_configs)

    # -- Define bounding box (RA/Dec) and edges --
    ra_min, ra_max, dec_min, dec_max = define_bbox(subcluster_configs, margin_frac)
    bbox_edges = [
        ((ra_min, dec_min), (ra_max, dec_min)),
        ((ra_max, dec_min), (ra_max, dec_max)),
        ((ra_max, dec_max), (ra_min, dec_max)),
        ((ra_min, dec_max), (ra_min, dec_min))
    ]

    # -- Find bisector intersections with the bounding box --
    segments = get_segments(bisectors, bbox_edges)

    # -- Build signature vector for each BCG across all bisectors --
    bcg_signatures = build_bcg_signatures(subcluster_configs, bisectors)

    # -- Classify segments into BCG-defined regions --
    bcg_regions = classify_segments(segments, bisectors, bcg_signatures, verbose=verbose)

   # --- Optional: Diagnostic Plotting --- #This plot is a pain, but can be a good first-look
    if plot:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MultipleLocator

        fig, ax = plt.subplots(figsize=(6, 6))

        # Plot BCG markers
        for i, c in enumerate(centers):
            ax.plot(c.ra.deg, c.dec.deg, 'k*', markersize=10, label=f"BCG {i+1}")
            ax.text(c.ra.deg, c.dec.deg, f"BCG {i+1}", fontsize=9, va='bottom', ha='left')

        # Set limits early so fill polygons use the same view
        ra_vals = np.array([c.ra.deg for c in centers])
        dec_vals = np.array([c.dec.deg for c in centers])
        ra_min, ra_max = ra_vals.min() - margin, ra_vals.max() + margin
        dec_min, dec_max = dec_vals.min() - margin, dec_vals.max() + margin
        ax.set_xlim(ra_min, ra_max)
        ax.set_ylim(dec_min, dec_max)

        # Plot segments and shaded regions
        for bcg_idx, segs in bcg_regions.items():
            if verbose:
                print(f"BCG {bcg_idx}: {len(segs)} segments")
            add_region_fill_clipped_to_signature(ax, segs, color=colors[bcg_idx],
                                                bcg_sig=bcg_signatures[bcg_idx],
                                                bisectors=bisectors)

        # Lock limits again (prevent tightening from patch rendering)
        ax.set_xlim(ra_min, ra_max)
        ax.set_ylim(dec_min, dec_max)
        ax.xaxis.set_major_locator(MultipleLocator(0.05))
        ax.invert_xaxis()
        ax.set_xlabel("R.A.")
        ax.set_ylabel("Decl.")

        plt.tight_layout()
        plt.show()
    if verbose:
        print(f"  Number of BCGs: {len(centers)}")
        print(f"  Number of bisectors: {len(bisectors)}")
        print(f"  BCG Regions are {bcg_regions}")
    return bcg_regions

def assign_subcluster_members_multi(subcluster_configs, galaxies_df, plot=False):
    """
    Assign galaxies to subclusters based on bisector geometry, radial cut, and (optional) redshift bounds.

    Each subcluster is defined as the unique region bounded by bisectors between all centers;
    galaxies are assigned by matching their "side" of each bisector to each BCG's signature.

    Parameters
    ----------
    subcluster_configs : list of dict
        List of subcluster configuration dictionaries.
    galaxies_df : pd.DataFrame
        DataFrame with columns 'RA', 'Dec', and (optionally) 'z'.
    plot : bool, optional
        Plot diagnostics.

    Returns
    -------
    region_list : list of pd.DataFrame
        List of DataFrames: one for each region/subcluster, with assigned members.
    bisectors : list of dict
        Bisector metadata (from get_bisectors), for possible plotting/debug.
    """

    centers = [sub['center'] for sub in subcluster_configs]
    n = len(centers)
    galaxy_coords = SkyCoord(ra=galaxies_df['RA'].values * u.deg, dec=galaxies_df['Dec'].values * u.deg)

    df_valid = galaxies_df.copy()

    # --- Bisector & signature calculations ---
    bisectors = get_bisectors(subcluster_configs)
    bcg_signatures = build_bcg_signatures(subcluster_configs, bisectors)

    galaxy_signatures = []
    for coord in galaxy_coords:
        sig = build_point_signature(coord, bisectors)
        galaxy_signatures.append(sig)
    galaxy_signatures = np.array(galaxy_signatures)  # shape (N_gal, N_bis)


    # --- Assign galaxies to subcluster regions ---
    region_ids = np.full(len(df_valid), -1)
    for region_id, bcg_sig in bcg_signatures.items():
        bcg_sig = np.array(bcg_sig)
        mask = (bcg_sig != 0)  # Only compare bisectors relevant to this BCG
        matches = np.all(galaxy_signatures[:, mask] == bcg_sig[mask], axis=1)
        region_ids[matches] = region_id

        print(f"Assigned {np.sum(matches)} galaxies to region {region_id}")

    # Another mediocre plot, but good for a first look
    if plot:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MultipleLocator

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(df_valid['RA'], df_valid['Dec'], c=region_ids, cmap='tab10', s=20, label='Galaxies')
        ax.scatter([c.ra.deg for c in centers], [c.dec.deg for c in centers],
                c='black', s=100, marker='x', label='Centers')

        for b in bisectors:
            anchor_vec = b['mid'].cartesian.xyz.value
            pole = b['pole']
            tangent = np.cross(pole, anchor_vec)
            tangent /= np.linalg.norm(tangent)
            # Parameterize great circle: anchor * cos(theta) + tangent * sin(theta)
            thetas = np.linspace(-0.5, 0.5, 200) * np.pi
            pts_xyz = (anchor_vec[:, None] * np.cos(thetas) +
                    tangent[:, None] * np.sin(thetas)).T

            pts_gc = SkyCoord(x=pts_xyz[:,0], y=pts_xyz[:,1], z=pts_xyz[:,2],
                            representation_type='cartesian', frame='icrs')
            pts_gc_sph = pts_gc.represent_as('spherical')
            ra = pts_gc_sph.lon.deg   # longitude (RA in deg)
            dec = pts_gc_sph.lat.deg  # latitude (Dec in deg)
            ax.plot(ra, dec, 'k--', lw=1, alpha=0.6)

        ra_vals = np.array([c.ra.deg for c in centers])
        dec_vals = np.array([c.dec.deg for c in centers])
        margin = 0.05
        ax.set_xlim(ra_vals.min() - margin, ra_vals.max() + margin)
        ax.set_ylim(dec_vals.min() - margin, dec_vals.max() + margin)

        ax.set_xlabel("RA [deg]")
        ax.xaxis.set_major_locator(MultipleLocator(0.05))
        ax.invert_xaxis()
        ax.set_ylabel("Dec [deg]")
        ax.set_aspect('equal', adjustable='datalim')
        plt.tight_layout()
        plt.show()


    # --- Gather DataFrames for each region ---
    region_list = []
    for i in range(n):
        region_list.append(df_valid[region_ids == i].copy())
    return region_list, bisectors

def filter_members_by_config(member_groups, subcluster_configs, spec=True):
    """
    Apply per-subcluster radius and (optionally) redshift cuts after region assignment.

    Parameters
    ----------
    member_groups : list of pd.DataFrame
        Region-assigned galaxy tables, each with 'RA', 'Dec', and optionally 'z'.
    subcluster_configs : list of dict
        Each dict contains 'z_range' (tuple of floats) and 'radius' (arcmin).
    spec : bool, optional
        If True, apply redshift cut as well as radius.

    Returns
    -------
    filtered : list of pd.DataFrame
        Filtered list of member groups.
    """
    centers = [sub['center'] for sub in subcluster_configs]
    filtered = []
    for i, df in enumerate(member_groups):
        if df.empty:
            filtered.append(df)
            continue

        zmin, zmax = subcluster_configs[i]['z_range']
        max_radius = subcluster_configs[i]['radius']

        coords = SkyCoord(ra=df['RA'].values * u.deg, dec=df['Dec'].values * u.deg)
        sep = coords.separation(centers[i]).arcmin

        if spec and 'z' in df.columns:
            print("\n------------------------------------------------------")
            print(f"Filtering Region {i+1} Spectroscopic Members")
            print(f"  Redshift range: {zmin:.4f} - {zmax:.4f}")
            print(f"      Radial cut: {max_radius} arcmin")
            z = df['z'].values
            has_z = np.isfinite(z)
            in_z = has_z & (z >= zmin) & (z <= zmax)
            keep = (sep < max_radius) & (in_z | ~has_z)
        else:
            print("\n------------------------------------------------------")
            print(f"Filtering Region {i+1} Photometric Members")
            print(f"      Radial cut: {max_radius} arcmin")
            keep = sep < max_radius


        print(f"        Region {i+1}: {keep.sum()} of {len(df)} members kept after radius/z cut")

        filtered.append(df[keep].copy())

    return filtered

def make_member_catalogs(cluster, spec_groups, phot_groups, subcluster_configs):

    # Output spectroscopic and photometric member catalogs for each subcluster
    output_dir = os.path.join(cluster.cluster_path, "Subcluster_Catalogs")
    os.makedirs(output_dir, exist_ok=True)

    for i, (spec_df, phot_df) in enumerate(zip(spec_groups, phot_groups)):
        sub_id = subcluster_configs[i].get("bcg_id", i+1)
        label = subcluster_configs[i].get("bcg_label", f"{sub_id}")
        # Safe file label
        safe_label = str(label).replace(" ", "_")
        # Output filenames
        spec_file = os.path.join(output_dir, f"spec_members_subcluster_{safe_label}.csv")
        phot_file = os.path.join(output_dir, f"phot_members_subcluster_{safe_label}.csv")
        # Write DataFrames
        spec_df.to_csv(spec_file, index=False)
        phot_df.to_csv(phot_file, index=False)
        print(f"  - Saved {len(spec_df)} spectroscopic and {len(phot_df)} photometric members for subcluster {label} to CSV.")

def make_combined_groups(cluster, spec_groups, phot_groups, subcluster_configs, combine_groups=None):

    # Output spectroscopic and photometric member catalogs for each subcluster
    output_dir = os.path.join(cluster.cluster_path, "Subcluster_Catalogs")
    os.makedirs(output_dir, exist_ok=True)

    spec_groups_copy = None
    combined_indices = []

    if combine_groups is not None:
        for group in combine_groups:
            print(f"\nCombining subclusters {group[0]} and {group[1]} into one catalog.")
            index_1 = [index for index, element in enumerate(subcluster_configs) if element['bcg_id'] == group[0]][0]
            index_2 = [index for index, element in enumerate(subcluster_configs) if element['bcg_id'] == group[1]][0]
            spec_combine = pd.concat([spec_groups[index_1], spec_groups[index_2]])
            phot_combine = pd.concat([phot_groups[index_1], phot_groups[index_2]])

            spec_file = os.path.join(output_dir, f"spec_members_subcluster_{group[0]}_{group[1]}.csv")
            phot_file = os.path.join(output_dir, f"phot_members_subcluster_{group[0]}_{group[1]}.csv")
            spec_combine.to_csv(spec_file, index=False)
            phot_combine.to_csv(phot_file, index=False)
            print(f"  - Saved combined spectroscopic and photometric members for subclusters {group[0]} and {group[1]} to CSV.")

            if spec_groups_copy is None:
                spec_groups_copy = spec_groups.copy()
                phot_groups_copy = phot_groups.copy()
            spec_groups_copy[index_1] = spec_combine
            phot_groups_copy[index_1] = phot_combine

            combined_indices.append((index_1, index_2))

        spec_groups_combined = [spec_groups_copy[i] for i in range(len(spec_groups_copy)) if i not in [idx[1] for idx in combined_indices]]
        phot_groups_combined = [phot_groups_copy[i] for i in range(len(phot_groups_copy)) if i not in [idx[1] for idx in combined_indices]]
    else:
        spec_groups_combined = spec_groups
        phot_groups_combined = phot_groups
        combined_indices = None

    return spec_groups_combined, phot_groups_combined, combined_indices
