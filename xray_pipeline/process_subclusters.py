#!/usr/bin/env python3
"""
process_subclusters.py

Subcluster Identification, Member Assignment, and Cluster Visualization Pipeline
-------------------------------------------------------------------------------
- Defines bisector-based sky regions around BCGs for unique subcluster partitioning.
- Assigns cluster galaxies to subclusters via signature matching (bisector-side logic).
- Applies per-region redshift/radius cuts, and generates a suite of spatial and redshift-visualization plots.


Usage:
    python process_subclusters.py cluster_id --subclusters 1 2 3 [options]

Example:
    python process_subclusters.py "1327" --subclusters 2 6 7 --radius 2.0 --z-range 0.420 0.455 --save-plots True --show-plots False --colors "tab:green" "gold" "tab:cyan"

Requirements:
    - astropy
    - matplotlib
    - numpy
    - pandas
    - scipy
    - cluster.py (local)
    - my_utils.py (local)
    - xray_plotting.py (local)
    - process_redshifts.py (local)

Arguments:
    # --- Cluster Arguments (for Cluster object) ---
    cluster_id             <str>      Cluster ID or RMJ/Abell name (e.g., '1327')
    --z-min                <float>    Minimum redshift for analysis
    --z-max                <float>    Maximum redshift for analysis
    --fov                  <float>    Field of view in arcmin (zoomed)
    --fov-full             <float>    Full field of view in arcmin
    --ra-offset            <float>    RA offset for image center (arcmin)
    --dec-offset           <float>    Dec offset for image center (arcmin)
    --density-levels       <int>      Number of density contour levels
    --density-skip         <int>      Number of density levels to skip
    --density-bandwidth    <float>    KDE bandwidth for density contours
    --xray-levels          <str>      X-ray contour levels (comma-separated)
    --xray-psf             <float>    X-ray PSF smoothing (arcsec)
    [Any other cluster-wide options]

    # --- Subcluster Arguments ---
    --subclusters          <int...>   List of BCG indices to define regions (e.g., 2 6 7)
    --radius               <float>    Search radius for subclusters [Mpc, default: 2.0]
    --z-range              <float float>  Redshift min and max for all subclusters
    --colors               <str...>   Colors for each subcluster (matplotlib names)
    --labels               <str...>   Labels for each subcluster
    [Supports per-subcluster overrides: e.g. --color_3 tab:cyan --z_range_2 0.43 0.45]

    # --- Figure/Output Arguments ---
    --save-plots           <bool>     Save all generated figures [default: True]
    --show-plots           <bool>     Display figures interactively [default: False]
    --plot-alt-regions     <bool>     Plot alternate region/histogram layout [default: False]
    --legend-loc           <str>      Legend location string (e.g. "upper right")
    [Any other plotting, output, or cosmetic options]

Notes:
    - The bisector signature logic uniquely assigns every galaxy to a subcluster region.
    - Output subcluster definitions (labels, colors, radius, z-range) are stored to a CSV for reproducibility.
    - Figures include: (1) region overlays, (2) redshift histograms, (3) 2-panel/3-panel/4-panel publication layouts.

"""


import os, argparse, yaml, ast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord, CartesianRepresentation, Angle, SkyCoord
from astropy.constants import c

import matplotlib as mpl
from matplotlib.patches import ConnectionPatch, Polygon
from matplotlib.ticker import FuncFormatter
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator, MultipleLocator, FuncFormatter, FormatStrFormatter

from collections import defaultdict

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from scipy.spatial import ConvexHull

from cluster import Cluster
from my_utils import load_dataframes, pop_prefixed_kwargs, str2bool, load_bcg_catalog, load_photo_coords
from xray_pipeline.process_redshifts import velocity_dispersion, plot_stacked_velocity_histograms, process_redshifts
from xray_pipeline.xray_plotting import plot_redshift_overlay, plot_optical, make_plots
# from run_cmd_pipeline import run_full_pipeline

import scipy.stats
from astropy.table import Table


plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14



# -- Helper Functions --

def build_subclusters(subclusters=(1,), cluster=None, **kwargs):
    """
    Build and update subcluster configuration for a given cluster.

    Hierarchy of parameter resolution for each field (e.g., color, radius, z_range):
        1. CLI per-subcluster override: e.g., color_2, radius_3
        2. CLI global override: e.g., --color tab:green --radius 2.5
        3. Existing CSV value (if present)
        4. Code default (palette, 2.0 for radius, etc.)

    Parameters
    ----------
    subclusters : list or tuple of int or dict
        Each entry is either a BCG index (1-based, int) or a dict with at least 'bcg_id'.
    cluster : Cluster
        Cluster object (must have attribute `subcluster_file` for saving).
    **kwargs :
        CLI-style overrides (color_2, radius_1, label_3, color, radius, label, z_range, etc).

    Returns
    -------
    output : list of dict
        List of subcluster dicts, one per subcluster in input.
    Raises
    ------
    ValueError
        If cluster is not provided, or required fields (e.g. z_range) are missing.
    Side Effects
    ------------
    Updates (or creates) the subcluster CSV at `cluster.subcluster_file` with the union of all seen BCGs.
    """
    if cluster is None:
        raise ValueError("Must supply a Cluster object.")


    ########################
    # Helper Methods
    ########################

    # -- Subcluster helpers --
    def _get_z_range(i, bcg_id, base):
        zr = _merged_field("z_range", i, bcg_id, base)
        if zr is not None:
            return zr
        # Cluster default (tuple of min/max)
        if cluster is not None and hasattr(cluster, "z_min") and hasattr(cluster, "z_max"):
            return (cluster.z_min, cluster.z_max)
        raise ValueError(f"Must specify z_range for subcluster {bcg_id} (index {i+1}).")

    def _kw_override(key, i, bcg_id, default=None):
        v = kwargs.get(f"{key}_{bcg_id}", None)
        if v is not None:
            return v
        v_all = kwargs.get(key, None)
        if v_all is not None:
            # Special handling for z_range/group_z_range (tuple of two floats)
            if key in {"z_range", "group_z_range"}:
                if isinstance(v_all, (list, tuple)) and len(v_all) == 2 and all(
                    isinstance(x, (int, float, np.floating, np.integer)) for x in v_all
                ):
                    return (float(v_all[0]), float(v_all[1]))
            # If list/tuple, select by subcluster index
            if isinstance(v_all, (list, tuple)):
                if len(v_all) > i:
                    return v_all[i]
                else:
                    return default
            # If scalar (float/int/str), use for all
            return v_all
        # Default for field (CSV checked in merged_field)
        return default
    
    def _merged_field(field, i, bcg_id, base, default=None):
        cli_val = _kw_override(field, i, bcg_id, None)
        if cli_val is not None:
            return cli_val
        csv_val = base.get(field, None)
        if csv_val is not None and csv_val != "":
            return csv_val
        return default


    # -- Combined group helpers --
    def _parse_combine_arg(combine):
        """
        Accepts: None; ['1+4','2+5+3']; [(1,4),(4,1)]; [(2,5,3)]
        Returns: list of lists with preserved order, e.g. [[1,4],[2,5,3]]
        """
        if not combine:
            return []
        groups = []
        for item in combine:
            if isinstance(item, (tuple, list)):
                groups.append([int(x) for x in item])
            else:
                parts = [int(p.strip()) for p in str(item).split("+") if p.strip()]
                groups.append(parts)
        return groups

    def _gid_from_members(members):
        """Canonical group id string using SORTED members (stable key for kwargs/CSV)."""
        return "_".join(str(x) for x in sorted(members))


    # -- Load and save --
    def _load_subcluster_csv(csv_path):
        if not os.path.exists(csv_path):
            return {}
        df = pd.read_csv(csv_path)
        configs = {}
        for _, row in df.iterrows():
            bcg_id = int(row["bcg_id"])
            cfg = {
                "bcg_id": bcg_id,
                "bcg_label": row.get("bcg_label", ""),
                "color": row.get("color", ""),
                "radius": float(row["radius"]) if "radius" in row and pd.notna(row["radius"]) else None,
                "z_range": (
                    float(row["z_range_min"]), float(row["z_range_max"])
                ) if "z_range_min" in row and "z_range_max" in row and pd.notna(row["z_range_min"]) and pd.notna(row["z_range_max"]) else None,
            }
    
            configs[bcg_id] = cfg
        return configs

    def _save_subcluster_csv(configs, csv_path):
        rows = []
        for bcg_id, cfg in sorted(configs.items()):
            z_range = cfg.get("z_range", (None, None))
            row = {
                "bcg_id": bcg_id,
                "bcg_label": cfg.get("bcg_label", ""),
                "color": cfg.get("color", ""),
                "radius": cfg.get("radius", ""),
                "z_range_min": z_range[0] if z_range else "",
                "z_range_max": z_range[1] if z_range else "",
            }
            # optional group metadata
            if "group_id" in cfg:       row["group_id"] = cfg["group_id"]
            if "is_dominant" in cfg:    row["is_dominant"] = int(cfg["is_dominant"])
            if "group_members" in cfg:
                gm = cfg.get("group_members", ())
                if isinstance(gm, (list, tuple)) and len(gm) > 0:
                    row["group_members"] = repr(tuple(int(x) for x in gm))
                else:
                    row["group_members"] = ""

            if "group_label" in cfg:    row["group_label"] = cfg["group_label"]
            if "group_color" in cfg:    row["group_color"] = cfg["group_color"]
            gz = cfg.get("group_z_range")
            if gz:
                row["group_z_range_min"] = gz[0]
                row["group_z_range_max"] = gz[1]
            rows.append(row)
        pd.DataFrame(rows).to_csv(csv_path, index=False)


    ########################
    # Build Subclusters
    ########################

    csv_path = cluster.subcluster_file

    # Load existing configs
    configs = _load_subcluster_csv(csv_path)
    _, _, bcg_df = load_dataframes(cluster)

    # Defaults
    default_colors = [
        "white", "tab:green", "tab:purple", "tab:cyan", "tab:pink", "gold", "tab:orange", "tab:green", "tab:purple", "tab:pink"
    ]
    default_labels = ["1", "2", "3", "4", "5", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    default_radius = 2.0


    output = []
    for j, sub in enumerate(subclusters):
        if isinstance(sub, dict):
            bcg_id = int(sub.get("bcg_id"))
        elif isinstance(sub, (int, np.integer)):
            bcg_id = int(sub)
        else:
            raise ValueError("Each subcluster must be an int or dict (with 'bcg_id').")

        color_idx = (bcg_id - 1) % len(default_colors)
        label_idx = bcg_id - 1
        base = configs.get(bcg_id, {})

        # Merge: priority is kwargs/subcluster > csv > default
        out = dict(base)
        out["bcg_id"]    = bcg_id
        out["bcg_label"] = _merged_field("bcg_label", j, bcg_id, base, default_labels[label_idx])
        out["color"]     = _merged_field("color", j, bcg_id, base, default_colors[color_idx])
        out["radius"]    = _merged_field("radius", j, bcg_id, base, default_radius)
        out["z_range"]   = _get_z_range(j, bcg_id, base)
        out['center'] = load_bcg_location(bcg_df, bcg_id=bcg_id)
        z_bcg = np.nan 
        if 'z' in bcg_df.columns:
            row = bcg_df[bcg_df['BCG_priority'] == bcg_id]
            if not row.empty:
                z_val = row['z'].iloc[0]
                if not pd.isna(z_val):
                    z_bcg = z_val
        out['z_bcg'] = z_bcg

        output.append(out)
        configs[bcg_id] = out  # Update/insert for CSV

        print(f"############ Subcluster {len(output)} ##############")
        print(f"  BCG ID:      {out['bcg_id']}")
        print(f"   Label:      {out['bcg_label']}")
        print(f"  Center:      {out['center'].to_string('decimal')}")
        print(f"   z_bcg:      {out['z_bcg']}")
        print(f"   Color:      {out['color']}")
        print(f"  Radius:      {out['radius']} Mpc")
        print()


    ########################
    # Build Combined Groups
    ########################

    combined = kwargs.pop("combined", None)
    combined_groups = _parse_combine_arg(combined)
    # --- Combined groups --
    by_id = {o["bcg_id"]: o for o in output}
    for bcg_id, ent in by_id.items():
        ent["group_id"]       = str(bcg_id)   # solo by default
        ent["is_dominant"]    = 1
        ent["group_members"]  = (bcg_id,)            # tuple; empty means no combine
        ent["group_label"]    = ent.get("bcg_label", str(bcg_id))   # reuse BCG label
        ent["group_color"]    = ent.get("color", "")                # reuse BCG color
        ent["group_z_range"]  = ent.get("z_range", None)            # reuse BCG z-range

    if combined_groups:
        by_id = {o["bcg_id"]: o for o in output}
        present = set(by_id.keys())

        # Global optional overrides
        global_glabel = kwargs.get("group_label", None)
        global_gcolor = kwargs.get("group_color", None)
        global_gz     = kwargs.get("group_z_range", None)  # tuple (zmin, zmax)

        for raw_members in combined_groups:
            members = [m for m in raw_members if m in present]
            if len(members) < 2:
                continue
            dominant = members[0]                  # ORDER MATTERS: dominant is first
            gid = _gid_from_members(members)       # canonical key (sorted)

            # Per-group overrides via kwargs: group_label_1_4, group_color_1_4, group_z_range_1_4
            per_label = kwargs.get(f"group_label_{gid}", None)
            per_color = kwargs.get(f"group_color_{gid}", None)
            per_gz    = kwargs.get(f"group_z_range_{gid}", None)

            # Defaults from dominant entry
            dom = by_id[dominant]
            dom_label = dom.get("bcg_label", gid)
            dom_color = dom.get("color", "")
            dom_zrng  = dom.get("z_range", None)

            group_label   = per_label if per_label is not None else (global_glabel if global_glabel is not None else dom_label)
            group_color   = per_color if per_color is not None else (global_gcolor if global_gcolor is not None else dom_color)
            group_z_range = per_gz    if per_gz    is not None else (global_gz    if global_gz    is not None else dom_zrng)

            tuple_members = tuple(members)
            for m in members:
                ent = by_id[m]
                ent["group_id"]      = gid
                ent["group_members"] = tuple_members 
                ent["is_dominant"]   = int(m == dominant)
                ent["group_label"]   = group_label
                ent["group_color"]   = group_color
                ent["group_z_range"] = group_z_range 

            print(f"############ Group {ent['group_id']} ##############")
            print(f" Members:    {', '.join(str(x) for x in members)}")
            print(f"   Label:      {ent['group_label']}")
            print(f"   Color:      {ent['group_color']}")
            print()


        # Fill in any missing group fields for singles
        for bcg_id, ent in by_id.items():
            ent.setdefault("group_id", str(bcg_id))
            ent.setdefault("is_dominant", 1)
            ent.setdefault("group_members", "")
            ent.setdefault("group_label", ent.get("group_label", "")) 
            ent.setdefault("group_color", ent.get("group_color", ""))

        # Update configs for saving
        for bcg_id, ent in by_id.items():
            configs[bcg_id] = ent


    # Save union of all BCGs
    _save_subcluster_csv(configs, csv_path)


    return output

def load_bcg_location(bcg_df, bcg_id=None, verbose=False):

    if verbose:
        print(f"BCGs {bcg_df}\n")
    if bcg_id is not None:
        bcg_row = bcg_df.loc[bcg_df['BCG_priority'] == bcg_id]
        if bcg_row.empty:
            raise ValueError(f"BCG {bcg_id} not found.")
        ra = bcg_row['RA'].values[0]
        dec = bcg_row['Dec'].values[0]

    return SkyCoord(ra=ra * u.deg, dec=dec * u.deg)

def los_velocity_diff(z1, z2, sig_z1=None, sig_z2=None):
    """
    Rest-frame line-of-sight velocity difference between two redshifts.

    Parameters
    ----------
    z1, z2 : float
        Redshifts of the two objects.
    sig_z1, sig_z2 : float, optional
        1-sigma uncertainties on z1 and z2.

    Returns
    -------
    dv_kms : float
        Signed rest-frame LOS velocity difference [km/s] (positive if z2 > z1).
    dv_err_kms : float or None
        Uncertainty on dv [km/s] if uncertainties provided, else None.
    z_mean : float
        Mean redshift used for the rest-frame correction.
    """
    c_kms = c.to(u.km / u.s).value
    z_mean = 0.5 * (z1 + z2)
    dz = (z2 - z1)
    dv_kms = c_kms * dz / (1.0 + z_mean)

    dv_err_kms = None
    if (sig_z1 is not None) and (sig_z2 is not None):
        dz_err = (sig_z1**2 + sig_z2**2) ** 0.5
        dv_err_kms = c_kms * dz_err / (1.0 + z_mean)
    return dv_kms, dv_err_kms, z_mean

def bcg_z_by_subcluster(cluster, subcluster_configs):
    """
    Build arrays aligned to subcluster order for BCG redshifts and uncertainties.

    Parameters
    ----------
    subcluster_configs : list[dict] or dict[int->dict]
        Each config must include 'bcg_id'.
    bcgs_dict : dict
        Output from load_bcg_catalog().

    Returns
    -------
    z_list : list[float or None]
    zerr_list : list[float or None]
    """
    bcgs_dict = load_bcg_catalog(cluster)

    # Normalize to list ordered by subcluster index (1..N)
    if isinstance(subcluster_configs, dict):
        # assume keys are bcg_id or 1..N; keep insertion order if already a dict
        ordered_cfgs = [subcluster_configs[k] for k in sorted(subcluster_configs.keys())]
    else:
        ordered_cfgs = list(subcluster_configs)

    z_list, zerr_list = [], []
    for cfg in ordered_cfgs:
        bid = int(cfg["bcg_id"])
        rec = bcgs_dict.get(bid)
        if rec is None:
            z_list.append(None)
            zerr_list.append(None)
        else:
            z_list.append(rec["z"])
            zerr_list.append(rec["z_err"])
    return z_list, zerr_list

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

def analyze_group(spec_groups):
    z_data = []
    vel_data = []
    for i, spec_members in enumerate(spec_groups):
        z_vals = spec_members['z'].values
        z_mean, sigma_v, velocities = velocity_dispersion(z_vals)
        vel_data.append((velocities, z_mean, sigma_v))
        z_data.append(z_vals)
        print(f"Subcluster {i+1}: N={len(spec_members)} | $\\bar{{z}}$ = {z_mean:.5f} | $\\sigma_v$ = {sigma_v:.1f} km/s")
    return z_data, vel_data

def make_stats_table(
    z_groups,
    *,
    bins: int = 12,
    prefix: str = "subcluster",
    make_plots: bool = True,
    save_plots: bool = True,
    show_plots: bool = True,
    save_path=None,
    ranges=None,  # optional list of (z_low, z_high) for display only
):
    """
    Compute normality diagnostics (KS, Anderson-Darling) and produce a LaTeX-ready
    stats table for each subcluster using the *actual* z values per subcluster.
    """

    def _ad_pvalue_normality(z: np.ndarray) -> tuple[float, float]:
        """
        Anderson-Darling normality test (unknown mean/variance) with p-value.
        Prefer statsmodels; fall back to a common approximation if statsmodels
        isn't installed.

        Returns
        -------
        ad_stat, ad_p : float, float
        """
        z = np.asarray(z, dtype=float)
        n = z.size
        if n < 3:
            return np.nan, np.nan

        # 1) Preferred: statsmodels provides (statistic, pvalue)
        try:
            from statsmodels.stats.diagnostic import normal_ad  # type: ignore
            ad2, p = normal_ad(z)
            return float(ad2), float(p)
        except Exception:
            pass

        # 2) Fallback: scipy statistic + Stephens-style approximate p-value
        #    (uses the adjusted statistic for estimated mu/sigma).
        #    Piecewise approximations are widely used; see e.g. summary here. :contentReference[oaicite:1]{index=1}
        anderson_res = scipy.stats.anderson(z, dist="norm")
        ad2 = float(anderson_res.statistic)

        # Adjusted statistic for estimated parameters (common correction)
        n = float(n)
        ad2_star = ad2 * (1.0 + 0.75 / n + 2.25 / (n * n))

        # Piecewise approximation for p-value
        if ad2_star < 0.2:
            p = 1.0 - np.exp(-13.436 + 101.14 * ad2_star - 223.73 * (ad2_star ** 2))
        elif ad2_star < 0.34:
            p = 1.0 - np.exp(-8.318 + 42.796 * ad2_star - 59.938 * (ad2_star ** 2))
        elif ad2_star < 0.6:
            p = np.exp(0.9177 - 4.279 * ad2_star - 1.38 * (ad2_star ** 2))
        elif ad2_star < 10:
            p = np.exp(1.2937 - 5.709 * ad2_star + 0.0186 * (ad2_star ** 2))
        else:
            p = 0.0

        # Numerical safety
        p = float(np.clip(p, 0.0, 1.0))
        return ad2, p

    stats_list = []
    stats_values = []

    if ranges is not None and len(ranges) != len(z_groups):
        raise ValueError("Length of 'ranges' must match length of 'z_groups' or be None.")

    for i, z_subset in enumerate(z_groups, start=1):
        z_subset = np.asarray(z_subset, dtype=float)
        N = z_subset.size

        if ranges is not None:
            z_low, z_high = ranges[i - 1]
        else:
            z_low, z_high = (np.nanmin(z_subset) if N > 0 else np.nan,
                             np.nanmax(z_subset) if N > 0 else np.nan)

        if N == 0:
            print(f"Skipping subcluster {i} (no z values provided).")
            stats_list.append([i, round(z_low, 3) if np.isfinite(z_low) else np.nan,
                                  round(z_high, 3) if np.isfinite(z_high) else np.nan,
                                  np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, "No data", 0])
            stats_values.append(None)
            continue

        subset_mean = float(np.mean(z_subset))
        subset_std = float(np.std(z_subset, ddof=1)) if N > 1 else 0.0

        if N >= 2 and subset_std > 0:
            ks_statistic, ks_p_value = scipy.stats.ks_1samp(
                z_subset,
                scipy.stats.norm(loc=subset_mean, scale=subset_std).cdf,
                method="auto",
            )
        else:
            ks_statistic, ks_p_value = np.nan, np.nan

        # AD stat + p-value
        ad_stat, ad_p_value = _ad_pvalue_normality(z_subset)

        # (Optional) keep your “critical value bucket” label for readability
        if N >= 3 and np.isfinite(ad_stat):
            anderson_res = scipy.stats.anderson(z_subset, dist="norm")
            sig_levels = anderson_res.significance_level
            crit_values = anderson_res.critical_values

            ad_significance = None
            for level, crit in zip(sig_levels[::-1], crit_values[::-1]):  # descending
                if ad_stat > crit:
                    ad_significance = level
                    break
            if ad_significance is None:
                ad_significance = "Fail to Reject Normality"
        else:
            ad_significance = "Insufficient N (<3)"

        stats_list.append([
            i,
            round(z_low, 3) if np.isfinite(z_low) else np.nan,
            round(z_high, 3) if np.isfinite(z_high) else np.nan,
            round(subset_mean, 5),
            round(subset_std, 6),
            round(ks_statistic, 4) if np.isfinite(ks_statistic) else np.nan,
            round(ks_p_value, 4) if np.isfinite(ks_p_value) else np.nan,
            round(ad_stat, 4) if np.isfinite(ad_stat) else np.nan,
            round(ad_p_value, 4) if np.isfinite(ad_p_value) else np.nan,
            ad_significance,
            N
        ])

        stats_values.append({
            "Subcluster": i,
            "z_min": z_low,
            "z_max": z_high,
            "Mean": subset_mean,
            "Std Dev": subset_std,
            "KS Stat": ks_statistic,
            "KS p-value": ks_p_value,
            "AD Stat": ad_stat,
            "AD p-value": ad_p_value,
            "Anderson Level": ad_significance,
            "N": N
        })

        print("")
        print(f"-------------- Stats Summary: Subcluster {i} --------------")
        print(f"                   N (count): {N}")
        print(f"                     Mean(z): {subset_mean}")
        print(f"                Std Dev (z): {subset_std}")
        print(f"          KS Test Statistic: {ks_statistic}")
        print(f"                 KS p-value: {ks_p_value}")
        print(f" Anderson-Darling Statistic: {ad_stat}")
        print(f"        Anderson-Darling p : {ad_p_value}")
        print(f"              Anderson Level: {ad_significance}")
        print("-----------------------------------------------------------")
        print("")

        # (your plotting code unchanged...)

    col_names = [
        "Subcluster", "z_min", "z_max", "Mean", "Std Dev",
        "KS Stat", "KS p-value",
        "AD Stat", "AD p-value",
        "Anderson Level", "N"
    ]
    stats_table = Table(rows=stats_list, names=col_names)
    print(stats_table)
    return stats_table, stats_values


# -- Build Regions --
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

def get_bisectors(subcluster_configs):
    """
    Compute all pairwise great-circle bisectors between centers (SkyCoord objects).

    Each bisector is the unique great circle that is **perpendicular** (at every point)
    to the arc connecting two BCGs (Brightest Cluster Galaxies) on the celestial sphere.

    For N centers, this function returns N(N-1)/2 bisectors. Each bisector divides the sky
    into two hemispheres: each BCG in the pair is assigned to the hemisphere containing itself.

    Parameters
    ----------
    subcluster_configs : list of dict
        List of subcluster configuration dictionaries, each containing at least a 'center' key with a SkyCoord value.

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
    centers = [sub['center'] for sub in subcluster_configs]

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

def define_bbox(subcluster_configs, margin_frac):
    """
    Compute a bounding box in RA/Dec that encloses all centers, with optional padding.

    Parameters
    ----------
    subcluster_configs : list of dict
        List of subcluster configuration dictionaries, each containing at least a 'center' key with a SkyCoord value.
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
    centers = [sub['center'] for sub in subcluster_configs]

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

def build_bcg_signatures(subcluster_configs, bisectors):
    """
    For each BCG, compute its signature vector with respect to all bisectors.
    
    Each entry is:
        +1 : BCG is on the positive side of the bisector (as defined by bisector pole)
        -1 : BCG is on the negative side
         0 : BCG not involved in this bisector

    Parameters
    ----------
    subcluster_configs : list of dict
        List of subcluster configuration dictionaries, each containing at least a 'center' key with a SkyCoord value.
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
    centers = [sub['center'] for sub in subcluster_configs]
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

# -- Subcluster Member Assignment --
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
        
        fig, ax = plt.subplots(figsize=(6, 6))
        scatter = ax.scatter(df_valid['RA'], df_valid['Dec'], c=region_ids, cmap='tab10', s=20, label='Galaxies')
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
            print(f"\n------------------------------------------------------")
            print(f"Filtering Region {i+1} Spectroscopic Members")
            print(f"  Redshift range: {zmin:.4f} - {zmax:.4f}")
            print(f"      Radial cut: {max_radius} arcmin")
            z = df['z'].values
            has_z = np.isfinite(z)
            in_z = has_z & (z >= zmin) & (z <= zmax)
            keep = (sep < max_radius) & (in_z | ~has_z)
        else:
            print(f"\n------------------------------------------------------")
            print(f"Filtering Region {i+1} Photometric Members")
            print(f"      Radial cut: {max_radius} arcmin")
            keep = sep < max_radius


        print(f"        Region {i+1}: {keep.sum()} of {len(df)} members kept after radius/z cut")

        filtered.append(df[keep].copy())

    return filtered

# -- Plotting Helpers --

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

    def _seg_key_exact(seg):
        """Order-independent exact key for a segment ((ra1,dec1),(ra2,dec2))."""
        return frozenset((seg[0], seg[1]))


    if combined_indices is None:
        colors = [sub.get('color') for sub in subcluster_configs]
    else:
        colors = [sub['group_color'] for sub in subcluster_configs]

    centers = [sub['center'] for sub in subcluster_configs]
    radii = [sub['radius'] for sub in subcluster_configs]

    # Build region segment list for each BCG (segment = pair of (RA, Dec))
    bcg_regions = assign_subcluster_regions(subcluster_configs)
    arc_points_dict = {}
    bcg_intersections = defaultdict(list)

    # Working copy we’ll use for plotting (original bcg_regions remains intact)
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
            print(f"Polygon creation failed: {e}")



# -- Figures and Plots --
def plot_subcluster_members_and_regions(
    cluster,
    subcluster_configs,
    spec_groups,
    combined_configs=None,
    spec_groups_combined=None,
    phot_groups=None,
    phot_groups_combined=None,
    fig=None,
    ax=None,
    legend_loc='upper right',
    combined_indices=None,
    show_legend=True,
    show_plots=True,
    save_plots=False,
    save_path=None,
    **kwargs
):
    """
    Plot subcluster member galaxies and bisector-defined region arcs on a WCS optical image.

    Parameters
    ----------
    cluster : Cluster
        Cluster object (must provide .get_optical_image(), .xray_file, .phot_file(), .bcg_file).
    subcluster_configs : list of dict
        List of subcluster definitions. Each dict should contain:
            - 'center': SkyCoord
            - 'color': matplotlib color (optional)
            - ...additional keys as needed.
    spec_groups : list of pd.DataFrame
        Each DataFrame contains spectroscopic members for a subcluster.
    phot_groups : list of pd.DataFrame, optional
        Each DataFrame contains photometric members for a subcluster (if available).
    fig : matplotlib.figure.Figure, optional
        Existing figure to plot on. Created if None.
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on. Created if None.
    legend_loc : str, optional
        Legend location [default: 'upper right'].
    show_legend : bool, optional
        Whether to display the legend [default: True].
    show_plots : bool, optional
        Whether to display the plot with plt.show() [default: True].
    save_plots : bool, optional
        Whether to save the plot [default: False].
    save_path : str or None, optional
        Path or directory for saving figure if save_plots is True.
    **kwargs :
        Passed to region arc plotting and scatter plotting (e.g., linewidths).

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes.Axes
        Axes with plotted subclusters and regions.
    """
    subcluster_kwargs = pop_prefixed_kwargs(kwargs, 'subcluster')

    if combined_configs is not None:
        colors = [sub.get('color') for sub in combined_configs]
    else:
        colors = [sub.get('color') for sub in subcluster_configs]

    if spec_groups_combined is not None:
        spec_groups = spec_groups_combined
    if phot_groups_combined is not None:
        phot_groups = phot_groups_combined

    optical_image_file = cluster.get_optical_image()

    # Load WCS
    with fits.open(optical_image_file) as hdul:
        wcs_optical = WCS(hdul[0].header, naxis=2)

    # Set up fig/ax
    if ax is None:
        fig = plt.figure(figsize=(7, 7)) if fig is None else fig
        ax = fig.add_subplot(111, projection=wcs_optical)

    fig, ax = plot_optical(
        optical_image_file,
        cluster=cluster,
        fig=fig, ax=ax,
        show_plots=False,
        save_plots=False,
        xray_fits_file=cluster.xray_file,
        photometric_file=cluster.get_phot_file(),
        bcg_file=cluster.bcg_file,
        **subcluster_kwargs
    )

    # Subcluster member points
    scatter_handles = []
    labels = []
    for i, spec in enumerate(spec_groups):
        color = colors[i] if colors[i] is not None else f"C{i}"
        sc1 = ax.scatter(spec['RA'], spec['Dec'], s=12,
                         edgecolors=color, facecolors=color, linewidths=1.2, zorder=11,
                         transform=ax.get_transform('icrs'))
        scatter_handles.append(sc1)
        labels.append(f"Subcluster {i+1}")
        if phot_groups is not None:
            phot = phot_groups[i]
            ax.scatter(phot['RA'], phot['Dec'], s=12,
                        edgecolors=color, facecolors=color, linewidths=1.2, zorder=11,
                        transform=ax.get_transform('icrs'))

    # spec_df = pd.read_csv(cluster.spec_file)
    # # Ensure numeric dtypes (coerce bad entries to NaN so they won't match)
    # for col in ["RA", "Dec", "z"]:
    #     spec_df[col] = pd.to_numeric(spec_df[col], errors="coerce")

    # mask_1 = spec_df["z"].between(0.480, 0.495, inclusive="both")
    # mask_2 = spec_df["z"].between(0.493, 0.510, inclusive="both")

    # print("mask_1 counts:", mask_1.sum(), "mask_2 counts:", mask_2.sum())

    # ax.scatter(spec_df.loc[mask_1, 'RA'], spec_df.loc[mask_1, 'Dec'], s=18,
    #                      edgecolors='tab:orange', facecolors='tab:orange', linewidths=1.2, zorder=11,
    #                      transform=ax.get_transform('icrs'))
    # ax.scatter(spec_df.loc[mask_2, 'RA'], spec_df.loc[mask_2, 'Dec'], s=10,
    #                      edgecolors='tab:green', facecolors='tab:green', linewidths=1.2, zorder=12,
    #                      transform=ax.get_transform('icrs'))
    # ax.scatter(spec_df["RA"], spec_df["Dec"], s=8,
    #                      edgecolors='lightgray', facecolors='none', linewidths=0.8, zorder=12,
    #                      transform=ax.get_transform('icrs'))

    # RA_phot, Dec_phot, Lum_phot = load_photo_coords(cluster.phot_file())
    # ax.scatter(RA_phot, Dec_phot, s=12, marker='x',
    #                      edgecolors='red', facecolors='red', linewidths=1.2, zorder=11,
    #                      transform=ax.get_transform('icrs'))

    # Subcluster region arcs
    bisectors = get_bisectors(subcluster_configs)
    bcg_signatures = build_bcg_signatures(subcluster_configs, bisectors)
    plot_bcg_region_arcs(
        ax,
        subcluster_configs,
        bisectors,
        bcg_signatures,
        combined_indices=combined_indices,
        transform=ax.get_transform('icrs'),
        **kwargs
    )

    if show_legend:
        l = ax.legend(handles=scatter_handles, labels=labels, loc=legend_loc, fontsize=12, frameon=True)
        l.set_zorder(20)

    if save_plots and save_path is not None:
        plt.tight_layout()
        save_file = os.path.join(save_path, "subcluster_regions.pdf") if os.path.isdir(save_path) else save_path
        fig.savefig(save_file, bbox_inches='tight')
    if show_plots:
        plt.tight_layout()
        plt.show()

    return fig, ax

def plot_redshift_and_subclusters_figure(
    cluster,
    subcluster_configs,
    spec_groups,
    phot_groups,
    combined_indices=None,
    combined_configs=None,
    spec_groups_combined=None,
    phot_groups_combined=None,
    fig=None, ax1=None, ax2=None,
    legend_loc='upper right',
    legend_loc_top=None,
    legend_loc_bottom=None,
    show_plots=True,
    save_plots=False,
    save_path=None,
    layout='vertical',   # 'vertical' (default) or 'horizontal'
    **kwargs
):
    """
    Plot a two-panel figure showing:
        1. Redshift-selected cluster members overlaid on the optical image (with X-ray and density contours).
        2. Subcluster assignments, with member galaxies and bisector-defined region arcs.

    Optionally arranges panels horizontally or vertically.
    Parameters
    ----------
    cluster : Cluster
        Cluster object (must have .spec_file, .xray_file, .phot_file(), .bcg_file, etc.)
    subcluster_configs : list of dict
        Each dict must have at least:
            - 'center': SkyCoord
            - 'radius': float (arcmin)
            - 'color': matplotlib color (optional, for region and members)
            - 'label': str (optional, for legend)
        Can include any additional config keys for advanced use.
    spec_groups, phot_groups : list of pd.DataFrame
        List of DataFrames with cluster member galaxies for each subcluster (matched in order to subcluster_configs).
    fig, ax1, ax2 : optional
        Optionally provide pre-existing Figure and Axes.
    legend_loc : str
        Location for the subcluster legend on the lower panel.
    show_plots, save_plots : bool
        Whether to display or save the plot.
    save_path : str or None
        Path to save the figure (if save_plots is True).
    **plot_kwargs :
        Any extra keyword arguments are passed to downstream plotting functions, including:
            - density_contour options
            - xray_contour options
            - colorbar options
            - region arc options
            - etc.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax1, ax2 : matplotlib.axes.Axes
        The axes for the two panels.
    sc : PathCollection
        The redshift scatter artist from ax1 (for colorbar).
    
    """

    if legend_loc_bottom is None:
        legend_loc_bottom = legend_loc
    if legend_loc_top is None:
        legend_loc_top = legend_loc

    optical_image_file = cluster.get_optical_image()

    # -- Load optical images and WCS --
    with fits.open(optical_image_file) as hdul:
        wcs_optical = WCS(hdul[0].header, naxis=2)


    colorbar = False
    # -- Setup figure and axes --
    if ax1 is None and ax2 is None:
        if layout == 'horizontal':
            fig = plt.figure(figsize=(16, 8))
            gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.01)
            ax2 = fig.add_subplot(gs[0], projection=wcs_optical)  # left: subclusters
            ax1 = fig.add_subplot(gs[1], projection=wcs_optical, sharey=ax2)  # right: redshift overlay
        else:
            fig = plt.figure(figsize=(8, 12))
            gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.08)
            ax1 = fig.add_subplot(gs[0], projection=wcs_optical)  # top: redshift overlay
            ax2 = fig.add_subplot(gs[1], projection=wcs_optical, sharex=ax1)
        colorbar = True

    # ---------------- Top Panel: Optical image with scalebar and legend ----------------

    fig, ax1, sc = plot_redshift_overlay(
                                        optical_image_file=optical_image_file,
                                        spectroscopy_file=cluster.spec_file,
                                        cluster=cluster,
                                        ax=ax1, fig=fig,
                                        z_low=cluster.z_min, z_high=cluster.z_max,
                                        xray_fits_file=cluster.xray_file,
                                        photometric_file=cluster.get_phot_file(),
                                        bcg_file=cluster.bcg_file,
                                        show_plots=False,
                                        save_plots=False,
                                        save_path=None,
                                        colorbar=False,
                                        density_linewidth = 0.5,
                                        xray_linewidth = 0.5,
                                        legend_loc=legend_loc_top,
                                        **kwargs
)


    if layout == 'horizontal':
        ax1.tick_params(axis='y', labelleft=False)
    else:
        ax1.tick_params(axis='x', labelbottom=False)

    # -- Colorbar placement --
    if colorbar:
        pos = ax1.get_position()
        if layout == 'horizontal':
            # Colorbar to right of ax1 (right panel)
            cbar_ax = fig.add_axes([pos.x1, pos.y0 + 0.02, 0.025, pos.height + 0.02])
        else:
            # Colorbar to right of ax1 (top panel)
            cbar_ax = fig.add_axes([pos.x1 + 0.05, pos.y0 + 0.022, 0.04, pos.height + 0.058])
        cbar = fig.colorbar(sc, cax=cbar_ax)
        cbar.set_label("Redshift (z)")
        cbar.locator = MultipleLocator(0.005)
        cbar.formatter = FormatStrFormatter('%.3f')
        cbar.update_ticks()

    
    # ----- Bottom Panel: Subcluster members and regions -----
    fig, ax2 = plot_subcluster_members_and_regions(
                                    cluster=cluster,
                                    subcluster_configs=subcluster_configs,
                                    combined_configs=combined_configs,
                                    spec_groups=spec_groups,
                                    phot_groups=phot_groups,
                                    spec_groups_combined=spec_groups_combined,
                                    phot_groups_combined=phot_groups_combined,
                                    combined_indices=combined_indices,
                                    fig=fig, ax=ax2,
                                    legend_loc="upper right",
                                    show_plots=False,
                                    save_plots=False,
                                    save_path=os.path.join(cluster.image_path, "subcluster_map.pdf"),
                                    **kwargs
    )


    if layout == 'horizontal':
        fig.subplots_adjust(wspace=0.04, left=0.09, right=0.88, top=0.96, bottom=0.09)
    else:
        fig.subplots_adjust(hspace=0.08, top=0.96, bottom=0.07, left=0.08, right=0.95)


    if save_path is not None and save_plots:
        save_file = os.path.join(save_path, "subcluster_z_scatter.pdf") if os.path.isdir(save_path) else save_path
        fig.savefig(save_file, bbox_inches='tight')
    if show_plots:
        plt.show()

    return fig, ax1, ax2, sc

def plot_stacked_redshift_histograms(
    cluster,
    subcluster_configs,
    spec_groups,
    bins=24,
    combined_color='darkorange',
    show_plots=True,
    save_plots=False,
    save_path=None
):
    """
    Plot vertically stacked redshift histograms for each subcluster, with velocity dispersion annotations
    and a combined redshift histogram for all subclusters, plus a zoomed-in histogram of all redshifts
    in a broader range. All histograms share consistent axes for direct comparison.

    Also outputs a LaTeX table of velocity dispersions for each subcluster and a stacked velocity histogram.

    Parameters
    ----------
    cluster : Cluster
        Cluster object with necessary metadata (spec_file, z_min, z_max, etc).
    subcluster_configs : list of dict
        Subcluster definitions (should include at least 'color', 'label', and optionally 'z_bcg' and 'bcg_label').
    spec_groups : list of pd.DataFrame
        Each DataFrame contains spectroscopic members for a subcluster (must include 'z').
    bins : int, optional
        Number of bins for histograms [default: 24].
    combined_color : str, optional
        Color for combined histogram [default: 'darkorange'].
    show_plots : bool, optional
        If True, display the figure [default: True].
    save_plots : bool, optional
        If True, save the figure as PDF [default: False].
    save_path : str, optional
        Path or directory for saving figure. If a directory, will be saved as "stacked_redshifts.pdf".

    Returns
    -------
    None
        Displays and/or saves the plot.
    """

    # Load spectroscopic data
    spec_df = pd.read_csv(cluster.spec_file)
    z = spec_df['z'].values

    # Prepare data per subcluster
    colors = [sub.get('color') for sub in subcluster_configs]
    z_data, vel_data = analyze_group(spec_groups)

    # Filter redshifts
    z_low = 0.0
    z_high = 1.5
    cluster_low = cluster.z_min
    cluster_high = cluster.z_max

    mask = (z_low < z) & (z < z_high)
    z_subset = z[mask].reshape(-1, 1)
    z_inset_mask = (z > cluster_low) & (z < cluster_high)
    z_inset = z[z_inset_mask]
    print(f"Total galaxies in cluster range ({cluster_low:.3f} to {cluster_high:.3f}): {len(z_inset)}")

    # Format figure
    n_subclusters = len(z_data)
    all_z = np.concatenate(z_data)
    zmin, zmax = all_z.min(), all_z.max()

    hist_bins = np.linspace(zmin, zmax, bins)

    # Get max count for each histogram
    y_max_list = []
    for z_vals in z_data:
        counts, _ = np.histogram(z_vals, bins=hist_bins)
        y_max_list.append(counts.max())
        
    y_max_list = [2 if x == 1 else x for x in y_max_list] # Prevent 1-member panels from collapsing

    # Reverse the order to plot Subcluster 1 on top
    z_data = z_data[::-1]
    vel_data = vel_data[::-1]
    y_max_list = y_max_list[::-1]
    colors = colors[::-1]
    subcluster_configs = subcluster_configs[::-1]

    # For the combined plot
    combined_counts, _ = np.histogram(all_z, bins=hist_bins)
    y_max_combined = combined_counts.max()

    ratio = y_max_list / y_max_combined
    ratio = ratio.tolist()
    height_ratios = ratio + [1.0, 1.0]

    # Figure size scales with number of subclusters
    fig_height = 5 + 5 * n_subclusters / 3


    fig = plt.figure(figsize=(6, fig_height), constrained_layout=True)
    gs = fig.add_gridspec(n_subclusters + 2, 1, height_ratios=height_ratios)
    axes = [fig.add_subplot(gs[i]) for i in range(n_subclusters + 2)]



    # Subcluster histograms
    for i, (z_vals, (_, z_mean, sigma_v)) in enumerate(zip(z_data, vel_data)):
        color = colors[i]
        if color == 'white':
            color = 'gainsboro'
        z_bcg = subcluster_configs[i].get('z_bcg', None)

        ax = axes[i]
        ax.hist(z_vals, bins=hist_bins, color=color, edgecolor='black', alpha=1.0)
        ax.set_ylim(0, y_max_list[i] * 1.1)
        ax.set_xticks([])
        ax.tick_params(labelbottom=False)
        # Use consistent tick spacing, e.g., every 4.     #Probably a param option
        # ax.yaxis.set_major_locator(MultipleLocator(4))
        ax.set_xlim(hist_bins[0]-0.002, hist_bins[-1]+0.002)


        # Hide the '0' label, but keep the tick
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: '' if x == 0 else f'{int(x)}'))

        # Plot mean redshift line
        ax.axvline(z_mean, color='black', linestyle='-', lw=1.6, label=f'$\\bar{{z}}$= {z_mean:.5f}')
        
        # Plot BCG redshift line if available
        if z_bcg is not None:
            ax.axvline(z_bcg, color='tab:red', linestyle='-', lw=1.6, label=f'BCG {subcluster_configs[i]["bcg_label"]}')

        ax.yaxis.set_major_locator(MaxNLocator(integer=True))


        ax.annotate(
                f"Subcluster {n_subclusters - i}\nN = {len(z_vals)}\n$\\sigma_v$ = {sigma_v:.0f} km/s",
                xy=(0, 1), xycoords='axes fraction',  # top-left corner of Axes
                xytext=(6, -6), textcoords='offset points',  # shift right and down
                ha='left', va='top',
                fontsize=10,
                bbox=dict(facecolor='white', edgecolor='gray', alpha=0.7)
                )


        ax.legend(loc='upper right', fontsize=10, frameon=True, framealpha=0.8)

    fig.get_layout_engine().set(hspace=0.0, h_pad =0.0)

    # Combined histogram
    ax_comb = axes[-2]
    ax_comb.hist(all_z, bins=hist_bins, color=combined_color, edgecolor='black', alpha = 1.0)

    ax_comb.set_ylim(0, y_max_combined * 1.1)

    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    ax_comb.xaxis.set_major_locator(MultipleLocator(0.005))
    ax_comb.xaxis.set_major_formatter(FormatStrFormatter('%.3f')) 
    ax_comb.tick_params(axis='x', which='major', pad=6)
    ax_comb.set_xlabel(' ',fontsize=6)

    colors = colors[::-1]
    subcluster_configs = subcluster_configs[::-1]


    for z_bcg, i in zip([subcluster_configs[i].get('z_bcg', None) for i in range(n_subclusters)], range(n_subclusters)):
        color = colors[i]
        if color == 'white':
            color = 'gainsboro'
        if z_bcg is not None:
            ax_comb.axvline(z_bcg, color=color, linestyle='-', lw=1.6, label=f'BCG {subcluster_configs[i]["bcg_label"]}')


    ax_comb.annotate(
                f"All Subclusters\nN = {len(all_z)}",
                xy=(0, 1), xycoords='axes fraction',  # top-left corner of Axes
                xytext=(6, -6), textcoords='offset points',  # shift right and down
                ha='left', va='top',
                fontsize=10,
                bbox=dict(facecolor='white', edgecolor='gray', alpha=0.7)
                )


    ax_comb.set_xlim(hist_bins[0]-0.002, hist_bins[-1]+0.002)
    ax_comb.legend(loc='upper right', fontsize=10, frameon=True, framealpha=0.8)

    fig.supylabel("Galaxy Counts", fontsize=16)
    axes[0].set_title(f" ", fontsize=6)

    ax_all_z = axes[-1]

    # Histogram of redshifts in zoom region
    ax_all_z.hist(z_subset, bins=80, density=False, alpha=0.6, color='b', edgecolor='black')
    ax_all_z.set_xlim(z_low, z_high)
    ax_all_z.set_xlabel('Redshift (z)')

    # Inset for cluster members
    axins = inset_axes(ax_all_z, width=1.5, height=1.5, loc='upper right', borderpad=1.5)

    axins.hist(z_inset, bins=12, color='darkorange', edgecolor='black')
    axins.set_title("Cluster Members", fontsize=10)
    axins.tick_params(labelsize=8)
    axins.set_xlabel('z', fontsize=9)
    axins.set_ylabel('Counts', fontsize=9)
    axins.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.3f}"))
    axins.yaxis.set_major_locator(MaxNLocator(integer=True))

    fig.canvas.draw() # Need to draw before making connection lines

    # Connection lines
    main_xlim = (cluster_low, cluster_high)
    main_ylim = ax_all_z.get_ylim()
    inset_xlim = axins.get_xlim()
    inset_ylim = axins.get_ylim()


    con1 = ConnectionPatch(xyA=(main_xlim[0], main_ylim[0]), coordsA=ax_all_z.transData,
                        xyB=(inset_xlim[0], inset_ylim[0]), coordsB=axins.transData,
                        color='gray', linewidth=1,zorder=3)
    con2 = ConnectionPatch(xyA=(main_xlim[1], main_ylim[0]), coordsA=ax_all_z.transData,
                        xyB=(inset_xlim[1], inset_ylim[0]), coordsB=axins.transData,
                        color='gray', linewidth=1,zorder=3)

    for con in [con1, con2]:
        ax_all_z.add_artist(con)


    if save_plots and save_path is not None:
        save_file = os.path.join(save_path, "stacked_redshifts.pdf") if os.path.isdir(save_path) else save_path
        fig.savefig(save_file, bbox_inches='tight')
    if show_plots:
        plt.show()

    plot_stacked_velocity_histograms(vel_data,color=colors,show_plots=show_plots, save_plots=save_plots)

def plot_redshift_histogram_heatmap(cluster, legend_loc="lower right", fig=None, ax1=None, ax2=None, legend_loc_bottom=None, show_plots=True, save_plots=False, save_path=None, cmap="viridis", **kwargs):

    if legend_loc_bottom is None:
        legend_loc_bottom = legend_loc

    optical_image_file = cluster.get_optical_image()

    # -- Load optical images and WCS --
    with fits.open(optical_image_file) as hdul:
        wcs_optical = WCS(hdul[0].header, naxis=2)


    colorbar = False
    # -- Setup figure and axes --
    if ax1 is None and ax2 is None:
        fig = plt.figure(figsize=(6, 9))
        gs = fig.add_gridspec(2, 1, height_ratios=[0.52, 1], hspace=0.32)
        ax1 = fig.add_subplot(gs[1], projection=wcs_optical)  # bottom: redshift overlay
        ax2 = fig.add_subplot(gs[0]) # top: histogram
        # fig.subplots_adjust(top=0.94, bottom=0.07, left=0.11, right=0.97)
        # fig.subplots_adjust(top=0.96, bottom=0.1, left=0.15, right=0.97)
        
        colorbar = True

    # ---------------- Bottom Panel: Optical image with scalebar and legend ----------------

    fig, ax1, handles, labels, sc = plot_redshift_overlay(
                                        optical_image_file=optical_image_file,
                                        spectroscopy_file=cluster.spec_file,
                                        cluster=cluster,
                                        ax=ax1, fig=fig,
                                        z_low=cluster.z_min, z_high=cluster.z_max,
                                        xray_fits_file=cluster.xray_file,
                                        photometric_file=cluster.get_phot_file(),
                                        bcg_file=cluster.bcg_file,
                                        scatter_cmap=cmap,
                                        show_legend=False,
                                        show_plots=False,
                                        save_plots=False,
                                        save_path=None,
                                        colorbar=False,
                                        density_linewidth = 0.8,
                                        xray_linewidth = 0.8,
                                        legend_loc=legend_loc_bottom,
    )


    ax1.set_aspect('equal') 
    l = ax1.legend(handles=handles, labels=labels, loc=legend_loc, fontsize=10, frameon=True)
    l.set_zorder(200)

    fig.subplots_adjust(top=0.98, bottom=0.1, left=0.2, right=0.94)
    # -- Colorbar placement --
    if colorbar:
        pos = ax1.get_position()
        # Colorbar above ax1 (bottom panel)
        cbar_ax = fig.add_axes([
            pos.x0,              # left edge
            pos.y1 + 0.04,       # slightly above top edge
            pos.width,           # width of axes
            0.015                # thickness of cbar
        ])
        cbar = fig.colorbar(sc, cax=cbar_ax, orientation='horizontal')
        
        cbar.locator = MultipleLocator(0.005)
        cbar.formatter = FormatStrFormatter('%.3f')
        cbar.update_ticks()

    # Load spectroscopic data
    spec_df = pd.read_csv(cluster.spec_file)
    z = spec_df['z'].values

    # Filter redshifts
    z_low = 0.0
    z_high = 1.5
    cluster_low = cluster.z_min
    cluster_high = cluster.z_max

    mask = (z_low < z) & (z < z_high)
    z_subset = z[mask].reshape(-1, 1)
    z_inset_mask = (z > cluster_low) & (z < cluster_high)
    z_inset = z[z_inset_mask]
    print(f"Total galaxies in cluster range ({cluster_low:.3f} to {cluster_high:.3f}): {len(z_inset)}")


    # Colormap for inset histogram
    cmap = plt.get_cmap(cmap) 

    zmin_cluster = np.min(z_inset)
    zmax_cluster = np.max(z_inset)

    # Normalize all to the histogram range
    norm_hist = mpl.colors.Normalize(vmin=zmin_cluster, vmax=zmax_cluster)
    color_location=zmin_cluster + (zmax_cluster - zmin_cluster)/4
    hist_color = cmap(norm_hist(color_location))
    

    # Full histogram of redshifts
    ax2.hist(z_subset, bins=80, density=False, alpha=1, color=hist_color, edgecolor='black')

    ax2.set_xlim(z_low, z_high)
    ax2.set_xlabel('Redshift (z)', labelpad=8)
    ax2.set_ylabel('Galaxy Counts', labelpad=36)


    pos = ax1.get_position()
    ax2.set_position([pos.x0, ax2.get_position().y0, pos.width, ax2.get_position().height])

    # Inset for cluster members
    axins = inset_axes(ax2, width=1.5, height=1.5, loc='upper right', borderpad=1.5)

    # Plot the histogram and get patches
    h = axins.hist(z_inset, bins=12, edgecolor='black')

    # Color each bar
    bin_centers = 0.5 * (h[1][:-1] + h[1][1:])
    colors = cmap(norm_hist(bin_centers))
    for patch, color in zip(h[2], colors):
        patch.set_facecolor(color)

    axins.set_title("Cluster Members", fontsize=10)
    axins.tick_params(labelsize=8)
    axins.set_xlabel('z', fontsize=9)
    axins.set_ylabel('Counts', fontsize=9)
    axins.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.3f}"))
    axins.yaxis.set_major_locator(MaxNLocator(integer=True))

    fig.canvas.draw() # Need to draw before making connection lines

    # Connection lines
    main_xlim = (cluster_low, cluster_high)
    main_ylim = ax2.get_ylim()
    inset_xlim = axins.get_xlim()
    inset_ylim = axins.get_ylim()


    con1 = ConnectionPatch(xyA=(main_xlim[0], main_ylim[0]), coordsA=ax2.transData,
                        xyB=(inset_xlim[0], inset_ylim[0]), coordsB=axins.transData,
                        color='gray', linewidth=1,zorder=3)
    con2 = ConnectionPatch(xyA=(main_xlim[1], main_ylim[0]), coordsA=ax2.transData,
                        xyB=(inset_xlim[1], inset_ylim[0]), coordsB=axins.transData,
                        color='gray', linewidth=1,zorder=3)

    for con in [con1, con2]:
        ax2.add_artist(con)


    if save_plots and save_path is not None:
        save_file = os.path.join(save_path, "redshifts_hist_hmap.pdf") if os.path.isdir(save_path) else save_path
        fig.savefig(save_file, dpi=450)
    if show_plots:
        plt.show()


def plot_2panel_optical_contours(
        cluster,
        subcluster_configs,
        fig=None, ax1=None, ax2=None,
        legend_loc='upper right',
        show_plots=True,
        save_plots=False,
        save_path=None,
        layout='vertical', 
        **kwargs
):
    """
    Plot a two-panel figure showing:
        1. Optical image, with no overlays.
        2. Optical image with xray and density contours.

    Optionally arranges panels horizontally or vertically.
    Parameters
    ----------
    cluster : Cluster
        Cluster object (must have .spec_file, .xray_file, .phot_file(), .bcg_file, etc.)
    subcluster_configs : list of dict
        Future use
    fig, ax1, ax2 : optional
        Optionally provide pre-existing Figure and Axes.
    legend_loc : str
        Location for the subcluster legend on the upper/ left panel.
    show_plots, save_plots : bool
        Whether to display or save the plot.
    save_path : str or None
        Path to save the figure (if save_plots is True).
    **plot_kwargs :
        Any extra keyword arguments are passed to downstream plotting functions, including:
            - density_contour options
            - xray_contour options
            - colorbar options
            - region arc options
            - etc.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax1, ax2 : matplotlib.axes.Axes
        The axes for the two panels.
    """


    optical_image_file = cluster.get_optical_image()

    # -- Load optical images and WCS --
    with fits.open(optical_image_file) as hdul:
        wcs_optical = WCS(hdul[0].header, naxis=2)

    # -- Setup figure and axes --
    if ax1 is None and ax2 is None:
        if layout == 'horizontal':
            fig = plt.figure(figsize=(14, 8))
            gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.01)
            ax1 = fig.add_subplot(gs[0], projection=wcs_optical)  # left: optical
            ax2 = fig.add_subplot(gs[1], projection=wcs_optical, sharey=ax1)  # right: contours
        else:
            fig = plt.figure(figsize=(6, 10.5))
            gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.02)
            ax1 = fig.add_subplot(gs[0], projection=wcs_optical)  # top: optical
            ax2 = fig.add_subplot(gs[1], projection=wcs_optical, sharex=ax1)  # bottom: contours

    # ---------------- Top (Left) Panel: Optical image with scalebar and legend ----------------

    fig, ax1 = plot_optical(
        optical_image_file,
        cluster=cluster,
        fig=fig, ax=ax1,
        show_plots=False,
        save_plots=False,
        legend_loc=legend_loc,
        **kwargs
    )


    if layout == 'horizontal':
        ax2.tick_params(axis='y', labelleft=False)
    else:
        ax1.tick_params(axis='x', labelbottom=False)



    # ----- Bottom (Right) Panel: Optical image with contours -----

    fig, ax2, handles, labels = plot_optical(
        optical_image_file,
        cluster=cluster,
        fig=fig, ax=ax2,
        show_plots=False,
        save_plots=False,
        xray_fits_file=cluster.xray_file,
        photometric_file=cluster.get_phot_file(),
        bcg_file=cluster.bcg_file,
        show_legend=False,
        show_scalebar=False,
        **kwargs
    )
    l = ax1.legend(handles=handles, labels=labels, loc=legend_loc, fontsize=10, frameon=True)
    l.set_zorder(20)

    if layout == 'horizontal':
        fig.subplots_adjust(wspace=0.04, left=0.09, right=0.92, top=0.96, bottom=0.09)
    else:
        fig.subplots_adjust(hspace=0.08, top=0.98, bottom=0.1, left=0.2, right=0.94)


    # plt.tight_layout()
    if save_path is not None and save_plots:
        save_file = os.path.join(save_path, "subcluster_z_scatter.pdf") if os.path.isdir(save_path) else save_path
        fig.savefig(save_file, dpi=450)
    if show_plots:
        plt.show()

    return fig, ax1, ax2

def plot_3panel_optical_subclusters_figure(
    cluster,
    subcluster_configs,
    spec_groups,
    phot_groups,
    combined_configs=None,
    spec_groups_combined=None,
    phot_groups_combined=None,
    combined_indices=None,
    fig=None,
    axes=None,
    layout='vertical', 
    show_legend=True,
    legend_loc='upper right',
    show_plots=True,
    save_plots=False,
    save_path=None,
    **kwargs
):
    """
    Plot a 3-panel figure (vertical or horizontal) for cluster images, using custom plot_optical overlays.

    Panels:
        1. Plain optical image (with optional overlays if provided via kwargs)
        2. Optical image + overlays (returns handles/labels for legend)
        3. Optical image + overlays (for regions, contours, etc)

    Parameters
    ----------
    cluster : Cluster
        Cluster object (must provide .get_optical_image(), .xray_file, .phot_file(), .bcg_file).
    fig : matplotlib.figure.Figure, optional
        Existing Figure object to use (otherwise created).
    axes : list of Axes or None
        Existing list of axes [ax1, ax2, ax3] (otherwise created).
    layout : str, optional
        'vertical' (default) or 'horizontal' for 3-panel arrangement.
    show_legend : bool, optional
        Whether to show legend on panel 2 [default: True].
    legend_loc : str, optional
        Legend location for panel 2 [default: 'upper right'].
    show_plots : bool, optional
        Whether to display the plot [default: True].
    save_plots : bool, optional
        Whether to save the plot [default: False].
    save_path : str or None, optional
        Path or directory to save the figure if save_plots is True.
    **kwargs :
        Passed to all plot_optical calls (xray, bcg, contour, etc).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    axes : list of matplotlib.axes.Axes
        List of axes [ax1, ax2, ax3].
    handles, labels : list, list
        Scatter handles/labels from panel 2 for legend building.
    """


    optical_image_file = cluster.get_optical_image()

    # -- Load WCS for axis projection
    with fits.open(optical_image_file) as hdul:
        wcs_optical = WCS(hdul[0].header, naxis=2)

    # Figure/axes setup
    if axes is not None and len(axes) == 3:
        ax1, ax2, ax3 = axes
    else:
        if layout == 'horizontal':
            fig = plt.figure(figsize=(21, 7)) if fig is None else fig
            gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.07)
            ax1 = fig.add_subplot(gs[0], projection=wcs_optical)
            ax2 = fig.add_subplot(gs[1], projection=wcs_optical, sharey=ax1)
            ax3 = fig.add_subplot(gs[2], projection=wcs_optical, sharey=ax1)
        else:
            fig = plt.figure(figsize=(7, 18)) if fig is None else fig
            gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1], hspace=0.01)
            ax1 = fig.add_subplot(gs[0], projection=wcs_optical)
            ax2 = fig.add_subplot(gs[1], projection=wcs_optical, sharex=ax1)
            ax3 = fig.add_subplot(gs[2], projection=wcs_optical, sharex=ax1)
    axes = [ax1, ax2, ax3]

    # Panel 1: Plain (or minimally decorated) optical image
    fig, ax1 = plot_optical(
        optical_image_file,
        cluster=cluster,
        fig=fig, ax=ax1,
        show_plots=False,
        save_plots=False,
        **kwargs
    )

    # Panel 2: Optical + overlays, return handles/labels
    fig, ax2, handles, labels = plot_optical(
        optical_image_file,
        cluster=cluster,
        fig=fig, ax=ax2,
        show_plots=False,
        save_plots=False,
        xray_fits_file=cluster.xray_file,
        photometric_file=cluster.get_phot_file(),
        bcg_file=cluster.bcg_file,
        show_legend=False,
        show_scalebar=False,
        **kwargs
    )

    if show_legend:
        leg = ax1.legend(handles=handles, labels=labels, loc=legend_loc, fontsize=10, frameon=True)
        leg.set_zorder(20)

    # Panel 3: Subcluster members and regions
    fig, ax3 = plot_subcluster_members_and_regions(
                                cluster=cluster,
                                subcluster_configs=subcluster_configs,
                                combined_configs=combined_configs,
                                spec_groups=spec_groups,
                                spec_groups_combined=spec_groups_combined,
                                phot_groups=phot_groups,
                                phot_groups_combined=phot_groups_combined,
                                combined_indices=combined_indices,
                                fig=fig, ax=ax3,
                                legend_loc="upper right",
                                show_plots=False,
                                save_plots=False,
                                save_path=None,
                                subcluster_show_scalebar=False,
                                **kwargs
    )
    if layout == 'horizontal':
        ax2.tick_params(axis='y', labelleft=False)
        ax3.tick_params(axis='y', labelleft=False)
    else:
        ax1.tick_params(axis='x', labelbottom=False)
        ax2.tick_params(axis='x', labelbottom=False)

    # Adjust figure spacing
    if layout == 'horizontal':
        fig.subplots_adjust(wspace=0.01, hspace=0.01, top=0.95, bottom=0.1, left=0.08, right=0.98)
    else:
        fig.subplots_adjust(hspace=0.08, top=0.98, bottom=0.07, left=0.1, right=0.99)

    if save_plots and save_path is not None:
        save_file = os.path.join(save_path, "cluster_3panel.pdf") if os.path.isdir(save_path) else save_path
        fig.savefig(save_file, bbox_inches='tight')
    if show_plots:
        plt.show()

    return fig, axes, handles, labels

def plot_combined_4panel_figure(
                        cluster,
                        subcluster_configs,
                        spec_groups,
                        phot_groups,
                        combined_configs=None,
                        spec_groups_combined=None,
                        phot_groups_combined=None,
                        combined_indices=None,
                        legend_loc="upper left",
                        show_plots=True,
                        save_plots=False,
                        save_path=None
):
    """
    Plot a 2x2 figure with subcluster overlays, density/X-ray contours, and redshift information.

    The layout is as follows:
        - Top left:  Optical image with legend
        - Bottom left: Optical image with bcgs and contours
        - Top right:  Redshift-selected member heatmap
        - Bottom right: Subcluster region assignments and bisector arcs


    Parameters
    ----------
    cluster : Cluster
        Cluster object with necessary file references and metadata.
    subcluster_configs : list of dict
        Subcluster configuration dictionaries.
    spec_groups : list of pd.DataFrame
        List of spectroscopic member DataFrames for each subcluster.
    phot_groups : list of pd.DataFrame
        List of photometric member DataFrames for each subcluster.
    legend_loc : str, optional
        Legend location for overlays [default: 'upper left'].
    show_plots : bool, optional
        Whether to display the figure [default: True].
    save_plots : bool, optional
        Whether to save the figure to file [default: False].
    save_path : str or None, optional
        Path or directory to save the figure if save_plots is True.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created Figure object.

    Notes
    -----
    - Panels are created using `plot_2panel_optical_contours` and `plot_redshift_and_subclusters_figure`.
    """
    optical_image_file = cluster.get_optical_image()

    # Load WCS
    with fits.open(optical_image_file) as hdul:
        wcs_optical = WCS(hdul[0].header, naxis=2)

    fig = plt.figure(figsize=(13.4, 12))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1], wspace=0.0, hspace=0.04)

    ax1 = fig.add_subplot(gs[0, 0], projection=wcs_optical)
    ax2 = fig.add_subplot(gs[1, 0], projection=wcs_optical)
    ax3 = fig.add_subplot(gs[0, 1], projection=wcs_optical)
    ax4 = fig.add_subplot(gs[1, 1], projection=wcs_optical)

    # LEFT COLUMN
    plot_2panel_optical_contours(
                                cluster=cluster,
                                subcluster_configs=subcluster_configs,
                                spec_groups=spec_groups,
                                phot_groups=phot_groups,
                                fig=fig, ax1=ax1, ax2=ax2,
                                legend_loc=legend_loc,
                                layout="vertical",
                                show_plots=False,
                                save_plots=False,
                                save_path=None
    )

    # RIGHT COLUMN
    _,_,_, sc= plot_redshift_and_subclusters_figure(
                                cluster=cluster,
                                subcluster_configs=subcluster_configs,
                                combined_configs=combined_configs,
                                spec_groups=spec_groups,
                                phot_groups=phot_groups,
                                spec_groups_combined=spec_groups_combined,
                                phot_groups_combined=phot_groups_combined,
                                combined_indices=combined_indices,
                                fig=fig, ax1=ax3, ax2=ax4,
                                legend_loc="lower left",
                                legend_loc_bottom="upper right",
                                show_plots=False,
                                save_plots=False,
                                save_path=None,
                                layout='vertical'
    )


    ax3.set_ylabel('')
    ax3.tick_params(axis='y', labelleft=False, pad=0)
    ax3.legend_.remove()

    ax4.set_ylabel('')
    ax4.tick_params(axis='y', labelleft=False, pad=0)


    fig.subplots_adjust(left=0.085, right=0.90) 


    # After scatter plotted in ax3
    pos = ax3.get_position()
    cbar_ax = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.015, pos.height])
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label("Redshift (z)")
    cbar.locator = MultipleLocator(0.005)
    cbar.formatter = FormatStrFormatter('%.3f')
    cbar.update_ticks()


    if save_plots and save_path is not None:
        save_file = os.path.join(save_path, "combined_4panel.pdf") if os.path.isdir(save_path) else save_path
        fig.savefig(save_file, bbox_inches='tight')
    if show_plots:
        plt.show()
    return fig
  
def plot_subcluster_regions_and_histograms(
    cluster,
    subcluster_configs,
    spec_groups,
    phot_groups,
    spec_groups_combined=None,
    phot_groups_combined=None,
    combined_configs=None,
    combined_indices=None,
    legend_loc="upper right",
    show_plots=True,
    save_plots=False,
    save_path=None,
    **kwargs
):
    """
    Plot a stacked figure with subcluster redshift histograms and a spatial regions panel.

    This function creates a figure with N histogram panels (one per subcluster) showing the redshift
    distribution for each region, followed by a bottom panel that overlays subcluster member galaxies,
    region boundaries, and photometric members on the optical image of the cluster.

    - Histogram panel heights are scaled dynamically by galaxy count.


    Parameters
    ----------
    cluster : Cluster
        Cluster object with necessary metadata (spec_file, z_min, z_max, etc).
    subcluster_configs : list of dict
        Subcluster definitions (with 'color', 'center', and optional 'z_bcg', 'bcg_label', etc).
    spec_groups : list of pd.DataFrame
        Each DataFrame contains spectroscopic members for a subcluster.
    phot_groups : list of pd.DataFrame
        Each DataFrame contains photometric members for a subcluster.
    legend_loc : str, optional
        Legend location for the spatial region panel [default: 'upper right'].
    show_plots : bool, optional
        Whether to display the plot [default: True].
    save_plots : bool, optional
        Whether to save the plot to file [default: False].
    save_path : str or None, optional
        Path or directory to save figure if save_plots is True.
    **kwargs :
        Additional arguments passed to region/member overlay plotting.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created Figure object.
    """

    

    # Prepare data per subcluster
    if combined_configs is not None:
        colors = [sub.get('color') for sub in combined_configs]
        z_group = spec_groups_combined
        print("Using combined subcluster configurations for colors and data.")
    else:
        colors = [sub.get('color') for sub in subcluster_configs]
        z_group = spec_groups

    print(f"Colors: {colors}")





    z_data, vel_data = analyze_group(z_group)


    # outputs['tex_path'] and outputs['pairs_csv'] have the written files
    

    n_subclusters = len(z_data)
    print(f"Number of subclusters: {n_subclusters}")
    print(f"Subcluster configs: {subcluster_configs}")
    hist_bins = np.linspace(min(np.concatenate(z_data)), max(np.concatenate(z_data)), 24)
    y_max_list = [np.histogram(zs, bins=hist_bins)[0].max() for zs in z_data]
    y_max_list = [2 if val == 1 else val for val in y_max_list]
    ratio = (np.array(y_max_list) * n_subclusters) / (sum(y_max_list) * 6)
    height_ratios = ratio.tolist() + [1]  # image panel taller

    fig = plt.figure(figsize=(6, 6.75 + n_subclusters))
    gs = GridSpec(n_subclusters + 1, 1, height_ratios=height_ratios, hspace=0.0)

    ax_list = [fig.add_subplot(gs[i]) for i in range(0, n_subclusters)]
    ax_img = fig.add_subplot(gs[-1], projection=WCS(fits.getheader(cluster.get_optical_image(), 0), naxis=2))



        # ---------------- Top Panels: Histograms ----------------
    hist_bins = np.linspace(min(np.concatenate(z_data)), max(np.concatenate(z_data)), 24)
    group_labels = []
    for i, ax in enumerate(ax_list):
        if subcluster_configs[i].get('group_label') in group_labels:
            j = i+1
        else:
            j = i
            group_labels.append(subcluster_configs[i].get('group_label'))
        zs = z_data[i]
        z_mean = vel_data[i][1]
        sigma_v = vel_data[i][2]
        color = colors[i] if colors[i] != 'white' else 'gainsboro'
        z_bcg = subcluster_configs[j].get('z_bcg', None)
        bcg_label = subcluster_configs[j].get('group_label', f"{j+1}")

        ax.hist(zs, bins=hist_bins, color=color, edgecolor='black')
        ax.axvline(z_mean, color='black', lw=1.5, linestyle='-', label=f"$\\bar{{z}}$ = {z_mean:.4f}")
        if z_bcg is not None:
            ax.axvline(z_bcg, color='tab:red', lw=1.5, linestyle='-', label=f"BCG {bcg_label}")
        ax.set_xlim(hist_bins[0]-0.002, hist_bins[-1]+0.002)
        ax.set_ylim(0, y_max_list[i]*1.1)
        ax.legend(fontsize=9, loc='upper right')
        ax.yaxis.set_major_locator(MultipleLocator(2))
        # ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax.annotate(f"Subcluster {i+1}\nN = {len(zs)}\n$\\sigma_v$ = {round(sigma_v, -1):.0f} km/s",
                    xy=(0, 1), xycoords='axes fraction', xytext=(6, -6),
                    textcoords='offset points', ha='left', va='top',
                    fontsize=9, bbox=dict(facecolor='white', edgecolor='gray', alpha=0.7))
        if i < n_subclusters - 1:
            ax.tick_params(labelbottom=False)
            ax.set_xticks([])
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: '' if x == 0 else f'{int(x)}'))
            ax.tick_params(axis='x', which='major', pad=0)
        else:
            ax.set_xlabel("Redshift (z)")


    # ---------------- Top Panel: Contours + Regions ----------------
    plot_subcluster_members_and_regions(
                                    cluster=cluster,
                                    subcluster_configs=subcluster_configs,
                                    combined_configs=combined_configs,
                                    spec_groups=spec_groups,
                                    spec_groups_combined=spec_groups_combined,
                                    phot_groups=phot_groups,
                                    phot_groups_combined=phot_groups_combined,
                                    fig=fig, ax=ax_img,
                                    legend_loc=legend_loc,
                                    combined_indices=combined_indices,
                                    show_plots=False,
                                    save_plots=False,
                                    save_path=os.path.join(cluster.image_path, "subcluster_map.pdf"),
                                    **kwargs
    )


    ax_img.set_xlabel('')
    ax_img.set_xticklabels([])
    ax_img.set_ylabel('Decl.')
    ax_img.set_xlabel('R.A.')
    ax_img.set_aspect('equal')
    ax_img.set_title(" ", fontsize=6)
    
    fig.subplots_adjust(top=0.96, bottom=0.02, left=0.2, right=0.94)
    # Get position of top and bottom histogram axes in figure coordinates
    bbox_top = ax_list[0].get_position()
    bbox_bot = ax_list[-1].get_position()

    # Compute center between top and bottom axes
    y_c = (bbox_top.y1 + bbox_bot.y0) / 2
    x_l = bbox_top.x0 - 0.16 

    fig.text(x_l, y_c, "Galaxy Counts", va='center', ha='center', rotation='vertical')

    

    if save_plots and save_path is not None:
        save_file = os.path.join(save_path, "subcluster_histograms.pdf") if os.path.isdir(save_path) else save_path
        fig.savefig(save_file, dpi=450)
    if show_plots:
        plt.show()
    return fig

def print_subcluster_deluxetable(spec_groups):
    """
    Print a LaTeX deluxetable summarizing subcluster properties.

    Parameters
    ----------
    spec_groups : list[array-like]
        Per-subcluster arrays of member galaxy redshifts.
    """
    z_data, vel_data = analyze_group(spec_groups)

    print(r"\begin{deluxetable}{cccc}")
    print(r"\tablecaption{RMJ0000 Subcluster Properties}\label{tab:RMJ0000_subclusters}")
    print(r"\tablehead{")
    print(r"\colhead{Subcluster} & \colhead{N} & \colhead{Mean $z$} & \colhead{$\sigma_v$ [km/s]}")
    print(r"}")
    print(r"\centering")
    print(r"\startdata")

    n_subclusters = len(z_data)
    for i in range(n_subclusters):
        sub_num = i + 1
        num_members = len(z_data[i])
        _, z_mean, sigma_v = vel_data[i]
        print(f"{sub_num} & {num_members} & {z_mean:.4f} & {sigma_v:.0f} \\\\")

    print(r"\enddata")
    print(r"\end{deluxetable}")

def build_subcluster_summary(
    cluster,
    configs,
    spec_groups,
    update_csv=True,
):
    """
    Summarize subcluster kinematics and BCG offsets with minimal I/O logic.

    This function:
      - Reads BCGs.csv from cluster.bcg_file (requires columns: BCG_priority, z, sigma_z)
      - Maps each subcluster's bcg_id to the BCG row where BCG_priority == bcg_id
      - Computes subcluster stats via your velocity_dispersion(zs)
      - Computes BCG delta v relative to the subcluster mean and its uncertainty from sigma_z
      - Writes a LaTeX deluxetable (.tex) into cluster.cluster_path
      - Writes pairwise BCG-BCG delta v CSV (using z and sigma_z) into cluster.cluster_path
      - Optionally updates subclusters.csv with bcg_dv_kms and bcg_dv_err_kms

    Parameters
    ----------
    cluster : Cluster
        Cluster object with:
          - bcg_file : str  -> path to BCGs.csv
          - cluster_path : str -> output directory
          - identifier : str   -> e.g., "RMJ 2135"
          - subcluster_file : str (optional) -> path to subclusters.csv for updates
    configs : list[dict] or dict
        Each subcluster config must include 'bcg_id'. If a dict is provided,
        rows are ordered by sorted key.
    spec_groups : list[array-like]
        Per-subcluster arrays of member galaxy redshifts.
    update_csv : bool
        If True and cluster.subcluster_file exists, add/overwrite bcg_dv_kms and
        bcg_dv_err_kms columns.

    Returns
    -------
    outputs : dict
        {
          'tex_path': str,
          'pairs_csv': str,
          'bcg_dv_kms': list[float or None],
          'bcg_dv_err_kms': list[float or None],
          'vel_data': list[tuple],  # (velocities, z_mean, sigma_v) per subcluster
        }

    Notes
    -----
    - Delta v is computed in the standard cluster rest frame:
        dv = c * (z2 - z1) / (1 + z_mean)
      where z_mean = (z1 + z2)/2, c in km/s.
    - For the BCG vs subcluster mean, the uncertainty uses only sigma_z of the BCG:
        sigma = c * sigma_z / (1 + z_mean)
    """

    # TODO: This function is a dumpster fire and should be burned to the ground at my earliest convenience.

    # ------------------------
    # Local helpers
    # ------------------------
    def _ordered(cfgs):
        if isinstance(cfgs, dict):
            return [cfgs[k] for k in sorted(cfgs.keys())]
        return list(cfgs)

    def _los_dv(z1, z2):
        zbar = (z1 + z2)/2
        return c.to('km/s').value * (z2 - z1) / (1.0 + zbar), zbar

    def _los_dv_err_from_sigmaz(sigmaz, z_ref):
        # z_ref is the redshift in the denominator; here we use the subcluster mean
        return None if sigmaz is None or (isinstance(sigmaz, float) and np.isnan(sigmaz)) \
            else c.to('km/s').value * sigmaz / (1.0 + z_ref)

    def _write_deluxetable_tex(cluster_name, vel_data, z_lists, bcg_z_list, bcg_id_str, cluster_props, out_path):

        lines = []
        lines.append(r"\begin{deluxetable}{cccccc}")
        label_id = cluster_name.replace(" ", "")
        lines.append(rf"\caption{{{label_id} Subcluster Properties}}\label{{tab:{label_id}_subclusters}}")
        lines.append(r"\tablehead{")
        lines.append(
            r"\colhead{Subcluster} & \colhead{N} & \colhead{BCG} & \colhead{BCG $z$} & "
            r"\colhead{Mean $z$} & \colhead{$\sigma_v$ [km s$^{-1}$]}"
        )
        lines.append(r"}")
        lines.append(r"\centering")
        lines.append(r"\startdata")
        lines.append(f"All & {cluster_props[2]} & --- & --- & {cluster_props[0]:.4f} & {round(cluster_props[1], -1):.0f} \\\\")
        for i, zs in enumerate(z_lists):
            nmem = len(zs)
            _, z_mean, sigma_v = vel_data[i]
            line = f"{i+1} & {nmem} & {bcg_id_str[i]} & {bcg_z_list[i]:.3f} & {z_mean:.4f} & {round(sigma_v, -1):.0f}  \\\\"
            lines.append(line)

        lines.append(r"\enddata}")
        # fix a missing brace if any LaTeX parser complains:
        lines[-1] = r"\enddata"
        lines.append(r"\end{deluxetable}")

        with open(out_path, "w") as f:
            f.write("\n".join(lines))

    def _write_bcg_pairs_csv(bcg_df, out_csv):
        # Keep only rows with BCG_priority and z
        df = bcg_df.copy()
        df = df[df["BCG_priority"].notna() & df["z"].notna()].copy()
        if df.empty:
            pd.DataFrame([], columns=[
                "bcg_i", "bcg_j", "z_i", "z_j", "sigma_z_i", "sigma_z_j",
                "z_mean_pair", "dv_kms", "dv_err_kms", "abs_dv_kms"
            ]).to_csv(out_csv, index=False)
            return

        # Ensure types
        df["BCG_priority"] = df["BCG_priority"].astype(int)
        if "sigma_z" not in df.columns:
            df["sigma_z"] = np.nan

        rows = []
        for i in range(len(df)):
            for j in range(i + 1, len(df)):
                pri_i = int(df.iloc[i]["BCG_priority"])
                pri_j = int(df.iloc[j]["BCG_priority"])
                zi = float(df.iloc[i]["z"])
                zj = float(df.iloc[j]["z"])
                szi = None if pd.isna(df.iloc[i]["sigma_z"]) else float(df.iloc[i]["sigma_z"])
                szj = None if pd.isna(df.iloc[j]["sigma_z"]) else float(df.iloc[j]["sigma_z"])

                dv, zbar = _los_dv(zi, zj)
                dv_err = None
                if (szi is not None) and (szj is not None):
                    dv_err = c.to('km/s').value * np.sqrt(szi**2 + szj**2) / (1.0 + zbar)

                rows.append({
                    "bcg_i": pri_i,
                    "bcg_j": pri_j,
                    "z_i": zi,
                    "z_j": zj,
                    "sigma_z_i": "" if szi is None else szi,
                    "sigma_z_j": "" if szj is None else szj,
                    "z_mean_pair": zbar,
                    "dv_kms": dv,
                    "dv_err_kms": "" if dv_err is None else dv_err,
                    "abs_dv_kms": abs(dv),
                })

        pd.DataFrame(rows).to_csv(out_csv, index=False)

    # ------------------------
    # Main
    # ------------------------

    bcg_df = pd.read_csv(cluster.bcg_file)

    # -- Read spectroscopy data --
    df_spec = pd.read_csv(cluster.spec_file)
    z_all = df_spec['z'].values

    # -- Select cluster members --
    mask_members = (z_all > cluster.z_min) & (z_all < cluster.z_max)
    z_members = z_all[mask_members]
    n_members = len(z_members)
    z_mean_all, sigma_v_all, _ = velocity_dispersion(z_members)
    cluster_props = [z_mean_all, sigma_v_all, n_members]


    required = {"BCG_priority", "z"}
    missing = required - set(bcg_df.columns)
    if missing:
        raise ValueError(f"BCGs.csv missing required column(s): {', '.join(sorted(missing))}")
    if "sigma_z" not in bcg_df.columns:
        bcg_df["sigma_z"] = np.nan

    # Map: bcg_id (from subcluster_configs) -> (z, sigma_z) via BCG_priority
    cfgs = _ordered(configs)
    # Ensure integer compare for priorities
    bcg_df["BCG_priority"] = bcg_df["BCG_priority"].astype(int)
    pri_to_vals = {
        int(row.BCG_priority): (float(row.z), (None if pd.isna(row.sigma_z) else float(row.sigma_z)))
        for _, row in bcg_df.iterrows()
    }

    bcg_z_list, bcg_sigz_list, bcg_id_str = [], [], []
    for cfg in cfgs:
        bcg_id_str.append(str(cfg.get("bcg_id", "---")))
        bid = int(cfg["bcg_id"])
        z_val, sig_val = pri_to_vals.get(bid, (None, None))
        bcg_z_list.append(z_val)
        bcg_sigz_list.append(sig_val)

    # Compute subcluster stats with your function
    z_data, vel_data = analyze_group(spec_groups)

    # Compute BCG delta v and sigma per subcluster
    bcg_dv_kms, bcg_dv_err_kms = [], []
    for i, (_, z_mean, _) in enumerate(vel_data):
        z_bcg = bcg_z_list[i]
        if z_bcg is None or (isinstance(z_bcg, float) and np.isnan(z_bcg)):
            bcg_dv_kms.append(None)
            bcg_dv_err_kms.append(None)
            continue
        dv = c.to('km/s').value * (z_bcg - z_mean) / (1.0 + z_mean)
        dv_err = _los_dv_err_from_sigmaz(bcg_sigz_list[i], z_mean)
        bcg_dv_kms.append(dv)
        bcg_dv_err_kms.append(dv_err)

        # dv_bcg = c.to('km/s').value * (z_bcg - z_mean) / (1.0 + z_mean)
        # sigz_bcg = bcg_sigz_list[i]
        # dv_err = _los_dv_err_from_sigmaz(sigz_bcg, z_mean)
        # bcg_col = rf"${dv_bcg:+.0f}$" if dv_err is None else rf"${dv_bcg:+.0f}\ \pm\ {dv_err:.0f}$"

    # Write deluxetable
    tex_path = os.path.join(cluster.cluster_path, f"{cluster.identifier.replace(' ', '')}_subclusters.tex")
    _write_deluxetable_tex(cluster.identifier, vel_data, z_data, bcg_z_list, bcg_id_str, cluster_props, tex_path)

    print_subcluster_deluxetable(spec_groups)

    # Pairwise BCG–BCG CSV
    pairs_csv = os.path.join(cluster.cluster_path, "BCG_velocity_pairs.csv")
    _write_bcg_pairs_csv(bcg_df, pairs_csv)

    # Optional: update subclusters.csv in place
    if update_csv and getattr(cluster, "subcluster_file", None) and os.path.exists(cluster.subcluster_file):
        try:
            sdf = pd.read_csv(cluster.subcluster_file)
            # Try to map by bcg_id if present; else by subcluster index order
            if "bcg_id" in sdf.columns:
                id_to_dv = {}
                id_to_dverr = {}
                for i, cfg in enumerate(cfgs):
                    bid = int(cfg["bcg_id"])
                    id_to_dv[bid] = bcg_dv_kms[i]
                    id_to_dverr[bid] = bcg_dv_err_kms[i]
                sdf["bcg_dv_kms"] = sdf["bcg_id"].map(id_to_dv)
                sdf["bcg_dv_err_kms"] = sdf["bcg_id"].map(id_to_dverr)
            else:
                # Fallback: assume row order corresponds to cfg order
                sdf["bcg_dv_kms"] = pd.Series(bcg_dv_kms[:len(sdf)])
                sdf["bcg_dv_err_kms"] = pd.Series(bcg_dv_err_kms[:len(sdf)])
            sdf.to_csv(cluster.subcluster_file, index=False)
        except Exception as e:
            print(f"Warning: could not update subclusters.csv with BCG dv columns: {e}")

    return {
        "tex_path": tex_path,
        "pairs_csv": pairs_csv,
        "bcg_dv_kms": bcg_dv_kms,
        "bcg_dv_err_kms": bcg_dv_err_kms,
        "vel_data": vel_data,
    }



## -- Driver -- ##
def analyze_cluster(
        cluster,
        subcluster_configs,
        legend_loc="upper right",
        manual_lists=None,
        combined_groups=None,
        show_plots=True,
        save_plots=False,
        save_path=None,
        plot_alt_regions=False,
        run_pipeline=False,
        verbose=False
    ):
    if save_path is None:
        save_path = cluster.image_path
    os.makedirs(save_path, exist_ok=True)

    # if run_pipeline:
    #     run_full_pipeline(cluster, manual_list=manual_list,save_plots=save_plots, show_plots=show_plots)
    #     process_redshifts(cluster, save_plots=save_plots, show_plots=show_plots)
    #     make_plots(cluster, save_plots=save_plots, show_plots=show_plots)

    spec_df, phot_df, bcg_df = load_dataframes(cluster)

    # Create first look at regions using dividing lines
    if verbose:
        bcg_regions = assign_subcluster_regions(subcluster_configs, plot=True)

    # Assign subcluster members based on dividing lines
    region_groups_spec, bisectors   = assign_subcluster_members_multi(subcluster_configs, spec_df)
    region_groups_phot, _ = assign_subcluster_members_multi(subcluster_configs, phot_df)

    # Filter members based on config (e.g., z_range, r_max)
    spec_groups = filter_members_by_config(region_groups_spec, subcluster_configs)  
    phot_groups = filter_members_by_config(region_groups_phot, subcluster_configs, spec=False)
    
    print(f"\nSpectroscopic groups: {[len(g) for g in spec_groups]}")
    print(f"Photometric groups: {[len(g) for g in phot_groups]}")

    # Build member catalogs
    make_member_catalogs(cluster, spec_groups, phot_groups, subcluster_configs)
    build_subcluster_summary(cluster=cluster, configs=subcluster_configs, spec_groups=spec_groups, update_csv=True)

    z_data, vel_data = analyze_group(spec_groups)
    stats_table, stats_values = make_stats_table(
        z_data,
        bins=12,
        prefix="subcluster",
        make_plots=True,
        save_plots=True,
        show_plots=True,
        save_path="figures",   # or Path(...)
        ranges=None            # or [(z1_low, z1_high), (z2_low, z2_high), ...] if you want display ranges
)

    print(stats_table)

    # Combine groups if specified
    spec_groups_combined, phot_groups_combined, combined_indices = make_combined_groups(cluster, spec_groups, phot_groups, subcluster_configs, combined_groups)
    combined_configs = [cfg for cfg in subcluster_configs if cfg.get("is_dominant") == 1]

    z_data, vel_data = analyze_group(spec_groups_combined)
    stats_table, stats_values = make_stats_table(
        z_data,
        bins=12,
        prefix="subcluster",
        make_plots=True,
        save_plots=True,
        show_plots=True,
        save_path="figures",   # or Path(...)
        ranges=None            # or [(z1_low, z1_high), (z2_low, z2_high), ...] if you want display ranges
)

    print(stats_table)


    
    if plot_alt_regions:
        # Save file based on subclusters used.
        subclust_ids = [str(sub['bcg_id']) for sub in subcluster_configs]
        subclust_part = "_".join(subclust_ids)
        filename = f"{cluster.identifier}_subcluster_histograms_{subclust_part}.pdf"
        save_file = os.path.join(save_path, filename)
        plot_subcluster_regions_and_histograms(
                                        cluster=cluster,
                                        subcluster_configs=subcluster_configs,
                                        combined_configs=combined_configs,
                                        spec_groups=spec_groups,
                                        phot_groups=phot_groups,
                                        spec_groups_combined=spec_groups_combined,
                                        phot_groups_combined=phot_groups_combined,
                                        legend_loc=legend_loc,
                                        combined_indices=combined_indices,
                                        show_plots=show_plots,
                                        save_plots=True,
                                        save_path=save_file
        )
        quit()

    clean_id = cluster.identifier.replace(" ", "_")

    plot_subcluster_members_and_regions(
                                    cluster=cluster,
                                    subcluster_configs=subcluster_configs,
                                    combined_configs=combined_configs,
                                    spec_groups=spec_groups,
                                    spec_groups_combined=spec_groups_combined,
                                    phot_groups=phot_groups,
                                    phot_groups_combined=phot_groups_combined,
                                    legend_loc=legend_loc,
                                    combined_indices=combined_indices,
                                    show_plots=show_plots,
                                    save_plots=save_plots,
                                    save_path=os.path.join(cluster.image_path, f"{clean_id}_subcluster_map.pdf")
    )


    plot_redshift_and_subclusters_figure(
                                    cluster=cluster,
                                    subcluster_configs=subcluster_configs,
                                    combined_configs=combined_configs,
                                    spec_groups=spec_groups,
                                    spec_groups_combined=spec_groups_combined,
                                    phot_groups=phot_groups,
                                    phot_groups_combined=phot_groups_combined,
                                    legend_loc=legend_loc,
                                    combined_indices=combined_indices,
                                    legend_loc_bottom="upper right",
                                    show_plots=show_plots,
                                    save_plots=save_plots,
                                    save_path=os.path.join(cluster.image_path, f"{clean_id}_subcluster_z_scatter_vert.pdf"),
                                    layout='vertical'
    )

    plot_redshift_and_subclusters_figure(
                                    cluster=cluster,
                                    subcluster_configs=subcluster_configs,
                                    combined_configs=combined_configs,
                                    spec_groups=spec_groups,
                                    spec_groups_combined=spec_groups_combined,
                                    phot_groups=phot_groups,
                                    phot_groups_combined=phot_groups_combined,
                                    legend_loc=legend_loc,
                                    combined_indices=combined_indices,
                                    legend_loc_bottom="upper right",
                                    show_plots=show_plots,
                                    save_plots=save_plots,
                                    save_path=os.path.join(cluster.image_path, f"{clean_id}_subcluster_z_scatter_hor.pdf"),
                                    layout='horizontal'
    )

    plot_2panel_optical_contours(
                                    cluster=cluster,
                                    subcluster_configs=subcluster_configs,
                                    spec_groups=spec_groups,
                                    phot_groups=phot_groups,
                                    layout="vertical",
                                    legend_loc=legend_loc,
                                    show_plots=show_plots,
                                    save_plots=save_plots,
                                    save_path=os.path.join(cluster.image_path, f"{clean_id}_2panel_optical_contours_vert.pdf")
    )

    plot_2panel_optical_contours(
                                    cluster=cluster,
                                    subcluster_configs=subcluster_configs,
                                    spec_groups=spec_groups,
                                    phot_groups=phot_groups,
                                    layout="horizontal",
                                    legend_loc=legend_loc,
                                    show_plots=show_plots,
                                    save_plots=save_plots,
                                    save_path=os.path.join(cluster.image_path, f"{clean_id}_2panel_optical_contours_hor.pdf")
    )

    plot_redshift_histogram_heatmap(
                                    cluster=cluster,
                                    legend_loc=legend_loc,
                                    show_plots=show_plots,
                                    save_plots=save_plots,
                                    save_path=os.path.join(cluster.image_path, f"{clean_id}_redshifts_hist_hmap.pdf")
    )


    plot_stacked_redshift_histograms(
                                    cluster=cluster,
                                    subcluster_configs=combined_configs,
                                    spec_groups=spec_groups_combined,
                                    show_plots=show_plots,
                                    save_plots=save_plots,
                                    save_path=os.path.join(cluster.image_path, f"{clean_id}_stacked_redshifts.pdf")
    )

    plot_3panel_optical_subclusters_figure(
                                    cluster=cluster,
                                    subcluster_configs=subcluster_configs,
                                    combined_configs=combined_configs,
                                    spec_groups=spec_groups,
                                    spec_groups_combined=spec_groups_combined,
                                    phot_groups=phot_groups,
                                    phot_groups_combined=phot_groups_combined,
                                    combined_indices=combined_indices,
                                    show_plots=show_plots,
                                    save_plots=save_plots,
                                    save_path=os.path.join(cluster.image_path, f"{clean_id}_3panel_optical.pdf")
    )

    plot_combined_4panel_figure(
                                    cluster=cluster,
                                    subcluster_configs=subcluster_configs,
                                    combined_configs=combined_configs,
                                    spec_groups=spec_groups,
                                    spec_groups_combined=spec_groups_combined,
                                    phot_groups=phot_groups,
                                    phot_groups_combined=phot_groups_combined,
                                    combined_indices=combined_indices,
                                    legend_loc=legend_loc,
                                    show_plots=show_plots,
                                    save_plots=save_plots,
                                    save_path=os.path.join(cluster.image_path, f"{clean_id}_combined_4panel.pdf")
    )





# -- CLI -- 

def parse_subcluster_kwargs(cli_args):
    """
    Parses unknown CLI args into a kwargs dict for build_subclusters.
    Accepts --color_4 value, --label_2 value, --z_range_3 min max, etc.
    """
    kw = {}
    i = 0
    while i < len(cli_args):
        arg = cli_args[i]
        if arg.startswith("--"):
            key = arg.lstrip("-")
            # Support --key val or --key=val
            if "=" in key:
                key, val = key.split("=", 1)
                i_add = 0
            else:
                # Check if it's a z_range_X which takes two values
                if key.startswith("z_range_"):
                    val = tuple(float(cli_args[i+1]), float(cli_args[i+2]))
                    i_add = 2
                else:
                    val = cli_args[i+1]
                    i_add = 1
            # Attempt to parse int/float
            if isinstance(val, str):
                try:
                    val = float(val) if "." in val else int(val)
                except Exception:
                    pass
            kw[key] = val
            i += 1 + i_add
        else:
            i += 1
    return kw

def main():
    parser = argparse.ArgumentParser(description="Analyze cluster substructures and build subcluster configs.")

    # -- Required arguments --
    parser.add_argument("cluster_id", type=str, help="Cluster ID (e.g. '1327') or full RMJ/Abell name.")
    parser.add_argument("--subclusters", nargs="+", type=int, required=True,
        help="List of BCG indices or subcluster IDs (e.g., 2 6 7).")

    # -- Cluster options --
    parser.add_argument("--z-min", type=float, default=None, help="Minimum redshift for analysis.")
    parser.add_argument("--z-max", type=float, default=None, help="Maximum redshift for analysis.")
    parser.add_argument("--fov", type=float, default=None, help="Zoomed field of view in arcmin.")
    parser.add_argument("--fov-full", type=float, default=None, help="Full field of view in arcmin.")
    parser.add_argument("--ra-offset", type=float, default=None, help="RA offset in arcmin.")
    parser.add_argument("--dec-offset", type=float, default=None, help="Dec offset in arcmin.")

    # -- Global subcluster options --
    parser.add_argument("--radius", type=float, default=None, help="Default search radius for all subclusters (Mpc).")
    parser.add_argument("--z-range", nargs=2, type=float, default=None, metavar=("ZMIN", "ZMAX"), help="Global z_range for subclusters.")
    parser.add_argument("--colors", nargs="+", default=None, help="List of colors for subclusters (e.g., tab:green gold tab:cyan).")
    parser.add_argument("--labels", nargs="+", default=None, help="List of labels for subclusters.")

    # -- X-ray/density/display options --
    parser.add_argument("--xray-levels", type=str, default=None, help="Comma-separated contour levels for X-ray (e.g. '0.5,0,12').")
    parser.add_argument("--psf", type=float, default=None, help="PSF for X-ray smoothing.")
    parser.add_argument("--density-levels", type=int, default=None, help="Number of density contour levels.")
    parser.add_argument("--density-skip", type=int, default=None, help="Number of density levels to skip.")
    parser.add_argument("--density-bandwidth", type=float, default=None, help="Bandwidth for density estimation.")

    parser.add_argument("--save-plots", type=str2bool, nargs="?", const=True, default=True, help="Save plots? [default: True]")
    parser.add_argument("--show-plots", type=str2bool, nargs="?", const=True, default=False, help="Show plots? [default: False]")
    parser.add_argument("--plot-alt-regions", type=str2bool, nargs="?", const=True, default=False, help="Plot alternative regions? [default: False]")
    parser.add_argument("--run-pipeline", type=str2bool, nargs="?", const=True, default=False, help="Run full pipeline (redshift processing, plots)? [default: False]")    

    # -- Catch-all for extra per-subcluster kwargs (e.g. --color_4) --
    parser.add_argument("--subcluster-kwargs", nargs=argparse.REMAINDER, help="Extra per-subcluster args (e.g. --color_4 tab:orange --label_2 Main)")

    # Parse known and unknown
    args, unknown = parser.parse_known_args()



    # -- Parse known CLI for cluster construction --
    cluster_kwargs = {}
    for key in ("fov", "fov_full", "ra_offset", "dec_offset", "density_levels", "density_skip", "density_bandwidth", "xray_levels", "psf", "z_min", "z_max"):
        val = getattr(args, key, None)
        if val is not None:
            cluster_kwargs[key.replace("-", "_")] = val

    # Parse xray_levels tuple
    if args.xray_levels is not None:
        cluster_kwargs["contour_levels"] = tuple(float(x) for x in args.xray_levels.split(","))
        print(f"Using X-ray contour levels: {cluster_kwargs['contour_levels']}")
    if args.density_levels is not None:
        cluster_kwargs["phot_levels"] = args.density_levels
    if args.density_skip is not None:
        cluster_kwargs["phot_skip"] = args.density_skip
    if args.density_bandwidth is not None:
        cluster_kwargs["bandwidth"] = args.density_bandwidth

    # -- Build the Cluster object --
    cluster = Cluster(args.cluster_id, **cluster_kwargs)
    cluster.populate(verbose=True)

    # -- Subcluster dictionary assembly --
    # Parse global subcluster options
    subcluster_kwargs = {}

    if args.radius is not None:
        subcluster_kwargs["radius"] = args.radius
    if args.z_range is not None:
        subcluster_kwargs["z_range"] = tuple(args.z_range)
    if args.colors is not None:
        subcluster_kwargs["color"] = args.colors
    if args.labels is not None:
        subcluster_kwargs["bcg_label"] = args.labels

    # Parse extra subcluster-keyword pairs from unknown or --subcluster-kwargs
    extra_kwargs = {}
    if args.subcluster_kwargs is not None:
        extra_kwargs = parse_subcluster_kwargs(args.subcluster_kwargs)
    elif unknown:
        extra_kwargs = parse_subcluster_kwargs(unknown)

    # Call builder function
    subcluster_configs = build_subclusters(
        subclusters=args.subclusters,
        cluster=cluster,
        **subcluster_kwargs,
        **extra_kwargs
    )

    # -- Call your analysis function, passing configs and cluster --
    analyze_cluster(
        cluster=cluster,
        subcluster_configs=subcluster_configs,
        save_plots=args.save_plots,
        show_plots=args.show_plots,
        plot_alt_regions=args.plot_alt_regions,
        run_pipeline=args.run_pipeline
    )

if __name__ == "__main__":
    main()

