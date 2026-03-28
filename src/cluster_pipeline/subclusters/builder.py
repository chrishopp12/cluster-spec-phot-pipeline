"""Subcluster configuration building and persistence."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

from astropy.coordinates import SkyCoord
import astropy.units as u

from cluster_pipeline.io.catalogs import read_bcg_csv, select_bcgs, bcg_basic_info, load_dataframes
from cluster_pipeline.constants import DEFAULT_COLORS, DEFAULT_LABELS, DEFAULT_RADIUS_MPC


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
        if cluster is not None:
            return cluster.resolve_z_range()
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
            if "group_id" in cfg:
                row["group_id"] = cfg["group_id"]
            if "is_dominant" in cfg:
                row["is_dominant"] = int(cfg["is_dominant"])
            if "group_members" in cfg:
                gm = cfg.get("group_members", ())
                if isinstance(gm, (list, tuple)) and len(gm) > 0:
                    row["group_members"] = repr(tuple(int(x) for x in gm))
                else:
                    row["group_members"] = ""

            if "group_label" in cfg:
                row["group_label"] = cfg["group_label"]
            if "group_color" in cfg:
                row["group_color"] = cfg["group_color"]
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
    default_colors = DEFAULT_COLORS
    default_labels = DEFAULT_LABELS
    default_radius = DEFAULT_RADIUS_MPC


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
