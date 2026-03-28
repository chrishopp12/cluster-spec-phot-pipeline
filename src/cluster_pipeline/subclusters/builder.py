#!/usr/bin/env python3
"""
builder.py

Stage 5: Subcluster Configuration Building
---------------------------------------------------------

Builds Subcluster objects from BCG candidates and configuration.

Data products
~~~~~~~~~~~~~
- Returns ``list[Subcluster]`` with BCG roles set (no members yet).
- Persistence is through ``Subcluster.to_config()`` / config.yaml,
  **not** CSV files.

Configuration sources (highest to lowest priority)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. Per-subcluster kwarg  (``color_2``, ``radius_3``)
2. Global kwarg          (``color``, ``radius``)
3. config.yaml value     (``config["subclusters"]``)
4. Code default          (``DEFAULT_COLORS``, ``DEFAULT_RADIUS_MPC``, etc.)

BCG sources (first match wins)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. ``bcgs`` argument passed directly
2. ``config["bcgs"]`` section in config.yaml
3. ``BCGs.csv`` on disk (via ``read_bcg_csv``)
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from cluster_pipeline.models.cluster import Cluster
from cluster_pipeline.models.bcg import BCG
from cluster_pipeline.models.subcluster import Subcluster
from cluster_pipeline.io.catalogs import read_bcg_csv
from cluster_pipeline.config import load_config
from cluster_pipeline.constants import DEFAULT_COLORS, DEFAULT_LABELS, DEFAULT_RADIUS_MPC

log = logging.getLogger(__name__)


# ======================================================================
# Public API
# ======================================================================

def build_subclusters(
    cluster: Cluster,
    *,
    bcgs: list[BCG] | None = None,
    config: dict | None = None,
    **kwargs,
) -> list[Subcluster]:
    """Build Subcluster objects from BCGs and configuration.

    Parameters
    ----------
    cluster : Cluster
        Cluster object (provides paths, ``resolve_z_range()``).
    bcgs : list[BCG] or None
        Pre-built BCG objects.  If None, loaded from ``config["bcgs"]``
        or ``BCGs.csv`` (see ``_load_bcgs``).
    config : dict or None
        Full config dict (typically from config.yaml).  If None, loaded
        from ``cluster.config_path``.
    **kwargs
        CLI-style overrides.  Supports both global (``color``,
        ``radius``) and per-subcluster (``color_2``, ``radius_3``)
        forms.  Also accepts ``combined`` for group definitions.

    Returns
    -------
    list[Subcluster]
        One Subcluster per entry in ``config["subclusters"]``, with
        BCG roles assigned but no member galaxies yet.

    Raises
    ------
    ValueError
        If no subclusters are defined in config, or a referenced BCG
        ID cannot be found.
    """
    # -- Load config if not provided --------------------------------
    if config is None:
        config = load_config(cluster.cluster_path)

    # -- Load BCGs --------------------------------------------------
    bcg_lookup = _load_bcgs(cluster, bcgs=bcgs, config=config)

    # -- Build individual subclusters -------------------------------
    sc_configs = config.get("subclusters", {})
    if not sc_configs:
        raise ValueError(
            "No subclusters defined.  Add a 'subclusters' section to "
            "config.yaml or pass subcluster definitions in the config dict."
        )

    subclusters: list[Subcluster] = []
    for idx, (sc_id, sc_cfg) in enumerate(sc_configs.items()):
        sc = _build_one(
            idx=idx,
            sc_id=int(sc_id),
            sc_cfg=sc_cfg,
            bcg_lookup=bcg_lookup,
            cluster=cluster,
            **kwargs,
        )
        subclusters.append(sc)
        log.info(
            "Subcluster %d  BCG=%d  label='%s'  color='%s'  "
            "radius=%.2f Mpc  z_range=%s",
            idx + 1, sc.bcg_id, sc.label, sc.color,
            sc.radius_mpc, sc.z_range,
        )

    # -- Apply group assignments ------------------------------------
    group_configs = config.get("groups", {})
    combined_kwarg = kwargs.pop("combined", None)
    _apply_groups(subclusters, group_configs, bcg_lookup, combined_kwarg, **kwargs)

    return subclusters


# ======================================================================
# BCG Loading
# ======================================================================

def _load_bcgs(
    cluster: Cluster,
    *,
    bcgs: list[BCG] | None = None,
    config: dict,
) -> dict[int, BCG]:
    """Return a ``{bcg_id: BCG}`` lookup from the first available source.

    Resolution order:

    1. ``bcgs`` argument (pre-built objects).
    2. ``config["bcgs"]`` section (YAML definitions).
    3. ``BCGs.csv`` on disk.

    Parameters
    ----------
    cluster : Cluster
        Used for ``cluster.bcg_file`` path when falling back to CSV.
    bcgs : list[BCG] or None
        Pre-built BCG objects.
    config : dict
        Full config dict, checked for a ``bcgs`` key.

    Returns
    -------
    dict[int, BCG]
        Mapping from ``bcg_id`` to ``BCG`` object.

    Raises
    ------
    ValueError
        If no BCGs can be found from any source.
    """
    # Source 1: directly provided
    if bcgs is not None:
        log.debug("Using %d BCG(s) passed directly.", len(bcgs))
        return {b.bcg_id: b for b in bcgs}

    # Source 2: config.yaml bcgs section
    bcg_section = config.get("bcgs", {})
    if bcg_section:
        log.debug("Loading BCGs from config.yaml 'bcgs' section.")
        lookup = {}
        for bid, bcfg in bcg_section.items():
            lookup[int(bid)] = BCG.from_config(int(bid), bcfg)

        # Enrich with matched data from BCGs.csv (redshifts, photometry)
        try:
            bcg_df = read_bcg_csv(cluster)
            if not bcg_df.empty:
                for _, row in bcg_df.iterrows():
                    bid = int(row.get("BCG_priority", 0))
                    if bid in lookup:
                        bcg = lookup[bid]
                        # Fill missing fields from CSV (CSV has matched spec+phot)
                        if bcg.z is None and pd.notna(row.get("z")):
                            bcg.z = float(row["z"])
                        if bcg.sigma_z is None and pd.notna(row.get("sigma_z")):
                            bcg.sigma_z = float(row["sigma_z"])
                        if bcg.probability is None and pd.notna(row.get("BCG_probability")):
                            bcg.probability = float(row["BCG_probability"])
                log.debug("Enriched BCGs with matched data from BCGs.csv")
        except Exception:
            pass  # BCGs.csv may not exist yet on first run

        return lookup

    # Source 3: BCGs.csv
    log.debug("Loading BCGs from %s", cluster.bcg_file)
    bcg_df = read_bcg_csv(cluster)
    if bcg_df.empty:
        raise ValueError(
            f"No BCGs found.  Checked: bcgs argument, config['bcgs'], "
            f"and {cluster.bcg_file}."
        )
    lookup = {}
    for _, row in bcg_df.iterrows():
        b = BCG.from_dataframe_row(row)
        lookup[b.bcg_id] = b
    return lookup


# ======================================================================
# Single Subcluster Builder
# ======================================================================

def _build_one(
    idx: int,
    sc_id: int,
    sc_cfg: dict[str, Any],
    bcg_lookup: dict[int, BCG],
    cluster: Cluster,
    **kwargs,
) -> Subcluster:
    """Create a single Subcluster from its config entry.

    Parameters
    ----------
    idx : int
        Zero-based position in the subcluster list (used for
        index-based kwarg lookups like ``color`` being a list).
    sc_id : int
        Subcluster / BCG identifier.
    sc_cfg : dict
        Config dict for this subcluster (from ``config["subclusters"][id]``).
    bcg_lookup : dict[int, BCG]
        All available BCGs keyed by ``bcg_id``.
    cluster : Cluster
        For ``resolve_z_range()`` fallback.
    **kwargs
        CLI overrides.

    Returns
    -------
    Subcluster
    """
    bcg_id = int(sc_cfg.get("bcg_id", sc_id))
    if bcg_id not in bcg_lookup:
        raise ValueError(
            f"Subcluster {sc_id} references bcg_id={bcg_id}, but that "
            f"BCG was not found.  Available: {sorted(bcg_lookup.keys())}"
        )
    primary_bcg = bcg_lookup[bcg_id]

    # -- Resolve per-field with the priority chain -------------------
    color_default = DEFAULT_COLORS[(bcg_id - 1) % len(DEFAULT_COLORS)]
    label_default = DEFAULT_LABELS[(bcg_id - 1) % len(DEFAULT_LABELS)]

    label = _resolve_field("label", idx, bcg_id, sc_cfg, kwargs, default=label_default)
    color = _resolve_field("color", idx, bcg_id, sc_cfg, kwargs, default=color_default)
    radius_mpc = _resolve_field(
        "radius_mpc", idx, bcg_id, sc_cfg, kwargs, default=DEFAULT_RADIUS_MPC,
    )
    # Also accept "radius" as a kwarg alias for "radius_mpc"
    if radius_mpc == DEFAULT_RADIUS_MPC:
        radius_alt = _resolve_field(
            "radius", idx, bcg_id, sc_cfg, kwargs, default=None,
        )
        if radius_alt is not None:
            radius_mpc = radius_alt
    radius_mpc = float(radius_mpc)

    z_range = _resolve_z_range(idx, bcg_id, sc_cfg, kwargs, cluster)

    return Subcluster(
        bcg_id=bcg_id,
        primary_bcg=primary_bcg,
        label=str(label),
        color=str(color),
        radius_mpc=radius_mpc,
        z_range=z_range,
    )


def _resolve_z_range(
    idx: int,
    bcg_id: int,
    sc_cfg: dict,
    kwargs: dict,
    cluster: Cluster,
) -> tuple[float, float]:
    """Resolve the redshift range for a subcluster.

    Tries (in order): per-subcluster kwarg, global kwarg, config value,
    then falls back to ``cluster.resolve_z_range()``.

    Parameters
    ----------
    idx : int
        Zero-based position index.
    bcg_id : int
        BCG identifier.
    sc_cfg : dict
        Subcluster config dict.
    kwargs : dict
        CLI overrides.
    cluster : Cluster
        Fallback source via ``resolve_z_range()``.

    Returns
    -------
    tuple[float, float]
        (z_min, z_max).
    """
    zr = _resolve_field("z_range", idx, bcg_id, sc_cfg, kwargs, default=None)
    if zr is not None:
        if isinstance(zr, list):
            zr = tuple(zr)
        return (float(zr[0]), float(zr[1]))
    return cluster.resolve_z_range()


# ======================================================================
# Group Handling
# ======================================================================

def _apply_groups(
    subclusters: list[Subcluster],
    group_configs: dict,
    bcg_lookup: dict[int, BCG],
    combined_kwarg: Any,
    **kwargs,
) -> None:
    """Apply group assignments to subclusters (in place).

    Handles both config.yaml group definitions and the legacy
    ``combined`` kwarg.

    For each group:
    - The first listed member is the dominant subcluster (defines
      the group's identity: label, color, stats).
    - All members get matching ``group_id``, ``group_members``,
      ``group_label``, ``group_color``.
    - The member closest to the boundary (if specified) gets set
      as ``border_bcg``.

    Parameters
    ----------
    subclusters : list[Subcluster]
        Modified in place.
    group_configs : dict
        From ``config["groups"]``, keyed by group ID.
    bcg_lookup : dict[int, BCG]
        All BCGs.
    combined_kwarg : list or None
        Legacy ``combined`` kwarg (e.g., ``["1+4", "2+5"]``).
    **kwargs
        Per-group overrides (``group_label_1_4``, etc.).
    """
    by_id = {sc.bcg_id: sc for sc in subclusters}

    # -- Solo defaults: every subcluster is its own group -----------
    for sc in subclusters:
        sc.group_id = str(sc.bcg_id)
        sc.group_members = (sc.bcg_id,)
        sc.is_dominant = True
        sc.group_label = None
        sc.group_color = None

    # -- Config-defined groups --------------------------------------
    # groups can be a list (from YAML) or a dict (legacy)
    if isinstance(group_configs, list):
        for gcfg in group_configs:
            members = [int(m) for m in gcfg.get("members", [])]
            primary = gcfg.get("primary_bcg", members[0] if members else None)
            gid = _group_id_from_members(members)
            # Reorder so primary is first (dominant)
            if primary in members:
                members = [primary] + [m for m in members if m != primary]
            _assign_group(gid, members, gcfg, by_id, bcg_lookup, **kwargs)
    elif isinstance(group_configs, dict):
        for gid, gcfg in group_configs.items():
            members = [int(m) for m in gcfg.get("members", [])]
            _assign_group(str(gid), members, gcfg, by_id, bcg_lookup, **kwargs)

    # -- Legacy combined kwarg (e.g., ["1+4", "2+5"]) ---------------
    combined_groups = _parse_combined_groups(combined_kwarg)
    for members in combined_groups:
        gid = _group_id_from_members(members)
        _assign_group(gid, members, {}, by_id, bcg_lookup, **kwargs)


def _assign_group(
    gid: str,
    members: list[int],
    gcfg: dict,
    by_id: dict[int, Subcluster],
    bcg_lookup: dict[int, BCG],
    **kwargs,
) -> None:
    """Assign group properties to a set of subclusters.

    Parameters
    ----------
    gid : str
        Group identifier string.
    members : list[int]
        BCG IDs in the group. First is dominant.
    gcfg : dict
        Group config dict (from config.yaml or empty).
    by_id : dict[int, Subcluster]
        Lookup of subclusters by bcg_id.
    bcg_lookup : dict[int, BCG]
        All BCGs.
    **kwargs
        CLI overrides for group-level fields.
    """
    # Filter to members that actually exist as subclusters
    present = [m for m in members if m in by_id]
    if len(present) < 2:
        return

    dominant_id = present[0]
    dominant_sc = by_id[dominant_id]
    member_tuple = tuple(present)

    # Resolve group display properties
    # Per-group kwarg key uses the canonical (sorted) group id
    canonical_gid = _group_id_from_members(present)

    group_label = (
        kwargs.get(f"group_label_{canonical_gid}")
        or kwargs.get("group_label")
        or gcfg.get("label")
        or dominant_sc.label
    )
    group_color = (
        kwargs.get(f"group_color_{canonical_gid}")
        or kwargs.get("group_color")
        or gcfg.get("color")
        or dominant_sc.color
    )

    # Border BCG: the member closest to the inter-group boundary.
    # Can be specified in config; otherwise left as None (primary is used).
    border_id = gcfg.get("border_bcg_id")

    for mid in present:
        sc = by_id[mid]
        sc.group_id = canonical_gid
        sc.group_members = member_tuple
        sc.is_dominant = (mid == dominant_id)
        sc.group_label = group_label
        sc.group_color = group_color

        # Add all group BCGs as member_bcgs
        sc.member_bcgs = [bcg_lookup[m] for m in present if m in bcg_lookup]

        # Set border BCG if specified and this is the dominant subcluster
        if border_id is not None and mid == dominant_id and border_id in bcg_lookup:
            sc.border_bcg = bcg_lookup[border_id]

    log.info(
        "Group %s  members=%s  dominant=%d  label='%s'  color='%s'",
        canonical_gid, present, dominant_id, group_label, group_color,
    )


# ======================================================================
# Helper: Field Resolution
# ======================================================================

def _resolve_field(
    field: str,
    idx: int,
    bcg_id: int,
    sc_cfg: dict,
    kwargs: dict,
    *,
    default: Any = None,
) -> Any:
    """Resolve a single field through the priority chain.

    Priority (highest to lowest):

    1. Per-subcluster kwarg: ``kwargs["{field}_{bcg_id}"]``
    2. Global kwarg (scalar or indexed): ``kwargs["{field}"]``
    3. Config value: ``sc_cfg["{field}"]``
    4. Default

    Parameters
    ----------
    field : str
        Field name (e.g., ``"color"``, ``"radius_mpc"``).
    idx : int
        Zero-based position index (for list-valued global kwargs).
    bcg_id : int
        BCG identifier (for per-subcluster kwargs).
    sc_cfg : dict
        Config dict for this subcluster.
    kwargs : dict
        CLI overrides.
    default : Any
        Fallback value if nothing else matches.

    Returns
    -------
    Any
        The resolved value.
    """
    # 1. Per-subcluster kwarg (e.g., color_2)
    per_sc = kwargs.get(f"{field}_{bcg_id}")
    if per_sc is not None:
        return per_sc

    # 2. Global kwarg (e.g., color="tab:green" or color=["white", "tab:green"])
    global_val = kwargs.get(field)
    if global_val is not None:
        # z_range is a tuple/list of two floats — don't index into it
        if field in ("z_range", "group_z_range"):
            if (
                isinstance(global_val, (list, tuple))
                and len(global_val) == 2
                and all(isinstance(x, (int, float)) for x in global_val)
            ):
                return tuple(float(x) for x in global_val)
        # List-valued global: pick by index position
        if isinstance(global_val, (list, tuple)):
            if idx < len(global_val):
                return global_val[idx]
            return default
        # Scalar: applies to all subclusters
        return global_val

    # 3. Config value
    cfg_val = sc_cfg.get(field)
    if cfg_val is not None:
        return cfg_val

    # 4. Default
    return default


# ======================================================================
# Helper: Combined Group Parsing
# ======================================================================

def _parse_combined_groups(combine: Any) -> list[list[int]]:
    """Parse the ``combined`` kwarg into lists of member BCG IDs.

    Accepts several formats:

    - ``None`` -> ``[]``
    - ``["1+4", "2+5+3"]`` -> ``[[1, 4], [2, 5, 3]]``
    - ``[(1, 4), (2, 5, 3)]`` -> ``[[1, 4], [2, 5, 3]]``

    Parameters
    ----------
    combine : None, list of str, or list of tuple/list
        Group definitions from CLI.

    Returns
    -------
    list[list[int]]
        Each inner list is a group with member BCG IDs.
        Order is preserved (first element = dominant).
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


def _group_id_from_members(members: list[int]) -> str:
    """Build a canonical group ID from sorted member BCG IDs.

    Parameters
    ----------
    members : list[int]
        BCG IDs in the group.

    Returns
    -------
    str
        e.g., ``"1_4"`` for members ``[4, 1]``.
    """
    return "_".join(str(x) for x in sorted(members))
