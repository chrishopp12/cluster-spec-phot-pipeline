#!/usr/bin/env python3
"""
run_xray_pipeline.py

Cluster X-ray Analysis Pipeline Driver
---------------------------------------------------------

This is the driver script for the x-ray analysis portion of the cluster analysis pipeline. It orchestrates the following high-level steps:

1) Process redshift distributions to identify subclusters within the specified redshift range.
2) Create x-ray specific plots, including surface brightness maps and contours, Gaussian gradient magnitude maps, and unsharp-masked images.
3) Identify spatial regions corresponding to subclusters and plot redshift distributions for galaxies within those regions.

Most of the science logic lives in the individual sub-pipelines. This script is intended to remain lightweight and easy to run from the command line.

Arguments can be passed by CLI or loaded from a configuration file.

Note: Currently, most of the driver function is actually handled within the process_subclusters module and this serves as a thin wrapper to set parameters and invoke that logic. Eventually, the scripts will be split to better isolate functionality.

"""



from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any
from cluster import Cluster
from my_utils import str2bool, read_json, string_to_numeric, find_first_val
from xray_pipeline.process_subclusters import analyze_cluster, build_subclusters

# ------------------------------------
# Defaults/ Constants
# ------------------------------------
DEFAULT_FOV = 6.0  # arcmin
DEFAULT_FOV_FULL = 30.0  # arcmin
DEFAULT_PSF = 10.0  # arcsec
DEFAULT_RA_OFFSET = 0.0  # arcmin
DEFAULT_DEC_OFFSET = 0.0  # arcmin
DEFAULT_BANDWIDTH = 0.1 
DEFAULT_PHOT_LEVELS = 10
DEFAULT_PHOT_SKIP = 1
DEFAULT_LEGEND_LOC = 'upper right'
DEFAULT_LEVELS = (1.0, 0.0, 12)  # (base level, min level, max level)

DEFAULT_CONFIG_FILE = 'cluster_configs.json'
DEFAULT_CONFIG_DIR = './../cluster_configs/' 

DEFAULT_SAVE_PLOTS = True
DEFAULT_SHOW_PLOTS = False
DEFAULT_PLOT_ALT_REGIONS = False
DEFAULT_RUN_PIPELINE = False

# ------------------------------------
# Helpers
# ------------------------------------

def _parse_subcluster_kwargs(cli_args: list[str]) -> dict[str, Any]:
    """
    Parses unknown CLI args into a kwargs dict for build_subclusters. Accepts inputs in the form --{key}_{index} value or --{key}_{index}=value. Keys include, but aren't limited to: color, label, radius, z_range, group_z_range, etc.

    Common Examples:
        --color_4 tab:orange
        --label_2 Main
        --z_range_3 0.35 0.4
        --group_z_range_1_4=0.3,0.5

    Parameters:
    -----------
    cli_args : list[str]
        List of CLI arguments (typically from argparse's unknown args).

    Returns:
    --------
    dict[str, Any]
        Dictionary of parsed keyword arguments.

    """
    def _parse_two_floats(raw: str) -> tuple[float, float]:
        parts = [p for p in raw.replace(",", " ").split() if p]
        if len(parts) != 2:
            raise ValueError(f"Expected two values, got: {parts}")
        return (float(parts[0]), float(parts[1]))


    Z_RANGE_KEYS = ("z_range_", "group_z_range_")

    kw = {}
    i = 0
    while i < len(cli_args):
        arg = cli_args[i]
        if not arg.startswith("--"):
            i += 1
            continue

        key = arg.lstrip("-")
        val: Any = None
        # Support --key val or --key=val
        if "=" in key:
            key, raw_val = key.split("=", 1)
            if key.startswith(Z_RANGE_KEYS):
                val = _parse_two_floats(raw_val)
            else:
                val = string_to_numeric(raw_val)
            i += 1
        else:
            if key.startswith(Z_RANGE_KEYS):

                # Expect two subsequent values
                if i + 2 >= len(cli_args):
                    raise ValueError(f"Expected two values after {key}, got fewer.")
                val = _parse_two_floats(f"{cli_args[i + 1]} {cli_args[i + 2]}")

                i += 3
            else:
                if i + 1 >= len(cli_args):
                    raise ValueError(f"Expected a value after {key}, got none.")
                val = string_to_numeric(cli_args[i + 1])
                i += 2
        
        # Alias handling for bcg_label and label
        if key == "label":
            key = "bcg_label"
        elif key.startswith("label_"):
            key = key.replace("label_", "bcg_label_")
        

        kw[key] = val


    return kw


def _process_z_range(val: Any) -> tuple[float, float] | list[tuple[float, float]]:
    """
    Processes z_range input into standardized format.
    Accept either:
      - [zmin, zmax] or (zmin, zmax)  -> (zmin, zmax)
      - [[zmin,zmax], [zmin,zmax], ...] -> [(zmin,zmax), ...]  (order matches subclusters list)

    Raises ValueError for invalid shapes.

    Parameters:
    -----------
    val : Any
        Input z_range value.

    Returns:
    --------
    tuple[float, float] | list[tuple[float, float]]
        Processed z_range value(s).
    """
    if val is None:
        raise ValueError("z_range value is None.")

    # Case 1: a single pair
    if isinstance(val, (list, tuple)) and len(val) == 2 and not any(isinstance(x, (list, tuple, dict)) for x in val):
        return (float(val[0]), float(val[1]))

    # Case 2: list of pairs
    if isinstance(val, (list, tuple)):
        out: list[tuple[float, float]] = []
        for item in val:
            if not (isinstance(item, (list, tuple)) and len(item) == 2):
                raise ValueError(f"Expected each z_range entry to be a 2-sequence, got: {item!r}")
            out.append((float(item[0]), float(item[1])))
        if not out:
            raise ValueError("z_range list is empty.")
        return out

    raise ValueError(f"Unsupported z_range type: {type(val).__name__}")


def _resolve_config_path(
        cluster_id: str,
        config: str | None,
        config_dir: str | None
    ) -> Path | None:
    """
    Resolves the configuration file path based on provided arguments.

    Priority:
    1) Explicit config file path from CLI.
    2) <config_dir>/<cluster_id>.json or <config_dir>/<cluster_id>_config.json
    3) <config_dir>/cluster_config.json

    Parameters:
    -----------
    cluster_id : str
        The cluster identifier.
    config : str | None
        Explicit config file path from CLI.
    config_dir : str | None
        Config directory from CLI.

    Returns:
    --------
    Path | None
        Resolved config file path or None if not found.
    """
    if config is not None:
        return Path(config).expanduser().resolve()
    
    cfg_dir = Path(find_first_val(config_dir, DEFAULT_CONFIG_DIR))
    cfg_dir = (Path(__file__).parent / cfg_dir).resolve() if not cfg_dir.is_absolute() else cfg_dir
    possible_names = [
        f"{cluster_id}.json",
        f"{cluster_id}_config.json",
        DEFAULT_CONFIG_FILE
    ]
    for name in possible_names:
        candidate = cfg_dir / name
        if candidate.is_file():
            return candidate.resolve()

    return None


def _select_cluster_block(config: dict[str, Any], cluster_id: str) -> dict[str, Any]:
    """
    Support "global config file" patterns and return the effective block for cluster_id.

    Accepted shapes:
      A) { "<cluster_id>": { ... }, "<cluster_id2>": { ... } }
      B) { ... }  (single block)

    Parameters:
    -----------
    config : dict[str, Any]
        Loaded JSON config.
    cluster_id : str
        Cluster identifier.

    Returns:
    --------
    dict[str, Any]
        Effective config block for the specified cluster_id.
    """
    # A) keyed by cluster_id
    if cluster_id in config and isinstance(config[cluster_id], dict):
        return config[cluster_id]

    # B) treat as a single block
    return config


def _normalize_config_dict(config: dict[str, Any]) -> dict[str, Any]:
    """
    Normalizes subcluster config dict keys to use consistent naming conventions.
    Rules:
      - "label" -> "bcg_label"
      - "label_X" -> "bcg_label_X"
      - Convert "z_range" and "group_z_range" lists to tuples.

    Parameters:
    -----------
    config : dict[str, Any]
        Input subcluster config dict.

    Returns:
    --------
    dict[str, Any]
        Normalized subcluster config dict.
    """

    out_dict: dict[str, Any] = dict(config)
    # Remove private keys
    out_dict = {k: v for k, v in out_dict.items() if not k.startswith("_")}


    # Alias label to bcg_label
    if "label" in config:
        out_dict["bcg_label"] = out_dict.pop("label")

    # Individual labels
    for key in list(out_dict.keys()):
        if key.startswith("label_"):
            new_key = key.replace("label_", "bcg_label_")
            out_dict[new_key] = out_dict.pop(key)
    # z_range and group_z_range
    for key in ("z_range", "group_z_range"):
        if key in out_dict and isinstance(out_dict[key], list) and len(out_dict[key]) == 2:
            out_dict[key] = (float(out_dict[key][0]), float(out_dict[key][1]))

    return out_dict


def _parse_levels(val: Any) -> tuple[float, float, float]:
    """
    Parses contour levels from various input formats into a tuple of three floats.

    Acceptable formats:
    - Comma-separated string: "1.0,0,12"
    - List or tuple of numbers: [1.0, 0, 12] or (1.0, 0, 12)

    Parameters:
    -----------
    val : Any
        Input value representing contour levels.

    Returns:
    --------
    tuple[float, float, float]
        Parsed contour levels as a tuple.
    """
    if val is None:
        return DEFAULT_LEVELS
    
    if isinstance(val, str):
        parts = [float(x.strip()) for x in val.split(",") if x.strip() != ""]
        if len(parts) != 3:
            raise ValueError(f"Expected three contour levels in string, got: {parts}")
        return (float(parts[0]), float(parts[1]), float(parts[2]))
    
    elif isinstance(val, (list, tuple)):
        if len(val) != 3:
            raise ValueError(f"Expected three contour levels in list/tuple, got: {val}")
        return (float(val[0]), float(val[1]), float(val[2]))
    else:
        raise TypeError(f"Unsupported type for contour levels: {type(val)}")


def build_parser() -> argparse.ArgumentParser:
    """
    Builds the argument parser for the x-ray pipeline driver script.

    Returns:
    -----------
    argparse.ArgumentParser
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Analyze cluster substructures and build subcluster configs.")

    # -- JSON config --
    # All defaults are None to prevent accidental overrides
    parser.add_argument("--config", type=str, default=None, help=f"Path to JSON config file [default: {DEFAULT_CONFIG_FILE}].")
    parser.add_argument("--config-dir", type=str, default=None, help=f"Directory for config files [default: {DEFAULT_CONFIG_DIR}].")

    # -- Required arguments --
    parser.add_argument("cluster_id", type=str, help="Cluster ID (e.g. '1327') or full RMJ/Abell name.")
    parser.add_argument("--subclusters", nargs="+", type=int, default=None,
        help="List of BCG indices or subcluster IDs (e.g., 2 6 7).")

    # -- Variants --
    parser.add_argument("--variant", type=str, default=None,
    help="Optional config variant name (e.g. fiducial, alt_core).")
    

    # -- Cluster options --
    parser.add_argument("--z-min", type=float, default=None, help="Minimum redshift for analysis.")
    parser.add_argument("--z-max", type=float, default=None, help="Maximum redshift for analysis.")
    parser.add_argument("--fov", type=float, default=None, help="Zoomed field of view in arcmin.")
    parser.add_argument("--fov-full", type=float, default=None, help="Full field of view in arcmin.")
    parser.add_argument("--ra-offset", type=float, default=None, help="RA offset in arcmin (Positive looks to the right).")
    parser.add_argument("--dec-offset", type=float, default=None, help="Dec offset in arcmin (Positive looks up).")

    # -- Global subcluster options --
    parser.add_argument("--radius", type=float, default=None, help="Default search radius for all subclusters (Mpc).")
    parser.add_argument("--z-range", nargs=2, type=float, default=None, metavar=("ZMIN", "ZMAX"), help="Global z_range for subclusters.")
    parser.add_argument("--colors", nargs="+", default=None, help="List of colors for subclusters (e.g., tab:green gold tab:cyan).")
    parser.add_argument("--labels", nargs="+", default=None, help="List of labels for subclusters.")
    parser.add_argument("--combined-groups", nargs="+", type=int, default=None, help="List of subcluster groups to combine (e.g., --combined 1 4 --combined 2 3).")
    parser.add_argument(
        "--manual-bcg",
        nargs=2,
        type=float,
        action="append",
        metavar=("RA", "DEC"),
        help="Manually specify a BCG position as RA DEC in degrees. Repeatable.",
    )

    # -- X-ray/density/display options --
    parser.add_argument("--xray-levels", type=str, default=None, help="Comma-separated contour levels for X-ray (e.g. '0.5,0,12').")
    parser.add_argument("--psf", type=float, default=None, help="PSF for X-ray smoothing.")
    parser.add_argument("--density-levels", type=int, default=None, help="Number of density contour levels.")
    parser.add_argument("--density-skip", type=int, default=None, help="Number of density levels to skip.")
    parser.add_argument("--density-bandwidth", type=float, default=None, help="Bandwidth for density estimation.")

    parser.add_argument("--save-plots", type=str2bool, nargs="?", const=True, default=None, help="Save plots? [default: True]")
    parser.add_argument("--show-plots", type=str2bool, nargs="?", const=True, default=None, help="Show plots? [default: False]")
    parser.add_argument("--plot-alt-regions", type=str2bool, nargs="?", const=True, default=None, help="Plot alternative regions? [default: False]")
    parser.add_argument("--run-pipeline", type=str2bool, nargs="?", const=True, default=None, help="Run full pipeline (redshift processing, plots)? [default: False]")
    parser.add_argument("--legend-loc", type=str, default=None, help="Legend location on plots (e.g. 'upper right').")
    parser.add_argument("--verbose", type=str2bool, nargs="?", const=True, default=None, help="Verbose output? [default: True]")    

    # -- Catch-all for extra per-subcluster kwargs (e.g. --color_4) --
    parser.add_argument("--subcluster-kwargs", nargs=argparse.REMAINDER, help="Extra per-subcluster args (e.g. --color_4 tab:orange --label_2 Main)")

    return parser

def main():
    """
    Main driver function for the x-ray pipeline.
    Parses arguments, constructs Cluster and subcluster configurations, and invokes analysis.
    """

    parser = build_parser()
    args, unknown = parser.parse_known_args()

    # -- Load config file if specified --
    config_path = _resolve_config_path(args.cluster_id, args.config, args.config_dir)
    config_raw: dict[str, Any] = {}
    if config_path is not None:
        print(f"Loading config from: {config_path}")
        config_raw = read_json(config_path)
    else:
        print("No config file found or specified; using CLI/default parameters.")

    # -- Extract proper cluster block --
    config = _select_cluster_block(config_raw, args.cluster_id) if config_raw else {}
    config = _normalize_config_dict(config)


    variant = args.variant
    variants = config.get("variants", {})

    if variant is not None:
        if variant not in variants:
            raise ValueError(f"Variant '{variant}' not found. Available: {list(variants)}")
        config = {**config, **variants[variant]}

    # -- Cluster kwargs --
    # Defaults, config, CLI (CLI takes precedence)
    fov = find_first_val(args.fov, config.get("fov"), DEFAULT_FOV)
    fov_full = find_first_val(args.fov_full, config.get("fov_full"), DEFAULT_FOV_FULL)
    ra_offset = find_first_val(args.ra_offset, config.get("ra_offset"), DEFAULT_RA_OFFSET)
    dec_offset = find_first_val(args.dec_offset, config.get("dec_offset"), DEFAULT_DEC_OFFSET)

    z_min = find_first_val(args.z_min, config.get("z_min"), config.get("zmin"), config.get("z-min"))
    z_max = find_first_val(args.z_max, config.get("z_max"), config.get("zmax"), config.get("z-max"))

    phot_levels = find_first_val(args.density_levels, config.get("density_levels"), config.get("phot_levels"), DEFAULT_PHOT_LEVELS)
    phot_skip = find_first_val(args.density_skip, config.get("density_skip"), config.get("phot_skip"), DEFAULT_PHOT_SKIP)
    bandwidth = find_first_val(args.density_bandwidth, config.get("density_bandwidth"), config.get("bandwidth"), DEFAULT_BANDWIDTH)

    psf = find_first_val(args.psf, config.get("psf"), DEFAULT_PSF)
    x_ray_levels_raw = find_first_val(args.xray_levels, config.get("xray_levels"), config.get("xray-levels"))
    contour_levels = _parse_levels(x_ray_levels_raw)

    cluster_kwargs: dict[str, Any] = {
        "fov": float(fov),
        "fov_full": float(fov_full),
        "ra_offset": float(ra_offset),
        "dec_offset": float(dec_offset),
        "psf": float(psf),
        "bandwidth": float(bandwidth),
        "phot_levels": int(phot_levels),
        "phot_skip": int(phot_skip),
        "contour_levels": contour_levels,
    }
    if z_min is not None:
        cluster_kwargs["z_min"] = float(z_min)
    if z_max is not None:
        cluster_kwargs["z_max"] = float(z_max)

    # -- Build the Cluster object --
    cluster = Cluster(args.cluster_id, **cluster_kwargs)
    cluster.populate(verbose=True)

    # -- Subcluster kwargs --
    subclusters = find_first_val(args.subclusters, config.get("subclusters"))
    if subclusters is None:
        raise ValueError("Subclusters must be specified via --subclusters or config file.")
    subclusters = [int(x) for x in subclusters]
    manual_list = [tuple(x) for x in (args.manual_bcg or [])]  # list[tuple[float, float]]

    # Global subcluster options
    radius = find_first_val(args.radius, config.get("radius"))
    z_range = find_first_val(args.z_range, config.get("z_range"), config.get("z-range"))
    colors = find_first_val(args.colors, config.get("colors"), config.get("color"))
    labels = find_first_val(args.labels, config.get("labels"), config.get("bcg_labels"), config.get("bcg_label"))

    subcluster_kwargs: dict[str, Any] = {}
    if labels is not None:
        if isinstance(labels, str):
            raise ValueError('Config key "labels" must be a list of strings, not a single string.')
        subcluster_kwargs["bcg_label"] = list(labels)
    if radius is not None:
        subcluster_kwargs["radius"] = float(radius)
    if z_range is not None:
        # Accept [zmin, zmax] or (zmin, zmax) or list of such pairs
        subcluster_kwargs["z_range"] = _process_z_range(z_range)
    if colors is not None:
        subcluster_kwargs["color"] = list(colors)
    if labels is not None:
        subcluster_kwargs["bcg_label"] = list(labels)

    # Subcluster groups
    combined = find_first_val(
        args.combined_groups,
        config.get("combined"),
        config.get("combined_groups"),
    )

    if combined is not None:
        subcluster_kwargs["combined"] = combined

    # Parse extra subcluster-keyword pairs from unknown or --subcluster-kwargs
    extra_kwargs: dict[str, Any] = {}

    if args.subcluster_kwargs is not None:
        extra_kwargs = _parse_subcluster_kwargs(args.subcluster_kwargs)
    elif unknown:
        extra_kwargs = _parse_subcluster_kwargs(unknown)

    # CLI overrides
    cli_remainder = args.subcluster_kwargs if args.subcluster_kwargs is not None else unknown
    if cli_remainder:
        extra_kwargs.update(_parse_subcluster_kwargs(cli_remainder))

    # Build subclusters
    subcluster_configs = build_subclusters(
        subclusters=subclusters,
        cluster=cluster,
        **subcluster_kwargs,
        **extra_kwargs
    )

    # -- Pipeline/ display options --
    save_plots = find_first_val(
        args.save_plots,
        config.get("save_plots"),
        config.get("save-plots"),
        DEFAULT_SAVE_PLOTS
    )
    show_plots = find_first_val(
        args.show_plots,
        config.get("show_plots"),
        config.get("show-plots"),
        DEFAULT_SHOW_PLOTS
    )
    plot_alt_regions = find_first_val(
        args.plot_alt_regions,
        config.get("plot_alt_regions"),
        config.get("plot-alt-regions"),
        DEFAULT_PLOT_ALT_REGIONS
    )
    run_pipeline = find_first_val(
        args.run_pipeline,
        config.get("run_pipeline"),
        config.get("run-pipeline"),
        DEFAULT_RUN_PIPELINE
    )
    legend_loc = find_first_val(
        config.get("legend_loc"),
        config.get("legend-loc"),
        DEFAULT_LEGEND_LOC
    )

    # -- Run analysis --
    analyze_cluster(
        cluster=cluster,
        subcluster_configs=subcluster_configs,
        manual_list=manual_list,
        legend_loc=legend_loc,
        combined_groups=combined,
        save_plots=save_plots,
        show_plots=show_plots,
        plot_alt_regions=plot_alt_regions,
        run_pipeline=run_pipeline
    )

if __name__ == "__main__":
    main()
