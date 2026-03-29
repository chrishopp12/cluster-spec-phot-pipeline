#!/usr/bin/env python3
"""
cli.py

Unified CLI for Cluster Pipeline
---------------------------------------------------------

Entry point for all pipeline operations. Installed as ``cluster-pipeline``
via pyproject.toml.

Commands:
  run   — Run pipeline stages for a cluster
  init  — Initialize a new cluster directory and config
  info  — Display cluster summary
  list  — List all known clusters from clusters.csv
"""

from __future__ import annotations

import os

import click

from cluster_pipeline.plotting.common import setup_plot_style


@click.group()
@click.version_option(package_name="cluster-pipeline")
def main():
    """Cluster Pipeline — uniform first-look analysis for galaxy cluster merger candidates."""
    pass


# ====================================================================
# run
# ====================================================================

@main.command()
@click.argument("cluster_id")
@click.option("--base-path", type=click.Path(exists=True, file_okay=False),
              default=None, help="Base directory for cluster data.")
@click.option("--stages", multiple=True,
              type=click.Choice(["spec", "phot", "matching", "redseq", "subclusters", "xray"], case_sensitive=False),
              help="Pipeline stages to run. Default: spec + phot + matching + redseq.")
@click.option("--save", is_flag=True, help="Persist CLI overrides back to config.yaml.")
@click.option("--save-plots/--no-save-plots", default=True, help="Save generated figures.")
@click.option("--show-plots/--no-show-plots", default=False, help="Display figures interactively.")
# Cluster overrides
@click.option("--ra", type=float, default=None, help="Override RA (degrees).")
@click.option("--dec", type=float, default=None, help="Override Dec (degrees).")
@click.option("--redshift", type=float, default=None, help="Override redshift.")
@click.option("--z-min", type=float, default=None, help="Minimum redshift for analysis.")
@click.option("--z-max", type=float, default=None, help="Maximum redshift for analysis.")
@click.option("--fov", type=float, default=None, help="Field of view (arcmin).")
@click.option("--psf", type=float, default=None, help="PSF smoothing (arcsec).")
@click.option("--survey", type=click.Choice(["legacy", "panstarrs"]), default=None)
# Xray / subcluster options
@click.option("--subclusters", multiple=True, type=int,
              help="BCG indices for subclusters (e.g., --subclusters 2 --subclusters 6).")
@click.option("--radius", type=float, default=None, help="Subcluster search radius (Mpc).")
def run(cluster_id, base_path, stages, save, save_plots, show_plots,
        ra, dec, redshift, z_min, z_max, fov, psf, survey,
        subclusters, radius):
    """Run pipeline stages for a cluster."""
    from cluster_pipeline.models.init_cluster import cluster_init
    from cluster_pipeline.config import load_config, save_config, merge_config

    setup_plot_style()

    # --- Build CLI overrides ---
    cli_overrides = {}
    for key, val in [("ra", ra), ("dec", dec), ("redshift", redshift),
                     ("z_min", z_min), ("z_max", z_max), ("fov", fov),
                     ("psf", psf), ("survey", survey), ("radius", radius)]:
        if val is not None:
            cli_overrides[key] = val

    # --- Initialize cluster ---
    cluster = cluster_init(cluster_id, base_path=base_path, verbose=True, **cli_overrides)

    # --- Load and merge config ---
    cfg = load_config(cluster.cluster_path)
    if cli_overrides:
        cfg = merge_config(cfg, cli_overrides)

    # --- Resolve z_range ---
    z_lo, z_hi = cluster.resolve_z_range(z_min=z_min, z_max=z_max)

    # --- Determine stages ---
    if not stages:
        stages = ("spec", "phot", "matching", "redseq")

    click.echo(f"\n{'='*50}")
    click.echo(f"  Cluster: {cluster.identifier}")
    click.echo(f"     Name: {cluster.name}")
    click.echo(f" Redshift: {cluster.redshift}")
    click.echo(f"  z range: [{z_lo:.4f}, {z_hi:.4f}]")
    click.echo(f"   Stages: {', '.join(stages)}")
    click.echo(f"{'='*50}\n")

    # --- Stage 1: Spectroscopy ---
    archival_df = None
    deimos_df = None
    if "spec" in stages:
        from cluster_pipeline.catalog.spectroscopy import run_spectroscopy
        click.echo("--- Stage 1: Spectroscopy ---")
        archival_df, deimos_df = run_spectroscopy(cluster)

    # --- Stage 2: Photometry ---
    phot_dfs = None
    if "phot" in stages:
        from cluster_pipeline.catalog.photometry import run_photometry
        click.echo("\n--- Stage 2: Photometry ---")
        phot_dfs = run_photometry(cluster, surveys=["legacy", "panstarrs"])

    # --- Stage 3: Matching ---
    bcgs = None
    if "matching" in stages:
        from cluster_pipeline.catalog.matching import run_matching
        click.echo("\n--- Stage 3: Catalog Matching ---")
        combined_df, matched_dfs, bcgs = run_matching(
            cluster,
            archival_df=archival_df,
            deimos_df=deimos_df,
            phot_dfs=phot_dfs,
        )

        # Supplement BCGs with manual entries from config.yaml
        if "bcgs" in cfg:
            from cluster_pipeline.models.bcg import BCG
            existing_ids = {b.bcg_id for b in bcgs}
            for bid, bcfg in cfg["bcgs"].items():
                bid = int(bid)
                if bid not in existing_ids:
                    manual_bcg = BCG.from_config(bid, bcfg)
                    bcgs.append(manual_bcg)
                    click.echo(f"  Added manual BCG {bid} ('{manual_bcg.label}') from config.yaml")

            # Rewrite BCGs.csv with the complete list
            _write_bcgs_csv(cluster, bcgs)

    # --- Stage 4: Red Sequence ---
    if "redseq" in stages:
        from cluster_pipeline.catalog.redsequence import run_redsequence
        click.echo("\n--- Stage 4: Red Sequence ---")
        members_df = run_redsequence(cluster)

        # CMD diagnostic plots (after red sequence fitting)
        from cluster_pipeline.plotting.cmd import run_cluster_plots, plot_all_color_magnitude

        click.echo("\n--- CMD Plots ---")
        # 2x3 CMD overview (both surveys)
        try:
            legacy_phot = os.path.join(cluster.photometry_path, "photometry_legacy.csv")
            panstarrs_phot = os.path.join(cluster.photometry_path, "photometry_panstarrs.csv")
            legacy_matched = os.path.join(cluster.photometry_path, "legacy_matched.csv")
            panstarrs_matched = os.path.join(cluster.photometry_path, "panstarrs_matched.csv")

            import pandas as pd
            leg_df = pd.read_csv(legacy_phot) if os.path.isfile(legacy_phot) else pd.DataFrame()
            pan_df = pd.read_csv(panstarrs_phot) if os.path.isfile(panstarrs_phot) else pd.DataFrame()
            leg_spec = pd.read_csv(legacy_matched) if os.path.isfile(legacy_matched) else None
            pan_spec = pd.read_csv(panstarrs_matched) if os.path.isfile(panstarrs_matched) else None

            if not leg_df.empty or not pan_df.empty:
                plot_all_color_magnitude(
                    cluster=cluster,
                    panstarrs_df=pan_df if not pan_df.empty else pd.DataFrame(columns=leg_df.columns),
                    legacy_df=leg_df if not leg_df.empty else pd.DataFrame(columns=pan_df.columns),
                    panstarrs_spec=pan_spec,
                    legacy_spec=leg_spec,
                    show_plots=show_plots,
                    save_plots=save_plots,
                    save_path=os.path.join(cluster.photometry_path, "Images", "CMDs.pdf"),
                )
                click.echo("    CMD overview (2x3): OK")
        except Exception as e:
            click.echo(f"    CMD overview FAILED: {e}")

        # Per-survey CMD plots
        for s in ["legacy", "panstarrs"]:
            for ct in ["g-r", "r-i", "g-i"]:
                try:
                    run_cluster_plots(
                        cluster, fov_size=cluster.fov, survey=s, color_type=ct,
                        ra_offset=cluster.ra_offset, dec_offset=cluster.dec_offset,
                        bandwidth=cluster.bandwidth,
                        save_plots=save_plots, show_plots=show_plots,
                    )
                    click.echo(f"    {s} {ct}: OK")
                except Exception as e:
                    click.echo(f"    {s} {ct}: FAILED ({e})")

    # --- Subcluster building + assignment + stats + plots ---
    subcluster_list = None
    if "subclusters" in stages or "xray" in stages:
        from cluster_pipeline.subclusters.builder import build_subclusters
        from cluster_pipeline.pipelines.xray import run_subcluster_analysis

        click.echo("\n--- Stage 5: Subcluster Building ---")

        # Inject CLI subcluster IDs into config if provided
        if subclusters:
            sc_dict = {}
            for sid in subclusters:
                entry = {"bcg_id": sid}
                if radius is not None:
                    entry["radius_mpc"] = radius
                sc_dict[sid] = entry
            cfg["subclusters"] = sc_dict

        subcluster_list = build_subclusters(cluster, bcgs=bcgs, config=cfg)

        click.echo("\n--- Stages 6-7: Subcluster Analysis ---")
        run_subcluster_analysis(cluster, subclusters=subcluster_list,
                                save_plots=save_plots, show_plots=show_plots)

    # --- X-ray processing (independent of subclusters) ---
    if "xray" in stages:
        from cluster_pipeline.pipelines.xray import run_xray_imaging

        click.echo("\n--- Stage 8: X-ray Processing ---")
        run_xray_imaging(cluster, save_plots=save_plots, show_plots=show_plots)

    # --- Persist config if --save ---
    if save:
        save_config(cluster.cluster_path, cfg)
        click.echo(f"\nConfig saved to {cluster.config_path}")

    click.echo("\nDone.")


# ====================================================================
# init
# ====================================================================

@main.command()
@click.argument("cluster_id")
@click.option("--base-path", type=click.Path(file_okay=False),
              default=None, help="Base directory for cluster data.")
@click.option("--ra", type=float, help="Right ascension (degrees).")
@click.option("--dec", type=float, help="Declination (degrees).")
@click.option("--redshift", type=float, default=None, help="Cluster redshift.")
def init(cluster_id, base_path, ra, dec, redshift):
    """Initialize a new cluster directory and config."""
    from cluster_pipeline.models.init_cluster import cluster_init
    from cluster_pipeline.config import save_config, get_default_config

    overrides = {}
    if ra is not None:
        overrides["ra"] = ra
    if dec is not None:
        overrides["dec"] = dec
    if redshift is not None:
        overrides["redshift"] = redshift

    cluster = cluster_init(cluster_id, base_path=base_path, verbose=True, **overrides)
    cluster.ensure_directories()

    # Build config from defaults + discovered values
    cfg = get_default_config()
    cfg["identifier"] = cluster.identifier
    if cluster.name:
        cfg["name"] = cluster.name
    if cluster.ra is not None:
        cfg["ra"] = cluster.ra
    if cluster.dec is not None:
        cfg["dec"] = cluster.dec
    if cluster.redshift is not None:
        cfg["redshift"] = cluster.redshift
    if cluster.z_min is not None and cluster.z_max is not None:
        cfg["z_range"] = [cluster.z_min, cluster.z_max]

    save_config(cluster.cluster_path, cfg)

    click.echo(f"\nInitialized {cluster.identifier}")
    click.echo(f"  Directory: {cluster.cluster_path}")
    click.echo(f"     Config: {cluster.config_path}")
    click.echo(cluster)


# ====================================================================
# info
# ====================================================================

@main.command()
@click.argument("cluster_id")
@click.option("--base-path", type=click.Path(exists=True, file_okay=False),
              default=None, help="Base directory for cluster data.")
def info(cluster_id, base_path):
    """Display summary information for a cluster."""
    from cluster_pipeline.models.init_cluster import cluster_init
    from cluster_pipeline.config import load_config

    cluster = cluster_init(cluster_id, base_path=base_path, verbose=False)
    click.echo(cluster)

    cfg = load_config(cluster.cluster_path)
    if cfg:
        click.echo("Config keys: " + ", ".join(cfg.keys()))
    else:
        click.echo("No config.yaml found. Run `cluster-pipeline init` to create one.")


# ====================================================================
# list
# ====================================================================

@main.command(name="list")
@click.option("--base-path", type=click.Path(exists=True, file_okay=False),
              default=None, help="Base directory for cluster data.")
def list_clusters(base_path):
    """List all known clusters from clusters.csv."""
    from cluster_pipeline.models.cluster import DEFAULT_BASE_PATH
    import pandas as pd

    bp = base_path or str(DEFAULT_BASE_PATH)
    csv_path = os.path.join(bp, "clusters.csv")

    if not os.path.exists(csv_path):
        click.echo(f"No clusters.csv found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        click.echo("clusters.csv is empty.")
        return

    click.echo(f"{'Identifier':<25} {'Name':<35} {'z':>8}")
    click.echo("-" * 70)
    for _, row in df.iterrows():
        ident = str(row.get("identifier", ""))
        name = str(row.get("name", ""))[:33]
        z = row.get("redshift", "")
        z_str = f"{float(z):.4f}" if z and str(z).strip() else ""
        click.echo(f"{ident:<25} {name:<35} {z_str:>8}")


# ====================================================================
# Helpers
# ====================================================================

def _write_bcgs_csv(cluster, bcgs: list) -> None:
    """Write the complete BCG list to BCGs.csv.

    Includes both redMaPPer and manual BCGs. This is a data product
    with all available photometry and spectroscopy for each BCG.
    """
    import pandas as pd

    rows = []
    for bcg in sorted(bcgs, key=lambda b: b.bcg_id):
        row = {
            "BCG_priority": bcg.bcg_id,
            "RA": bcg.ra,
            "Dec": bcg.dec,
            "BCG_probability": bcg.probability,
            "z": bcg.z,
            "sigma_z": bcg.sigma_z,
            "label": bcg.label,
        }
        # Include photometry if available (from spec+phot matching)
        for attr in ("gmag", "rmag", "imag", "phot_source",
                     "lum_weight_g", "lum_weight_r", "lum_weight_i",
                     "g_r", "r_i", "g_i"):
            row[attr] = getattr(bcg, attr, None)
        rows.append(row)

    df = pd.DataFrame(rows)
    path = cluster.bcg_file
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Wrote BCGs.csv ({len(df)} BCGs) -> {path}")
