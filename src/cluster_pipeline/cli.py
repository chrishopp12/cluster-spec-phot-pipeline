"""Unified CLI for cluster-pipeline."""

from __future__ import annotations

import os

import click

from cluster_pipeline.plotting.common import setup_plot_style


@click.group()
@click.version_option(package_name="cluster-pipeline")
def main():
    """Cluster Pipeline — uniform first-look analysis for galaxy cluster merger candidates."""
    pass


# ------------------------------------
# run
# ------------------------------------

@main.command()
@click.argument("cluster_id")
@click.option("--base-path", type=click.Path(exists=True, file_okay=False),
              default=None, help="Base directory for cluster data (default: ~/XSorter/Clusters/).")
@click.option("--stages", multiple=True, type=click.Choice(["spec", "phot", "xray"], case_sensitive=False),
              help="Pipeline stages to run. Omit to run all.")
@click.option("--save", is_flag=True, help="Persist CLI overrides back to config.yaml.")
@click.option("--save-plots/--no-save-plots", default=True, help="Save generated figures.")
@click.option("--show-plots/--no-show-plots", default=False, help="Display figures interactively.")
# Spec/phot options
@click.option("--skip-redshifts", is_flag=True, help="Skip archival redshift queries.")
@click.option("--skip-photometry", is_flag=True, help="Skip archival photometry queries.")
@click.option("--skip-catalogs", is_flag=True, help="Skip catalog building.")
@click.option("--skip-cmd", is_flag=True, help="Skip color-magnitude fitting.")
@click.option("--skip-plots", is_flag=True, help="Skip plot generation.")
# Cluster overrides
@click.option("--ra", type=float, default=None, help="Override RA (degrees).")
@click.option("--dec", type=float, default=None, help="Override Dec (degrees).")
@click.option("--redshift", type=float, default=None, help="Override redshift.")
@click.option("--z-min", type=float, default=None, help="Minimum redshift for analysis.")
@click.option("--z-max", type=float, default=None, help="Maximum redshift for analysis.")
@click.option("--fov", type=float, default=None, help="Field of view (arcmin).")
@click.option("--psf", type=float, default=None, help="PSF smoothing (arcsec).")
@click.option("--survey", type=click.Choice(["legacy", "panstarrs"]), default=None, help="Photometry survey.")
# Xray options
@click.option("--subclusters", multiple=True, type=int, help="BCG indices for subclusters (e.g., --subclusters 2 6 7).")
@click.option("--radius", type=float, default=None, help="Subcluster search radius (Mpc).")
def run(cluster_id, base_path, stages, save, save_plots, show_plots,
        skip_redshifts, skip_photometry, skip_catalogs, skip_cmd, skip_plots,
        ra, dec, redshift, z_min, z_max, fov, psf, survey,
        subclusters, radius):
    """Run pipeline stages for a cluster."""
    from cluster_pipeline.models.cluster import Cluster
    from cluster_pipeline.config import load_config, save_config, merge_config

    setup_plot_style()

    # Determine which stages to run
    if not stages:
        stages = ("spec", "phot")  # Default to spec+phot if nothing specified

    # Build CLI overrides dict (only non-None values)
    cli_overrides = {}
    for key, val in [("ra", ra), ("dec", dec), ("redshift", redshift),
                     ("z_min", z_min), ("z_max", z_max), ("fov", fov),
                     ("psf", psf), ("survey", survey), ("radius", radius)]:
        if val is not None:
            cli_overrides[key] = val

    # Create and populate the Cluster
    cluster = Cluster(cluster_id, base_path=base_path, **cli_overrides)
    cluster.populate(verbose=True)

    # Load and merge config
    cfg = load_config(cluster.cluster_path)
    if cli_overrides:
        cfg = merge_config(cfg, cli_overrides)

    # Resolve z_range
    z_lo, z_hi = cluster.resolve_z_range(z_min=z_min, z_max=z_max)

    click.echo(f"\n{'='*50}")
    click.echo(f"  Cluster: {cluster.identifier}")
    click.echo(f"     Name: {cluster.name}")
    click.echo(f" Redshift: {cluster.redshift}")
    click.echo(f"  z range: [{z_lo:.4f}, {z_hi:.4f}]")
    click.echo(f"   Stages: {', '.join(stages)}")
    click.echo(f"{'='*50}\n")

    # Run spec/phot stages
    if "spec" in stages or "phot" in stages:
        from cluster_pipeline.pipelines.spec_phot import run_full_pipeline

        run_full_pipeline(
            cluster,
            z_min=z_lo,
            z_max=z_hi,
            show_plots=show_plots,
            save_plots=save_plots,
            skip_redshifts=skip_redshifts or ("spec" not in stages),
            skip_photometry=skip_photometry or ("phot" not in stages),
            skip_catalogs=skip_catalogs,
            skip_cmd=skip_cmd,
            skip_plots=skip_plots,
        )

    # Run xray stage
    if "xray" in stages:
        from cluster_pipeline.subclusters.builder import build_subclusters
        from cluster_pipeline.pipelines.xray import analyze_cluster

        subcluster_ids = subclusters if subclusters else cfg.get("subclusters_ids", (1,))
        subcluster_configs = build_subclusters(
            subclusters=subcluster_ids,
            cluster=cluster,
            radius=radius,
        )
        analyze_cluster(
            cluster,
            subcluster_configs,
            show_plots=show_plots,
            save_plots=save_plots,
        )

    # Persist config if --save
    if save:
        save_config(cluster.cluster_path, cfg)
        click.echo(f"\nConfig saved to {cluster.config_path}")

    click.echo("\nDone.")


# ------------------------------------
# init
# ------------------------------------

@main.command()
@click.argument("cluster_id")
@click.option("--base-path", type=click.Path(file_okay=False),
              default=None, help="Base directory for cluster data.")
@click.option("--ra", type=float, help="Right ascension (degrees).")
@click.option("--dec", type=float, help="Declination (degrees).")
@click.option("--redshift", type=float, default=None, help="Cluster redshift.")
def init(cluster_id, base_path, ra, dec, redshift):
    """Initialize a new cluster directory and config."""
    from cluster_pipeline.models.cluster import Cluster
    from cluster_pipeline.config import save_config, get_default_config

    kwargs = {}
    if ra is not None:
        kwargs["ra"] = ra
    if dec is not None:
        kwargs["dec"] = dec
    if redshift is not None:
        kwargs["redshift"] = redshift

    cluster = Cluster(cluster_id, base_path=base_path, **kwargs)
    cluster.populate(verbose=True)

    # Create config with defaults + discovered values
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

    save_config(cluster.cluster_path, cfg)

    click.echo(f"\nInitialized {cluster.identifier}")
    click.echo(f"  Directory: {cluster.cluster_path}")
    click.echo(f"  Config: {cluster.config_path}")
    click.echo(cluster)


# ------------------------------------
# info
# ------------------------------------

@main.command()
@click.argument("cluster_id")
@click.option("--base-path", type=click.Path(exists=True, file_okay=False),
              default=None, help="Base directory for cluster data.")
def info(cluster_id, base_path):
    """Display summary information for a cluster."""
    from cluster_pipeline.models.cluster import Cluster
    from cluster_pipeline.config import load_config

    cluster = Cluster(cluster_id, base_path=base_path)
    cluster.populate(verbose=False)

    click.echo(cluster)

    cfg = load_config(cluster.cluster_path)
    if cfg:
        click.echo("Config keys: " + ", ".join(cfg.keys()))
    else:
        click.echo("No config.yaml found. Run `cluster-pipeline init` to create one.")


# ------------------------------------
# list
# ------------------------------------

@main.command(name="list")
def list_clusters():
    """List all known clusters from clusters.csv."""
    from cluster_pipeline.models.cluster import Cluster
    import pandas as pd

    csv_path = os.path.join(Cluster.DEFAULT_BASE_PATH, Cluster.MASTER_FILE_NAME)
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
