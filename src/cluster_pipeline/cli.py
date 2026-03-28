"""Unified CLI for cluster-pipeline."""

import click


@click.group()
@click.version_option(package_name="cluster-pipeline")
def main():
    """Cluster Pipeline — uniform first-look analysis for galaxy cluster merger candidates."""
    pass


@main.command()
@click.argument("cluster_id")
@click.option("--stages", multiple=True, help="Pipeline stages to run (spec, phot, xray).")
@click.option("--save", is_flag=True, help="Persist CLI overrides back to config.yaml.")
@click.option("--save-plots", is_flag=True, default=True, help="Save generated figures.")
@click.option("--show-plots", is_flag=True, default=False, help="Display figures interactively.")
def run(cluster_id, stages, save, save_plots, show_plots):
    """Run pipeline stages for a cluster."""
    click.echo(f"[placeholder] Would run stages {stages or '(all)'} for {cluster_id}")


@main.command()
@click.argument("cluster_id")
@click.option("--ra", type=float, help="Right ascension (degrees).")
@click.option("--dec", type=float, help="Declination (degrees).")
def init(cluster_id, ra, dec):
    """Initialize a new cluster directory and config."""
    click.echo(f"[placeholder] Would initialize {cluster_id}")


@main.command()
@click.argument("cluster_id")
def info(cluster_id):
    """Display summary information for a cluster."""
    click.echo(f"[placeholder] Would show info for {cluster_id}")


@main.command(name="list")
def list_clusters():
    """List all known clusters."""
    click.echo("[placeholder] Would list clusters from clusters.csv")
