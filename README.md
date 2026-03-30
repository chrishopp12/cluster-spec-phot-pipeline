# Cluster Pipeline

Uniform first-look analysis pipeline for galaxy cluster merger candidates. Designed for the [X-SORTER](https://arxiv.org/) (X-ray Survey Of meRging clusTErs in Redmapper) project, but broadly applicable to multi-wavelength cluster analysis.

Given a cluster identifier, the pipeline retrieves archival spectroscopy and photometry, fits the red sequence, identifies subclusters via BCG-bisector geometry, computes velocity dispersions, and generates publication-ready figures with X-ray overlays.

## Installation

Requires Python 3.11+.

```bash
# Editable install (recommended for development)
pip install -e .

# With test dependencies
pip install -e ".[dev]"
```

This installs the `cluster-pipeline` command-line tool.

## Quick Start

```bash
# Initialize a new cluster
cluster-pipeline init "RMJ 1327" --ra 201.85 --dec 53.78 --redshift 0.254

# Run the full pipeline (all 9 stages)
cluster-pipeline run "RMJ 1327"

# Run specific stages
cluster-pipeline run "RMJ 1327" --stages spec phot matching redseq
cluster-pipeline run "RMJ 1327" --stages subclusters --subclusters 2 6 7
cluster-pipeline run "RMJ 1327" --stages xray

# List known clusters
cluster-pipeline list

# Show cluster summary
cluster-pipeline info "RMJ 1327"
```

## Pipeline Stages

| Stage | Name | Description | Key Outputs |
|-------|------|-------------|-------------|
| 1 | Spectroscopy | Retrieve archival redshifts (NED, SDSS, DESI, DEIMOS) | `archival_z.csv`, `deimos.csv` |
| 2 | Photometry | Query Legacy Survey + Pan-STARRS photometry | `photometry_{survey}.csv` |
| 3 | Matching | Cross-match spectroscopic and photometric catalogs, identify BCGs | `combined_redshifts.csv`, `BCGs.csv` |
| 4 | Red Sequence | Fit red sequence, select cluster members | `cluster_members.csv` |
| 5 | Subclusters | Build subclusters from BCG pairs via bisector geometry | `Subcluster` objects |
| 6 | Assignment | Assign galaxies to subcluster regions | Per-subcluster member catalogs |
| 7 | Statistics | Velocity dispersions, redshift analysis, summary tables | LaTeX deluxetables |
| 8 | X-ray | Process XMM images: smoothing, GGM edge detection, contours | Smoothed FITS, GGM panels |
| 9 | Plots | Multi-panel publication figures | ~60 PDF figures per cluster |

By default, all stages run. Use `--stages` to select a subset.

## Configuration

Each cluster has a YAML config file at `~/Clusters/{id}/config.yaml`. The config stores cluster identity, analysis parameters, BCG definitions, and subcluster groupings.

```bash
# Override parameters for a single run
cluster-pipeline run "RMJ 1327" --fov 8.0 --redshift 0.255

# Persist overrides to config.yaml
cluster-pipeline run "RMJ 1327" --fov 8.0 --save
```

### Key Parameters

| Flag | Description | Default |
|------|-------------|---------|
| `--fov` | Cluster FOV for subcluster-scale images (arcmin) | 6.0 |
| `--fov-full` | Wide-field FOV for full-cluster images (arcmin) | 30.0 |
| `--ra-offset` | RA offset for image centering (arcmin) | 0.0 |
| `--dec-offset` | Dec offset for image centering (arcmin) | 0.0 |
| `--psf` | PSF smoothing for X-ray images (arcsec) | 8.0 |
| `--survey` | Photometry survey (`legacy` or `panstarrs`) | `legacy` |
| `--radius` | Subcluster search radius (Mpc) | 2.5 |
| `--z-min` / `--z-max` | Redshift range for member selection | redshift +/- 0.15 |

## Data Layout

```
~/Clusters/{cluster_id}/
    config.yaml           Analysis configuration
    BCGs.csv              BCG catalog with photometry
    Redshifts/            Per-source + combined redshift catalogs
    Photometry/           Per-survey photometry + matched catalogs
    Members/              Cluster + subcluster member catalogs
    Xray/                 X-ray FITS images
    Images/               Publication figures
    Tables/               LaTeX deluxetables + machine-readable tables
```

## Package Structure

```
src/cluster_pipeline/
    cli.py          Unified click CLI
    config.py       YAML config load/save/merge
    constants.py    All shared defaults
    models/         Cluster, Subcluster, BCG, Region data classes
    catalog/        Spectroscopy, photometry, matching, red sequence
    subclusters/    Bisector geometry, member assignment, statistics
    xray/           X-ray image processing and analysis
    plotting/       All visualization (separated from analysis)
    io/             CSV, FITS, LaTeX I/O
    utils/          SkyCoord helpers, cosmology, NED/SIMBAD resolvers
    pipelines/      Stage orchestration
```

## Requirements

Core dependencies (installed automatically):

- numpy, pandas, scipy, scikit-learn, scikit-image
- astropy, astroquery, reproject
- matplotlib
- click, pyyaml, requests

External data requirements:
- Archival spectroscopy queries require network access (NED, SDSS, DESI)
- Photometry queries require CASJobs credentials (Legacy Survey, Pan-STARRS)
- X-ray analysis requires XMM FITS images placed in `{cluster}/Xray/`

## Testing

```bash
pytest                    # Run all tests
pytest tests/test_config.py   # Config system tests
pytest tests/test_models.py   # Data model tests
```

Integration testing uses test data at `pipeline_v2_tests/RMJ_1327/`:

```bash
cluster-pipeline run "RMJ 1327" --base-path path/to/pipeline_v2_tests
```

## License

This project is part of ongoing research at UC Davis. Contact for usage terms.
