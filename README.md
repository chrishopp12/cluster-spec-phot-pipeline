# Cluster Pipeline

Uniform first-look analysis pipeline for galaxy cluster merger candidates. Designed for the [X-SORTER](https://doi.org/10.48550/arXiv.2603.05596) (X-ray Survey Of meRging clusTErs in Redmapper) project, but broadly applicable to multi-wavelength cluster analysis.

Given a cluster identifier, the pipeline retrieves archival spectroscopy and photometry, fits the red sequence, identifies subclusters via BCG-bisector geometry, computes velocity dispersions, and generates publication-ready figures with X-ray and radio overlays.

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
cluster-pipeline init RMJ_0219 --ra 34.9673 --dec 1.4978 --redshift 0.365

# Run the full pipeline (all 10 stages)
cluster-pipeline run RMJ_0219

# Run specific stages (comma-separated, or repeat the flag)
cluster-pipeline run RMJ_0219 --stages spec,phot,matching,redseq
cluster-pipeline run RMJ_0219 --stages subclusters --subclusters 1,2
cluster-pipeline run RMJ_0219 --stages xray
cluster-pipeline run RMJ_0219 --stages radio

# List known clusters
cluster-pipeline list

# Show cluster summary
cluster-pipeline info RMJ_0219
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
| 10 | Radio | Overlay radio continuum contours on the optical + X-ray + density + BCG figure | `Radio/Images/{id}_radio_overlay.pdf` |

By default, all stages run. Use `--stages` to select a subset. X-ray (8) and radio
(10) are independent imaging stages: each needs only its user-provided FITS and is
skipped if absent.

## Configuration

Each cluster has a YAML config file at `{base_path}/{id}/config.yaml`. The config stores cluster identity, analysis parameters, BCG definitions, and subcluster groupings.

```bash
# Override parameters for a single run
cluster-pipeline run RMJ_0219 --fov 8.0 --redshift 0.366

# Persist overrides to config.yaml
cluster-pipeline run RMJ_0219 --fov 8.0 --save
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
| `--legend-loc` | Legend corner for field figures (optical/X-ray/radio) | `upper right` |
| `--radius` | Subcluster search radius (arcmin) | 2.5 |
| `--z-min` / `--z-max` | Redshift range for member selection | redshift +/- 0.15 |

The legend corner is a per-cluster property (set by where the cluster sits in the field) and is shared across all `plot_optical` field figures, so optical, X-ray, and radio legends stay in the same place.

### Radio Overlay

The radio stage overlays continuum contours on the optical + X-ray + density + BCG composite. Place a radio FITS image under `{cluster}/Radio/` and point `radio.filename` at it in `config.yaml` — the path is relative to `Radio/`, so it may include a team-product subdirectory (e.g. `PSZ2..._FITS/...T50kpcSUB.fits`). Contours follow a geometric `start_sigma * sigma * step^k` ladder over the robust (MAD) noise of the map.

```bash
cluster-pipeline run RMJ_0219 --stages radio \
    --radio-file lotss_image.fits --radio-fov 20 --radio-start-sigma 3 --save
```

| Flag | Description | Default |
|------|-------------|---------|
| `--radio-file` | Radio FITS path, relative to `Radio/` | — |
| `--radio-fov` | Radio overlay zoom FOV (arcmin) | cluster FOV |
| `--radio-start-sigma` | Lowest contour, in units of sigma | 4.0 |
| `--radio-n-levels` | Number of contour levels | 12 |
| `--radio-contour-step` | Geometric step between levels | sqrt(2) |

Like X-ray, radio FITS are user-provided; the stage is skipped if none is configured. Per-cluster contour tuning, compact-source masking, and styling live in the `radio:` block of `config.yaml`.

## Data Layout

Each cluster lives in its own directory under the base path (resolved as `--base-path` > `CLUSTER_BASE_PATH` env var > `./clusters`):

```
{base_path}/{cluster_id}/
    config.yaml           Analysis configuration
    BCGs.csv              BCG catalog with photometry
    Redshifts/            Per-source + combined redshift catalogs
    Photometry/           Per-survey photometry + matched catalogs
    Members/              Cluster + subcluster member catalogs
    Xray/                 X-ray FITS images
    Radio/                Radio FITS images (+ Radio/Images/ overlay output)
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
    radio/          Radio image processing (noise, masking, contour ladder)
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
- Radio overlays require a continuum FITS image placed in `{cluster}/Radio/`

## Testing

```bash
pytest                        # Unit + integration tests
pytest -m "not integration"   # Unit tests only (fast, no pipeline run)
pytest tests/test_config.py   # Config system tests
pytest tests/test_models.py   # Data model tests
pytest tests/test_radio.py    # Radio image utilities
```

The unit tests are self-contained (no data required). An offline integration test
(`tests/test_integration.py`) runs the matching → red-sequence → subclusters slice
against a minimal committed fixture at `tests/data/RMJ_0219/` — RMJ 0219 cut to the
central 4.5′ with archival redshifts only, plus a BCG seed catalog so matching skips
its redMaPPer lookup. No network, CASJobs credentials, or X-ray FITS image are needed.

Reproduce the integration run by hand (copy first so products don't land in the fixture):

```bash
cp -r tests/data /tmp/cp_test
cluster-pipeline run RMJ_0219 --base-path /tmp/cp_test \
    --stages matching,redseq,subclusters --no-save-plots
```

## License

This project is part of ongoing research at UC Davis. Contact for usage terms.
