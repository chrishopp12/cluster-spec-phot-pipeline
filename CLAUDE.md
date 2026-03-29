# Cluster Pipeline v2

## Architecture

Installable Python package (`src/cluster_pipeline/`) for uniform first-look analysis of galaxy cluster merger candidates.

### Package Structure

```
src/cluster_pipeline/
  models/        - Cluster, Subcluster, BCG, Region data classes
  catalog/       - Spectroscopic/photometric retrieval, matching, red sequence
  subclusters/   - Bisector geometry, member assignment, statistics, builder
  xray/          - X-ray image processing and analysis
  plotting/      - ALL visualization (separated from analysis)
  io/            - CSV, FITS, LaTeX I/O
  utils/         - SkyCoord helpers, cosmology, NED/SIMBAD resolvers
  pipelines/     - Stage orchestration (xray driver)
  config.py      - YAML config load/save/merge
  constants.py   - Single source of truth for all defaults
  cli.py         - Unified click CLI
```

### Pipeline Stages

```
Stage 1: Spectroscopy (catalog/spectroscopy.py)     → archival_z.csv, deimos.csv
Stage 2: Photometry (catalog/photometry.py)          → photometry_{survey}.csv
Stage 3: Matching (catalog/matching.py)              → combined_redshifts.csv, {survey}_matched.csv, BCGs.csv
Stage 4: Red Sequence (catalog/redsequence.py)       → Members/cluster_members.csv
Stage 5: Subcluster Building (subclusters/builder.py) → list[Subcluster] objects
Stage 6: Member Assignment (subclusters/assignment.py) → subcluster member catalogs
Stage 7: Statistics (subclusters/statistics.py)       → velocity dispersions, LaTeX tables
Stage 8: X-ray Processing (xray/image.py + analysis.py) → smoothed images, GGM
Stage 9: Plots (plotting/*.py)                        → publication figures
```

### Object Model

Hybrid: structural classes for hierarchy, DataFrames for bulk galaxy data.

- **Cluster** — lean dataclass (~190 lines): metadata + path properties
- **Subcluster** — captures primary/border/member BCG role distinctions
- **BCG** — lightweight, ~5-10 per cluster, from_config/from_dataframe_row
- **Region** — bisector geometry result container with matches() method
- Galaxy catalogs stay as DataFrames (10k+ rows, vectorized ops)

### Configuration

Two-layer system:
- **YAML** (`~/Clusters/{id}/config.yaml`) — single source of truth per cluster
  - Stores: cluster identity, analysis params, all BCGs (including manual), subcluster definitions, groups
- **CLI args** — ephemeral overrides for current run only
- **`--save`** flag persists CLI overrides back to YAML
- `clusters.csv` remains as lightweight index only

### Data Products

Organized per-cluster:
```
{cluster_path}/
  config.yaml           - Analysis configuration
  BCGs.csv              - Self-contained BCG catalog (with photometry)
  Redshifts/            - Archived per-source + combined redshifts
  Photometry/           - Per-survey photometry + matched catalogs
  Members/              - Cluster + subcluster member catalogs (deduplicated)
  Xray/                 - X-ray FITS images
  Images/               - Subcluster plots
  Tables/               - LaTeX deluxetables + machine-readable tables
```

### Import Layering (to prevent circular imports)

```
models/     -> constants, utils only
io/         -> models, constants, utils
catalog/    -> models, io, utils
subclusters/-> models, io, utils, catalog (for matching utilities)
xray/       -> models, io, utils
plotting/   -> models, io, utils, subclusters, xray, catalog
pipelines/  -> everything (orchestration layer)
cli.py      -> pipelines, models, config
```

### Conventions

- **Columns**: RA, Dec (caps), z, sigma_z, gmag, rmag, imag, g_r, r_i, g_i, lum_weight_r, spec_source, phot_source, member_type
- **Missing values**: NaN (never -9999 or inf)
- **SkyCoord**: use `utils/coordinates.py` helpers (`make_skycoord`, `skycoord_from_df`)
- **Plotting**: `setup_plot_style()` from CLI entry points (not at import time)
- **Defaults**: all in `constants.py` (not hardcoded in modules)
- **Docstrings**: NumPy-style, detailed module headers (see catalog/photometry.py as reference)
- **No Claude co-author on commits**

## Commands

```bash
pip install -e .                                           # Install dev mode
pip install -e ".[dev]"                                    # With test deps
cluster-pipeline run <ID> --stages spec phot matching redseq  # Optical pipeline
cluster-pipeline run <ID> --stages subclusters --subclusters 2 6 7  # Subcluster analysis
cluster-pipeline run <ID> --stages xray                    # X-ray processing
cluster-pipeline run <ID> --base-path /path/to/data        # Custom data location
cluster-pipeline init <ID> --ra RA --dec DEC               # Initialize new cluster
cluster-pipeline info <ID>                                 # Show cluster summary
cluster-pipeline list                                      # List all clusters
pytest                                                     # Run tests
```

## Key Files

- `constants.py` — change defaults here, not in individual modules
- `config.py` — YAML config system (load, save, merge with CLI)
- `models/cluster.py` — lean dataclass, `cluster_init()` in `models/init_cluster.py`
- `models/subcluster.py` — primary_bcg vs border_bcg role distinction
- `subclusters/geometry.py` — bisector math (the most delicate code)
- `subclusters/builder.py` — builds list[Subcluster] from config + BCGs
- `io/tables.py` — all LaTeX table generation in one place
- `plotting/subclusters.py` — 8 multi-panel figure functions

## Testing

- `pipeline_v2_tests/RMJ_1327/` — test cluster with copied data (protects real data)
- `tests/test_config.py` — 15 tests for YAML config system
- `tests/test_models.py` — 24 tests for BCG/Region/Subcluster classes
- Always use `--base-path .../pipeline_v2_tests` when testing

## Refactor Status

- v1 tagged on main branch (X-SORTER paper)
- v2 refactor on `v2-refactor` branch
- All v1 code removed (shims deleted)
- Full pipeline runs end-to-end on RMJ 1327
- ~60 plots, 7 table files, 10+ catalog files generated per cluster run
- Systematic cleanup pass completed (2026-03-28):
  - Dead code: 773 lines removed from pipelines/xray.py, obsolete functions deleted
  - CLI: `--verbose/--no-verbose` and `--diagnostics/--no-diagnostics` flags added
  - Constants: all analysis defaults centralized in constants.py
  - Structure: build_subcluster_summary decomposed, plotting moved out of statistics.py
  - Headers: all modules have full reference-style headers (see catalog/photometry.py)
  - Exception messages include type(e).__name__ for debugging
  - Test data at ~/Clusters/pipeline_v2_tests/RMJ_1327/
