# Cluster Pipeline v2

## Architecture

Installable Python package (`src/cluster_pipeline/`) for uniform first-look analysis of galaxy cluster merger candidates.

### Package Structure

```
src/cluster_pipeline/
  models/        - Cluster, Subcluster, BCG, Region data classes
  catalog/       - Spectroscopic/photometric data retrieval and catalog building
  subclusters/   - Bisector geometry, member assignment, statistics
  xray/          - X-ray image processing and analysis
  plotting/      - ALL visualization (separated from analysis)
  io/            - CSV, FITS, LaTeX I/O
  utils/         - SkyCoord helpers, cosmology, NED/SIMBAD resolvers
  pipelines/     - Stage orchestration (spec_phot, xray)
  config.py      - YAML config load/save/merge
  constants.py   - Single source of truth for all defaults
  cli.py         - Unified click CLI
```

### Object Model

Hybrid: structural classes for hierarchy, DataFrames for bulk galaxy data.

- **Cluster** - central metadata + path management
- **Subcluster** - captures primary/border/member BCG role distinctions
- **BCG** - lightweight, ~5-10 per cluster
- **Region** - bisector geometry result container
- Galaxy catalogs stay as DataFrames (10k+ rows, vectorized ops)

### Configuration

Two-layer system:
- **YAML** (`~/Clusters/{id}/config.yaml`) - single source of truth per cluster
- **CLI args** - ephemeral overrides for current run only
- **`--save`** flag persists CLI overrides back to YAML
- `clusters.csv` remains as lightweight index

### Import Layering (to prevent circular imports)

```
models/     -> constants, utils only
io/         -> models, constants, utils
catalog/    -> models, io, utils
subclusters/-> models, io, utils
xray/       -> models, io, utils
plotting/   -> models, io, utils, subclusters, xray, catalog
pipelines/  -> everything (orchestration layer)
```

### Conventions

- All SkyCoord operations use `utils/coordinates.py` helpers
- All defaults come from `constants.py` (not hardcoded in modules)
- All plotting functions live in `plotting/` (separated from analysis)
- `setup_plot_style()` called from CLI entry points (not at module import)
- X-ray filename configurable via YAML `xray.filename` key

## Commands

```bash
pip install -e .                              # Install in development mode
pip install -e ".[dev]"                       # With test dependencies
cluster-pipeline run <ID> --stages spec phot  # Run specific stages
cluster-pipeline run <ID> --stages xray       # Run X-ray analysis
cluster-pipeline info <ID>                    # Show cluster summary
cluster-pipeline init <ID> --ra RA --dec DEC  # Initialize new cluster
cluster-pipeline list                         # List all clusters
pytest                                        # Run tests
```

## Key Files

- `constants.py` - change defaults here, not in individual modules
- `config.py` - YAML config system (load, save, merge with CLI)
- `models/subcluster.py` - primary_bcg vs border_bcg role distinction
- `subclusters/geometry.py` - bisector math (the most delicate code)

## Refactor Notes

- v1 tagged on main branch (X-SORTER paper submission)
- v2 refactor on `v2-refactor` branch
- Old `my_utils.py` kept as re-export shim during migration (remove in Phase 5)
- No Claude co-author on commits
