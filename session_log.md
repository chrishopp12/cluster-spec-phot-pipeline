# Session Log — Cluster Pipeline v2 Refactor

Tracks all decisions, rationale, and progress. Started 2026-03-28.

---

### Coding Note: No Claude co-author on commits

## 2026-03-28 — Planning Session

### Design Decisions

1. **Installable package** with pyproject.toml and src/ layout.
   - Enables clean imports, pip install -e for dev, modern standard.

2. **Hybrid object model** — structural classes + DataFrame bulk data.
   - Cluster, Subcluster, BCG, Region as classes.
   - Bulk galaxy data stays in DataFrames (10k+ galaxies, vectorized ops essential).
   - No Galaxy base class.
   - Rationale: classes capture relationships (which BCG plays which role), DataFrames handle bulk filtering/sorting.

3. **Foundation-first build order** — 5 phases, 30 commits.
   - Phase 1: Skeleton + infrastructure
   - Phase 2: Data model
   - Phase 3: Spec/phot pipeline migration
   - Phase 4: X-ray pipeline decomposition
   - Phase 5: New features + polish

4. **Two-layer config** — persistent YAML + ephemeral CLI overrides.
   - Per-cluster YAML at ~/Clusters/{id}/config.yaml is single source of truth.
   - CLI args override for current run only; --save persists.
   - BCG data never lost when running with subset.
   - clusters.csv stays as lightweight index (Excel viewable).

5. **Unified CLI** via click — `cluster-pipeline run/info/init/list`.

6. **Two-track testing** — pytest for algorithms, manual verification for I/O/plotting.

7. **Config lives with cluster data** (~/Clusters/{id}/config.yaml), not in repo.

### Exploration Findings

- ~14,800 lines across 14 files.
- process_subclusters.py (3,827 lines) is the worst offender — geometry + analysis + stats + plotting tangled.
- my_utils.py (2,518 lines) is a kitchen-sink utility file.
- Cluster class (840 lines) is actually well-structured — better than expected.
- spec_phot_pipeline is the organizational template to follow.
- Bisector geometry is mathematically sound — needs extraction, not rewriting.
- Duplicate make_stats_table() in two files.
- Hardcoded X-ray filename in 5+ places.
- PSF default inconsistency: 8.0 (Cluster) vs 10.0 (run_xray_pipeline).

### Key Risk Areas Identified

1. Bisector geometry state shared through dicts — extract exactly as-is first.
2. Duplicate plot_optical may have drifted — diff before choosing canonical.
3. Circular imports — enforce layering convention.
4. matplotlib rcParams at import time — move to setup_plot_style().

---

## Progress

### Phase 1: Skeleton + Infrastructure

- [ ] Step 1.0: Branch + CLAUDE.md + session_log.md
- [ ] Step 1.1: Package skeleton + pyproject.toml
- [ ] Step 1.2: constants.py
- [ ] Step 1.3: Decompose my_utils.py
- [ ] Step 1.4: config.py + test_config.py
- [ ] Step 1.5: CLI skeleton
