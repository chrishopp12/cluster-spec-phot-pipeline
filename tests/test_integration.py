"""Offline end-to-end integration test.

Runs the matching -> red-sequence -> subclusters slice of the pipeline against
the minimal committed fixture in ``tests/data/RMJ_0219`` and asserts that each
stage produces well-formed data products.

The fixture is RMJ 0219 cut to the central 4.5 arcmin, using archival (public)
redshifts only plus a BCG seed catalog (``BCGs.csv`` with redshifts blanked). The
BCG seed lets matching skip its redMaPPer VizieR lookup, so the slice needs no
network, no CASJobs credentials, and no X-ray FITS image. The spectroscopy,
photometry, and X-ray stages are intentionally not exercised (they require external
resources). Outputs go to a pytest ``tmp_path`` and are auto-removed; the committed
fixture stays input-only.
"""

import shutil
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from cluster_pipeline.cli import main

FIXTURE = Path(__file__).parent / "data" / "RMJ_0219"


@pytest.mark.integration
@pytest.mark.filterwarnings("ignore::UserWarning")  # subcluster plotting noise
def test_offline_pipeline_slice(tmp_path):
    """matching -> redseq -> subclusters runs offline and yields valid products."""
    # Copy the input-only fixture into a writable temporary base path.
    shutil.copytree(FIXTURE, tmp_path / "RMJ_0219")

    res = CliRunner().invoke(
        main,
        ["run", "RMJ_0219", "--base-path", str(tmp_path),
         "--stages", "matching,redseq,subclusters",
         "--no-diagnostics", "--no-save-plots", "--no-verbose"],
        catch_exceptions=False,
    )
    assert res.exit_code == 0, f"CLI run failed:\n{res.output}"

    run = tmp_path / "RMJ_0219"

    # --- Stage 3: matching ---
    combined = pd.read_csv(run / "Redshifts" / "combined_redshifts.csv")
    assert {"RA", "Dec", "z", "spec_source"} <= set(combined.columns)
    assert len(combined) >= 20

    matched = pd.read_csv(run / "Photometry" / "legacy_matched.csv")
    assert {"RA", "Dec", "z", "gmag", "rmag", "g_r"} <= set(matched.columns)
    assert len(matched) >= 20

    bcgs = pd.read_csv(run / "BCGs.csv")
    assert {"BCG_priority", "RA", "Dec"} <= set(bcgs.columns)
    assert len(bcgs) >= 2

    # --- Stage 4: red sequence ---
    members = pd.read_csv(run / "Members" / "cluster_members.csv")
    assert {"RA", "Dec", "member_type"} <= set(members.columns)
    assert len(members) >= 50  # rich red-sequence selection for this cluster

    # --- Stages 5-7: subclusters (config.yaml defines two: BCGs 1 and 2) ---
    subs = pd.read_csv(run / "subclusters.csv")
    assert len(subs) == 2
    assert {"bcg_id", "bcg_label", "bcg_dv_kms"} <= set(subs.columns)
    for bcg_id in (1, 2):
        mf = run / "Members" / f"subcluster_{bcg_id}_members.csv"
        assert mf.is_file(), f"missing {mf.name}"
        assert len(pd.read_csv(mf)) >= 1
