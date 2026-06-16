"""Tests for the Anderson-Darling normality computation in make_stats_table.

Exercises the deterministic Monte-Carlo AD p-value path (scipy goodness_of_fit)
directly — the integration fixture is too sparse (N<3 components) to reach it.
"""
import numpy as np

from cluster_pipeline.subclusters.statistics import make_stats_table


def _fit_covering(z):
    """A single Gaussian fit (z_low, z_high, mean, std, weight) covering all of z."""
    return [(float(z.min()) - 0.01, float(z.max()) + 0.01,
             float(np.mean(z)), float(np.std(z, ddof=1)), 1.0)]


def test_ad_schema_and_determinism():
    """AD p-value is finite, reproducible, and uses the new single-method schema."""
    z = np.random.default_rng(0).normal(0.41, 0.005, 40)
    fits = _fit_covering(z)

    t1, _ = make_stats_table(z, fits, make_plots=False, verbose=False)
    t2, _ = make_stats_table(z, fits, make_plots=False, verbose=False)

    # New single-method schema (old dual-p-value columns are gone)
    assert "AD p-value" in t1.colnames
    assert "Normality" in t1.colnames
    assert "AD p-value (interpolated)" not in t1.colnames
    assert "Anderson Level" not in t1.colnames

    # The Monte-Carlo path actually ran (finite p-value for N >= 3)
    ad_p = float(t1["AD p-value"][0])
    assert np.isfinite(ad_p)
    assert 0.0 <= ad_p <= 1.0

    # Seeded => identical across calls
    assert float(t2["AD p-value"][0]) == ad_p


def test_ad_verdict_label():
    """The Normality verdict is one of the p-value-derived labels."""
    z = np.random.default_rng(1).normal(0.41, 0.005, 30)
    t, _ = make_stats_table(z, _fit_covering(z), make_plots=False, verbose=False)
    assert t["Normality"][0] in ("Reject normality", "Consistent with normal")


def test_ad_degenerate_is_nan():
    """A zero-variance component yields NaN (guarded, no crash, no divide warning)."""
    z = np.full(5, 0.41)
    t, _ = make_stats_table(z, _fit_covering(z), make_plots=False, verbose=False)
    assert not np.isfinite(float(t["AD p-value"][0]))
    assert t["Normality"][0] == "Insufficient data"
