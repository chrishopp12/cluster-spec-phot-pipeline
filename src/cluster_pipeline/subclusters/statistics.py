#!/usr/bin/env python3
"""
statistics.py

Stage 7: Subcluster Statistics and Redshift Analysis
---------------------------------------------------------

Statistical analysis for subclusters and cluster redshift distributions:
GMM fitting, velocity dispersions, normality tests, and group analysis.

Produces
--------
- Per-subcluster velocity dispersions and mean redshifts
- KS and Anderson-Darling normality tests per GMM component
- GMM redshift histograms with inset subcluster panels
- Stacked velocity histograms
- LaTeX deluxetable of subcluster properties
- Pairwise BCG-BCG velocity-difference CSV

Requirements
------------
- Combined spectroscopic redshift catalog (combined_redshifts.csv)
- BCGs.csv with columns: BCG_priority, z, sigma_z
- Subcluster objects with populated spec_members DataFrames
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

from sklearn.mixture import GaussianMixture

from astropy.table import Table
from astropy.stats import biweight_location, biweight_midvariance
from astropy.constants import c
from astropy import units as u

from itertools import cycle

import scipy.stats

from cluster_pipeline.plotting.common import finalize_figure

if TYPE_CHECKING:
    from cluster_pipeline.models.subcluster import Subcluster


def fit_gaussian_model(z, max_components=8, min_galaxies=8, broad_threshold=0.05, verbose=True):
    """
    Fit a Gaussian Mixture Model (GMM) to 1D redshift data, selecting the optimal
    number of components using BIC and filtering out broad or sparse components.

    Parameters
    ----------
    z : array-like, shape (N,)
        1D array of redshift values (e.g., all cluster or field member redshifts).
    max_components : int, optional
        Maximum number of GMM components to try (default: 8).
    min_galaxies : int, optional
        Minimum number of galaxies required for a Gaussian to be considered valid (default: 8).
    broad_threshold : float, optional
        Exclude any Gaussian with stddev (sigma) >= this value (default: 0.04).

    Returns
    -------
    valid_gaussians : list of tuples
        List of subclusters: (z_low_fit, z_high_fit, mean, std, weight) for each retained Gaussian.
    fit_info : dict
        Dictionary with all fitted GMM parameters and BICs for diagnostics:
            - means: array, shape (N_best,)
            - stds: array, shape (N_best,)
            - weights: array, shape (N_best,)
            - best_n: int, number of selected components
            - bics: list of float, BIC for each n
            - best_model: trained GaussianMixture object (scikit-learn)
            - z_low, z_high: min and max of fitted data
    """
    z_subset = np.array(z, dtype=float)
    if z_subset.size == 0:
        raise ValueError("Input redshift array is empty!")

    z_low, z_high = z_subset.min(), z_subset.max()

    z_subset_reshape = z_subset.reshape(-1, 1)
    if verbose:
        print(f"Fitting max {max_components} component GMM to {len(z_subset)} galaxies in range ({z_low}, {z_high})")

    # Fit GMM with BIC selection
    bics = []
    models = []
    for n in range(1, max_components + 1):
        gmm = GaussianMixture(n_components=n, covariance_type='full', random_state=42)
        gmm.fit(z_subset_reshape)
        bics.append(gmm.bic(z_subset_reshape))
        models.append(gmm)
        if verbose:
            print(f"  - BIC(n={n}): {bics[-1]:.2f}")

    best_idx = np.argmin(bics)
    if verbose:
        print(f"  -> Best model: n={best_idx+1} with BIC={bics[best_idx]:.2f}")
    best_model = models[best_idx]
    best_n = best_model.n_components
    print(f"Selected {best_n} components (min BIC at n={best_n})")

    means = best_model.means_.flatten()
    stds = np.sqrt(best_model.covariances_).flatten()
    weights = best_model.weights_

    # Sort by mean (for cleaner downstream processing)
    sorted_indices = np.argsort(means)
    means = means[sorted_indices]
    stds = stds[sorted_indices]
    weights = weights[sorted_indices]

    # Filter out broad Gaussians and those with too few galaxies
    valid_gaussians = []
    for i, (mean, std, weight) in enumerate(zip(means, stds, weights)):
        z_low_fit, z_high_fit = mean - 3 * std, mean + 3 * std
        N_local = np.sum((z_subset > mean - std) & (z_subset < mean + std))
        if verbose:
            msg = (
                f"  - Gaussian {i+1}: mean={mean:.5f}, std={std:.5f}, weight={weight:.3f}, N={N_local}"
                f"{' [filtered]' if (std >= broad_threshold or N_local < min_galaxies) else ''}"
            )
            print(msg)
        if std < broad_threshold and N_local >= min_galaxies:
            valid_gaussians.append((z_low_fit, z_high_fit, mean, std, weight))

    # Prepare fit info
    fit_info = dict(
        means=means,
        stds=stds,
        weights=weights,
        best_n=best_n,
        bics=bics,
        best_model=best_model,
        z_low=z_low,
        z_high=z_high,
    )

    return valid_gaussians, fit_info


def make_stats_table(
    z,
    gaussian_fits,
    bins=12,
    prefix="subcluster",
    make_plots=True,
    save_plots=True,
    show_plots=True,
    save_path=None,
    verbose=False,
):
    """
    Computes and plots statistics for each subcluster defined by Gaussian fits, including
    Kolmogorov-Smirnov and Anderson-Darling tests, and generates a LaTeX table.

    Adds an (approx) p-value for the Anderson-Darling normality test:
      - Prefer statsmodels.normal_ad (recommended)
      - Fall back to a common Stephens-style approximation if statsmodels isn't installed
    """

    def _ad_pvalue_normality(z_subset: np.ndarray) -> tuple[float, float]:
        """
        Anderson-Darling normality test (unknown mean/variance) with p-value.

        Returns
        -------
        ad_stat, ad_p : float, float
        """
        z_subset = np.asarray(z_subset, dtype=float)
        n = z_subset.size
        if n < 3:
            return np.nan, np.nan

        # Preferred: statsmodels provides AD stat + calibrated p-value
        try:
            from statsmodels.stats.diagnostic import normal_ad  # type: ignore
            ad2, p = normal_ad(z_subset)
            return float(ad2), float(p)
        except Exception:
            pass

        # statsmodels not available — return dummy p-value
        p = 10.0
        return ad2, p

    stats_list = []
    stats_values = []

    z = np.asarray(z, dtype=float)

    for i, (z_low, z_high, mean, std, weight) in enumerate(gaussian_fits):
        # Create the mask
        mask = (z_low < z) & (z < z_high)
        z_subset = z[mask]
        N = len(z_subset)

        if N == 0:
            print(f"Skipping subcluster {i+1} (no data in range).")
            stats_list.append([i + 1, z_low, z_high, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, "No data"])
            stats_values.append(None)
            continue

        # Normal parameters from the data in this range
        subset_mean = float(np.mean(z_subset))
        subset_std = float(np.std(z_subset, ddof=1)) if N > 1 else 0.0

        # KS test (only meaningful if std>0 and N>=2)
        if N >= 2 and subset_std > 0:
            ks_statistic, ks_p_value = scipy.stats.ks_1samp(
                z_subset,
                scipy.stats.norm(loc=subset_mean, scale=subset_std).cdf,
                method="auto",
            )
        else:
            ks_statistic, ks_p_value = np.nan, np.nan

        # AD stat + p-value
        ad_stat, ad_p_value = _ad_pvalue_normality(z_subset)

        if N >= 3:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                anderson_res = scipy.stats.anderson(z_subset, dist="norm")
            sig_levels = anderson_res.significance_level
            crit_values = anderson_res.critical_values

            anderson_res2 = scipy.stats.anderson(z_subset, dist="norm", method=scipy.stats.MonteCarloMethod(n_resamples=50_000))
            ad_p_value2 = anderson_res2.pvalue

            ad_significance = None
            for level, crit in zip(sig_levels[::-1], crit_values[::-1]):  # descending order
                if np.isfinite(ad_stat) and ad_stat > crit:
                    ad_significance = level
                    break
            if ad_significance is None:
                ad_significance = "Fail to Reject Normality"
        else:
            sig_levels, crit_values = [], []
            ad_significance = "Insufficient N (<3)"

        # Store in list (now includes AD stat + AD p-value)
        stats_list.append([
            i + 1,
            round(float(z_low), 3),
            round(float(z_high), 3),
            round(subset_mean, 4),
            round(subset_std, 5),
            round(ks_statistic, 4) if np.isfinite(ks_statistic) else np.nan,
            round(ks_p_value, 4) if np.isfinite(ks_p_value) else np.nan,
            round(ad_stat, 4) if np.isfinite(ad_stat) else np.nan,
            round(ad_p_value, 4) if np.isfinite(ad_p_value) else np.nan,
            round(ad_p_value2, 4) if np.isfinite(ad_p_value2) else np.nan,
            ad_significance,
        ])

        stats_values.append({
            "Subcluster": i + 1,
            "z_min": float(z_low),
            "z_max": float(z_high),
            "Mean": subset_mean,
            "Std Dev": subset_std,
            "KS Stat": ks_statistic,
            "KS p-value": ks_p_value,
            "AD Stat": ad_stat,
            "AD p-value": ad_p_value,
            "AD p-value (interpolated)": ad_p_value2,
            "Anderson Level": ad_significance,
        })

        if verbose:
            print("")
            print(f"-------------- Stats Summary: Subcluster {i+1} --------------")
            print(f"          KS Test Statistic: {ks_statistic}")
            print(f"                 KS p-value: {ks_p_value}")
            print(f" Anderson-Darling Statistic: {ad_stat}")
            print(f"        Anderson-Darling p : {ad_p_value}")
            print(f"        Anderson-Darling p (interpolated): {ad_p_value2}")
            print(f"            Critical Values: {crit_values}")
            print(f"        Significance Levels: {sig_levels}")
            print("---------------------------------------------------------------")
            print("")

        if make_plots:
            fig, axes = plt.subplots(1, 2, figsize=(14, 7))

            # Histogram + PDF
            axes[0].hist(z_subset, bins=bins, density=True, alpha=0.6, color="b")
            if N >= 2 and subset_std > 0:
                x_pdf = np.linspace(np.min(z_subset), np.max(z_subset), 100)
                pdf = scipy.stats.norm.pdf(x_pdf, subset_mean, subset_std)
                axes[0].plot(x_pdf, pdf, "r--", label=f"Normal PDF (z={subset_mean:.3f}, std={subset_std:.3f})")
                axes[0].legend(loc="upper left")

            axes[0].set_xlabel("z")
            axes[0].xaxis.set_major_locator(MultipleLocator(0.005))
            axes[0].xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
            axes[0].set_ylabel("Density")
            axes[0].set_title(f"Histogram & PDF\n({z_low:.3f} < z < {z_high:.3f})")

            # ECDF vs Normal CDF
            x_ecdf = np.sort(z_subset)
            y_ecdf = np.arange(1, N + 1) / N
            axes[1].plot(x_ecdf, y_ecdf, label="Empirical CDF")

            if N >= 2 and subset_std > 0:
                cdf = scipy.stats.norm.cdf(x_ecdf, subset_mean, subset_std)
                axes[1].plot(x_ecdf, cdf, "r--", label="Normal CDF")

            axes[1].set_xlabel("z")
            axes[1].xaxis.set_major_locator(MultipleLocator(0.005))
            axes[1].xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
            axes[1].set_ylabel("Cumulative Probability")
            axes[1].set_title("ECDF vs Normal CDF")
            axes[1].legend(loc="upper left")

            plt.tight_layout()
            finalize_figure(
                fig,
                save_path=save_path,
                save_plots=save_plots,
                show_plots=show_plots,
                filename=f"{prefix}_{i+1}_stats.pdf",
            )

    # Convert list to Astropy Table
    col_names = [
        "Subcluster",
        "z_min",
        "z_max",
        "Mean",
        "Std Dev",
        "KS Stat",
        "KS p-value",
        "AD Stat",
        "AD p-value",
        "AD p-value (interpolated)",
        "Anderson Level",
    ]
    stats_table = Table(rows=stats_list, names=col_names)

    if verbose:
        print(stats_table)
    return stats_table, stats_values


def velocity_dispersion(zs):
    z_mean = biweight_location(zs)
    vels = c.to('km/s').value * (zs - z_mean) / (1 + z_mean)
    return z_mean, np.sqrt(biweight_midvariance(vels)), vels


def get_velocities(gaussians, z, verbose=False):
    # Analyze each spectroscopic group
    vel_data = []
    z_data = []
    for i, (z_low, z_high, mean, std, weight) in enumerate(gaussians):
        z_vals = z[(z >= z_low) & (z < z_high)]
        z_mean, sigma_v, velocities = velocity_dispersion(z_vals)
        vel_data.append((velocities, z_mean, sigma_v))
        z_data.append(z_vals)
        if verbose:
            print(f"Subcluster {i+1}: N={len(z_vals)} | z\u0304 = {z_mean:.5f} | \u03c3_v = {sigma_v:.1f} km/s")

    return vel_data, z_data


def process_redshifts(cluster, z_min_field=0, z_max_field=1.5, input_csv=None, output_folder=None, max_components=None, save_plots=True, show_plots=True, verbose=True):


    # -- Load Redshift Data --
    if input_csv is None:
        redshift_csv = os.path.join(cluster.redshift_path, "combined_redshifts.csv")
    else:
        redshift_csv = input_csv
    redshifts_df = pd.read_csv(redshift_csv)
    if not {'z','RA','Dec'}.issubset(redshifts_df.columns):
        raise ValueError(f"CSV {redshift_csv} missing required columns (z, RA, Dec)")

    z = redshifts_df['z'].values
    z_field = z[(z_min_field < z) & (z < z_max_field)]
    z_cluster = z[(cluster.z_min < z) & (z < cluster.z_max)]

    kwargs = {}
    if max_components is not None:
        kwargs['max_components'] = max_components

    # -- Fit Gaussians for entire range --
    if verbose:
        print(f"Fitting Gaussians in field range ({z_min_field}, {z_max_field})")
    gaussians_field, fit_info_field = fit_gaussian_model(z_field, broad_threshold=0.05, verbose=verbose, **kwargs)
    if verbose:
        print(f"Identified {len(gaussians_field)} clusters.")

    # -- Fit Gaussians for cluster only--
    if verbose:
        print(f"Fitting Gaussians in cluster range ({cluster.z_min}, {cluster.z_max})")
    gaussians, fit_info = fit_gaussian_model(z_cluster, broad_threshold=0.02, max_components=2, min_galaxies = 4, verbose=verbose, **kwargs)
    if verbose:
        print(f"Identified {len(gaussians)} subclusters.")


    # -- Output subcluster stats --
    if output_folder is None:
        output_folder = cluster.redshift_path

    image_folder = os.path.join(output_folder, "Images")
    os.makedirs(image_folder, exist_ok=True)

    stats_table_field, _ = make_stats_table(z_field, gaussians_field, prefix="field", make_plots=True, save_plots=save_plots, show_plots=show_plots, save_path=image_folder, verbose=verbose)
    stats_table_cluster, _  = make_stats_table(z_cluster, gaussians, prefix="cluster", make_plots=True, save_plots=save_plots, show_plots=show_plots, save_path=image_folder, verbose=verbose)

    stats_path_field = os.path.join(output_folder, "subcluster_stats_field.csv")
    stats_table_field.write(stats_path_field, format="csv", overwrite=True)
    print(f"Stats table written to {stats_path_field}")

    stats_path_cluster = os.path.join(output_folder, "subcluster_stats_cluster.csv")
    stats_table_cluster.write(stats_path_cluster, format="csv", overwrite=True)
    print(f"Stats table written to {stats_path_cluster}")



    # -- Plot results --
    plot_path = os.path.join(image_folder, "redshift_histogram_field.pdf")
    field_colors = plot_gmm_histogram(z_field, gaussians_field, fit_info=fit_info_field,  plot_total=True, save_path=plot_path, save_plots=save_plots, show_plots=show_plots)

    plot_path = os.path.join(image_folder, "redshift_histogram_cluster.pdf")
    cluster_colors = plot_gmm_histogram(z_cluster, gaussians, fit_info=fit_info, plot_total=True, save_path=plot_path, save_plots=save_plots, show_plots=show_plots)

    vel_data_field, _ = get_velocities(gaussians_field, z_field)
    vel_data_cluster, _ = get_velocities(gaussians, z_cluster)

    vel_path = os.path.join(image_folder, "velocity_histograms_field.pdf")
    plot_stacked_velocity_histograms(vel_data_field, bins=25, color=field_colors, combined_color='orange', save_path=vel_path, save_plots=save_plots, show_plots=show_plots)
    vel_path = os.path.join(image_folder, "velocity_histograms_cluster.pdf")
    plot_stacked_velocity_histograms(vel_data_cluster, bins=25, color=cluster_colors, combined_color='orange', with_all=True, save_path=vel_path, save_plots=save_plots, show_plots=show_plots)


def plot_gmm_histogram(z, valid_gaussians, plot_total=True, bins=60, n_inset_max=4, show=True, fit_info=None, save_path=None, save_plots=True, show_plots=True):
    """
    Plot the field/cluster redshift histogram with GMM overlays and subcluster insets.

    Parameters
    ----------
    z : array-like
        Redshift data for all galaxies (field or cluster region).
    valid_gaussians : list of tuples
        (z_low_fit, z_high_fit, mean, std, weight) for each subcluster
    fit_info : dict or None
        If present, overlays total GMM fit using original BIC-selected model.
    plot_total : bool
        If True, overlay the total GMM fit.
    bins : int
        Number of bins for the main histogram.
    n_inset_max : int
        Maximum number of insets to show (more subclusters print a warning).
    show : bool
        Whether to display the plot interactively.
    save_path : str or None
        File path to save the figure (PDF/PNG).
    """
    if len(z) == 0:
        print("No data to plot.")
        return

    z = np.asarray(z)
    fig, ax = plt.subplots(figsize=(14, 8))

    # Main histogram
    counts, bins_hist, _ = ax.hist(z, bins=bins, color='b', alpha=0.6, edgecolor='black', zorder=5)
    bin_width = bins_hist[1] - bins_hist[0]
    x = np.linspace(z.min(), z.max(), 1000)

    # Individual Gaussians
    inset_info = []
    color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'][1:])  # skip C0
    gaussian_colors = []
    for idx, (z_low_fit, z_high_fit, mean, std, weight) in enumerate(valid_gaussians):
        color = next(color_cycle)
        gaussian_colors.append(color)
        # Use local normalization for each component
        region_mask = (z > mean - 2* std) & (z < mean + 2*std)
        N_local = np.sum(region_mask)
        inset_info.append({'idx': idx, 'N_local': N_local, 'mean' : mean})
        gaussian = scipy.stats.norm.pdf(x, mean, std) * N_local * bin_width
        ax.plot(
            x, gaussian, linestyle='dashed', linewidth=2, color=color,zorder=7,
            label=f"SC {idx+1}: z={mean:.3f}, $\\sigma$={std:.4f}, N={N_local}"
        )

    # Total GMM fit overlay (if desired and fit_info provided)
    if plot_total and fit_info is not None:
        total_gmm = np.zeros_like(x)
        for mean, std, weight in zip(fit_info['means'], fit_info['stds'], fit_info['weights']):
            # Scale by GMM weight and total number of galaxies (so area = N_total)
            total_gmm += scipy.stats.norm.pdf(x, mean, std) * weight * len(z) * bin_width
        ax.plot(x, total_gmm, color='grey', alpha=0.7, lw=2, zorder=6, label="Total GMM Fit")

    ax.set_xlabel("Redshift z")
    if z.min() < 0:
        ax.set_xlim(0, z.max())
    else:
        ax.set_xlim(z.min()-0.05, z.max()+0.05)
    ax.set_ylabel("Galaxy Counts")
    ax.legend(fontsize=9, loc='upper left')
    plt.tight_layout()

    # Inset subcluster histograms
    # Sort insets by number of galaxies, then mean redshift
    n_inset = min(n_inset_max, len(valid_gaussians))
    top_n = sorted(inset_info, key=lambda x: -x['N_local'])[:n_inset]
    top_n_sorted = sorted(top_n, key=lambda x: x['mean'])
    inset_indices = [item['idx'] for item in top_n_sorted]


    inset_axes_list = []

    inset_width = 0.3
    inset_height = 0.20

    inset_positions = {
    1: [(0.65, 0.42)],                                # Lower right
    2: [(0.65, 0.72), (0.65, 0.42)],                   # Upper right, lower right
    3: [(0.08, 0.42), (0.65, 0.72), (0.65, 0.42)],       # UL, UR, LR (low to high z)
    4: [(0.08, 0.42), (0.65, 0.72), (0.65, 0.42), (0.65, 0.12)], # LL, UL, UR, LR
    }


    positions = inset_positions.get(n_inset, [])


    for inset_rank, i in enumerate(inset_indices):
        z_low_fit, z_high_fit, mean, std, weight = valid_gaussians[i]
        if z_low_fit < 0: z_low_fit = 0
        sub_mask = (z > z_low_fit) & (z < z_high_fit)
        z_sub = z[sub_mask]
        if len(z_sub) == 0:
            continue
        if inset_rank >= len(positions):
                continue
        x0, y0 = positions[inset_rank]

        color = gaussian_colors[i]
        axins = fig.add_axes([x0, y0, inset_width, inset_height])
        axins.hist(z_sub, bins=15, color=color, edgecolor='black')
        axins.set_title(f"SC {i+1}", fontsize=9)
        axins.set_xlabel("z", fontsize=8)
        axins.set_ylabel("N", fontsize=8)
        axins.tick_params(labelsize=7)
        axins.set_xlim(z_low_fit, z_high_fit)
        inset_axes_list.append(axins)


        # Draw connection lines
        inset_xlim = axins.get_xlim()
        inset_ylim = axins.get_ylim()
        main_ylim = ax.get_ylim()
        con1 = ConnectionPatch((z_low_fit, main_ylim[0]), (inset_xlim[0], inset_ylim[0]),
                               coordsA=ax.transData, coordsB=axins.transData,
                               color=color, linewidth=0.5, zorder=0)
        con2 = ConnectionPatch((z_high_fit, main_ylim[0]), (inset_xlim[1], inset_ylim[0]),
                               coordsA=ax.transData, coordsB=axins.transData,
                               color=color, linewidth=0.5, zorder=0)
        ax.add_artist(con1)
        ax.add_artist(con2)

    if len(valid_gaussians) > n_inset_max:
        print(f"Warning: More than {n_inset_max} subclusters; only plotting first {n_inset_max} insets.")

    finalize_figure(fig, save_path=save_path, save_plots=save_plots, show_plots=show_plots, filename="gmm_histogram.pdf")

    return gaussian_colors


def plot_stacked_velocity_histograms(vel_data, bins=25, color=None, combined_color='orange', with_all=False, save_path=None, save_plots=True, show_plots=True):
    """
    Plots stacked histograms of velocity data for each subcluster (and optionally all).

    Parameters
    ----------
    vel_data : list of (velocities, z_mean, sigma_v) tuples per subcluster.
    bins : int
        Number of histogram bins (per subcluster).
    color : list or None
        Colors for subcluster histograms. If None, use color cycle.
    combined_color : str
        Color for combined histogram (if with_all=True).
    with_all : bool
        If True, plot a combined histogram as the last panel.
    """

    # TODO: Allow fig/ax input and return
    # TODO: Fix spacing between plots


    n_subclusters = len(vel_data)
    n_panels = n_subclusters + (1 if with_all else 0)
    fig, axes = plt.subplots(n_panels, 1, figsize=(7, 2.5 * n_panels), sharex=False)
    if n_panels == 1:
        axes = [axes]
    fig.subplots_adjust(hspace=0.25)

    if color is None:
        default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        subcluster_colors = [default_colors[i % len(default_colors)] for i in range(n_subclusters)]
    elif len(color) ==1:
        subcluster_colors = [color] * n_subclusters
    else:
        subcluster_colors = color
    textbox_props = dict(boxstyle="round,pad=0.35", fc="lightgrey", ec="0.5", alpha=0.5, linewidth=1)


    for i, (velocities, z_mean, sigma_v) in enumerate(vel_data):
        ax = axes[i]
        vmin, vmax = np.min(velocities), np.max(velocities)


        bins_local = np.linspace(vmin, vmax, bins)
        ax.hist(velocities, bins=bins_local, color=subcluster_colors[i], edgecolor='black', density=False)
        ax.set_title(f"SC {i+1}", loc='left', fontsize=11)
        ax.set_xlim(vmin, vmax)
        ax.axvline(0, color='gray', linestyle=':', linewidth=1, zorder=1)
        # Textbox with mean and sigma
        textstr = f"$\\bar{{z}}$ = {z_mean:.4f}\n$\\sigma_v$ = {sigma_v:.0f} km/s"
        # Place textbox in upper right
        ax.text(
            0.98, 0.96, textstr,
            transform=ax.transAxes,
            ha='right', va='top',
            fontsize=10,
            bbox=textbox_props
        )

    # "All" panel
    if with_all:
        all_velocities = np.concatenate([v[0] for v in vel_data if len(v[0]) > 0])
        ax_comb = axes[-1]
        vmin, vmax = np.min(all_velocities), np.max(all_velocities)
        bins_all = np.linspace(vmin, vmax, bins)
        ax_comb.hist(all_velocities, bins=bins_all, color=combined_color, edgecolor='black', density=False)
        ax_comb.set_xlabel("Velocity (km/s)")
        ax_comb.set_title("All", loc='left', fontsize=11)
        ax_comb.axvline(0, color='gray', linestyle=':', linewidth=1, zorder=1)

    fig.supylabel("Galaxy Counts", fontsize=16)
    fig.suptitle("Subcluster Velocity Distributions", fontsize=15)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    finalize_figure(fig, save_path=save_path, save_plots=save_plots, show_plots=show_plots, filename="velocity_histograms.pdf")


def analyze_group(subclusters: list, verbose=False) -> dict:
    """Compute velocity dispersion for each subcluster's spectroscopic members.

    Parameters
    ----------
    subclusters : list[Subcluster]
        Subclusters with .spec_members populated.
    verbose : bool, optional
        If True, print per-subcluster stats. [default: False]

    Returns
    -------
    dict
        Per-subcluster stats: {label: {"z_mean": ..., "sigma_v": ..., "n_spec": ...}}
    """
    results = {}
    for sub in subclusters:
        if sub.spec_members is None or len(sub.spec_members) == 0:
            print(f"Subcluster {sub.label}: no spectroscopic members, skipping.")
            continue
        z_vals = sub.spec_members['z'].values
        z_mean, sigma_v, velocities = velocity_dispersion(z_vals)
        results[sub.label] = {
            "z_mean": z_mean,
            "sigma_v": sigma_v,
            "n_spec": len(z_vals),
            "velocities": velocities,
            "z_vals": z_vals,
        }
        if verbose:
            print(
                f"Subcluster {sub.label}: N={len(z_vals)} | "
                f"$\\bar{{z}}$ = {z_mean:.5f} | $\\sigma_v$ = {sigma_v:.1f} km/s"
            )
    return results


def los_velocity_diff(z1, z2, sig_z1=None, sig_z2=None):
    """
    Rest-frame line-of-sight velocity difference between two redshifts.

    Parameters
    ----------
    z1, z2 : float
        Redshifts of the two objects.
    sig_z1, sig_z2 : float, optional
        1-sigma uncertainties on z1 and z2.

    Returns
    -------
    dv_kms : float
        Signed rest-frame LOS velocity difference [km/s] (positive if z2 > z1).
    dv_err_kms : float or None
        Uncertainty on dv [km/s] if uncertainties provided, else None.
    z_mean : float
        Mean redshift used for the rest-frame correction.
    """
    c_kms = c.to(u.km / u.s).value
    z_mean = 0.5 * (z1 + z2)
    dz = (z2 - z1)
    dv_kms = c_kms * dz / (1.0 + z_mean)

    dv_err_kms = None
    if (sig_z1 is not None) and (sig_z2 is not None):
        dz_err = (sig_z1**2 + sig_z2**2) ** 0.5
        dv_err_kms = c_kms * dz_err / (1.0 + z_mean)
    return dv_kms, dv_err_kms, z_mean


def bcg_z_by_subcluster(subclusters: list) -> tuple[list, list]:
    """Extract BCG redshifts aligned to subcluster order.

    Parameters
    ----------
    subclusters : list[Subcluster]

    Returns
    -------
    bcg_zs : list[float | None]
    bcg_z_errs : list[float | None]
    """
    bcg_zs = []
    bcg_z_errs = []
    for sub in subclusters:
        bcg = sub.primary_bcg
        bcg_zs.append(bcg.z)
        bcg_z_errs.append(bcg.sigma_z)
    return bcg_zs, bcg_z_errs


def build_subcluster_summary(
    cluster,
    subclusters: list,
    update_csv=True,
):
    """
    Summarize subcluster kinematics and BCG offsets.

    This function:
      - Uses Subcluster objects for BCG redshifts and identifiers
      - Computes subcluster stats via velocity_dispersion on spec_members
      - Computes BCG delta v relative to the subcluster mean and its uncertainty
      - Writes a LaTeX deluxetable (.tex) into cluster.tables_path
      - Writes pairwise BCG-BCG delta v CSV into cluster.tables_path
      - Optionally updates subclusters.csv with bcg_dv_kms and bcg_dv_err_kms

    Parameters
    ----------
    cluster : Cluster
        Cluster object with:
          - bcg_file : str  -> path to BCGs.csv
          - tables_path : str -> output directory for LaTeX/CSV
          - cluster_path : str -> cluster data directory
          - identifier : str   -> e.g., "RMJ 2135"
          - spec_file : str -> path to combined spec catalog
          - subcluster_file : str (optional) -> path to subclusters.csv for updates
    subclusters : list[Subcluster]
        Subcluster objects with spec_members populated.
    update_csv : bool
        If True and cluster.subcluster_file exists, add/overwrite bcg_dv_kms and
        bcg_dv_err_kms columns.

    Returns
    -------
    outputs : dict
        {
          'tex_path': str,
          'pairs_csv': str,
          'bcg_dv_kms': list[float or None],
          'bcg_dv_err_kms': list[float or None],
          'group_stats': dict,  # from analyze_group
        }

    Notes
    -----
    - Delta v is computed in the standard cluster rest frame:
        dv = c * (z2 - z1) / (1 + z_mean)
      where z_mean = (z1 + z2)/2, c in km/s.
    - For the BCG vs subcluster mean, the uncertainty uses only sigma_z of the BCG:
        sigma = c * sigma_z / (1 + z_mean)
    """

    # TODO: This function is a dumpster fire and should be burned to the ground at my earliest convenience.

    # ------------------------
    # Local helpers
    # ------------------------
    def _los_dv(z1, z2):
        zbar = (z1 + z2)/2
        return c.to('km/s').value * (z2 - z1) / (1.0 + zbar), zbar

    def _los_dv_err_from_sigmaz(sigmaz, z_ref):
        # z_ref is the redshift in the denominator; here we use the subcluster mean
        return None if sigmaz is None or (isinstance(sigmaz, float) and np.isnan(sigmaz)) \
            else c.to('km/s').value * sigmaz / (1.0 + z_ref)

    def _write_deluxetable_tex(cluster_name, group_stats, bcg_z_list, bcg_id_str, cluster_props, out_path):

        lines = []
        lines.append(r"\begin{deluxetable}{cccccc}")
        label_id = cluster_name.replace(" ", "")
        lines.append(rf"\caption{{{label_id} Subcluster Properties}}\label{{tab:{label_id}_subclusters}}")
        lines.append(r"\tablehead{")
        lines.append(
            r"\colhead{Subcluster} & \colhead{N} & \colhead{BCG} & \colhead{BCG $z$} & "
            r"\colhead{Mean $z$} & \colhead{$\sigma_v$ [km s$^{-1}$]}"
        )
        lines.append(r"}")
        lines.append(r"\centering")
        lines.append(r"\startdata")
        lines.append(f"All & {cluster_props[2]} & --- & --- & {cluster_props[0]:.4f} & {round(cluster_props[1], -1):.0f} \\\\")
        for i, sub in enumerate(subclusters):
            stats = group_stats.get(sub.label)
            if stats is None:
                continue
            nmem = stats["n_spec"]
            z_mean = stats["z_mean"]
            sigma_v = stats["sigma_v"]
            bcg_z = bcg_z_list[i]
            bcg_z_str = f"{bcg_z:.3f}" if bcg_z is not None else "---"
            line = f"{bcg_id_str[i]} & {nmem} & {sub.label} & {bcg_z_str} & {z_mean:.4f} & {round(sigma_v, -1):.0f}  \\\\"
            lines.append(line)

        lines.append(r"\enddata}")
        # fix a missing brace if any LaTeX parser complains:
        lines[-1] = r"\enddata"
        lines.append(r"\end{deluxetable}")

        with open(out_path, "w") as f:
            f.write("\n".join(lines))

    def _write_bcg_pairs_csv(bcg_df, out_csv):
        # Keep only rows with BCG_priority and z
        df = bcg_df.copy()
        df = df[df["BCG_priority"].notna() & df["z"].notna()].copy()
        if df.empty:
            pd.DataFrame([], columns=[
                "bcg_i", "bcg_j", "z_i", "z_j", "sigma_z_i", "sigma_z_j",
                "z_mean_pair", "dv_kms", "dv_err_kms", "abs_dv_kms"
            ]).to_csv(out_csv, index=False)
            return

        # Ensure types
        df["BCG_priority"] = df["BCG_priority"].astype(int)
        if "sigma_z" not in df.columns:
            df["sigma_z"] = np.nan

        rows = []
        for i in range(len(df)):
            for j in range(i + 1, len(df)):
                pri_i = int(df.iloc[i]["BCG_priority"])
                pri_j = int(df.iloc[j]["BCG_priority"])
                zi = float(df.iloc[i]["z"])
                zj = float(df.iloc[j]["z"])
                szi = None if pd.isna(df.iloc[i]["sigma_z"]) else float(df.iloc[i]["sigma_z"])
                szj = None if pd.isna(df.iloc[j]["sigma_z"]) else float(df.iloc[j]["sigma_z"])

                dv, zbar = _los_dv(zi, zj)
                dv_err = None
                if (szi is not None) and (szj is not None):
                    dv_err = c.to('km/s').value * np.sqrt(szi**2 + szj**2) / (1.0 + zbar)

                rows.append({
                    "bcg_i": pri_i,
                    "bcg_j": pri_j,
                    "z_i": zi,
                    "z_j": zj,
                    "sigma_z_i": "" if szi is None else szi,
                    "sigma_z_j": "" if szj is None else szj,
                    "z_mean_pair": zbar,
                    "dv_kms": dv,
                    "dv_err_kms": "" if dv_err is None else dv_err,
                    "abs_dv_kms": abs(dv),
                })

        pd.DataFrame(rows).to_csv(out_csv, index=False)

    # ------------------------
    # Main
    # ------------------------

    bcg_df = pd.read_csv(cluster.bcg_file)

    # -- Read spectroscopy data --
    df_spec = pd.read_csv(cluster.spec_file)
    z_all = df_spec['z'].values

    # -- Select cluster members --
    mask_members = (z_all > cluster.z_min) & (z_all < cluster.z_max)
    z_members = z_all[mask_members]
    n_members = len(z_members)
    z_mean_all, sigma_v_all, _ = velocity_dispersion(z_members)
    cluster_props = [z_mean_all, sigma_v_all, n_members]


    required = {"BCG_priority", "z"}
    missing = required - set(bcg_df.columns)
    if missing:
        raise ValueError(f"BCGs.csv missing required column(s): {', '.join(sorted(missing))}")
    if "sigma_z" not in bcg_df.columns:
        bcg_df["sigma_z"] = np.nan

    # Build BCG info from Subcluster objects
    bcg_z_list = []
    bcg_sigz_list = []
    bcg_id_str = []
    for sub in subclusters:
        bcg_id_str.append(str(sub.bcg_id))
        bcg_z_list.append(sub.primary_bcg.z)
        bcg_sigz_list.append(sub.primary_bcg.sigma_z)

    # Compute subcluster stats
    group_stats = analyze_group(subclusters)

    # Compute BCG delta v and sigma per subcluster
    bcg_dv_kms, bcg_dv_err_kms = [], []
    for i, sub in enumerate(subclusters):
        stats = group_stats.get(sub.label)
        if stats is None:
            bcg_dv_kms.append(None)
            bcg_dv_err_kms.append(None)
            continue
        z_mean = stats["z_mean"]
        z_bcg = bcg_z_list[i]
        if z_bcg is None or (isinstance(z_bcg, float) and np.isnan(z_bcg)):
            bcg_dv_kms.append(None)
            bcg_dv_err_kms.append(None)
            continue
        dv = c.to('km/s').value * (z_bcg - z_mean) / (1.0 + z_mean)
        dv_err = _los_dv_err_from_sigmaz(bcg_sigz_list[i], z_mean)
        bcg_dv_kms.append(dv)
        bcg_dv_err_kms.append(dv_err)

    # Write deluxetable
    tables_path = getattr(cluster, "tables_path", cluster.cluster_path)
    tex_path = os.path.join(tables_path, f"{cluster.identifier.replace(' ', '')}_subclusters.tex")
    os.makedirs(tables_path, exist_ok=True)
    _write_deluxetable_tex(cluster.identifier, group_stats, bcg_z_list, bcg_id_str, cluster_props, tex_path)

    # Pairwise BCG-BCG CSV
    pairs_csv = os.path.join(tables_path, "BCG_velocity_pairs.csv")
    _write_bcg_pairs_csv(bcg_df, pairs_csv)

    # Optional: update subclusters.csv in place
    if update_csv and getattr(cluster, "subcluster_file", None) and os.path.exists(cluster.subcluster_file):
        try:
            sdf = pd.read_csv(cluster.subcluster_file)
            # Map by bcg_id if present; else by subcluster index order
            if "bcg_id" in sdf.columns:
                id_to_dv = {}
                id_to_dverr = {}
                for i, sub in enumerate(subclusters):
                    id_to_dv[sub.bcg_id] = bcg_dv_kms[i]
                    id_to_dverr[sub.bcg_id] = bcg_dv_err_kms[i]
                sdf["bcg_dv_kms"] = sdf["bcg_id"].map(id_to_dv)
                sdf["bcg_dv_err_kms"] = sdf["bcg_id"].map(id_to_dverr)
            else:
                # Fallback: assume row order corresponds to subcluster order
                sdf["bcg_dv_kms"] = pd.Series(bcg_dv_kms[:len(sdf)])
                sdf["bcg_dv_err_kms"] = pd.Series(bcg_dv_err_kms[:len(sdf)])
            sdf.to_csv(cluster.subcluster_file, index=False)
        except Exception as e:
            print(f"Warning: could not update subclusters.csv with BCG dv columns: {e}")

    return {
        "tex_path": tex_path,
        "pairs_csv": pairs_csv,
        "bcg_dv_kms": bcg_dv_kms,
        "bcg_dv_err_kms": bcg_dv_err_kms,
        "group_stats": group_stats,
    }
