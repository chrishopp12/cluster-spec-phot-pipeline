#!/usr/bin/env python3
"""
process_redshifts.py

Cluster Redshift Processing Suite
----------------------------------
Fits Gaussian Mixture Models (GMM) to spectroscopic cluster redshift catalogs, identifies subclusters, 
outputs diagnostic statistics, and produces redshift and velocity dispersion histograms.

Usage:
    python process_redshifts.py "cluster_id" [options]

Example:
    python process_redshifts.py "RMJ1234.5+6789" --output_folder ./outputs --n_components 4

Requirements:
    - scikit-learn
    - astropy
    - matplotlib
    - pandas
    - cluster.py (local module)
    - my_utils.py (local module)
Arguments:
    cluster_id           <str>    Cluster identifier (e.g. "1327" or "RMJ1234.5+6789")
    --input_csv          <str>    Input CSV of redshifts (optional)
    --output_folder      <str>    Output path for all plots and tables
    --zmin_cluster       <float>  Cluster min redshift (override default)
    --zmax_cluster       <float>  Cluster max redshift (override default)
    --zmin_field         <float>  Field min redshift [0]
    --zmax_field         <float>  Field max redshift [1.5]
    --max_components     <int>    Number of GMM components (0=auto/BIC)
    --save_plots         <bool>   Save plots [True]
    --show_plots         <bool>   Show interactive plots [False]
Notes:
    - Gaussian fits, subcluster statistics, and velocity dispersions use robust biweight estimators.
    - Diagnostic images, ECDF plots, and tables are saved in the output_folder.
"""


import os
import argparse

from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

from sklearn.mixture import GaussianMixture

from astropy.table import Table
from astropy.stats import biweight_location, biweight_midvariance
from astropy.constants import c

from itertools import cycle

import scipy as scipy

from cluster import Cluster
from my_utils import str2bool, finalize_figure


plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14


def fit_gaussian_model(z, max_components=8, min_galaxies=8, broad_threshold=0.05):
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
    print(f"Fitting max {max_components} component GMM to {len(z_subset)} galaxies in range ({z_low}, {z_high})")

    # Fit GMM with BIC selection
    bics = []
    models = []
    for n in range(1, max_components + 1):
        gmm = GaussianMixture(n_components=n, covariance_type='full', random_state=42)
        gmm.fit(z_subset_reshape)
        bics.append(gmm.bic(z_subset_reshape))
        models.append(gmm)
        print(f"  - BIC(n={n}): {bics[-1]:.2f}")

    best_idx = np.argmin(bics)
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

# def make_stats_table(z, gaussian_fits, bins=12, prefix="subcluster", make_plots=True, save_plots=True, show_plots=True, save_path=None):
#     """
#     Computes and plots statistics for each subcluster defined by Gaussian fits, including
#     Kolmogorov-Smirnov and Anderson-Darling tests, and generates a LaTeX table.

#     Parameters
#     ----------
#     z : array-like
#         Full redshift dataset.
#     gaussian_fits : list of tuples
#         List of tuples containing Gaussian fit parameters for subclusters, where each tuple is (z_low, z_high, mean, std, weight).
#     bins : int
#         Number of bins to use in ECDF/ CDF plots
#     make_plots : bool, optional
#         Whether to generate diagnostic plots for each subcluster (default is True).
#     plot_dir : path
#         Directory to store output images

#     Returns
#     -------
#     stats_table : astropy.table.Table
#         Astropy Table containing statistics for each subcluster.
#     """
#     stats_list = []
#     stats_values = []

#     for i, (z_low, z_high, mean, std, weight) in enumerate(gaussian_fits):
#         # Create the mask
#         mask = (z_low < z) & (z < z_high)
#         z_subset = z[mask]
#         N = len(z_subset)

#         if N == 0:
#             print(f"Skipping subcluster {i+1} (no data in range).")
#             stats_list.append([i+1, z_low, z_high, np.nan, np.nan, 0, np.nan, np.nan, None])
#             stats_values.append(None)
#             continue

#         # plot_stats(z, float(z_low), float(z_high))
#         # Compute Normal distribution parameters
#         subset_mean, subset_std = np.mean(z_subset), np.std(z_subset)

#         # Perform Kolmogorov-Smirnov test
#         ks_statistic, ks_p_value = scipy.stats.ks_1samp(
#             z_subset, scipy.stats.norm(loc=subset_mean, scale=subset_std).cdf, method='auto'
#         )

#         # Perform Anderson-Darling test
#         anderson_res = scipy.stats.anderson(z_subset)

#         # Find the lowest significance level at which normality is rejected
#         ad_stat = anderson_res.statistic
#         sig_levels = anderson_res.significance_level
#         crit_values = anderson_res.critical_values

#         # Default to "None" (meaning normality is not rejected)
#         ad_significance = None

#         for level, crit in zip(sig_levels[::-1], crit_values[::-1]):  # Check in descending order
#             if ad_stat > crit:
#                 ad_significance = level  # Store the first level where normality is rejected
#                 break

#         if ad_significance is None:
#             ad_significance = "Fail to Reject Normality"

#         # Store in list
#         stats_list.append([
#             i + 1,
#             round(z_low, 3),
#             round(z_high, 3),
#             round(subset_mean, 4),
#             round(subset_std, 5),
#             round(ks_statistic, 4),
#             round(ks_p_value, 4),
#             ad_significance  # Store actual AD significance level
#         ])

#         stats_values.append({
#             "Subcluster": i + 1,
#             "z_min": z_low,
#             "z_max": z_high,
#             "Mean": subset_mean,
#             "Std Dev": subset_std,
#             "KS Stat": ks_statistic,
#             "KS p-value": ks_p_value,
#             "Anderson Level": ad_significance
#         })


#         print("")
#         print(f"-------------- Stats Summary: Subcluster {i+1} --------------")
#         print(f"          KS Test Statistic: {ks_statistic}")
#         print(f"                 KS p-value: {ks_p_value}")
#         print(f" Anderson-Darling Statistic: {ad_stat}")
#         print(f"            Critical Values: {crit_values}")
#         print(f"        Significance Levels: {sig_levels}")
#         print("---------------------------------------------------------------")
#         print("")


#         if make_plots:
#             fig, axes = plt.subplots(1, 2, figsize=(14, 7))
#             # Histogram + PDF
#             x_pdf = np.linspace(np.min(z_subset), np.max(z_subset), 100)
#             pdf = scipy.stats.norm.pdf(x_pdf, subset_mean, subset_std)
#             axes[0].hist(z_subset, bins=bins, density=True, alpha=0.6, color='b')
#             axes[0].plot(x_pdf, pdf, 'r--', label=f'Normal PDF (z={subset_mean:.3f}, std={subset_std:.3f})')
#             axes[0].set_xlabel('z')
#             axes[0].xaxis.set_major_locator(MultipleLocator(0.005))
#             axes[0].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
#             axes[0].set_ylabel('Density')
#             axes[0].set_title(f'Histogram & PDF\n({z_low:.3f} < z < {z_high:.3f})')
#             axes[0].legend(loc='upper left')

#             # ECDF vs Normal CDF
#             x_ecdf = np.sort(z_subset)
#             y_ecdf = np.arange(1, N + 1) / N
#             axes[1].plot(x_ecdf, y_ecdf, label='Empirical CDF')
#             cdf = scipy.stats.norm.cdf(x_ecdf, subset_mean, subset_std)
#             axes[1].plot(x_ecdf, cdf, 'r--', label='Normal CDF')
#             axes[1].set_xlabel('z')
#             axes[1].xaxis.set_major_locator(MultipleLocator(0.005))
#             axes[1].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
#             axes[1].set_ylabel('Cumulative Probability')
#             axes[1].set_title('ECDF vs Normal CDF')
#             axes[1].legend(loc='upper left')

#             plt.tight_layout()
#             finalize_figure(fig, save_path=save_path, save_plots=save_plots, show_plots=show_plots, filename=f"{prefix}_{i+1}_stats.pdf")


#     # Convert list to Astropy Table
#     col_names = ["Subcluster", "z_min", "z_max", "Mean", "Std Dev", "KS Stat", "KS p-value", "Anderson Level"]
#     stats_table = Table(rows=stats_list, names=col_names)

#     # Print Table
#     print(stats_table)

#     return stats_table, stats_values



def make_stats_table(
    z,
    gaussian_fits,
    bins=12,
    prefix="subcluster",
    make_plots=True,
    save_plots=True,
    show_plots=True,
    save_path=None,
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

        # # Fallback: SciPy AD stat + piecewise p-value approximation
        # anderson_res = scipy.stats.anderson(z_subset, dist="norm")
        # ad2 = float(anderson_res.statistic)

        # # Adjusted statistic for estimated parameters (common correction)
        # n = float(n)
        # ad2_star = ad2 * (1.0 + 0.75 / n + 2.25 / (n * n))

        # # Piecewise approximation
        # if ad2_star < 0.2:
        #     p = 1.0 - np.exp(-13.436 + 101.14 * ad2_star - 223.73 * (ad2_star ** 2))
        # elif ad2_star < 0.34:
        #     p = 1.0 - np.exp(-8.318 + 42.796 * ad2_star - 59.938 * (ad2_star ** 2))
        # elif ad2_star < 0.6:
        #     p = np.exp(0.9177 - 4.279 * ad2_star - 1.38 * (ad2_star ** 2))
        # elif ad2_star < 10:
        #     p = np.exp(1.2937 - 5.709 * ad2_star + 0.0186 * (ad2_star ** 2))
        # else:
        #     p = 0.0

        # p = float(np.clip(p, 0.0, 1.0))
        p = 10.0  # Dummy value to indicate failure
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
            stats_list.append([i + 1, z_low, z_high, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, "No data"])
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

        # Keep your existing "Anderson Level" via SciPy critical values (if possible)
        if N >= 3:
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

    print(stats_table)
    return stats_table, stats_values



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

def velocity_dispersion(zs):
    z_mean = biweight_location(zs)
    vels = c.to('km/s').value * (zs - z_mean) / (1 + z_mean)
    return z_mean, np.sqrt(biweight_midvariance(vels)), vels

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

def get_velocities(gaussians, z):
    # Analyze each spectroscopic group
    vel_data = []
    z_data = []
    for i, (z_low, z_high, mean, std, weight) in enumerate(gaussians):
        z_vals = z[(z >= z_low) & (z < z_high)]
        z_mean, sigma_v, velocities = velocity_dispersion(z_vals)
        vel_data.append((velocities, z_mean, sigma_v))
        z_data.append(z_vals)
        print(f"Subcluster {i+1}: N={len(z_vals)} | z̄ = {z_mean:.5f} | σ_v = {sigma_v:.1f} km/s")

    return vel_data, z_data


# ------ Driver Function -------
def process_redshifts(cluster, z_min_field=0, z_max_field=1.5, input_csv=None, output_folder=None, max_components=None, save_plots=True, show_plots=True):


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
    print(f"Fitting Gaussians in field range ({z_min_field}, {z_max_field})")
    gaussians_field, fit_info_field = fit_gaussian_model(z_field, broad_threshold=0.05, **kwargs)
    print(f"Identified {len(gaussians_field)} clusters.")

    # -- Fit Gaussians for cluster only--
    print(f"Fitting Gaussians in cluster range ({cluster.z_min}, {cluster.z_max})")
    gaussians, fit_info = fit_gaussian_model(z_cluster, broad_threshold=0.02, max_components=2, min_galaxies = 4, **kwargs)
    print(f"Identified {len(gaussians)} subclusters.")


    # -- Output subcluster stats --
    if output_folder is None:
        output_folder = cluster.redshift_path

    image_folder = os.path.join(output_folder, "Images")
    os.makedirs(image_folder, exist_ok=True)

    stats_table_field, _ = make_stats_table(z_field, gaussians_field, prefix="field", make_plots=True, save_plots=save_plots, show_plots=show_plots, save_path=image_folder)
    stats_table_cluster, _  = make_stats_table(z_cluster, gaussians, prefix="cluster", make_plots=True, save_plots=save_plots, show_plots=show_plots, save_path=image_folder)

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


# def main():
#     parser = argparse.ArgumentParser(description="Fit Gaussian Mixture Models to cluster redshift catalog.")
#     parser.add_argument("cluster_id", type=str, help="Cluster ID (e.g. '1327') or full RMJ/Abell name.")
#     parser.add_argument("--input_csv", type=str, default=None, help="Input redshift CSV (default: standard location)")
#     parser.add_argument("--output-folder", type=str, default=None, help="Where to save plots/results")
#     parser.add_argument("--zmin_cluster", type=float, default=None, help="Minimum redshift for cluster membership [default: z - 0.015]")
#     parser.add_argument("--zmax_cluster", type=float, default=None, help="Maximum redshift for cluster membership [default: z + 0.015]")
#     parser.add_argument("--zmin_field", type=float, default=0, help="Minimum overall redshift [default: 0]")
#     parser.add_argument("--zmax_field", type=float, default=1.5, help="Maximum overall redshift [default: 1.5]")
#     parser.add_argument("--max_components", type=int, default=None, help="Number of Gaussians (default: 8)")
#     parser.add_argument("--save-plots", type=str2bool, nargs="?", const=True, default=True, help="Save figures? [default: True]")
#     parser.add_argument("--show-plots", type=str2bool, nargs="?", const=True, default=False, help="Show interactive figures? [default: False]")
#     args = parser.parse_args()

#     cluster = Cluster(args.cluster_id)
#     cluster.populate(verbose=True)
#     if args.zmin_cluster is not None:
#         cluster.z_min = args.zmin_cluster
#     if args.zmax_cluster is not None:
#         cluster.z_max = args.zmax_cluster
#     if cluster.z_min is None or np.isnan(cluster.z_min):
#         cluster.z_min = cluster.redshift - 0.015
#     if cluster.z_max is None or np.isnan(cluster.z_max):
#         cluster.z_max = cluster.redshift + 0.015

#     process_redshifts(cluster, max_components=args.max_components, z_min_field=args.zmin_field, z_max_field=args.zmax_field, input_csv=args.input_csv, output_folder=args.output_folder, save_plots=args.save_plots, show_plots=args.show_plots)


# if __name__ == "__main__":
#     main()



# cluster_id = "RMJ 0003"
# cluster_range = (0.36,0.385)

# cluster_id = "RMJ 0219"
# cluster_range = (0.355,0.375)

# cluster_id = "RMJ 0801"
# cluster_range = (0.485,0.520)

# cluster_id = "RMJ 0829"
# cluster_range = (0.38,0.41)

# cluster_id = "RMJ 0926"
# cluster_range = (0.445,0.475)

# cluster_id = "RMJ 1043"
# cluster_range = (0.42,0.44)

# cluster_id = "RMJ 1219"
# cluster_range = (0.535,0.566)

# cluster_id = "RMJ 1257"
# cluster_range = (0.515,0.54)
# cluster_range = (0.52,0.54)

# cluster_id = "RMJ 1635"
# cluster_range = (0.465,0.485)

# cluster_id = "RMJ 2321"
# cluster_range = (0.48,0.495)
# cluster_range = (0.495,0.51)
# cluster_range = (0.47, 0.52)
# cluster_range = (0.48, 0.492)
# cluster_range = (0.492, 0.51)
# cluster_range = (0.48, 0.492)
# cluster_range = (0.48, 0.51)


# save_plots = False
# show_plots = True



# cluster = Cluster(cluster_id, z_min=cluster_range[0], z_max=cluster_range[1])
# cluster.populate(verbose=True)


# process_redshifts(cluster, save_plots=save_plots, show_plots=show_plots)


