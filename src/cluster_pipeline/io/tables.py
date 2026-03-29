#!/usr/bin/env python3
"""
tables.py

LaTeX Table Generation and Output Formatting
---------------------------------------------------------

Centralizes all LaTeX deluxetable generation and machine-readable table
exports. All table creation functions live here so they are easy to find
and maintain.

Table types:
  - Spectroscopic redshift tables (full catalog + new DEIMOS)
  - BCG information tables
  - Cluster summary one-liners
  - Subcluster statistics (added during Stage 7 refactoring)

Requirements:
  - pandas, numpy

Notes:
  - All functions accept DataFrames and Cluster objects as input.
  - ``emit_latex()`` is the low-level writer used by all table functions.
  - Tables are written to the cluster's Tables/ directory by default.
"""

from __future__ import annotations

import os

import pandas as pd
import numpy as np

from cluster_pipeline.models.cluster import Cluster


# ====================================================================
# Low-level writer
# ====================================================================

def emit_latex(
    latex_str: str,
    save_tex: bool = True,
    print_tex: bool = False,
    save_path: str | os.PathLike | None = None,
) -> None:
    """Write a LaTeX string to a .tex file and/or print to console.

    Parameters
    ----------
    latex_str : str
        The LaTeX table text.
    save_tex : bool
        If True, write to ``save_path``. [default: True]
    print_tex : bool
        If True, print the LaTeX string to the console. [default: False]
    save_path : str or Path or None
        Output file path. Required if ``save_tex=True``.
    """
    if print_tex:
        print(latex_str)

    if save_tex and save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            f.write(latex_str)
        print(f"[saved] LaTeX written to {save_path}")


# ====================================================================
# Redshift tables
# ====================================================================

def export_redshift_table(
    cluster: Cluster,
    combined_df: pd.DataFrame,
    *,
    save_tex: bool = True,
    print_tex: bool = False,
) -> None:
    """Generate a LaTeX deluxetable (first 5 rows preview) and machine-readable
    table for the full spectroscopic redshift catalog.

    Parameters
    ----------
    cluster : Cluster
        Cluster object for paths and metadata.
    combined_df : pd.DataFrame
        Combined redshift catalog with columns: RA, Dec, z, sigma_z, spec_source.
    save_tex : bool
        If True, save LaTeX to Tables/redshifts_table.tex. [default: True]
    print_tex : bool
        If True, print LaTeX to console. [default: False]

    Notes
    -----
    - The LaTeX table shows only the first 5 rows as a preview.
    - A full machine-readable table is always written to Tables/redshifts_table.dat.
    """
    cluster.ensure_directories()
    clean_id = cluster.identifier.replace(" ", "")

    header = (
        "\\begin{deluxetable}{ccccc}\n"
        f"\\tablecaption{{{clean_id} Spectroscopic Redshifts}}"
        f"\\label{{tab:{clean_id}_redshifts}}\n"
        "\\tablehead{\n"
        "  \\colhead{RA [deg]} & \\colhead{Dec [deg]} & \\colhead{Redshift} & "
        "\\colhead{Error} & \\colhead{Source} \\\\\n"
        "}\n"
        "\\centering\n"
        "\\startdata\n"
    )

    body_lines = []
    for _, row in combined_df.head(5).iterrows():
        ra = f"{row['RA']:.5f}"
        dec = f"{row['Dec']:.5f}"
        z = f"{row['z']:.5f}"
        sigma = row.get("sigma_z", 0.0)
        sigma_str = "" if (sigma == 0.0 or pd.isna(sigma)) else f"{sigma:.1e}"
        source = row.get("spec_source", "Unknown")
        if pd.isna(source):
            source = "Unknown"
        body_lines.append(f"{ra} & {dec} & {z} & {sigma_str} & {source} \\\\ \\hline")

    footer = "\\enddata\n\\end{deluxetable}"
    latex_str = header + "\n".join(body_lines) + "\n" + footer

    tex_path = os.path.join(cluster.tables_path, "redshifts_table.tex")
    emit_latex(latex_str, save_tex=save_tex, print_tex=print_tex, save_path=tex_path)

    # Machine-readable full table
    dat_path = os.path.join(cluster.tables_path, "redshifts_table.dat")
    combined_df.to_csv(dat_path, sep="\t", index=False)
    print(f"Machine-readable table written to {dat_path}")


def export_new_redshift_table(
    cluster: Cluster,
    new_spectroscopy_df: pd.DataFrame,
    *,
    save_tex: bool = True,
    print_tex: bool = False,
) -> None:
    """Generate a LaTeX deluxetable and machine-readable table for newly obtained
    spectroscopic redshifts (e.g., from DEIMOS).

    Parameters
    ----------
    cluster : Cluster
        Cluster object for paths and metadata.
    new_spectroscopy_df : pd.DataFrame
        DataFrame with columns: RA, Dec, z, sigma_z (and optionally spec_source).
    save_tex : bool
        If True, save LaTeX to Tables/{id}_new_redshifts_table.tex. [default: True]
    print_tex : bool
        If True, print LaTeX to console. [default: False]

    Notes
    -----
    - Includes cluster identifier as a column (for multi-cluster tables).
    - Omits spec_source column (these are all from the same new observation).
    """
    cluster.ensure_directories()
    clean_id = cluster.name or cluster.identifier

    header = (
        "\\begin{deluxetable}{ccccc}\n"
        f"\\tablecaption{{{clean_id} New Spectroscopic Redshifts}}"
        f"\\label{{tab:{clean_id}_new_redshifts}}\n"
        "\\tablehead{\n"
        "  \\colhead{Cluster} & \\colhead{RA [deg]} & "
        "\\colhead{Dec [deg]} & \\colhead{Redshift} & "
        "\\colhead{Error} \\\\\n"
        "}\n"
        "\\centering\n"
        "\\startdata\n"
    )

    body_lines = []
    for _, row in new_spectroscopy_df.head(5).iterrows():
        ra = f"{row['RA']:.5f}"
        dec = f"{row['Dec']:.5f}"
        z = f"{row['z']:.5f}"
        sigma = row.get("sigma_z", 0.0)
        sigma_str = "" if (sigma == 0.0 or pd.isna(sigma)) else f"{sigma:.1e}"
        body_lines.append(f"{clean_id} & {ra} & {dec} & {z} & {sigma_str} \\\\ \\hline")

    footer = "\\enddata\n\\end{deluxetable}"
    latex_str = header + "\n".join(body_lines) + "\n" + footer

    cluster_path_name = cluster.identifier.replace(" ", "")
    tex_path = os.path.join(cluster.tables_path, f"{cluster_path_name}_new_redshifts_table.tex")
    emit_latex(latex_str, save_tex=save_tex, print_tex=print_tex, save_path=tex_path)

    # Machine-readable table (add cluster column, drop source)
    machine_df = new_spectroscopy_df.copy()
    machine_df.insert(0, "Cluster", clean_id)
    machine_df = machine_df.drop(columns=["spec_source"], errors="ignore")

    dat_path = os.path.join(cluster.tables_path, f"{cluster_path_name}_new_redshifts_table.dat")
    machine_df.to_csv(dat_path, sep="\t", index=False)
    print(f"Machine-readable table written to {dat_path}")


# ====================================================================
# BCG tables
# ====================================================================

def export_bcg_table(
    cluster: Cluster,
    bcg_df: pd.DataFrame,
    *,
    mag_col: str = "rmag",
    save_tex: bool = True,
    print_tex: bool = False,
) -> None:
    """Generate a LaTeX deluxetable with BCG candidate information.

    Parameters
    ----------
    cluster : Cluster
        Cluster object for paths and metadata.
    bcg_df : pd.DataFrame
        BCG DataFrame with columns: BCG_priority, BCG_probability, z,
        RA, Dec, and optionally the magnitude column.
    mag_col : str
        Which magnitude column to include. [default: "rmag"]
    save_tex : bool
        If True, save to Tables/BCGs_table.tex. [default: True]
    print_tex : bool
        If True, print LaTeX to console. [default: False]

    Notes
    -----
    - Table includes tablenotes for spectroscopic source references.
    - BCGs are sorted by priority (rank).
    """
    cluster.ensure_directories()
    clean_id = cluster.identifier.replace(" ", "")

    lines = []
    lines.append(r"\begin{deluxetable}{cccccc}[hb]")
    lines.append(f"    \\caption{{{clean_id} BCG Information}}\\label{{tab:{clean_id}_BCG}}")
    lines.append("    \\tablehead{")
    lines.append(
        "        \\colhead{BCG} & \\colhead{Probability} & \\colhead{Redshift} & "
        f"\\colhead{{{mag_col}\\tablenotemark{{\\footnotesize d}}}} & "
        "\\colhead{RA {[}deg{]}} & \\colhead{Dec {[}deg{]}} "
    )
    lines.append("    }")
    lines.append("    \\centering")
    lines.append("    \\startdata")

    sort_col = "BCG_priority" if "BCG_priority" in bcg_df.columns else bcg_df.columns[0]
    for _, row in bcg_df.sort_values(sort_col).iterrows():
        priority = int(row.get("BCG_priority", 0))
        mag = row.get(mag_col, np.nan)
        z = row.get("z", np.nan)
        prob = row.get("BCG_probability", np.nan)
        ra = row.get("RA", np.nan)
        dec = row.get("Dec", np.nan)

        z_str = f"{z:.3f}" if pd.notnull(z) else "---"
        prob_str = f"{prob:.4f}" if pd.notnull(prob) else "---"
        mag_str = f"{mag:.2f}" if pd.notnull(mag) else "---"
        ra_str = f"{ra:.5f}" if pd.notnull(ra) else "---"
        dec_str = f"{dec:.5f}" if pd.notnull(dec) else "---"

        lines.append(
            f"    {priority}  & {prob_str} & {z_str}\\tablenotemark{{\\footnotesize a}} "
            f"& {mag_str} & {ra_str} & {dec_str} \\\\ \\hline"
        )

    lines.append("    \\enddata")
    lines.append("\\tablenotetext{a}{SDSS DR18 \\citep{almeida2023eighteenth}}")
    lines.append("\\tablenotetext{b}{DESI DR1 \\citep{abdul2025data}}")
    lines.append("\\tablenotetext{c}{DEIMOS (This work)}")
    lines.append("\\tablenotetext{d}{DESI Legacy Survey DR10 \\citep{dey2019overview}}")
    lines.append("\\end{deluxetable}")

    latex_str = "\n".join(lines)
    tex_path = os.path.join(cluster.tables_path, "BCGs_table.tex")
    emit_latex(latex_str, save_tex=save_tex, print_tex=print_tex, save_path=tex_path)


# ====================================================================
# Cluster summary
# ====================================================================

def export_cluster_summary_row(cluster: Cluster) -> str:
    """Return a single LaTeX table row summarizing the cluster.

    Parameters
    ----------
    cluster : Cluster
        Cluster object with name, redshift, richness, coords.

    Returns
    -------
    str
        LaTeX-formatted table row string.

    Notes
    -----
    - Intended for use in multi-cluster summary tables.
    - Also prints the row to stdout.
    """
    z = cluster.redshift or 0.0
    z_err = cluster.redshift_err or 0.0
    rich = cluster.richness or 0.0
    rich_err = cluster.richness_err or 0.0
    ra = cluster.coords.ra.deg if cluster.coords else 0.0
    dec = cluster.coords.dec.deg if cluster.coords else 0.0

    row = (
        f"\\textbf{{{cluster.name}}} & {z:.4f} & {z_err:.4f} & "
        f"{rich:.2f} & {rich_err:.2f} & {ra:.4f} & {dec:.4f} \\\\ \\hline"
    )
    return row
