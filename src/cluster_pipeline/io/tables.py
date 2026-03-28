"""LaTeX table and output formatting utilities."""

from __future__ import annotations
import os


def emit_latex(
        latex_str: str,
        save_tex: bool = True,
        print_tex: bool = False,
        save_path: str | os.PathLike | None = None,
    ) -> None:
    """
    Print the LaTeX string and optionally save it to a .tex file.

    Parameters
    ----------
    latex_str : str
        The LaTeX table text to print (and optionally save).
    save_tex : bool
        - None/False: don't write a file
        - True: write to `save_path`
    print_tex : bool
        If True, print the LaTeX string to the console.
    save_path : str | None | os.PathLike
        Fallback path used when save_tex is True.
    """
    if print_tex:
        print(latex_str)

    # Optionally save
    if save_tex:
        with open(save_path, "w") as f:
            f.write(latex_str)
        print(f"[saved] LaTeX written to {save_path}")
