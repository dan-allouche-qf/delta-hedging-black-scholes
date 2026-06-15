"""Plotting helpers. Every convergence plot carries Monte-Carlo error bars so the
reader can see whether an apparent effect is signal or sampling noise.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from .analysis import mc_summary

__all__ = [
    "FREQUENCY_LABELS",
    "plot_error_histograms",
    "plot_convergence",
]

#: Calendar-day convention used by the project (1 year = 12 months,
#: 1 month = 4 weeks, 1 week = 7 days), so over T = 0.25y: n = 84 = calendar-daily.
#: A real trading quarter has ~63 trading days; this is documented, not hidden.
FREQUENCY_LABELS = {
    0: "No hedging",
    1: "Single (static)",
    3: "Monthly",
    12: "Weekly",
    84: "Calendar-daily",
    252: "~4x / trading day",
}


def _label(n: int) -> str:
    return FREQUENCY_LABELS.get(n, f"n = {n}")


def plot_error_histograms(
    results: dict[int, NDArray],
    title: str,
    color: str = "steelblue",
    ncols: int = 3,
):
    """Grid of error histograms with mean and zero reference lines."""
    ns = sorted(results)
    nrows = int(np.ceil(len(ns) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for idx, n in enumerate(ns):
        ax = axes[idx]
        errs = results[n]
        s = mc_summary(errs)
        ax.hist(errs, bins=50, density=True, alpha=0.75, color=color, edgecolor="black")
        ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Error = 0")
        ax.axvline(
            s["mean"], color="black", linewidth=2,
            label=f"Mean = {s['mean']:.4f} ± {s['sem']:.4f}",
        )
        ax.set_title(f"n = {n} ({_label(n)})", fontweight="bold")
        ax.set_xlabel("Hedging error")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    for j in range(len(ns), len(axes)):
        fig.delaxes(axes[j])
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_convergence(
    results: dict[int, NDArray],
    title_prefix: str = "",
    color: str = "steelblue",
):
    """Two panels: mean error with 95% CI error bars, and std on a log-log axis."""
    ns = [n for n in sorted(results) if n > 0]
    summaries = {n: mc_summary(results[n]) for n in ns}
    means = [summaries[n]["mean"] for n in ns]
    sems = [summaries[n]["sem"] for n in ns]
    stds = [summaries[n]["std"] for n in ns]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.errorbar(
        ns, means, yerr=[1.96 * s for s in sems], fmt="o-", capsize=4,
        color=color, linewidth=2, markersize=7,
    )
    ax1.axhline(0, color="red", linestyle="--", linewidth=1, label="Error = 0")
    ax1.set_xscale("log")
    ax1.set_xlabel("Number of hedging periods n")
    ax1.set_ylabel("Mean error (95% CI)")
    ax1.set_title(f"{title_prefix}Mean error vs n (with MC error bars)", fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(ns, stds, "o-", color=color, linewidth=2, markersize=7)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Number of hedging periods n")
    ax2.set_ylabel("Std of error")
    ax2.set_title(f"{title_prefix}Std of error vs n (log-log)", fontweight="bold")
    ax2.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    return fig
