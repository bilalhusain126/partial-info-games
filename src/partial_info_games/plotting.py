"""
Plot style: STIX fonts, minimal frames, and the colours of the figures. Call use_style() once.

Learned controls are drawn in LEARNED and exact ones in dashed black.
plot_controls draws them along a few validation paths, one panel per player.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

LEARNED = "#4C72B0"      # steel blue


def use_style():
    """Sets a serif (STIX) matplotlib style for all subsequent figures."""
    mpl.rcParams.update({
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "lines.linewidth": 1.4,
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


def plot_controls(learned: list, exact: list, dt: float, titles: list, paths=(0, 1, 2), ylabel: str = 'control'):
    """
    Plots the controls of each player along a few validation paths: learned (solid, LEARNED) and exact
    (dashed black).

    Args:
        learned, exact: Controls per player, lists of [M, N] tensors.
        dt: Time step of the grid.
        titles: Panel title per player.
        paths: Indices of the paths to plot. Default (0, 1, 2).
        ylabel: Label of the vertical axis. Default 'control'.
    """
    t = np.arange(learned[0].shape[1]) * dt
    fig, axes = plt.subplots(1, len(learned), figsize=(10, 3.2), sharey=True)
    for i, ax in enumerate(axes):
        for n, path in enumerate(paths):
            ax.plot(t, learned[i][path].cpu(), color=LEARNED, lw=2.6, alpha=0.45, label='learned' if n == 0 else None)
            ax.plot(t, exact[i][path].cpu(), 'k--', lw=0.8, label='exact' if n == 0 else None)
        ax.set_xlabel('$t$')
        ax.set_title(titles[i])
    axes[0].set_ylabel(ylabel)
    axes[0].legend()
    plt.show()
