"""
Plot style: STIX fonts, minimal frames, and the colours of the figures. Call use_style() once.

Learned controls are drawn in LEARNED, exact ones in dashed black, and linear-strategy ones in LINEAR.
"""

import matplotlib as mpl

LEARNED = "#4C72B0"      # steel blue
LINEAR = "#C44E52"       # muted red


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
