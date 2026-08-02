"""Figure style: a single place that defines how every plot in the paper looks.

The categorical palette is an Okabe-Ito derivative reordered so that no adjacent
pair falls below the colour-vision-deficiency separation floor. It was checked
computationally rather than by eye (worst adjacent pair deltaE = 9.6 under
deuteranopia, >= 8 required; normal-vision floor 20.0). Because several hues sit
below 3:1 contrast against white, every categorical chart also carries a direct
label or a legend entry, so identity is never conveyed by colour alone.

Sequential fields use a single perceptually-uniform hue ramp; signed quantities
(differences, residuals, Jacobians) use a diverging map with a neutral midpoint.
Never a rainbow map for magnitude.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# --------------------------------------------------------------------------- #
# Palette
# --------------------------------------------------------------------------- #

CATEGORICAL: List[str] = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#E69F00",  # orange
    "#7F3C8D",  # purple
    "#56B4E9",  # sky blue
    "#CC79A7",  # reddish purple
]

INK = "#1a1a1a"
INK_SECONDARY = "#4d4d4d"
INK_MUTED = "#7a7a7a"
GRID = "#d9d9d9"
SURFACE = "#ffffff"

#: Fixed model -> colour assignment. Colour follows the entity, never its rank,
#: so a figure that drops a baseline does not repaint the survivors.
MODEL_COLORS: Dict[str, str] = {
    "nmo": "#0072B2",
    "gnn": "#D55E00",
    "stagate": "#009E73",
    "spagcn": "#E69F00",
    "graph_transformer": "#7F3C8D",
    "gp": "#56B4E9",
    "autoencoder": "#CC79A7",
    "neural_field": "#F0E442",
    "gp_multiscale": "#8DA0CB",
    "tangram": "#B15928",
    "spage": "#999999",
    "mean": INK_MUTED,
}

# Sequential (magnitude): single hue, light -> dark.
SEQ = LinearSegmentedColormap.from_list(
    "nmo_seq", ["#f2f7fb", "#c6dcec", "#8dbcd8", "#4f95c1", "#1f6da3", "#0b4b78"]
)
# Diverging (signed): two poles, neutral grey midpoint -- no hue at the centre.
DIV = LinearSegmentedColormap.from_list(
    "nmo_div", ["#0b4b78", "#4f95c1", "#c6dcec", "#f0f0f0", "#f6c9a8", "#d98850", "#8c3d06"]
)
# Latent channels: perceptually uniform, but distinct from the expression ramp
LATENT = "viridis"


def set_style(fontsize: int = 8, usetex: bool = False) -> None:
    """NeurIPS-appropriate defaults: small type, thin recessive axes, no chartjunk."""
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "font.size": fontsize,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "axes.titlesize": fontsize + 1,
            "axes.labelsize": fontsize,
            "axes.labelcolor": INK,
            "axes.edgecolor": INK_SECONDARY,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.prop_cycle": mpl.cycler(color=CATEGORICAL),
            "grid.color": GRID,
            "grid.linewidth": 0.4,
            "grid.alpha": 0.7,
            "xtick.labelsize": fontsize - 1,
            "ytick.labelsize": fontsize - 1,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "legend.fontsize": fontsize - 1,
            "legend.frameon": False,
            "legend.handlelength": 1.4,
            "lines.linewidth": 1.6,
            "lines.markersize": 4,
            "text.usetex": usetex,
            "text.color": INK,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
        }
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

# NeurIPS text width is 5.5 in; a two-column-style figure spans that.
WIDTH_FULL = 5.5
WIDTH_HALF = 2.65


def spatial_axes(ax) -> None:
    """Tissue maps are images: equal aspect, no grid, no ticks, no frame."""
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


def scatter_field(
    ax, coords: np.ndarray, values: np.ndarray, cmap=None, s: float = 2.0,
    vmin=None, vmax=None, robust: bool = True, diverging: bool = False,
):
    """Plot an irregularly sampled scalar field as a tissue map."""
    if robust and vmin is None and vmax is None:
        if diverging:
            m = float(np.nanpercentile(np.abs(values), 99))
            vmin, vmax = -m, m
        else:
            vmin, vmax = np.nanpercentile(values, [1, 99])
    cmap = cmap if cmap is not None else (DIV if diverging else SEQ)
    h = ax.scatter(
        coords[:, 0], coords[:, 1], c=values, s=s, cmap=cmap, vmin=vmin, vmax=vmax,
        linewidths=0, rasterized=True,
    )
    spatial_axes(ax)
    return h


def add_colorbar(fig, handle, ax, label: str = "", pad: float = 0.02, shrink: float = 0.85):
    cb = fig.colorbar(handle, ax=ax, pad=pad, shrink=shrink, aspect=18)
    cb.set_label(label, fontsize=7, color=INK_SECONDARY)
    cb.ax.tick_params(labelsize=6, width=0.5, color=INK_SECONDARY)
    cb.outline.set_visible(False)
    return cb


def bar_with_error(
    ax, labels: Sequence[str], means: Sequence[float], errs: Sequence[float],
    colors: Sequence[str] = None, horizontal: bool = True, value_fmt: str = "{:.3f}",
    label_values: bool = True,
):
    """Bar chart with error bars and direct value labels.

    Direct labels are the secondary encoding that makes the palette legible for
    readers who cannot separate the hues, and they satisfy the contrast-relief
    requirement for the lighter categorical steps.
    """
    colors = list(colors) if colors is not None else CATEGORICAL[: len(labels)]
    pos = np.arange(len(labels))
    if horizontal:
        b = ax.barh(pos, means, xerr=errs, color=colors, height=0.68,
                    error_kw=dict(ecolor=INK_SECONDARY, elinewidth=0.8, capsize=2))
        ax.set_yticks(pos); ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        if label_values:
            # Place the label on the outside of the bar end, so it never sits on
            # top of the bar or its error whisker for negative values.
            span = (max(means) - min(means)) if len(means) > 1 else max(abs(max(means)), 1e-9)
            pad = 0.04 * max(span, 1e-9)
            for p, m, e in zip(pos, means, errs):
                out = m - (e or 0) - pad if m < 0 else m + (e or 0) + pad
                ax.text(out, p, value_fmt.format(m), va="center",
                        ha="right" if m < 0 else "left", fontsize=6, color=INK_SECONDARY)
            lo = min(list(means) + [0]); hi = max(list(means) + [0])
            ax.set_xlim(lo - 0.42 * max(span, 1e-9), hi + 0.42 * max(span, 1e-9))
    else:
        b = ax.bar(pos, means, yerr=errs, color=colors, width=0.68,
                   error_kw=dict(ecolor=INK_SECONDARY, elinewidth=0.8, capsize=2))
        ax.set_xticks(pos); ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.grid(axis="x", visible=False)
        if label_values:
            for p, m, e in zip(pos, means, errs):
                ax.text(p, m + (e or 0), value_fmt.format(m), ha="center", va="bottom",
                        fontsize=6, color=INK_SECONDARY)
    return b


def panel_label(ax, letter: str, dx: float = -0.12, dy: float = 1.06):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="top", ha="left", color=INK)


def savefig(fig, path, formats=("pdf", "png")):
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for f in formats:
        p = path.with_suffix("." + f)
        fig.savefig(p)
        out.append(p)
    plt.close(fig)
    return out


def scale_bar(ax, coords, coord_scale_um: float, length_um: float = 1000.0,
              label: Optional[str] = None) -> None:
    """Draw a physical scale bar on a tissue map.

    A spatial figure without one gives the reader no way to judge the length
    scales the method is arguing about, which for a paper about diffusion
    lengths is the whole point.
    """
    import numpy as np
    frac = float(length_um) / float(coord_scale_um)      # coords are in [-1, 1]
    x0, x1 = np.nanmin(coords[:, 0]), np.nanmax(coords[:, 0])
    y0 = np.nanmin(coords[:, 1])
    span = x1 - x0
    if not np.isfinite(frac) or frac <= 0 or frac > 0.9 * span:
        return
    xs = x1 - frac - 0.04 * span
    ys = y0 - 0.03 * (np.nanmax(coords[:, 1]) - y0)
    ax.plot([xs, xs + frac], [ys, ys], lw=1.4, color=INK, solid_capstyle="butt",
            clip_on=False, zorder=6)
    ax.text(xs + frac / 2, ys - 0.02 * span,
            label or (f"{length_um/1000:g} mm" if length_um >= 1000
                      else f"{length_um:g} $\\mu$m"),
            ha="center", va="top", fontsize=5.0, color=INK, clip_on=False)
