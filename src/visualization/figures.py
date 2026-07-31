"""Publication figures. Each function returns a matplotlib Figure.

Figures that need a fitted model take a ``(model, section)`` pair; figures that
summarise experiments take the corresponding results JSON.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch

from ..data.preprocess import MORPHOGEN_PATHWAYS, pathway_gene_mask
from ..evaluation.metrics import pearson_per_gene
from .style import (
    CATEGORICAL, DIV, GRID, INK, INK_MUTED, INK_SECONDARY, LATENT, MODEL_COLORS,
    SEQ, WIDTH_FULL, WIDTH_HALF, add_colorbar, bar_with_error, panel_label,
    scatter_field, set_style, spatial_axes,
)


# --------------------------------------------------------------------------- #
# Figure 1 -- architecture schematic
# --------------------------------------------------------------------------- #


def figure1_overview() -> plt.Figure:
    """Schematic of the encode -> evolve -> decode pipeline."""
    set_style()
    fig, ax = plt.subplots(figsize=(WIDTH_FULL, 2.05))
    ax.set_xlim(0, 100); ax.set_ylim(-2, 34)
    ax.axis("off")

    def box(x, y, w, h, title, sub, color, alpha=0.13):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5,rounding_size=1.4",
                                    linewidth=1.0, edgecolor=color,
                                    facecolor=color, alpha=alpha, zorder=1))
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5,rounding_size=1.4",
                                    linewidth=1.0, edgecolor=color, facecolor="none", zorder=3))
        ax.text(x + w / 2, y + h - 2.4, title, ha="center", va="top", fontsize=6.8,
                fontweight="bold", color=INK, zorder=4)
        ax.text(x + w / 2, y + h - 8.0, sub, ha="center", va="top", fontsize=5.4,
                color=INK_SECONDARY, linespacing=1.45, zorder=4)

    def arrow(x0, x1, y=15, label=""):
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=8,
                                     linewidth=1.0, color=INK_SECONDARY, zorder=5))
        if label:
            ax.text((x0 + x1) / 2, y + 1.4, label, ha="center", va="bottom",
                    fontsize=5.6, color=INK_MUTED, style="italic", zorder=6)

    box(0.5, 6, 20, 26, "Spatial encoder",
        "spots $(p_i,\\, g_i)$\ngraph kernel integral\nGaussian splat\nspectral (FNO) blocks",
        CATEGORICAL[0])
    box(29.5, 6, 18, 26, "Latent field $z(x,y)$",
        "continuous field on\na regular lattice\n+ occupancy support", CATEGORICAL[2])
    box(56.5, 6, 22, 26, "Morphogen operator",
        "$\\partial_t z = \\nabla\\!\\cdot\\!(D_\\theta \\nabla z) + f_\\theta(z)$\nStrang splitting;\nexact spectral\ndiffusion solve",
        CATEGORICAL[1])
    box(87.0, 6, 12.5, 26, "Decoder",
        "read-out at\nany $(x,y)$\n$\\rightarrow\\ \\hat{g}$", CATEGORICAL[4])

    arrow(22.0, 28.0, y=18, label="encode")
    arrow(49.0, 55.0, y=18, label="evolve")
    arrow(80.0, 85.5, y=18, label="decode")

    ax.text(50, -1.4, "trained by masked spatial reconstruction; held-out regions are never seen by the encoder",
            ha="center", va="bottom", fontsize=5.6, color=INK_MUTED, style="italic")
    return fig


# --------------------------------------------------------------------------- #
# Figure 2 -- reconstruction examples
# --------------------------------------------------------------------------- #


def figure2_reconstruction(
    section, pred_nmo: np.ndarray, preds_other: Dict[str, np.ndarray],
    visible: np.ndarray, genes: Optional[Sequence[str]] = None, n_genes: int = 3,
) -> plt.Figure:
    """Ground truth vs. predictions for a few well-reconstructed genes."""
    set_style()
    true = section.numpy_expr(denorm=True)
    coords = section.coords.detach().cpu().numpy()
    held = visible == 0

    if genes is None:
        # Show genes spanning the performance range, not only the best ones:
        # a cherry-picked top-k panel would misrepresent typical behaviour.
        r = pearson_per_gene(pred_nmo[held], true[held])
        order = np.argsort(-np.nan_to_num(r, nan=-1))
        valid = order[np.isfinite(r[order])]
        if len(valid) >= n_genes:
            quantiles = np.linspace(0, 0.75, n_genes)  # best -> lower quartile
            gi = np.array([valid[int(q * (len(valid) - 1))] for q in quantiles])
        else:
            gi = valid[:n_genes]
    else:
        name2i = {g: i for i, g in enumerate(section.gene_names)}
        gi = np.array([name2i[g] for g in genes if g in name2i][:n_genes])

    others = list(preds_other.items())
    ncol = 2 + len(others)
    fig, axes = plt.subplots(len(gi), ncol, figsize=(WIDTH_FULL, 1.35 * len(gi)),
                             squeeze=False)

    for r_i, g in enumerate(gi):
        vmin, vmax = np.nanpercentile(true[:, g], [2, 98])
        h = scatter_field(axes[r_i][0], coords, true[:, g], vmin=vmin, vmax=vmax, s=1.6)
        # outline the held-out region on the ground-truth panel
        axes[r_i][0].scatter(coords[held, 0], coords[held, 1], s=1.6, facecolors="none",
                             edgecolors="#00000022", linewidths=0.15, rasterized=True)
        axes[r_i][0].set_ylabel(section.gene_names[g], fontsize=7, style="italic",
                                color=INK, rotation=0, ha="right", va="center", labelpad=2)
        scatter_field(axes[r_i][1], coords, pred_nmo[:, g], vmin=vmin, vmax=vmax, s=1.6)
        for c_i, (nm, pr) in enumerate(others):
            scatter_field(axes[r_i][2 + c_i], coords, pr[:, g], vmin=vmin, vmax=vmax, s=1.6)

        rr = pearson_per_gene(pred_nmo[held][:, [g]], true[held][:, [g]])[0]
        axes[r_i][1].text(0.5, -0.06, f"r = {rr:.2f}", transform=axes[r_i][1].transAxes,
                          ha="center", va="top", fontsize=6, color=INK_SECONDARY)
        for c_i, (nm, pr) in enumerate(others):
            rr2 = pearson_per_gene(pr[held][:, [g]], true[held][:, [g]])[0]
            axes[r_i][2 + c_i].text(0.5, -0.06, f"r = {rr2:.2f}",
                                    transform=axes[r_i][2 + c_i].transAxes, ha="center",
                                    va="top", fontsize=6, color=INK_SECONDARY)

    titles = ["Measured", "NMO (ours)"] + [n for n, _ in others]
    for c, t in enumerate(titles):
        axes[0][c].set_title(t, fontsize=7.5, color=INK, pad=4)
    fig.subplots_adjust(wspace=0.06, hspace=0.30)
    return fig


# --------------------------------------------------------------------------- #
# Figure 3 -- latent morphogen fields
# --------------------------------------------------------------------------- #


def figure3_latent_fields(model, section, n_show: int = 6) -> plt.Figure:
    """Latent channels of the relaxed field, plus their pathway alignment."""
    set_style()
    visible = section.mask(["train", "val", "test"])
    with torch.no_grad():
        z0, occ = model.encode(section.coords, section.expr * visible.view(-1, 1),
                               section.edge_index, visible)
        zT = model.evolve(z0)
    Z = zT.detach().cpu().numpy()[0]                     # (C, H, W)
    occ_np = occ.detach().cpu().numpy()[0, 0]
    support = occ_np > np.percentile(occ_np, 45)

    # rank channels by spatial structure inside the tissue support
    energy = np.array([np.nanstd(np.where(support, Z[c], np.nan)) for c in range(Z.shape[0])])
    order = np.argsort(-energy)[:n_show]

    # pathway alignment: correlate each channel (sampled at spots) with the
    # mean expression of each morphogen pathway's genes
    from ..models.layers import grid_gather

    with torch.no_grad():
        z_at = grid_gather(zT[0], section.coords).cpu().numpy()      # (N, C)
    expr = section.numpy_expr(denorm=True)
    pw_names, corr = [], []
    for pw in MORPHOGEN_PATHWAYS:
        m = pathway_gene_mask(section.gene_names, pw)
        if m.sum() < 4:
            continue
        score = expr[:, m].mean(1)
        pw_names.append(f"{pw} ({int(m.sum())})")
        corr.append([
            np.corrcoef(z_at[:, c], score)[0, 1] if np.std(z_at[:, c]) > 1e-9 else np.nan
            for c in order
        ])
    corr = np.array(corr) if corr else np.zeros((1, len(order)))

    nrow = 2
    fig = plt.figure(figsize=(WIDTH_FULL, 2.9))
    gs = fig.add_gridspec(nrow, n_show, height_ratios=[1.0, 0.95], hspace=0.42, wspace=0.08)

    for i, c in enumerate(order):
        ax = fig.add_subplot(gs[0, i])
        field = np.where(support, Z[c], np.nan)
        m = np.nanpercentile(np.abs(field), 98)
        ax.imshow(field, origin="lower", cmap=DIV, vmin=-m, vmax=m, interpolation="bilinear")
        spatial_axes(ax)
        ax.set_title(f"$z_{{{c}}}$", fontsize=7, color=INK, pad=2)
        if i == 0:
            panel_label(ax, "a", dx=-0.10, dy=1.30)

    axh = fig.add_subplot(gs[1, :])
    im = axh.imshow(corr, cmap=DIV, vmin=-np.nanmax(np.abs(corr)), vmax=np.nanmax(np.abs(corr)),
                    aspect="auto")
    axh.set_xticks(range(len(order)))
    axh.set_xticklabels([f"$z_{{{c}}}$" for c in order], fontsize=6.5)
    axh.set_yticks(range(len(pw_names)))
    axh.set_yticklabels(pw_names, fontsize=6.5)
    axh.grid(False)
    for s in axh.spines.values():
        s.set_visible(False)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            if np.isfinite(corr[i, j]):
                axh.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=5.4,
                         color=INK if abs(corr[i, j]) < 0.55 else "white")
    axh.set_title("Pearson correlation between latent channel and morphogen-pathway score",
                  fontsize=7, color=INK, pad=4)
    panel_label(axh, "b", dx=-0.075, dy=1.30)
    cb = fig.colorbar(im, ax=axh, pad=0.012, aspect=10, shrink=0.9)
    cb.ax.tick_params(labelsize=5.5, width=0.4)
    cb.outline.set_visible(False)
    return fig


# --------------------------------------------------------------------------- #
# Figure 4 -- transfer
# --------------------------------------------------------------------------- #


def _agg(records: List[Dict], key: str, group: Sequence[str]) -> "object":
    import pandas as pd

    df = pd.DataFrame(records)
    if df.empty or key not in df.columns:
        return None
    return df.groupby(list(group))[key].agg(["mean", "std", "count"]).reset_index()


def figure4_transfer(exp2: List[Dict], exp3: Optional[List[Dict]] = None) -> plt.Figure:
    """Zero-shot transfer performance across tissue and across resolution."""
    set_style()
    panels = [("Mouse brain $\\rightarrow$ human breast\n(cross-tissue, cross-species)", exp2)]
    if exp3:
        panels.append(("Visium $\\rightarrow$ Xenium\n(cross-resolution)", exp3))

    fig, axes = plt.subplots(1, len(panels), figsize=(WIDTH_FULL, 2.35), squeeze=False)
    order = ["oracle", "zero_shot", "decoder_finetune", "floor"]
    pretty = {"oracle": "in-domain oracle", "zero_shot": "zero-shot",
              "decoder_finetune": "decoder-only fine-tune", "floor": "training mean"}

    for pi, (title, recs) in enumerate(panels):
        ax = axes[0][pi]
        a = _agg(recs, "pearson_mean", ["setting", "model"])
        if a is None:
            continue
        labels, means, errs, colors = [], [], [], []
        for setting in order:
            sub = a[a["setting"] == setting]
            for _, row in sub.sort_values("mean", ascending=False).iterrows():
                if setting == "floor" and row["model"] != "mean":
                    continue
                nm = "mean" if setting == "floor" else row["model"]
                labels.append(f"{nm}  ({pretty[setting]})")
                means.append(float(row["mean"]))
                errs.append(float(row["std"]) if np.isfinite(row["std"]) else 0.0)
                colors.append(MODEL_COLORS.get(nm, INK_MUTED))
        bar_with_error(ax, labels, means, errs, colors, horizontal=True)
        ax.set_xlabel("Pearson $r$ (held-out locations)")
        ax.set_title(title, fontsize=7.5, pad=5)
        panel_label(ax, "ab"[pi], dx=-0.62, dy=1.13)
    fig.subplots_adjust(wspace=1.5)
    return fig


# --------------------------------------------------------------------------- #
# Figure 5 -- dynamics
# --------------------------------------------------------------------------- #


def figure5_dynamics(model, section, channel: Optional[int] = None) -> plt.Figure:
    """Relaxation trajectory, learned diffusion tensors, and dispersion relation."""
    set_style()
    visible = section.mask(["train"])
    with torch.no_grad():
        z0, occ = model.encode(section.coords, section.expr * visible.view(-1, 1),
                               section.edge_index, visible)
        zT, traj = model.evolve(z0, return_traj=True)
    T = len(traj) - 1
    occ_np = occ.detach().cpu().numpy()[0, 0]
    support = occ_np > np.percentile(occ_np, 45)

    Z0 = traj[0].detach().cpu().numpy()[0]
    if channel is None:
        channel = int(np.argmax([np.nanstd(np.where(support, Z0[c], np.nan))
                                 for c in range(Z0.shape[0])]))

    show = [0, max(T // 3, 1), max(2 * T // 3, 2), T] if T >= 3 else list(range(T + 1))
    fig = plt.figure(figsize=(WIDTH_FULL, 3.25))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.05], hspace=0.55, wspace=0.62)

    fields = [traj[t].detach().cpu().numpy()[0, channel] for t in show]
    m = np.nanpercentile(np.abs(np.stack(fields)), 98)
    for i, (t, f) in enumerate(zip(show, fields)):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(np.where(support, f, np.nan), origin="lower", cmap=DIV, vmin=-m, vmax=m,
                  interpolation="bilinear")
        spatial_axes(ax)
        ax.set_title(f"$t = {t}\\,\\Delta t$", fontsize=7, pad=2)
        if i == 0:
            panel_label(ax, "a", dx=-0.10, dy=1.32)

    # -- diffusion tensor ellipses --
    ax2 = fig.add_subplot(gs[1, 0])
    if model.operator.diffusion is not None:
        D = model.operator.diffusion.tensor().detach().cpu().numpy()
        scale = section.coord_scale_um
        for c in range(D.shape[0]):
            ev, evec = np.linalg.eigh(D[c])
            ang = math.degrees(math.atan2(evec[1, 1], evec[0, 1]))
            # semi-axes as diffusion lengths over the full horizon, in microns
            hz = model.cfg.dynamics.dt * model.cfg.dynamics.n_steps
            a, b = np.sqrt(2 * np.maximum(ev, 0) * hz) * scale
            ax2.add_patch(Ellipse((0, 0), 2 * b, 2 * a, angle=ang, facecolor="none",
                                  edgecolor=CATEGORICAL[0], alpha=0.42, linewidth=0.7))
        lim = float(np.nanpercentile(
            np.sqrt(2 * np.maximum(np.linalg.eigvalsh(D), 0) * hz) * scale, 97)) * 1.35
        ax2.set_xlim(-lim, lim); ax2.set_ylim(-lim, lim)
        ax2.set_aspect("equal")
    ax2.set_xlabel("$\\mu$m"); ax2.set_ylabel("$\\mu$m")
    ax2.set_title("Diffusion tensors", fontsize=7, pad=4)
    panel_label(ax2, "b", dx=-0.34, dy=1.30)

    # -- dispersion relation --
    ax3 = fig.add_subplot(gs[1, 1:3])
    rep = model.operator.linear_stability(zT, coord_scale_um=section.coord_scale_um)
    k, g = rep["k"], rep["growth_rate"]
    ax3.plot(k, g, color=CATEGORICAL[1], linewidth=1.6)
    ax3.axhline(0, color=INK_SECONDARY, linewidth=0.6, linestyle="--")
    if rep["growth_max"] > 0 and rep["k_max"] > 1e-6:
        lam = 2 * np.pi / rep["k_max"] * section.coord_scale_um
        ax3.axvline(rep["k_max"], color=INK_MUTED, linewidth=0.6, linestyle=":")
        ax3.annotate(f"$k^* = {rep['k_max']:.1f}$\n$\\lambda \\approx {lam:.0f}\\,\\mu$m",
                     xy=(rep["k_max"], rep["growth_max"]), xytext=(6, -2),
                     textcoords="offset points", fontsize=6, color=INK_SECONDARY)
    ax3.set_xlabel("wavenumber $|k|$ (normalised units)")
    ax3.set_ylabel("max Re $\\lambda(k)$")
    ax3.set_title("Dispersion relation of the learned operator", fontsize=7, pad=4)
    panel_label(ax3, "c", dx=-0.13, dy=1.30)

    # -- diffusion length distribution --
    ax4 = fig.add_subplot(gs[1, 3])
    L = model.diffusion_length_um(section.coord_scale_um)
    if L.size:
        ax4.hist(np.asarray(L).ravel(), bins=14, color=CATEGORICAL[2], alpha=0.85,
                 edgecolor="white", linewidth=0.4)
    ax4.set_xlabel("diffusion length ($\\mu$m)")
    ax4.set_ylabel("channels")
    ax4.set_title("Length scales", fontsize=7, pad=4)
    panel_label(ax4, "d", dx=-0.36, dy=1.30)
    return fig


# --------------------------------------------------------------------------- #
# Figure 6 -- ablations
# --------------------------------------------------------------------------- #


ABLATION_LABELS = {
    "full": "Full model",
    "no_pde": "$-$ PDE constraints",
    "no_diffusion": "$-$ diffusion term",
    "no_reaction": "$-$ reaction term",
    "no_bio_reg": "$-$ biological regularisers",
    "no_dynamics": "$-$ dynamics ($T=0$)",
    "isotropic_diffusion": "isotropic $D$",
    "state_dependent_diffusion": "state-dependent $D$",
    "discrete_gnn": "discrete GNN operator",
    "latent_8": "latent $C=8$",
    "latent_16": "latent $C=16$",
    "latent_32": "latent $C=32$",
    "latent_64": "latent $C=64$",
}


def figure6_ablations(records: List[Dict]) -> plt.Figure:
    """Ablation deltas relative to the full model, plus the latent-width sweep."""
    set_style()
    import pandas as pd

    df = pd.DataFrame(records)
    a = df.groupby("variant")["pearson_mean"].agg(["mean", "std"]).reset_index()
    full = float(a.loc[a["variant"] == "full", "mean"].iloc[0]) if (a["variant"] == "full").any() else np.nan

    structural = [v for v in ["no_dynamics", "no_diffusion", "no_reaction", "no_pde",
                              "no_bio_reg", "isotropic_diffusion",
                              "state_dependent_diffusion", "discrete_gnn"]
                  if v in set(a["variant"])]
    latent = [v for v in ["latent_8", "latent_16", "latent_32", "latent_64"]
              if v in set(a["variant"])]

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_FULL, 2.3),
                             gridspec_kw={"width_ratios": [1.55, 1.0]})

    sub = a[a["variant"].isin(structural)].copy()
    sub["delta"] = sub["mean"] - full
    sub = sub.sort_values("delta")
    colors = [CATEGORICAL[1] if d < 0 else CATEGORICAL[2] for d in sub["delta"]]
    bar_with_error(
        axes[0], [ABLATION_LABELS.get(v, v) for v in sub["variant"]],
        list(sub["delta"]), list(np.nan_to_num(sub["std"])), colors,
        horizontal=True, value_fmt="{:+.3f}",
    )
    axes[0].axvline(0, color=INK_SECONDARY, linewidth=0.7)
    axes[0].set_xlabel("$\\Delta$ Pearson $r$ vs. full model")
    axes[0].set_title(f"Ablations (full model $r$ = {full:.3f})", fontsize=7.5, pad=5)
    panel_label(axes[0], "a", dx=-0.58, dy=1.12)

    if latent:
        s2 = a[a["variant"].isin(latent)].copy()
        s2["C"] = [int(v.split("_")[1]) for v in s2["variant"]]
        # The base configuration already uses C = 32, so the 'full' run *is* the
        # C = 32 point of the sweep; re-running it would duplicate work.
        base_C = 32
        if base_C not in set(s2["C"]) and (a["variant"] == "full").any():
            row = a.loc[a["variant"] == "full"].iloc[0]
            s2 = pd.concat([s2, pd.DataFrame([{
                "variant": "full", "mean": row["mean"], "std": row["std"], "C": base_C,
            }])], ignore_index=True)
        s2 = s2.sort_values("C")
        axes[1].errorbar(s2["C"], s2["mean"], yerr=np.nan_to_num(s2["std"]),
                         marker="o", color=CATEGORICAL[0], capsize=2, linewidth=1.5,
                         markersize=4.5, ecolor=INK_SECONDARY, elinewidth=0.8)
        axes[1].set_xscale("log", base=2)
        axes[1].set_xticks(s2["C"]); axes[1].set_xticklabels(s2["C"])
        axes[1].set_xlabel("latent channels $C$")
        axes[1].set_ylabel("Pearson $r$")
        axes[1].set_title("Latent width", fontsize=7.5, pad=5)
        panel_label(axes[1], "b", dx=-0.24, dy=1.12)
    fig.subplots_adjust(wspace=0.95)
    return fig


# --------------------------------------------------------------------------- #
# Figure 7 -- benchmark summary (main table as a chart)
# --------------------------------------------------------------------------- #


def figure7_benchmark(records: List[Dict]) -> plt.Figure:
    set_style()
    import pandas as pd

    df = pd.DataFrame(records)
    metrics = [("pearson_mean", "Pearson $r$ $\\uparrow$"), ("rmse", "RMSE $\\downarrow$"),
               ("ssim_mean", "SSIM $\\uparrow$"), ("morans_i_abs_error", "|$\\Delta$ Moran's $I$| $\\downarrow$")]
    metrics = [(k, l) for k, l in metrics if k in df.columns]
    fig, axes = plt.subplots(1, len(metrics), figsize=(WIDTH_FULL, 2.1))
    for i, (key, label) in enumerate(metrics):
        a = df.groupby("model")[key].agg(["mean", "std"]).reset_index()
        asc = key in ("rmse", "morans_i_abs_error", "mae")
        a = a.sort_values("mean", ascending=asc)
        bar_with_error(
            axes[i], list(a["model"]), list(a["mean"]), list(np.nan_to_num(a["std"])),
            [MODEL_COLORS.get(m, INK_MUTED) for m in a["model"]], horizontal=True,
            value_fmt="{:.3f}", label_values=(i == 0),
        )
        axes[i].set_xlabel(label, fontsize=7)
        if i > 0:
            axes[i].set_yticklabels([])
    fig.subplots_adjust(wspace=0.18)
    return fig
