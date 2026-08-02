"""Publication figures. Each function returns a matplotlib Figure.

Figures that need a fitted model take a ``(model, section)`` pair; figures that
summarize experiments take the corresponding results JSON.
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
from ..models.baselines import DISPLAY_NAMES
from .style import (
    CATEGORICAL, DIV, GRID, INK, INK_MUTED, INK_SECONDARY, LATENT, MODEL_COLORS,
    SEQ, WIDTH_FULL, WIDTH_HALF, add_colorbar, bar_with_error, panel_label,
    scale_bar, scatter_field, set_style, spatial_axes,
)


# --------------------------------------------------------------------------- #
# Figure 1 -- architecture schematic
# --------------------------------------------------------------------------- #


def figure1_overview(model=None, section=None) -> plt.Figure:
    """The pipeline shown on real tissue rather than as a flow chart.

    A box-and-arrow schematic tells a reader what the components are called,
    which the text already does. What it cannot show is that each stage produces
    something: measured expression on an irregular point set, a continuous latent
    field on a lattice, that field after the operator has acted, and a prediction
    at coordinates the encoder never saw. Every panel here is real output from a
    trained checkpoint; only the arrows are drawn.

    Falls back to the schematic when no checkpoint is supplied, so the figure
    still builds on a fresh clone with no results.
    """
    set_style()
    if model is None or section is None:
        return _figure1_schematic()

    import torch
    vis = section.mask("train")
    with torch.no_grad():
        z0, occ = model.encode(section.coords, section.expr * vis.view(-1, 1),
                               section.edge_index, vis)
        zT = model.evolve(z0)
        out = model(section.coords, section.expr * vis.view(-1, 1),
                    query_coords=section.coords, edge_index=section.edge_index,
                    point_mask=vis)
    xy = section.coords.cpu().numpy()
    true = section.numpy_expr(denorm=False)
    pred = out["pred"].detach().cpu().numpy()
    held = ~vis.cpu().numpy().astype(bool)

    # the gene with the most spatial structure, so the panels show signal
    from ..evaluation.metrics import morans_i, spatial_weights
    W = spatial_weights(xy, 8)
    gi = int(np.nanargmax(morans_i(true, W)))

    Z0 = z0.detach().cpu().numpy()[0]
    ZT = zT.detach().cpu().numpy()[0]
    occ_np = occ.detach().cpu().numpy()[0, 0]
    ch = int(np.argmax([np.nanstd(np.where(occ_np > np.percentile(occ_np, 45),
                                           Z0[c], np.nan)) for c in range(Z0.shape[0])]))

    fig = plt.figure(figsize=(WIDTH_FULL, 1.42))
    gs = fig.add_gridspec(1, 5, wspace=0.28,
                          width_ratios=[1, 1, 1, 1, 1.05])

    ax = fig.add_subplot(gs[0, 0])
    scatter_field(ax, xy, np.where(held, np.nan, true[:, gi]), s=1.6)
    scale_bar(ax, xy, section.coord_scale_um, 1000.0)
    ax.set_title("measured, masked", fontsize=6, pad=3)
    # equal aspect shrinks this axes box, so the usual offset lands on the title
    panel_label(ax, "a", dx=-0.26, dy=1.18)

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(Z0[ch], cmap=LATENT, origin="lower", rasterized=True)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("encoded $\\mathbf{z}_0$", fontsize=6, pad=3)
    panel_label(ax, "b", dx=-0.10, dy=1.13)

    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(ZT[ch], cmap=LATENT, origin="lower", rasterized=True)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("evolved $\\mathbf{z}_T$", fontsize=6, pad=3)
    panel_label(ax, "c", dx=-0.10, dy=1.13)

    ax = fig.add_subplot(gs[0, 3])
    scatter_field(ax, xy, pred[:, gi], s=1.6)
    ax.set_title("decoded", fontsize=6, pad=3)
    panel_label(ax, "d", dx=-0.26, dy=1.18)

    ax = fig.add_subplot(gs[0, 4])
    ax.scatter(true[held, gi], pred[held, gi], s=1.5, alpha=0.35,
               color=CATEGORICAL[0], linewidths=0, rasterized=True)
    lo = float(min(true[held, gi].min(), pred[held, gi].min()))
    hi = float(max(true[held, gi].max(), pred[held, gi].max()))
    ax.plot([lo, hi], [lo, hi], color=INK_MUTED, lw=0.7, ls="--")
    r = float(np.corrcoef(true[held, gi], pred[held, gi])[0, 1])
    ax.text(0.04, 0.95, f"$r={r:.2f}$\n{section.gene_names[gi]}", transform=ax.transAxes,
            fontsize=5.4, color=INK, va="top", linespacing=1.3)
    ax.set_xlabel("measured", fontsize=5.6); ax.set_ylabel("predicted", fontsize=5.6)
    ax.tick_params(labelsize=5)
    ax.set_title("held-out, one gene", fontsize=6, pad=3)
    panel_label(ax, "e", dx=-0.30, dy=1.13)

    fig.subplots_adjust(left=0.045, right=0.985, top=0.78, bottom=0.20)
    # Arrows go in the gaps between panels, derived from the drawn axes rather
    # than from guessed figure coordinates -- hard-coded positions land on top of
    # the panels as soon as any width ratio changes.
    fig.canvas.draw()
    boxes = [a.get_position() for a in fig.axes]
    for left, right in zip(boxes[:3], boxes[1:4]):
        gap_l, gap_r = left.x1, right.x0
        if gap_r - gap_l < 0.012:
            continue
        y = (left.y0 + left.y1) / 2
        fig.patches.append(plt.matplotlib.patches.FancyArrow(
            gap_l + 0.1 * (gap_r - gap_l), y, 0.8 * (gap_r - gap_l), 0,
            transform=fig.transFigure, width=0.004, head_width=0.022,
            head_length=0.10 * (gap_r - gap_l), color=INK_MUTED,
            length_includes_head=True))
    return fig


def _figure1_schematic() -> plt.Figure:
    """Fallback used when no trained checkpoint is available."""
    set_style()
    fig, ax = plt.subplots(figsize=(WIDTH_FULL, 1.0))
    ax.axis("off")
    ax.text(0.5, 0.5, "encode $\\rightarrow$ evolve under "
                      "$\\partial_t\\mathbf{z} = \\nabla\\!\\cdot\\!"
                      "(\\mathbf{D}_\\theta\\nabla\\mathbf{z}) + "
                      "f_\\theta(\\mathbf{z})$ $\\rightarrow$ decode",
            ha="center", va="center", fontsize=8, color=INK)
    return fig


def figure2_reconstruction(
    section, pred_nmo: np.ndarray, preds_other: Dict[str, np.ndarray],
    visible: np.ndarray, genes: Optional[Sequence[str]] = None, n_genes: int = 2,
) -> plt.Figure:
    """Ground truth vs. predictions for a few well-reconstructed genes."""
    set_style()
    true = section.numpy_expr(denorm=True)
    coords = section.coords.detach().cpu().numpy()
    held = visible == 0

    if genes is None:
        # Show genes spanning the performance range, not only the best ones:
        # a cherry-picked top-k panel would misrepresent typical behavior.
        r = pearson_per_gene(pred_nmo[held], true[held])
        order = np.argsort(-np.nan_to_num(r, nan=-1))
        valid = order[np.isfinite(r[order])]
        if len(valid) >= n_genes:
            quantiles = np.linspace(0, 0.6, n_genes)  # best -> lower quartile
            gi = np.array([valid[int(q * (len(valid) - 1))] for q in quantiles])
        else:
            gi = valid[:n_genes]
    else:
        name2i = {g: i for i, g in enumerate(section.gene_names)}
        gi = np.array([name2i[g] for g in genes if g in name2i][:n_genes])

    others = list(preds_other.items())
    ncol = 2 + len(others)
    fig, axes = plt.subplots(len(gi), ncol, figsize=(WIDTH_FULL, 0.66 * len(gi)),
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

    titles = ["Measured", "NRDO (ours)"] + [DISPLAY_NAMES.get(n, n)
                                        for n, _ in others]
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
    fig = plt.figure(figsize=(WIDTH_FULL, 2.35))
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

    fig, axes = plt.subplots(1, len(panels), figsize=(WIDTH_FULL, 1.90), squeeze=False)
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
    fig = plt.figure(figsize=(WIDTH_FULL, 1.85))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.95], hspace=0.62, wspace=0.62)

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
    ax3 = fig.add_subplot(gs[1, 1:4])
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
    ax3.set_xlabel("wavenumber $|k|$ (normalized units)")
    ax3.set_ylabel("max Re $\\lambda(k)$")
    ax3.set_title("Dispersion relation of the learned operator", fontsize=7, pad=4)
    panel_label(ax3, "c", dx=-0.13, dy=1.30)

    return fig


# --------------------------------------------------------------------------- #
# Figure 6 -- ablations
# --------------------------------------------------------------------------- #


ABLATION_LABELS = {
    "full": "Full model",
    "no_pde": "$-$ PDE constraints",
    "no_diffusion": "$-$ diffusion term",
    "no_reaction": "$-$ reaction term",
    "no_bio_reg": "$-$ biological regularizers",
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

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_FULL, 2.05),
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


# --------------------------------------------------------------------------- #
# Combined results panel (benchmark + ablation + control), for the main text
# --------------------------------------------------------------------------- #


def figure_results_panel(bench: List[Dict], abl: List[Dict], matched: Dict[str, float]) -> plt.Figure:
    """Three-panel summary: benchmark, ablation deltas, matched-budget control.

    Combining these into one float keeps three results on one page in a
    9-page workshop format, where separate figures would not fit.
    """
    set_style()
    import pandas as pd

    fig, axes = plt.subplots(1, 3, figsize=(WIDTH_FULL, 2.10),
                             gridspec_kw={"width_ratios": [1.15, 1.15, 0.8]})

    # (a) benchmark Pearson
    b = pd.DataFrame(bench).groupby("model")["pearson_mean"].agg(["mean", "std"]).reset_index()
    b = b.sort_values("mean")
    bar_with_error(axes[0], [DISPLAY_NAMES.get(m, m).replace(" (ours)", "*") for m in b["model"]],
                   list(b["mean"]), list(np.nan_to_num(b["std"])),
                   [MODEL_COLORS.get(m, INK_MUTED) for m in b["model"]],
                   horizontal=True, value_fmt="{:.3f}", label_values=False)
    axes[0].set_xlabel("Pearson $r$, held-out blocks")
    axes[0].set_title("Masked reconstruction", fontsize=7.5, pad=4)
    panel_label(axes[0], "a", dx=-0.72, dy=1.14)

    # (b) ablation deltas
    a = pd.DataFrame(abl).groupby("variant")["pearson_mean"].agg(["mean", "std"]).reset_index()
    full = float(a.loc[a["variant"] == "full", "mean"].iloc[0])
    keep = ["no_dynamics", "no_diffusion", "no_reaction", "no_pde", "no_bio_reg",
            "isotropic_diffusion", "discrete_gnn"]
    a = a[a["variant"].isin(keep)].copy()
    a["delta"] = a["mean"] - full
    a = a.sort_values("delta")
    short = {"discrete_gnn": "discrete GNN", "no_bio_reg": "$-$ bio. regularizers",
             "no_pde": "$-$ PDE terms", "isotropic_diffusion": "isotropic $D$",
             "no_dynamics": "$-$ dynamics", "no_diffusion": "$-$ diffusion",
             "no_reaction": "$-$ reaction"}
    bar_with_error(axes[1], [short.get(v, ABLATION_LABELS.get(v, v)) for v in a["variant"]],
                   list(a["delta"]), list(np.nan_to_num(a["std"])),
                   [CATEGORICAL[1] if d < 0 else CATEGORICAL[2] for d in a["delta"]],
                   horizontal=True, value_fmt="{:+.3f}")
    axes[1].axvline(0, color=INK_SECONDARY, linewidth=0.7)
    axes[1].set_xlabel("$\\Delta r$ vs. full (reduced budget)")
    axes[1].set_title("Ablations", fontsize=7.5, pad=4)
    panel_label(axes[1], "b", dx=-0.78, dy=1.14)

    # (c) matched-budget control
    names = list(matched.keys())
    vals = [matched[k] for k in names]
    cols = [MODEL_COLORS["nmo"] if "full" in k else CATEGORICAL[1] for k in names]
    axes[2].bar(range(len(names)), vals, color=cols, width=0.62)
    for i, v in enumerate(vals):
        axes[2].text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=6, color=INK_SECONDARY)
    axes[2].set_xticks(range(len(names)))
    axes[2].set_xticklabels(["full", "$-$dyn", "$-$rxn"][: len(names)], fontsize=6.5)
    axes[2].set_ylim(min(vals) * 0.93, max(vals) * 1.06)
    axes[2].set_ylabel("Pearson $r$")
    axes[2].set_title("Matched budget", fontsize=7.5, pad=4)
    axes[2].grid(axis="x", visible=False)
    panel_label(axes[2], "c", dx=-0.42, dy=1.14)

    fig.subplots_adjust(wspace=1.55)
    return fig


def figure_datasets(processed_dir="data/processed", keys=None) -> plt.Figure:
    """Tissue-section overview: one panel per dataset, colored by total counts.

    Standard orientation figure for spatial-omics papers -- it shows at a glance
    that the sections differ in geometry, sampling density and physical extent,
    which is what the transfer experiments are actually varying.
    """
    import anndata as ad
    set_style()
    keys = keys or ["visium_mouse_brain", "visium_human_breast", "xenium_mouse_brain",
                    "merfish_allen_40", "mosta_embryo_E9.5"]
    titles = {"visium_mouse_brain": "Visium\nmouse brain",
              "visium_human_breast": "Visium\nhuman breast",
              "xenium_mouse_brain": "Xenium\nmouse brain",
              "merfish_allen_40": "MERFISH\nmouse brain",
              "mosta_embryo_E9.5": "Stereo-seq\nembryo E9.5"}
    avail = [k for k in keys if (Path(processed_dir) / f"{k}.h5ad").exists()]
    fig, axes = plt.subplots(1, len(avail), figsize=(WIDTH_FULL, 1.35))
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, avail):
        a = ad.read_h5ad(Path(processed_dir) / f"{k}.h5ad")
        xy = np.asarray(a.obsm["spatial_um"] if "spatial_um" in a.obsm else a.obsm["spatial"])
        tot = np.asarray(a.X.sum(1)).ravel()
        n = a.n_obs
        s = float(np.clip(2200.0 / max(n, 1), 0.06, 3.0))
        scatter_field(ax, xy, tot, s=s)
        # equal aspect shrinks each axes box by a different amount; anchoring
        # north keeps the panel titles on a common baseline
        ax.set_anchor("N")
        ext = (np.ptp(xy[:, 0]) / 1000.0, np.ptp(xy[:, 1]) / 1000.0)
        ax.set_title(titles.get(k, k), fontsize=6.2, color=INK, pad=3, linespacing=1.2)
        ax.text(0.5, -0.04, f"{n:,} loc · {a.n_vars} genes\n{ext[0]:.1f}$\\times${ext[1]:.1f} mm",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=5.2, color=INK_SECONDARY, linespacing=1.25)
        del a
    fig.subplots_adjust(wspace=0.06, top=0.80, bottom=0.14)
    return fig


def figure_numerics(stability: List[Dict], cost: List[Dict],
                    downstream: Optional[List[Dict]] = None) -> plt.Figure:
    """Stability envelope, cost at matched accuracy, and downstream effect."""
    set_style()
    import pandas as pd
    S = pd.DataFrame(stability); K = pd.DataFrame(cost)
    order = ["strang-spectral", "euler-spectral", "strang-fd5", "euler-fd5"]
    cols = {o: CATEGORICAL[i] for i, o in enumerate(order)}
    n = 3 if downstream else 2
    fig, axes = plt.subplots(1, n, figsize=(WIDTH_FULL, 1.95),
                             gridspec_kw={"width_ratios": [1.25, 1.0] + ([0.95] if downstream else [])})

    # (a) largest stable dt per scheme, against the CFL bound
    ax = axes[0]
    for i, o in enumerate(order):
        s = S[S["scheme"] == o]
        top = s[s["stable"]]["dt"].max() if s["stable"].any() else np.nan
        ax.barh(i, top, color=cols[o], height=0.62)
        ax.text(top * 1.15, i, f"{top:.3g}", va="center", fontsize=6, color=INK_SECONDARY)
    cfl = float(S["cfl_limit"].iloc[0])
    ax.axvline(cfl, color=INK_SECONDARY, ls="--", lw=0.8)
    ax.text(cfl, len(order) - 0.35, " CFL bound", fontsize=5.6, color=INK_SECONDARY, va="top")
    ax.set_xscale("log"); ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=6.2)
    ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.set_xlabel("largest stable $\\Delta t$"); ax.set_title("Stability envelope", fontsize=7.5, pad=4)
    panel_label(ax, "a", dx=-0.52, dy=1.15)

    # (b) steps needed to reach a fixed accuracy
    ax = axes[1]
    base = K[K["scheme"] == "strang-spectral"]["n_steps"].iloc[0]
    vals = [K[K["scheme"] == o]["n_steps"].iloc[0] for o in order]
    ax.bar(range(len(order)), vals, color=[cols[o] for o in order], width=0.62)
    for i, v in enumerate(vals):
        if v: ax.text(i, v, f"{v/base:.0f}$\\times$", ha="center", va="bottom",
                      fontsize=6, color=INK_SECONDARY)
    ax.set_yscale("log"); ax.set_xticks(range(len(order)))
    ax.set_xticklabels(["Str-sp", "Eul-sp", "Str-fd", "Eul-fd"], fontsize=6)
    ax.set_ylabel("steps to rel. err $<10^{-2}$")
    ax.set_title("Cost at matched accuracy", fontsize=7.5, pad=4)
    ax.grid(axis="x", visible=False)
    panel_label(ax, "b", dx=-0.30, dy=1.15)

    # (c) downstream held-out reconstruction
    if downstream:
        D = pd.DataFrame([d for d in downstream if not d.get("diverged")])
        ax = axes[2]
        if not D.empty:
            g = D.groupby("scheme")["pearson_mean"].agg(["mean", "std"]).reindex(order).dropna()
            ax.bar(range(len(g)), g["mean"], yerr=np.nan_to_num(g["std"]),
                   color=[cols[o] for o in g.index], width=0.62,
                   error_kw=dict(ecolor=INK_SECONDARY, elinewidth=0.8, capsize=2))
            for i, v in enumerate(g["mean"]):
                ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=6, color=INK_SECONDARY)
            ax.set_xticks(range(len(g)))
            ax.set_xticklabels([o.split("-")[0][:3] + "-" + o.split("-")[1][:2] for o in g.index],
                               fontsize=6)
            ax.set_ylim(max(0, g["mean"].min() * 0.9), g["mean"].max() * 1.10)
        ax.set_ylabel("Pearson $r$")
        ax.set_title("Held-out accuracy", fontsize=7.5, pad=4)
        ax.grid(axis="x", visible=False)
        panel_label(ax, "c", dx=-0.36, dy=1.15)
    fig.subplots_adjust(wspace=0.85)
    return fig


def figure_multisection(records: List[Dict], reference: str = "nmo") -> plt.Figure:
    """Per-section paired view of the benchmark, plus effect sizes."""
    set_style()
    import pandas as pd
    from ..evaluation.statistics import paired_comparison

    df = pd.DataFrame([r for r in records if "pearson_mean" in r and not r.get("failed")])
    per = df.groupby(["section", "model"])["pearson_mean"].mean().unstack("model")
    others = [c for c in per.columns if c != reference]
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH_FULL, 1.35),
                             gridspec_kw={"width_ratios": [1.30, 1.0, 0.80]})

    # (a) NRDO vs each baseline, one point per section
    ax = axes[0]
    for i, o in enumerate(others):
        s = per[[reference, o]].dropna()
        ax.scatter(s[o], s[reference], s=11, alpha=0.8,
                   color=MODEL_COLORS.get(o, CATEGORICAL[i % 7]),
                   label=DISPLAY_NAMES.get(o, o), linewidths=0)
    lim = [float(np.nanmin(per.values)) * 0.95, float(np.nanmax(per.values)) * 1.05]
    ax.plot(lim, lim, ls="--", lw=0.8, color=INK_SECONDARY)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xlabel("baseline Pearson $r$"); ax.set_ylabel("NRDO Pearson $r$")
    ax.set_title(f"{len(per)} sections", fontsize=7.5, pad=4)
    # no legend: colors match the model ordering in panels (b) and (c)
    panel_label(ax, "a", dx=-0.24, dy=1.16)

    # (b) paired difference with bootstrap CI
    ax = axes[1]
    res = paired_comparison(df, reference, "pearson_mean")
    res = sorted(res, key=lambda r: r.mean_diff)
    y = np.arange(len(res))
    ax.errorbar([r.mean_diff for r in res], y,
                xerr=[[r.mean_diff - r.ci_lo for r in res], [r.ci_hi - r.mean_diff for r in res]],
                fmt="o", ms=4, color=CATEGORICAL[0], ecolor=INK_SECONDARY,
                elinewidth=1.0, capsize=2, lw=0)
    ax.axvline(0, color=INK_SECONDARY, lw=0.8, ls="--")
    ax.set_yticks(y); ax.set_yticklabels(
        [DISPLAY_NAMES.get(r.other, r.other).replace(" (non-spatial)","").replace(" (SIREN)","").replace(" (GCN)","") for r in res], fontsize=6)
    ax.set_xlabel("$\\Delta r$ vs. NRDO (95% CI)")
    ax.set_title("Paired difference", fontsize=7.5, pad=4)
    ax.grid(axis="y", visible=False)
    panel_label(ax, "b", dx=-0.62, dy=1.14)

    # (c) sections won
    ax = axes[2]
    ax.barh(y, [r.n_reference_wins / r.n_sections * 100 for r in res],
            color=[CATEGORICAL[2] if r.n_reference_wins / r.n_sections > 0.5 else CATEGORICAL[1]
                   for r in res], height=0.62)
    ax.axvline(50, color=INK_SECONDARY, lw=0.8, ls="--")
    for i, r in enumerate(res):
        ax.text(r.n_reference_wins / r.n_sections * 100 + 2, i,
                f"{r.n_reference_wins}/{r.n_sections}", va="center", fontsize=5.6,
                color=INK_SECONDARY)
    ax.set_yticks(y); ax.set_yticklabels([])
    ax.set_xlim(0, 118); ax.set_xlabel("% sections NRDO wins")
    ax.set_title("Win rate", fontsize=7.5, pad=4)
    ax.grid(axis="y", visible=False)
    panel_label(ax, "c", dx=-0.10, dy=1.14)
    fig.subplots_adjust(wspace=1.15)
    return fig


def figure_biology(records: List[Dict]) -> plt.Figure:
    """Biological preservation across models."""
    set_style()
    import pandas as pd
    df = pd.DataFrame([r for r in records if "ari_predicted" in r])
    panels = [("ari_retention", "ARI retention $\\uparrow$"),
              ("marker_auroc_predicted", "marker AUROC $\\uparrow$"),
              ("neighborhood_preservation", "$k$-NN preservation $\\uparrow$"),
              ("gearys_c_abs_error", "$|\\Delta$ Geary's $C|$ $\\downarrow$")]
    panels = [(k, l) for k, l in panels if k in df.columns]
    fig, axes = plt.subplots(1, len(panels), figsize=(WIDTH_FULL, 1.95))
    for ax, (key, label) in zip(np.atleast_1d(axes), panels):
        g = df.groupby("model")[key].agg(["mean", "std"])
        asc = "error" in key
        g = g.sort_values("mean", ascending=asc)
        bar_with_error(ax, [DISPLAY_NAMES.get(m, m) for m in g.index],
                       list(g["mean"]), list(np.nan_to_num(g["std"])),
                       [MODEL_COLORS.get(m, INK_MUTED) for m in g.index],
                       horizontal=True, value_fmt="{:.3f}", label_values=False)
        ax.set_xlabel(label, fontsize=6.5)
        ax.tick_params(labelsize=5.6)
    for ax in np.atleast_1d(axes)[1:]:
        ax.set_yticklabels([])
    fig.subplots_adjust(wspace=0.15)
    return fig


def figure_robustness(records: List[Dict]) -> plt.Figure:
    """Degradation curves under each corruption axis."""
    set_style()
    import pandas as pd
    df = pd.DataFrame([r for r in records if "pearson_mean" in r and not r.get("failed")])
    axes_list = [a for a in ["noise", "dropout", "density", "knn"] if a in set(df["axis"])]
    fig, axs = plt.subplots(1, len(axes_list), figsize=(WIDTH_FULL, 1.85), squeeze=False)
    labels = {"noise": "noise $\\sigma$", "dropout": "fraction dropped",
              "density": "fraction retained", "knn": "$k$ neighbors"}
    for ax, axis in zip(axs[0], axes_list):
        d = df[df["axis"] == axis]
        for m in sorted(d["model"].unique()):
            g = d[d["model"] == m].groupby("level")["pearson_mean"].agg(["mean", "std"])
            ax.errorbar(g.index, g["mean"], yerr=np.nan_to_num(g["std"]),
                        marker="o", ms=3.2, lw=1.3, capsize=1.8,
                        color=MODEL_COLORS.get(m, INK_MUTED),
                        label=DISPLAY_NAMES.get(m, m), ecolor=INK_SECONDARY, elinewidth=0.7)
        ax.set_xlabel(labels.get(axis, axis), fontsize=6.5)
        if axis == "knn":
            ax.set_xscale("log", base=2)
        ax.tick_params(labelsize=5.6)
    axs[0][0].set_ylabel("Pearson $r$", fontsize=6.5)
    axs[0][-1].legend(fontsize=5.0, loc="best", handletextpad=0.3, borderpad=0.2)
    fig.subplots_adjust(wspace=0.30)
    return fig


def figure_spectral(records: List[Dict]) -> plt.Figure:
    """Accuracy against spatial fidelity as the spectral weight is swept.

    Plotted as a trade-off curve rather than two line charts, because the
    question is not how either metric behaves alone but what one costs the
    other. Up and to the left is better: high correlation, low Moran's I error.
    """
    import pandas as pd
    set_style()
    df = pd.DataFrame([r for r in records if not r.get("failed")])
    if df.empty:
        return plt.figure()
    df["mode"] = (df["mode"].fillna("full") if "mode" in df.columns
                  else "full")
    g = (df.groupby(["mode", "spectral_weight"])
           [["pearson_mean", "morans_i_abs_error", "morans_i_pred", "morans_i_true"]]
           .mean().reset_index())

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_FULL, 2.0))
    styles = {"full": ("o-", "absolute spectrum, all bands"),
              "shape": ("s-", "normalized spectrum, high band")}

    ax = axes[0]
    for mode, sub in g.groupby("mode"):
        sub = sub.sort_values("spectral_weight")
        mk, lab = styles.get(mode, ("^-", mode))
        ax.plot(sub["morans_i_abs_error"], sub["pearson_mean"], mk, ms=3.4, lw=1.0,
                label=lab, alpha=0.9)
        for _, r in sub.iterrows():
            if r["spectral_weight"] > 0:
                ax.annotate(f"{r['spectral_weight']:g}",
                            (r["morans_i_abs_error"], r["pearson_mean"]),
                            fontsize=5.0, color=INK_SECONDARY,
                            xytext=(2.5, 2.5), textcoords="offset points")
    base = g[g["spectral_weight"] == 0]
    if len(base):
        ax.scatter(base["morans_i_abs_error"], base["pearson_mean"], s=26,
                   facecolors="none", edgecolors=INK, lw=0.9, zorder=5)
        ax.annotate("$\\lambda=0$", (float(base["morans_i_abs_error"].iloc[0]),
                                    float(base["pearson_mean"].iloc[0])),
                    fontsize=5.2, color=INK, xytext=(3, -7),
                    textcoords="offset points")
    ax.set_xlabel("$|\\Delta I|$  (lower is better)")
    ax.set_ylabel("Pearson $r$")
    ax.set_title("Accuracy against spatial fidelity", fontsize=7, pad=4)
    ax.legend(fontsize=5.0, frameon=False, loc="lower left")
    panel_label(ax, "a", dx=-0.24, dy=1.10)

    ax = axes[1]
    it = float(g["morans_i_true"].mean())
    for mode, sub in g.groupby("mode"):
        sub = sub.sort_values("spectral_weight")
        mk, lab = styles.get(mode, ("^-", mode))
        ax.plot(np.maximum(sub["spectral_weight"], 3e-4), sub["morans_i_pred"],
                mk, ms=3.4, lw=1.0, label=lab, alpha=0.9)
    ax.axhline(it, color=CATEGORICAL[3], lw=0.9, ls="--")
    ax.annotate(f"measured $I = {it:.2f}$", (3e-4, it), fontsize=5.0, color=CATEGORICAL[3],
                xytext=(1, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("spectral weight $\\lambda$  (0 plotted at left)")
    ax.set_ylabel("$I_{\\mathrm{pred}}$")
    ax.set_title("Predicted autocorrelation never reaches the data", fontsize=7, pad=4)
    panel_label(ax, "b", dx=-0.24, dy=1.10)

    fig.subplots_adjust(wspace=0.42, bottom=0.24, top=0.84)
    return fig


def figure_evidence(converged, robustness, spectral, biology) -> plt.Figure:
    """The four supporting results in one panel.

    Each of these previously lived either in prose or in its own appendix
    figure. Together they answer the four questions a reader of this benchmark
    actually asks: is the ordering an artifact of the training budget, does the
    continuous formulation buy anything a graph cannot, can the over-smoothing be
    trained away, and does the reconstruction still support downstream analysis.
    """
    import pandas as pd
    set_style()
    fig, axes = plt.subplots(1, 4, figsize=(WIDTH_FULL, 1.18))

    # (a) budget vs convergence -- the ordering inverts
    ax = axes[0]
    c = pd.DataFrame([r for r in converged if not r.get("failed")])
    if not c.empty:
        g = c.groupby("model")["pearson_mean"].agg(["mean", "std"])
        budget = {"nmo": 0.218, "stagate": 0.225, "gnn": 0.210, "autoencoder": 0.213}
        order = [m for m in ["nmo", "stagate", "gnn", "autoencoder"] if m in g.index]
        x = np.arange(len(order))
        ax.plot(x, [budget[m] for m in order], "o--", ms=3.2, lw=0.9,
                color=INK_MUTED, label="200 epochs")
        ax.errorbar(x, [g.loc[m, "mean"] for m in order],
                    yerr=[g.loc[m, "std"] for m in order], fmt="o-", ms=3.2, lw=1.0,
                    color=CATEGORICAL[0], capsize=1.5, label="converged")
        ax.set_xticks(x)
        ax.set_xticklabels(["NRDO", "STAGATE", "GCN", "AE"][:len(order)],
                           fontsize=5, rotation=30, ha="right")
        ax.set_ylabel("Pearson $r$", fontsize=6)
        ax.legend(fontsize=5.0, frameon=False, loc="lower right")
    ax.set_title("budget flips the order", fontsize=6.2, pad=3)
    ax.tick_params(labelsize=5)
    panel_label(ax, "a", dx=-0.30, dy=1.14)

    # (b) robustness to sampling density -- where a field beats a lattice
    ax = axes[1]
    rb = pd.DataFrame([r for r in robustness if not r.get("failed")])
    if not rb.empty and "density" in set(rb["axis"]):
        d = rb[rb["axis"] == "density"]
        piv = d.pivot_table(index="level", columns="model", values="pearson_mean")
        for m, lab in [("nmo", "NRDO"), ("stagate", "STAGATE"), ("gnn", "GCN")]:
            if m in piv.columns:
                sub = piv[m].dropna().sort_index()
                ax.plot(sub.index, sub.values / sub.values[-1], "o-", ms=3.0, lw=1.0,
                        label=lab, color=MODEL_COLORS.get(m, CATEGORICAL[0]))
        ax.set_xscale("log")
        ax.set_xticks([0.125, 0.25, 0.5, 1.0])
        ax.set_xticklabels(["1/8", "1/4", "1/2", "1"], fontsize=5)
        ax.minorticks_off()
        ax.set_xlabel("fraction of locations", fontsize=6)
        ax.set_ylabel("retained accuracy", fontsize=6)
        ax.legend(fontsize=5.0, frameon=False, loc="lower right")
    ax.set_title("density robustness", fontsize=6.2, pad=3)
    ax.tick_params(labelsize=5)
    panel_label(ax, "b", dx=-0.32, dy=1.14)

    # (c) spectral matching -- a bound, not a fix
    ax = axes[2]
    sp = pd.DataFrame([r for r in spectral if not r.get("failed")])
    if not sp.empty:
        sp["mode"] = sp["mode"].fillna("full") if "mode" in sp.columns else "full"
        g = sp.groupby(["mode", "spectral_weight"])[
            ["pearson_mean", "morans_i_abs_error", "morans_i_true"]].mean().reset_index()
        for mode, mk, lab in [("full", "o-", "absolute"), ("shape", "s-", "normalized")]:
            sub = g[g["mode"] == mode].sort_values("spectral_weight")
            if len(sub):
                ax.plot(sub["morans_i_abs_error"], sub["pearson_mean"], mk, ms=3.0,
                        lw=0.9, label=lab)
        ax.set_xlabel("$|\\Delta I|$", fontsize=6)
        ax.set_ylabel("Pearson $r$", fontsize=6)
        ax.legend(fontsize=5.0, frameon=False, loc="lower left")
    ax.set_title("accuracy vs fidelity", fontsize=6.2, pad=3)
    ax.tick_params(labelsize=5)
    panel_label(ax, "c", dx=-0.32, dy=1.14)

    # (d) biology -- separates where accuracy does not
    ax = axes[3]
    b = pd.DataFrame([r for r in biology if "ari_retention" in r])
    if not b.empty:
        piv = b.pivot_table(index="section", columns="model", values="ari_retention")
        pairs = [(m, lab) for m, lab in [("stagate", "STAGATE"), ("gnn", "GCN"),
                                         ("gp_multiscale", "GP-mb"),
                                         ("neural_field", "SIREN")]
                 if m in piv.columns]
        for i, (m, lab) in enumerate(pairs):
            sub = piv[["nmo", m]].dropna()
            ax.scatter(np.full(len(sub), i) + np.random.default_rng(0).normal(0, .05, len(sub)),
                       sub["nmo"] - sub[m], s=7, alpha=0.75, linewidths=0,
                       color=MODEL_COLORS.get(m, CATEGORICAL[i % 6]))
        ax.axhline(0, color=INK_MUTED, lw=0.7, ls="--")
        ax.set_xticks(range(len(pairs)))
        ax.set_xticklabels([l for _, l in pairs], fontsize=5, rotation=30, ha="right")
        ax.set_ylabel("$\\Delta$ ARI retention", fontsize=6)
    ax.set_title("domain structure", fontsize=6.2, pad=3)
    ax.tick_params(labelsize=5)
    panel_label(ax, "d", dx=-0.34, dy=1.14)

    fig.subplots_adjust(left=0.075, right=0.99, top=0.79, bottom=0.30, wspace=0.62)
    return fig


def figure_per_gene(per_gene: Dict) -> plt.Figure:
    """Where the advantage lives: per-gene margin against spatial structure.

    The continuity argument predicts the advantage should vanish on genes with
    no spatial structure, and that is a sharper claim than any aggregate number
    because it says where the model should *stop* winning. The left panel shows
    every gene; the right shows the quartile means with the across-seed range,
    so the reader can see the least-structured quartile straddling zero rather
    than take it on assertion.
    """
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_FULL, 1.7),
                             gridspec_kw=dict(width_ratios=[1.25, 1.0]))
    seeds = per_gene.get("seeds", [])
    if not seeds:
        for ax in axes:
            ax.axis("off")
        return fig

    ax = axes[0]
    # Every gene, not the quartile means: the right panel already shows those,
    # and plotting the same four numbers twice would make the pair decorative.
    gx = np.asarray(per_gene.get("gene_structure", []), dtype=float)
    ga = np.array([np.nan if v is None else v
                   for v in per_gene.get("gene_advantage", [])], dtype=float)
    m = np.isfinite(gx) & np.isfinite(ga)
    ax.scatter(gx[m], ga[m], s=1.6, alpha=0.28, linewidths=0,
               color=MODEL_COLORS.get("nmo", "#1f77b4"), rasterized=True)
    # running mean over structure, so the trend is visible through the cloud
    if m.sum() > 40:
        o = np.argsort(gx[m])
        xs, ys = gx[m][o], ga[m][o]
        w = max(25, len(xs) // 25)
        k = np.ones(w) / w
        ax.plot(xs[w - 1:], np.convolve(ys, k, mode="valid"),
                color="0.1", lw=1.0, label=f"running mean ({w} genes)")
        ax.legend(fontsize=5.0, frameon=False, loc="upper left")
    ax.axhline(0, color="0.55", lw=0.6, ls="--", zorder=0)
    ax.set_xlabel("Moran's $I$ of measured expression")
    ax.set_ylabel("per-gene $\\Delta r$ vs best baseline")
    rl, rh = per_gene.get("corr_lo"), per_gene.get("corr_hi")
    ax.set_title(f"{int(m.sum())} genes, $r$ = {rl:+.2f} to {rh:+.2f} across seeds"
                 if rl is not None else f"{int(m.sum())} genes", fontsize=6.5)

    ax = axes[1]
    nq = len(seeds[0]["bins"])
    means = [np.mean([s["bins"][i]["delta"] for s in seeds]) for i in range(nq)]
    lo = [min(s["bins"][i]["delta"] for s in seeds) for i in range(nq)]
    hi = [max(s["bins"][i]["delta"] for s in seeds) for i in range(nq)]
    x = np.arange(nq)
    ax.bar(x, means, color=["0.75", "0.65", "0.5", "0.25"], width=0.68)
    ax.errorbar(x, means, yerr=[np.array(means) - np.array(lo),
                                np.array(hi) - np.array(means)],
                fmt="none", ecolor="0.15", elinewidth=0.7, capsize=1.6)
    ax.axhline(0, color="0.15", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(["Q1\nleast", "Q2", "Q3", "Q4\nmost"], fontsize=5.5)
    ax.set_xlabel("spatial-structure quartile")
    ax.set_ylabel("$\\Delta r$")
    ax.set_title("no structure, no advantage", fontsize=6.5)
    fig.tight_layout(pad=0.35)
    return fig
