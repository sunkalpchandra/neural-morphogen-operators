"""Generate the paper's LaTeX tables directly from results JSON.

Nothing in ``paper/tables/`` is hand-written: every number is produced here from
the run artifacts, so the paper cannot drift from the experiments.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..models.baselines import DISPLAY_NAMES

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

HIGHER_IS_BETTER = {
    "pearson_mean": True, "pearson_median": True, "spearman_mean": True,
    "ssim_mean": True, "morans_i_corr": True, "pearson_per_location": True,
    "rmse": False, "mae": False, "morans_i_abs_error": False,
}

METRIC_LABELS = {
    "pearson_mean": r"Pearson $r$ $\uparrow$",
    "pearson_median": r"median $r$ $\uparrow$",
    "spearman_mean": r"Spearman $\rho$ $\uparrow$",
    "rmse": r"RMSE $\downarrow$",
    "mae": r"MAE $\downarrow$",
    "ssim_mean": r"SSIM $\uparrow$",
    "morans_i_abs_error": r"$|\Delta I|$ $\downarrow$",
    "morans_i_corr": r"$r(I)$ $\uparrow$",
    "pearson_per_location": r"$r_{\mathrm{loc}}$ $\uparrow$",
}


def load_json_glob(pattern: str, root: str | Path = ".") -> List[Dict]:
    """Load and concatenate every JSON list matching a glob."""
    out: List[Dict] = []
    for p in sorted(Path(root).glob(pattern)):
        try:
            data = json.loads(p.read_text())
            if isinstance(data, list):
                out.extend(data)
        except Exception:
            continue
    return out


def dedupe(records: List[Dict], keys: Sequence[str]) -> List[Dict]:
    """Drop duplicate records sharing the same key tuple.

    The transfer experiments are sharded one job per model, and each shard
    independently recomputes the shared reference points (the training-mean
    floor). Without this, the floor would be counted once per shard and its
    across-seed standard deviation would be wrong.
    """
    seen, out = set(), []
    for r in records:
        k = tuple(str(r.get(x)) for x in keys)
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def _esc(s: str) -> str:
    return str(s).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def _fmt(mean: float, std: float, prec: int = 3, bold: bool = False) -> str:
    if not np.isfinite(mean):
        return "--"
    s = f"{mean:.{prec}f}"
    if np.isfinite(std) and std > 0:
        s += f"\\,\\tiny{{$\\pm$ {std:.{prec}f}}}"
    return f"\\textbf{{{s}}}" if bold else s


def _wrap(body: str, caption: str, label: str, colspec: str, header: str,
          note: str = "", small: bool = True, position: str = "t") -> str:
    size = "\\small\n" if small else ""
    note_line = f"\\\\[2pt]\n\\multicolumn{{{colspec.count('c') + colspec.count('l') + colspec.count('r')}}}{{p{{0.95\\linewidth}}}}{{\\scriptsize {note}}}\n" if note else ""
    return (
        f"\\begin{{table}}[{position}]\n\\centering\n{size}"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
        f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{header}\\\\\n\\midrule\n"
        f"{body}\\bottomrule\n{note_line}\\end{{tabular}}\n\\end{{table}}\n"
    )


# --------------------------------------------------------------------------- #
# Table: dataset inventory
# --------------------------------------------------------------------------- #


def is_derived(key: str) -> bool:
    """True for objects built *from* a source dataset for a specific experiment
    (e.g. the Visium section rebuilt with a target gene panel force-included).
    These are not separate datasets and must not be counted or listed as such."""
    return "__panel_" in key


def table_datasets(summary_path: str | Path, out: Path) -> str:
    s = {k: v for k, v in json.loads(Path(summary_path).read_text()).items()
         if not is_derived(k)}
    rows = []
    for key, r in sorted(s.items()):
        extent = r.get("extent_um")
        ext = f"{extent[0]}$\\times${extent[1]}" if extent else "--"
        rows.append(
            f"{_esc(key)} & {_esc(r.get('technology','')[:22])} & {_esc(r.get('organism','').split()[0])} & "
            f"{r['n_obs']:,} & {r['n_vars']:,} & {ext} & {_esc(r.get('resolution',''))} \\\\\n"
        )
    tex = _wrap(
        "".join(rows),
        "Processed spatial transcriptomics datasets. Locations and genes are counts "
        "after QC and gene selection; extent is the physical bounding box of the section.",
        "tab:datasets", "lllrrll",
        "dataset & technology & organism & locations & genes & extent ($\\mu$m) & resolution",
    )
    out.write_text(tex)
    return tex


# --------------------------------------------------------------------------- #
# Table 1: main benchmark
# --------------------------------------------------------------------------- #


def table_benchmark(
    records: List[Dict], out: Path,
    metrics: Sequence[str] = ("pearson_mean", "spearman_mean", "rmse", "ssim_mean",
                              "morans_i_abs_error"),
    section_label: str = "",
) -> str:
    df = pd.DataFrame(records)
    if df.empty:
        return ""
    metrics = [m for m in metrics if m in df.columns]
    agg = df.groupby("model")[list(metrics) + ["n_params"]].agg(["mean", "std"])

    # order rows: baselines by performance, ours last for emphasis
    order = [m for m in agg.index if m != "nmo"]
    order = sorted(order, key=lambda m: -agg.loc[m, ("pearson_mean", "mean")])
    if "nmo" in agg.index:
        order.append("nmo")

    best = {}
    for m in metrics:
        col = agg[(m, "mean")]
        best[m] = col.idxmax() if HIGHER_IS_BETTER.get(m, True) else col.idxmin()

    rows = []
    for model in order:
        cells = [
            _fmt(agg.loc[model, (m, "mean")], agg.loc[model, (m, "std")],
                 bold=(best[m] == model))
            for m in metrics
        ]
        npar = agg.loc[model, ("n_params", "mean")]
        name = DISPLAY_NAMES.get(model, model)
        if model == "nmo":
            rows.append("\\midrule\n")
        rows.append(f"{_esc(name)} & {npar/1e6:.2f}M & " + " & ".join(cells) + " \\\\\n")

    n_seeds = int(df.groupby("model").size().max())
    header = "model & params & " + " & ".join(METRIC_LABELS.get(m, m) for m in metrics)
    tex = _wrap(
        "".join(rows),
        f"Masked spatial reconstruction on {_esc(section_label)}. Contiguous tissue blocks "
        f"are hidden from every model; metrics are computed on held-out locations only. "
        f"Mean $\\pm$ s.d. over {n_seeds} seeds. All models share the identical training "
        f"loop, masks and evaluation code. $|\\Delta I|$ is the mean absolute error in "
        f"Moran's $I$ between predicted and measured expression maps: it penalises "
        f"over-smoothing that correlation alone does not detect.",
        "tab:benchmark", "l" + "r" * (len(metrics) + 1), header,
    )
    out.write_text(tex)
    return tex


# --------------------------------------------------------------------------- #
# Table 2: transfer
# --------------------------------------------------------------------------- #


SETTING_LABELS = {
    "source_in_domain": "source (in-domain)",
    "zero_shot": "target (zero-shot)",
    "decoder_finetune": "target (decoder-only fine-tune)",
    "oracle": "target (in-domain oracle)",
    "floor": "target (training-mean floor)",
}


def table_transfer(records: List[Dict], out: Path, caption: str, label: str) -> str:
    df = pd.DataFrame(records)
    if df.empty:
        return ""
    metrics = [m for m in ("pearson_mean", "rmse", "ssim_mean") if m in df.columns]
    agg = df.groupby(["setting", "model"])[metrics].agg(["mean", "std"])

    rows = []
    for setting in ["source_in_domain", "zero_shot", "decoder_finetune", "oracle", "floor"]:
        if setting not in agg.index.get_level_values(0):
            continue
        sub = agg.loc[setting]
        sub = sub.sort_values(("pearson_mean", "mean"), ascending=False)
        rows.append(f"\\multicolumn{{{len(metrics)+2}}}{{l}}{{\\emph{{{SETTING_LABELS[setting]}}}}} \\\\\n")
        for model in sub.index:
            cells = [_fmt(sub.loc[model, (m, "mean")], sub.loc[model, (m, "std")]) for m in metrics]
            nm = DISPLAY_NAMES.get(model, model)
            rows.append(f"\\quad {_esc(nm)} & & " + " & ".join(cells) + " \\\\\n")
        rows.append("\\addlinespace\n")

    ns = int(df["n_shared_genes"].iloc[0]) if "n_shared_genes" in df.columns else 0
    header = "model & & " + " & ".join(METRIC_LABELS.get(m, m) for m in metrics)
    tex = _wrap(
        "".join(rows),
        caption + f" Shared gene vocabulary: {ns} genes. Mean $\\pm$ s.d. over seeds.",
        label, "ll" + "r" * len(metrics), header,
    )
    out.write_text(tex)
    return tex


# --------------------------------------------------------------------------- #
# Table 3: ablations
# --------------------------------------------------------------------------- #


ABLATION_ORDER = [
    "full", "no_dynamics", "no_diffusion", "no_reaction", "no_pde", "no_bio_reg",
    "isotropic_diffusion", "state_dependent_diffusion", "discrete_gnn",
    "latent_8", "latent_16", "latent_32", "latent_64",
]

ABLATION_TEX = {
    "full": "Full model",
    "no_dynamics": r"$-$ dynamics ($T{=}0$)",
    "no_diffusion": r"$-$ diffusion term",
    "no_reaction": r"$-$ reaction term",
    "no_pde": r"$-$ PDE constraints",
    "no_bio_reg": r"$-$ biological regularisers",
    "isotropic_diffusion": r"isotropic $D$",
    "state_dependent_diffusion": r"state-dependent $D$",
    "discrete_gnn": r"discrete GNN operator",
    "latent_8": r"latent $C{=}8$",
    "latent_16": r"latent $C{=}16$",
    "latent_32": r"latent $C{=}32$",
    "latent_64": r"latent $C{=}64$",
}


def table_ablations(records: List[Dict], out: Path) -> str:
    df = pd.DataFrame(records)
    if df.empty:
        return ""
    metrics = [m for m in ("pearson_mean", "rmse", "ssim_mean") if m in df.columns]
    agg = df.groupby("variant")[metrics + ["n_params"]].agg(["mean", "std"])
    full = agg.loc["full", ("pearson_mean", "mean")] if "full" in agg.index else np.nan

    rows = []
    for v in ABLATION_ORDER:
        if v not in agg.index:
            continue
        if v == "latent_8":
            rows.append("\\midrule\n")
        cells = [_fmt(agg.loc[v, (m, "mean")], agg.loc[v, (m, "std")],
                      bold=(v == "full")) for m in metrics]
        d = agg.loc[v, ("pearson_mean", "mean")] - full
        dcell = "--" if v == "full" else f"{d:+.3f}"
        rows.append(f"{ABLATION_TEX.get(v, _esc(v))} & {agg.loc[v, ('n_params','mean')]/1e6:.2f}M & "
                    + " & ".join(cells) + f" & {dcell} \\\\\n")

    header = "variant & params & " + " & ".join(METRIC_LABELS.get(m, m) for m in metrics) \
             + r" & $\Delta r$"
    tex = _wrap(
        "".join(rows),
        "Ablations on the mouse-brain Visium section. Each variant removes exactly one "
        "ingredient with all else held fixed. $\\Delta r$ is the change in held-out "
        "Pearson $r$ relative to the full model. The \\emph{$-$ dynamics} row is the "
        "critical control: it disables the operator entirely ($T{=}0$), leaving an "
        "encoder--decoder that never integrates the PDE.",
        "tab:ablations", "lr" + "r" * (len(metrics) + 1), header,
    )
    out.write_text(tex)
    return tex


# --------------------------------------------------------------------------- #
# Table 4: perturbation
# --------------------------------------------------------------------------- #


def table_bead(records: List[Dict], out: Path) -> str:
    df = pd.DataFrame(records)
    if df.empty:
        return ""
    agg = df.groupby("pathway").agg(
        n_genes=("n_pathway_genes", "mean"),
        rank=("held_out_mean_rank_pct", "mean"),
        rank_sd=("held_out_mean_rank_pct", "std"),
        null=("null_mean_rank_pct", "mean"),
        p=("p_value", "mean"),
        width=("response_halfwidth_um", "mean"),
        width_sd=("response_halfwidth_um", "std"),
    ).reset_index()

    rows = []
    for _, r in agg.iterrows():
        star = "$^{*}$" if r["p"] < 0.05 else ""
        rows.append(
            f"{_esc(r['pathway'])} & {int(r['n_genes'])} & "
            f"{_fmt(r['rank'], r['rank_sd'], 1)}{star} & {r['null']:.1f} & {r['p']:.3f} & "
            f"{_fmt(r['width'], r['width_sd'], 0)} \\\\\n"
        )
    tex = _wrap(
        "".join(rows),
        "In-silico morphogen source (\\emph{bead implant}). The latent perturbation "
        "direction is defined from a random half of each pathway's genes; the table "
        "reports where the \\emph{held-out} half ranks in the response magnitude "
        "distribution (lower is stronger; 50\\% is chance). $p$ is a permutation test "
        "over random gene sets of matched size. Response half-width is the distance at "
        "which the predicted response falls to half its peak. $^{*}p<0.05$.",
        "tab:bead", "lrrrrr",
        r"pathway & genes & held-out rank (\%) & null (\%) & $p$ & half-width ($\mu$m)",
    )
    out.write_text(tex)
    return tex


def table_development(records: List[Dict], out: Path) -> str:
    """Stereo-seq E9.5 -> E10.5 field-level forecasting."""
    df = pd.DataFrame(records)
    if df.empty:
        return ""
    rows = []
    ref = df[df["model"].isin(["persistence", "mean"])]
    for _, r in ref.iterrows():
        rows.append(
            f"{_esc(r['display'])} & -- & {r['field_pearson_mean']:.3f} & "
            f"{r['field_rmse']:.3f} & {100*r['frac_genes_positive']:.0f}\\% \\\\\n"
        )
    nmo = df[df["model"] == "nmo"]
    if not nmo.empty:
        rows.append("\\midrule\n")
        agg = nmo.groupby("horizon").agg(
            m=("field_pearson_mean", "mean"), s=("field_pearson_mean", "std"),
            rm=("field_rmse", "mean"), fp=("frac_genes_positive", "mean")).reset_index()
        for _, r in agg.iterrows():
            rows.append(
                f"NMO & $T{{=}}{int(r['horizon'])}$ & {_fmt(r['m'], r['s'])} & "
                f"{r['rm']:.3f} & {100*r['fp']:.0f}\\% \\\\\n"
            )
    tex = _wrap(
        "".join(rows),
        "Developmental forecasting on Stereo-seq: the E9.5 field is encoded, "
        "integrated forward for $T$ operator steps, and scored against the "
        "measured E10.5 field. \\textbf{Consecutive stages are different embryos}, "
        "so there is no cell-to-cell correspondence and the comparison is made at "
        "the level of the rasterised field after isotropic normalisation to a "
        "common frame; it inherits the error of that coarse registration. "
        "\\emph{Persistence} predicts E10.5 $=$ E9.5 and is the reference the "
        "operator must beat for its dynamics to carry temporal information.",
        "tab:development", "llrrr",
        r"model & horizon & field $r$ $\uparrow$ & field RMSE $\downarrow$ & genes with $r>0$",
    )
    out.write_text(tex)
    return tex


def table_perturbseq(records: List[Dict], out: Path) -> str:
    df = pd.DataFrame([r for r in records if "error" not in r])
    if df.empty:
        return ""
    agg = df.groupby("model").agg(
        n=("n_perturbations", "mean"),
        rho=("mean_spearman", "mean"), rho_sd=("mean_spearman", "std"),
        null=("mean_null_spearman", "mean"),
        pos=("frac_positive", "mean"),
        p=("wilcoxon_p", "mean"),
    ).reset_index().sort_values("rho", ascending=False)

    rows = []
    for _, r in agg.iterrows():
        rows.append(
            f"{_esc(DISPLAY_NAMES.get(r['model'], r['model']))} & {int(r['n'])} & "
            f"{_fmt(r['rho'], r['rho_sd'])} & {r['null']:+.3f} & {r['pos']*100:.0f}\\% & "
            f"{r['p']:.3g} \\\\\n"
        )
    tex = _wrap(
        "".join(rows),
        "Agreement between the model's counterfactual gene--gene responses and measured "
        "CRISPRa responses from Norman et al. (2019). \\textbf{This is an out-of-context "
        "probe}: the Perturb-seq data are non-spatial K562 cells while the operator is "
        "fitted to human breast tissue, so it tests whether the reaction module encodes "
        "generic transcriptional coupling, not whether it recovered tissue-specific "
        "signalling. Null is a within-perturbation label permutation; $p$ is a Wilcoxon "
        "signed-rank test against that null.",
        "tab:perturbseq", "lrrrrr",
        r"model & \#perturb. & Spearman $\rho$ & null & \% positive & $p$",
    )
    out.write_text(tex)
    return tex


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #


def build_all(results_root: str | Path = "results", out_dir: str | Path = "paper/tables",
              processed_summary: str | Path = "data/processed/SUMMARY.json") -> Dict[str, str]:
    results_root, out_dir = Path(results_root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    made: Dict[str, str] = {}

    if Path(processed_summary).exists():
        made["datasets"] = table_datasets(processed_summary, out_dir / "tab_datasets.tex")

    exp1 = load_json_glob("exp1/**/*.json", results_root)
    exp1 = [r for r in exp1 if "model" in r and "pearson_mean" in r]
    if exp1:
        sect = exp1[0].get("section", "")
        made["benchmark"] = table_benchmark(exp1, out_dir / "tab_benchmark.tex", section_label=sect)

    for pattern, label, cap, name in [
        ("exp2/**/*.json", "tab:transfer_tissue",
         "Cross-tissue, cross-species transfer: adult mouse brain $\\rightarrow$ human breast carcinoma.",
         "tab_transfer_tissue.tex"),
        ("exp3/**/*.json", "tab:transfer_resolution",
         "Cross-resolution transfer: 55\\,$\\mu$m Visium spots $\\rightarrow$ single-cell Xenium.",
         "tab_transfer_resolution.tex"),
    ]:
        recs = [r for r in load_json_glob(pattern, results_root) if "setting" in r]
        recs = dedupe(recs, ["model", "seed", "setting", "target"])
        if recs:
            made[label] = table_transfer(recs, out_dir / name, cap, label)

    exp5 = [r for r in load_json_glob("exp5/**/*.json", results_root) if "variant" in r]
    exp5 = dedupe(exp5, ["variant", "seed", "section"])
    if exp5:
        made["ablations"] = table_ablations(exp5, out_dir / "tab_ablations.tex")

    bead = load_json_glob("exp4/**/bead_implant.json", results_root)
    if bead:
        made["bead"] = table_bead(bead, out_dir / "tab_bead.tex")
    ps = load_json_glob("exp4/**/perturbseq_consistency.json", results_root)
    if ps:
        made["perturbseq"] = table_perturbseq(ps, out_dir / "tab_perturbseq.tex")

    dev = load_json_glob("exp6/**/forecast.json", results_root)
    if dev:
        made["development"] = table_development(dev, out_dir / "tab_development.tex")

    return made


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="paper/tables")
    a = p.parse_args()
    made = build_all(a.results, a.out)
    print(f"wrote {len(made)} tables to {a.out}: {sorted(made)}")
