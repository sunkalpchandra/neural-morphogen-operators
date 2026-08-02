"""Which genes does the model win on, and is there a pattern?

The paper argues its advantage comes from modelling expression as a continuous
field rather than from message passing. If that is right, the advantage should
concentrate on genes whose true expression is spatially structured, and vanish
on genes that are spatially unstructured -- for those there is no field to
exploit, and a model that ignores geometry should do just as well.

That is a falsifiable prediction and this tests it. For every gene we compute the
held-out Pearson r of each model and the Moran's I of the *measured* expression,
then ask whether the per-gene advantage tracks spatial structure.

Each seed is compared against whichever baseline is strongest for that seed, not
against a fixed rival, so the comparison cannot be flattered by picking a weak
opponent. Reports the correlation, a quartile breakdown and the extreme genes in
both directions, so a null result would be as visible as a positive one.

    python scripts/per_gene_analysis.py --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import (  # noqa: E402
    morans_i, pearson_per_gene, spatial_weights)
from src.losses.objectives import LossWeights  # noqa: E402
from src.models.baselines import build_baseline  # noqa: E402
from src.models.nmo import build_nmo  # noqa: E402
from src.training.dataset import load_section  # noqa: E402
from src.training.trainer import TrainConfig, Trainer  # noqa: E402
from src.utils.common import Config, get_device  # noqa: E402

N_QUARTILES = 4


def _load_model(name, cfg, sec, ckpt):
    model = (build_nmo(cfg.model.to_dict(), n_genes=sec.n_genes) if name == "nmo"
             else build_baseline(name, n_genes=sec.n_genes, hidden=128, latent=32))
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("model", state))
    return model.eval()


def _one_seed(seed, cfg, sec, device, test, true, section):
    """Per-gene r for every model with a checkpoint at this seed."""
    ckpts = {c.parent.name.split("__")[1]: c
             for c in Path("results").glob(
                 f"exp1/*/runs/{section}__*__seed{seed}/best.pt")}
    if "nmo" not in ckpts:
        return None

    per_gene = {}
    for name, ck in sorted(ckpts.items()):
        try:
            model = _load_model(name, cfg, sec, ck)
            tr = Trainer(model, sec,
                         TrainConfig(**{**cfg.train.to_dict(), "seed": seed}),
                         LossWeights(**cfg.loss.to_dict()), None, device,
                         is_nmo=(name == "nmo"))
            pred = tr.predict(tr.train_visible)
            per_gene[name] = pearson_per_gene(pred[test], true[test])
        except Exception as exc:
            print(f"  seed {seed} {name}: skipped ({type(exc).__name__})")
    return per_gene if len(per_gene) >= 2 else None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--section", default="visium_mouse_brain")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--out", default="results/audit/per_gene.json")
    a = p.parse_args()

    cfg = Config.load(a.config)
    device = get_device("cpu")
    sec = load_section(Path(cfg.data.processed_dir) / f"{a.section}.h5ad",
                       device=device)

    test = sec.mask("test").bool().cpu().numpy()
    true = sec.numpy_expr(denorm=True)
    coords = sec.coords.cpu().numpy()
    names = (list(sec.gene_names) if sec.gene_names is not None
             else [str(i) for i in range(true.shape[1])])

    # Spatial structure of the measured signal on held-out locations only, so it
    # describes exactly the data the models were scored against.
    struct = morans_i(true[test], spatial_weights(coords[test]))

    seeds_out, advantages = [], []
    for seed in a.seeds:
        pg = _one_seed(seed, cfg, sec, device, test, true, a.section)
        if pg is None:
            print(f"seed {seed}: skipped")
            continue
        ours = pg["nmo"]
        rivals = {k: v for k, v in pg.items() if k != "nmo"}
        best_name = max(rivals, key=lambda k: float(np.nanmean(rivals[k])))
        adv = ours - rivals[best_name]
        ok = np.isfinite(adv) & np.isfinite(struct)
        rho = float(np.corrcoef(struct[ok], adv[ok])[0, 1])

        qs = np.quantile(struct[ok], np.linspace(0, 1, N_QUARTILES + 1))
        bins = []
        for i in range(N_QUARTILES):
            hi_incl = i == N_QUARTILES - 1
            m = ok & (struct >= qs[i]) & (struct <= qs[i + 1] if hi_incl
                                          else struct < qs[i + 1])
            if m.any():
                bins.append(dict(
                    quartile=i + 1, lo=float(qs[i]), hi=float(qs[i + 1]),
                    n=int(m.sum()), nrdo=float(np.nanmean(ours[m])),
                    rival=float(np.nanmean(rivals[best_name][m])),
                    delta=float(np.nanmean(ours[m] - rivals[best_name][m]))))

        seeds_out.append(dict(seed=seed, best_baseline=best_name,
                              n_genes=int(ok.sum()), corr=rho, bins=bins,
                              mean_r={k: float(np.nanmean(v))
                                      for k, v in pg.items()}))
        advantages.append(np.where(ok, adv, np.nan))
        print(f"seed {seed}: vs {best_name:<10} corr={rho:+.3f}  "
              f"top-quartile delta={bins[-1]['delta']:+.4f}")

    if not seeds_out:
        print("no seeds produced results")
        return 1

    # Genes ranked by advantage averaged over seeds, so one seed's noise cannot
    # put a gene at the top of the list.
    mean_adv = np.nanmean(np.vstack(advantages), axis=0)
    fin = np.flatnonzero(np.isfinite(mean_adv))
    fin = fin[np.argsort(mean_adv[fin])]
    losses = [(names[i], float(mean_adv[i])) for i in fin[:10]]
    gains = [(names[i], float(mean_adv[i])) for i in fin[::-1][:10]]

    rhos = [s["corr"] for s in seeds_out]
    tops = [s["bins"][-1]["delta"] for s in seeds_out]
    lows = [s["bins"][0]["delta"] for s in seeds_out]
    print(f"\nacross {len(seeds_out)} seeds: corr {min(rhos):+.3f}..{max(rhos):+.3f}")
    print(f"  most-structured quartile:  {min(tops):+.4f}..{max(tops):+.4f}")
    print(f"  least-structured quartile: {min(lows):+.4f}..{max(lows):+.4f}")
    print(f"  largest mean gains:  "
          f"{', '.join(f'{n} {v:+.3f}' for n, v in gains[:5])}")
    print(f"  largest mean losses: "
          f"{', '.join(f'{n} {v:+.3f}' for n, v in losses[:5])}")

    dest = Path(a.out); dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(dict(
        section=a.section, seeds=seeds_out,
        corr_lo=min(rhos), corr_hi=max(rhos),
        top_quartile_lo=min(tops), top_quartile_hi=max(tops),
        bottom_quartile_lo=min(lows), bottom_quartile_hi=max(lows),
        top_gains=gains, top_losses=losses,
        # Per-gene vectors so a figure can show the distribution rather than
        # replotting the quartile means a second time. Rounded to keep the
        # artifact small; the analysis above uses full precision.
        gene_structure=[round(float(x), 4) for x in struct],
        gene_advantage=[None if not np.isfinite(v) else round(float(v), 4)
                        for v in mean_adv]), indent=2))
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
