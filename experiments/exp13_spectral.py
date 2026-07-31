"""Experiment 13 -- does spectral matching buy spatial fidelity, and at what cost?

The reconstruction's clearest weakness is that it is too smooth: it leads on
correlation and on SSIM while carrying the largest error in Moran's $I$ of any
model tested. That is the expected signature of a dissipative operator, and the
paper previously deferred a remedy to future work.

The remedy implemented here adds a term matching the radially averaged power
spectrum of the predicted field to that of the measured field
(``src/losses/objectives.py::spectral_match``). Matching the spectrum rather
than penalizing smoothness directly is what makes the term targeted: a total
variation or gradient penalty pushes energy into *every* frequency, including
the noise floor, whereas the spectral term asks only that the reconstruction
distribute its energy across scales the way the measurement does.

The question is a trade-off, not a win, so this sweeps the weight and reports
the whole curve. A weight that fixes Moran's $I$ by destroying Pearson $r$ is
not a contribution, and if that is what the data show we report it as such.

    python experiments/exp13_spectral.py --weights 0 0.001 0.01 0.1 1.0
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.losses.objectives import LossWeights
from src.models.nmo import build_nmo
from src.training.dataset import load_section
from src.training.trainer import Trainer, TrainConfig
from src.utils.common import Config, ExperimentLogger, get_device, set_seed

METRICS = ["pearson_mean", "spearman_mean", "rmse", "ssim_mean",
           "morans_i_abs_error", "morans_i_pred", "morans_i_true",
           "gearys_c_abs_error"]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--section", default="visium_mouse_brain")
    p.add_argument("--weights", type=float, nargs="+",
                   default=[0.0, 0.001, 0.01, 0.1, 1.0])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--mode", default="full", choices=["full", "shape"],
                   help="'full' matches the whole spectrum in absolute terms; "
                        "'shape' matches the normalized high-band, which is the "
                        "form that isolates spatial structure from amplitude")
    p.add_argument("--out-dir", default="results/exp13")
    a = p.parse_args()

    cfg = Config.load(a.config)
    device = get_device(cfg.experiment.get("device", "auto"))
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    out_path = out / "spectral_sweep.json"
    rows: List[Dict] = json.loads(out_path.read_text()) if out_path.exists() else []
    done = {(r["spectral_weight"], r["seed"], r.get("mode", "full"))
            for r in rows if not r.get("failed")}

    todo = [(w, s) for w in a.weights for s in a.seeds
            if (w, s, a.mode) not in done]
    print(f"{len(todo)} of {len(a.weights) * len(a.seeds)} runs remaining")

    for k, (weight, seed) in enumerate(todo, 1):
        try:
            set_seed(seed)
            sec = load_section(Path(cfg.data.processed_dir) / f"{a.section}.h5ad",
                               device=device)
            model = build_nmo(cfg.model.to_dict(), n_genes=sec.n_genes)
            lw = LossWeights(**{**cfg.loss.to_dict(), "spectral": float(weight),
                                "spectral_mode": a.mode})
            lg = ExperimentLogger(out / "runs" / f"{a.mode}_w{weight}__s{seed}",
                                  cfg.to_dict())
            tr = Trainer(model, sec, TrainConfig(**{**cfg.train.to_dict(), "seed": seed,
                                                    "epochs": a.epochs}),
                         lw, lg, device, is_nmo=True)
            res = tr.fit()["test"]
            rows.append(dict(spectral_weight=float(weight), seed=seed, mode=a.mode,
                             section=a.section,
                             **{m: float(res[m]) for m in METRICS if m in res}))
            print(f"[{k}/{len(todo)}] {a.mode}/w={weight:<7} s{seed}  "
                  f"r={res['pearson_mean']:.4f}  "
                  f"|dI|={res.get('morans_i_abs_error', float('nan')):.4f}  "
                  f"I_pred={res.get('morans_i_pred', float('nan')):.4f}  "
                  f"(I_true={res.get('morans_i_true', float('nan')):.4f})", flush=True)
        except Exception as exc:
            print(f"[{k}/{len(todo)}] FAIL w={weight} s{seed}: "
                  f"{type(exc).__name__}: {exc}", flush=True)
            rows.append(dict(spectral_weight=float(weight), seed=seed, mode=a.mode,
                             failed=True,
                             error=str(exc)[:160]))
        out_path.write_text(json.dumps(rows, indent=2, default=float))

    ok = [r for r in rows if not r.get("failed")]
    if ok:
        import pandas as pd
        df = pd.DataFrame(ok).groupby(["mode", "spectral_weight"])[
            [m for m in METRICS if m in ok[0]]].mean()
        print("\n" + df.round(4).to_string())
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
