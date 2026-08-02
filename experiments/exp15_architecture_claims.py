"""Experiment 15 -- test the architectural claims the paper makes in passing.

Several components are described in the method section as doing something
specific, and none of those descriptions has been tested. Each is cheap to
check and each is a claim a reviewer could ask about:

``decoder``     "receives no positional input, so all spatial structure passes
                through the field". If the decoder could read coordinates
                directly it could memorise a coordinate-to-expression map and
                every ablation would be uninterpretable.
``occupancy``   the splat emits an occupancy map said to distinguish "tissue
                with no signal" from "no measurement here", called essential for
                the masked task. Never ablated.
``aux_z0``      a small term supervising the pre-relaxation read-out, said to
                keep the encoder conditioned without letting the model bypass
                the operator.
``bandwidth``   the splat bandwidth is described as learnable. Whether it
                actually moves, and whether moving matters, is a separate
                question -- an earlier sweep found an eightfold change in it
                alters nothing downstream.
``knn``         the encoder's graph neighbourhood size, never varied.

Each variant is trained at the benchmark budget on the primary section, so the
numbers are comparable to Table~\\ref{tab:benchmark}.

    python experiments/exp15_architecture_claims.py --seeds 0 1
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.losses.objectives import LossWeights
from src.models.nmo import build_nmo
from src.training.dataset import load_section
from src.training.trainer import Trainer, TrainConfig
from src.utils.common import Config, ExperimentLogger, get_device, set_seed

#: name -> (model overrides, loss overrides, note)
VARIANTS: Dict[str, Dict] = {
    "full":            dict(model={}, loss={}),
    "decoder_sees_xy": dict(model={"decoder_sees_coords": True}, loss={}),
    "no_aux_z0":       dict(model={}, loss={"aux_z0": 0.0}),
    # knn is a property of the section's graph, not of the model: the encoder
    # only consults cfg.knn_k when no edge_index is supplied, and every
    # experiment supplies one. Varying the model field tested nothing.
    "knn4":            dict(model={}, loss={}, knn=4),
    "knn16":           dict(model={}, loss={}, knn=16),
    "splat_sigma_half": dict(model={"splat_sigma": 0.5}, loss={}),
    "splat_sigma_2x":  dict(model={"splat_sigma": 2.0}, loss={}),
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--section", default="visium_mouse_brain")
    p.add_argument("--variants", nargs="+", default=list(VARIANTS))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--out-dir", default="results/exp15")
    a = p.parse_args()

    cfg = Config.load(a.config)
    device = get_device(cfg.experiment.get("device", "auto"))
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    out_path = out / "architecture.json"
    rows: List[Dict] = json.loads(out_path.read_text()) if out_path.exists() else []
    done = {(r["variant"], r["seed"]) for r in rows if not r.get("failed")}

    todo = [(v, s) for v in a.variants for s in a.seeds if (v, s) not in done]
    print(f"{len(todo)} of {len(a.variants) * len(a.seeds)} runs remaining")

    for k, (variant, seed) in enumerate(todo, 1):
        spec = VARIANTS[variant]
        try:
            set_seed(seed)
            sec = load_section(Path(cfg.data.processed_dir) / f"{a.section}.h5ad",
                               device=device, knn_k=spec.get("knn", 8))
            mcfg = {**copy.deepcopy(cfg.model.to_dict()), **spec["model"]}
            model = build_nmo(mcfg, n_genes=sec.n_genes)
            lw = LossWeights(**{**cfg.loss.to_dict(), **spec["loss"]})
            lg = ExperimentLogger(out / "runs" / f"{variant}__s{seed}", cfg.to_dict())
            tr = Trainer(model, sec, TrainConfig(**{**cfg.train.to_dict(), "seed": seed,
                                                    "epochs": a.epochs}),
                         lw, lg, device, is_nmo=True)
            res = tr.fit()["test"]
            sig = model.encoder.splat.log_sigma.exp().item()
            rows.append(dict(variant=variant, seed=seed, section=a.section,
                             knn=int(spec.get("knn", 8)),
                             n_edges=int(sec.edge_index.shape[1]),
                             learned_splat_sigma=float(sig),
                             **{m: float(res[m]) for m in
                                ("pearson_mean", "rmse", "ssim_mean",
                                 "morans_i_abs_error") if m in res}))
            print(f"[{k}/{len(todo)}] {variant:<18} s{seed}  "
                  f"r={res['pearson_mean']:.4f}  sigma={sig:.3f}", flush=True)
        except Exception as exc:
            print(f"[{k}/{len(todo)}] FAIL {variant} s{seed}: "
                  f"{type(exc).__name__}: {exc}", flush=True)
            rows.append(dict(variant=variant, seed=seed, failed=True,
                             error=str(exc)[:160]))
        out_path.write_text(json.dumps(rows, indent=2, default=float))

    ok = [r for r in rows if not r.get("failed")]
    if ok:
        import pandas as pd
        df = pd.DataFrame(ok).groupby("variant")[["pearson_mean", "morans_i_abs_error",
                                                  "learned_splat_sigma"]].mean()
        base = df.loc["full", "pearson_mean"] if "full" in df.index else np.nan
        df["delta_r"] = df["pearson_mean"] - base
        print("\\n" + df.round(4).to_string())
    print(f"\\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
