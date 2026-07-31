"""Experiment 8 -- benchmark every model on every available tissue section.

The original benchmark evaluated one section, which is the principal limitation
of the empirical side of the paper: a single tissue cannot distinguish a method
that is better in general from one that suits a particular anatomy. This
experiment runs the identical masked-reconstruction protocol across the full
processed pool, so that the unit of analysis becomes the section and the
comparison can be made with a paired test (``src/evaluation/statistics.py``).

Cost is controlled by subsampling large sections to a fixed budget of locations
and by a reduced epoch count; both are applied identically to every model, so
the comparison remains internally valid. Runs are keyed by
``(section, model, seed)`` and skipped if already present, so the experiment is
resumable and can be spread over several invocations.

    python experiments/exp8_multisection.py --seeds 0 1 --epochs 250
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.losses.objectives import LossWeights
from src.models.baselines import BASELINES, DISPLAY_NAMES, build_baseline
from src.models.nmo import build_nmo
from src.training.dataset import load_section, subsample_section
from src.training.trainer import Trainer, TrainConfig
from src.utils.common import Config, ExperimentLogger, get_device, set_seed

ALL_MODELS = ["nmo", "gnn", "spagcn", "stagate", "graph_transformer",
              "gp", "gp_multiscale", "neural_field", "autoencoder"]


def discover_sections(processed: Path, exclude_substr: Sequence[str] = ("__panel_", "perturb_")
                      ) -> List[str]:
    """Every processed spatial section, excluding derived and non-spatial objects."""
    out = []
    for f in sorted(processed.glob("*.h5ad")):
        if any(s in f.stem for s in exclude_substr):
            continue
        out.append(f.stem)
    return out


def build(model_type: str, cfg: Config, n_genes: int):
    if model_type == "nmo":
        return build_nmo(cfg.model.to_dict(), n_genes=n_genes)
    return build_baseline(model_type, n_genes=n_genes, hidden=128,
                          latent=cfg.model.get("latent_channels", 32))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--sections", nargs="+", default=None)
    p.add_argument("--models", nargs="+", default=ALL_MODELS)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--max-locations", type=int, default=4000)
    p.add_argument("--out-dir", default="results/exp8")
    p.add_argument("--shard", type=int, default=0, help="this worker's index")
    p.add_argument("--n-shards", type=int, default=1)
    a = p.parse_args()

    cfg = Config.load(a.config)
    device = get_device(cfg.experiment.get("device", "auto"))
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    processed = Path(cfg.data.processed_dir)
    sections = a.sections or discover_sections(processed)
    print(f"{len(sections)} sections x {len(a.models)} models x {len(a.seeds)} seeds")

    # one results file per shard so parallel workers never write the same file
    out_path = out / f"results_shard{a.shard}.json"
    rows: List[Dict] = json.loads(out_path.read_text()) if out_path.exists() else []
    done = {(r["section"], r["model"], r["seed"]) for r in rows}
    # also treat runs recorded by *other* shards as done, so a re-shard is safe
    for f in out.glob("results_shard*.json"):
        try:
            for r in json.loads(f.read_text()):
                done.add((r["section"], r["model"], r["seed"]))
        except Exception:
            pass

    jobs = [(s, m, sd) for s in sections for m in a.models for sd in a.seeds]
    jobs = [j for i, j in enumerate(jobs) if i % a.n_shards == a.shard]
    todo = [j for j in jobs if j not in done]
    print(f"shard {a.shard}/{a.n_shards}: {len(todo)} of {len(jobs)} jobs remaining")

    for k, (section, model_type, seed) in enumerate(todo, 1):
        t0 = time.time()
        try:
            set_seed(seed)
            sec = load_section(processed / f"{section}.h5ad", device=device)
            if a.max_locations and sec.n_obs > a.max_locations:
                sec = subsample_section(sec, a.max_locations, seed=seed).to(device)
            model = build(model_type, cfg, sec.n_genes)
            lg = ExperimentLogger(out / "runs" / f"{section}__{model_type}__seed{seed}",
                                  cfg.to_dict())
            tr = Trainer(model, sec, TrainConfig(**{**cfg.train.to_dict(), "seed": seed,
                                                   "epochs": a.epochs}),
                         LossWeights(**cfg.loss.to_dict()), lg, device,
                         is_nmo=(model_type == "nmo"))
            r = tr.fit()
            rows.append(dict(section=section, model=model_type,
                             display=DISPLAY_NAMES.get(model_type, model_type),
                             seed=seed, n_params=r["n_params"],
                             n_obs_used=sec.n_obs, n_genes_used=sec.n_genes,
                             wall_s=round(time.time() - t0, 1), **r["test"]))
            print(f"[{k}/{len(todo)}] {section:<22} {model_type:<18} s{seed} "
                  f"r={r['test']['pearson_mean']:.4f} ({time.time()-t0:.0f}s)", flush=True)
        except Exception as exc:
            print(f"[{k}/{len(todo)}] FAIL {section} {model_type} s{seed}: "
                  f"{type(exc).__name__}: {exc}", flush=True)
            rows.append(dict(section=section, model=model_type, seed=seed,
                             failed=True, error=str(exc)[:160]))
        out_path.write_text(json.dumps(rows, indent=2, default=float))
    print(f"shard {a.shard} complete: {len(rows)} records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
