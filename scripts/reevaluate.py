"""Recompute stored metrics from saved checkpoints with the current metric code.

Why this exists
---------------
Metric definitions occasionally need to change mid-project (here: the SSIM
rasterisation lattice is now chosen from the number of held-out locations, so
that bins are roughly singly occupied instead of mostly empty). Once that
happens, results written by earlier runs are no longer comparable with later
ones, and silently mixing them would corrupt every table.

Rather than re-train, we reload each checkpoint and re-score it. Training is the
expensive part; evaluation is a single forward pass. This guarantees that every
number in the paper was produced by one version of the metric code.

    python scripts/reevaluate.py --results results/exp1 --section visium_mouse_brain
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import evaluate_prediction
from src.models.baselines import build_baseline
from src.models.nmo import build_nmo
from src.training.dataset import load_section, subsample_section
from src.training.trainer import Trainer, TrainConfig
from src.utils.common import Config, get_device


def rescore(ckpt: Path, cfg: Config, section_name: str, model_type: str,
            device, max_locations: int | None = None) -> Dict:
    sec = load_section(Path(cfg.data.processed_dir) / f"{section_name}.h5ad", device=device)
    if max_locations:
        sec = subsample_section(sec, max_locations, seed=0).to(device)

    model = (build_nmo(cfg.model.to_dict(), n_genes=sec.n_genes) if model_type == "nmo"
             else build_baseline(model_type, n_genes=sec.n_genes, hidden=128,
                                 latent=cfg.model.get("latent_channels", 32)))
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model = model.to(device).eval()

    tr = Trainer(model, sec, TrainConfig(epochs=0), device=device,
                 is_nmo=(model_type == "nmo"))
    return tr.evaluate("test", compute_ssim=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--results", default="results/exp1")
    p.add_argument("--section", default="visium_mouse_brain")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    cfg = Config.load(a.config)
    device = get_device(cfg.experiment.get("device", "auto"))
    root = Path(a.results)

    n_updated = 0
    for jf in sorted(root.glob("**/*.json")):
        try:
            recs: List[Dict] = json.loads(jf.read_text())
        except Exception:
            continue
        if not isinstance(recs, list) or not recs or "model" not in recs[0]:
            continue

        changed = False
        for rec in recs:
            model_type, seed = rec["model"], rec["seed"]
            sect = rec.get("section", a.section)
            ck = jf.parent / "runs" / f"{sect}__{model_type}__seed{seed}" / "best.pt"
            if not ck.exists():
                print(f"  [miss] {ck.parent.name}")
                continue
            new = rescore(ck, cfg, sect, model_type, device)
            old_ssim = rec.get("ssim_mean")
            for k, v in new.items():
                rec[k] = v
            print(f"  [ok] {sect} {model_type:<18} seed{seed}  "
                  f"r={new['pearson_mean']:.4f}  ssim {old_ssim if old_ssim is None else round(old_ssim,4)}"
                  f" -> {new['ssim_mean']:.4f} (grid {new.get('ssim_grid')}, "
                  f"occ {new.get('ssim_occupancy', 0):.2f})")
            changed = True
            n_updated += 1
        if changed and not a.dry_run:
            jf.write_text(json.dumps(recs, indent=2, default=float))
    print(f"\nrescored {n_updated} runs" + (" (dry run, nothing written)" if a.dry_run else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
