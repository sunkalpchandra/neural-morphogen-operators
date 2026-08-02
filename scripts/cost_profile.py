"""Peak memory and throughput per model, to sit alongside the accuracy numbers.

The converged comparison already reports wall-clock (Table~\\ref{tab:converged}).
Memory is the other half of the cost question and was never measured: a reader
deciding whether this is affordable on their section needs to know what it
allocates, not only how long it takes.

Measures peak resident set over a fixed number of training epochs on one section,
sampling RSS from a background thread, since PyTorch's own counters cover only
CUDA/MPS allocations and most of the footprint here is CPU-side.

Run this on an otherwise idle machine: a concurrent training job inflates both
numbers and the result is not comparable across models measured at different
times.

    python scripts/cost_profile.py --steps 30
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import threading
import time
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.losses.objectives import LossWeights  # noqa: E402
from src.models.baselines import build_baseline  # noqa: E402
from src.models.nmo import build_nmo  # noqa: E402
from src.training.dataset import load_section  # noqa: E402
from src.training.trainer import TrainConfig, Trainer  # noqa: E402
from src.utils.common import Config, count_parameters, get_device  # noqa: E402

MODELS = ["nmo", "autoencoder", "gnn", "spagcn", "stagate",
          "graph_transformer", "neural_field"]


class _PeakRSS:
    """Sample RSS from a thread; torch's counters miss the CPU-side footprint."""

    def __init__(self, interval: float = 0.01):
        self.interval, self.peak, self._stop = interval, 0, False

    def __enter__(self):
        import psutil
        self._p = psutil.Process()
        self.base = self._p.memory_info().rss

        def loop():
            while not self._stop:
                self.peak = max(self.peak, self._p.memory_info().rss)
                time.sleep(self.interval)

        self._t = threading.Thread(target=loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop = True
        self._t.join(timeout=1.0)

    @property
    def peak_mb(self) -> float:
        return max(0.0, (self.peak - self.base)) / 1e6


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--section", default="visium_mouse_brain")
    p.add_argument("--models", nargs="+", default=MODELS)
    p.add_argument("--steps", type=int, default=30,
                   help="training epochs to time")
    p.add_argument("--out", default="results/audit/cost_profile.json")
    a = p.parse_args()

    try:
        import psutil  # noqa: F401
    except ImportError:
        print("psutil not installed; skipping (pip install psutil)")
        return 0

    cfg = Config.load(a.config)
    device = get_device("cpu")
    sec = load_section(Path(cfg.data.processed_dir) / f"{a.section}.h5ad",
                       device=device)

    rows = []
    print(f"{'model':<18}{'params':>10}{'peak MB':>10}{'s/epoch':>9}")
    for name in a.models:
        gc.collect()
        try:
            model = (build_nmo(cfg.model.to_dict(), n_genes=sec.n_genes)
                     if name == "nmo"
                     else build_baseline(name, n_genes=sec.n_genes,
                                         hidden=128, latent=32))
            tr = Trainer(model, sec,
                         TrainConfig(**{**cfg.train.to_dict(), "seed": 0,
                                        "epochs": a.steps}),
                         LossWeights(**cfg.loss.to_dict()), None, device,
                         is_nmo=(name == "nmo"))
            with _PeakRSS() as m:
                t0 = time.perf_counter()
                for _ in range(a.steps):
                    tr.train_epoch()
                dt = time.perf_counter() - t0
            rows.append(dict(model=name, params=count_parameters(model),
                             peak_mb=round(m.peak_mb, 1),
                             s_per_epoch=round(dt / a.steps, 4)))
            print(f"{name:<18}{count_parameters(model):>10,}{m.peak_mb:>10.1f}"
                  f"{dt / a.steps:>9.4f}")
        except Exception as exc:
            print(f"{name:<18}  skipped ({type(exc).__name__}: {str(exc)[:40]})")

    if rows:
        base = next((r for r in rows if r["model"] == "nmo"), None)
        if base:
            for r in rows:
                r["mem_vs_nmo"] = round(r["peak_mb"] / max(base["peak_mb"], 1e-9), 2)
                r["time_vs_nmo"] = round(r["s_per_epoch"] / max(base["s_per_epoch"], 1e-9), 2)
        dest = Path(a.out); dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
