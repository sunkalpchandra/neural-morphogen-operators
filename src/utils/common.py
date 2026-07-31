"""Shared infrastructure: config loading, seeding, device selection, logging."""

from __future__ import annotations

import json
import logging
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


class Config(dict):
    """Attribute-accessible nested dict, loaded from YAML.

    Supports single-level inheritance via a top-level ``defaults:`` key holding
    a path (relative to the config file) to a parent config, and dotted-path
    overrides from the command line.
    """

    def __getattr__(self, k: str) -> Any:
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def __setattr__(self, k: str, v: Any) -> None:
        self[k] = v

    @staticmethod
    def _wrap(obj: Any) -> Any:
        if isinstance(obj, dict):
            return Config({k: Config._wrap(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [Config._wrap(v) for v in obj]
        return obj

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        path = Path(path)
        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
        parent = raw.pop("defaults", None)
        if parent:
            base = cls.load((path.parent / parent).resolve())
            raw = deep_merge(base, raw)
        return cls._wrap(raw)

    def override(self, pairs: List[str]) -> "Config":
        """Apply ``a.b.c=value`` overrides; values parsed as YAML scalars."""
        for pair in pairs:
            if "=" not in pair:
                raise ValueError(f"override must be key=value, got {pair!r}")
            key, val = pair.split("=", 1)
            node: Any = self
            parts = key.split(".")
            for p in parts[:-1]:
                if p not in node:
                    node[p] = Config()
                node = node[p]
            node[parts[-1]] = _parse_scalar(val)
        return self

    def to_dict(self) -> Dict[str, Any]:
        def unwrap(o: Any) -> Any:
            if isinstance(o, dict):
                return {k: unwrap(v) for k, v in o.items()}
            if isinstance(o, list):
                return [unwrap(v) for v in o]
            return o

        return unwrap(self)

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for p in dotted.split("."):
            if not isinstance(node, dict) or p not in node:
                return default
            node = node[p]
        return node


def _parse_scalar(val: str):
    """Parse a CLI override value.

    ``yaml.safe_load`` follows YAML 1.1 and leaves ``1e-3`` as the *string*
    ``'1e-3'`` (a float needs ``1.0e-3``). Silently passing a string learning
    rate into the optimiser is the kind of bug that costs a day, so we retry an
    explicit float/int conversion before giving up and keeping the string.
    """
    parsed = yaml.safe_load(val)
    if isinstance(parsed, str):
        try:
            return int(parsed)
        except ValueError:
            pass
        try:
            return float(parsed)
        except ValueError:
            pass
    return parsed


def deep_merge(base: Dict, over: Dict) -> Dict:
    out = dict(base)
    for k, v in over.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed every RNG we touch.

    Note: full bitwise determinism is not achievable on the MPS backend, and
    ``torch.use_deterministic_algorithms`` is therefore requested in
    warn-only mode. Reported variability across seeds is handled by running
    multiple seeds and reporting mean +/- std, which is the honest approach.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def get_device(prefer: str = "auto") -> torch.device:
    """Select a compute device.

    ``auto`` resolves to CUDA when present and otherwise to **CPU**, even on
    Apple silicon where MPS is available. This is deliberate: the NMO forward
    pass is dominated by 2-D FFTs (the spectral diffusion solve), and at the
    grid sizes used here (64x64 to 128x128) ``torch.fft`` on the MPS backend is
    measured to be ~20x slower than on CPU. Pass ``device: mps`` explicitly to
    override.
    """
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def environment_report() -> Dict[str, Any]:
    """Everything needed to reproduce / explain a result."""

    def _git(*args: str) -> Optional[str]:
        try:
            return subprocess.check_output(
                ["git", *args], stderr=subprocess.DEVNULL, text=True
            ).strip()
        except Exception:
            return None

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": bool(
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        ),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
    }


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #


def get_logger(name: str = "nmo", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)-7s %(message)s", "%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(level)
        logger.propagate = False
    return logger


class ExperimentLogger:
    """Filesystem-backed run logger: JSONL metrics + config + env snapshot.

    Deliberately dependency-free (no wandb/tensorboard) so that every result in
    the paper can be regenerated offline from the run directory alone.
    """

    def __init__(self, run_dir: str | Path, config: Optional[Dict] = None, name: str = "nmo"):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.log = get_logger(name)
        self._t0 = time.time()
        if config is not None:
            (self.run_dir / "config.yaml").write_text(yaml.safe_dump(_plain(config), sort_keys=False))
        (self.run_dir / "environment.json").write_text(json.dumps(environment_report(), indent=2))

    def log_metrics(self, step: int, split: str = "train", **kw: Any) -> None:
        rec = {"step": step, "split": split, "wall_s": round(time.time() - self._t0, 2)}
        rec.update({k: _scalar(v) for k, v in kw.items()})
        with open(self.metrics_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")

    def info(self, msg: str) -> None:
        self.log.info(msg)

    def save_json(self, name: str, obj: Any) -> Path:
        p = self.run_dir / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(_plain(obj), indent=2))
        return p

    def read_metrics(self) -> List[Dict]:
        if not self.metrics_path.exists():
            return []
        return [json.loads(l) for l in self.metrics_path.read_text().splitlines() if l.strip()]


def _scalar(v: Any) -> Any:
    if isinstance(v, torch.Tensor):
        return v.detach().float().mean().item() if v.numel() > 1 else v.item()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def _plain(o: Any) -> Any:
    if is_dataclass(o) and not isinstance(o, type):
        return _plain(asdict(o))
    if isinstance(o, dict):
        return {k: _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return _scalar(o)
    if isinstance(o, Path):
        return str(o)
    return o


# --------------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------------- #


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    metrics: Optional[Dict] = None,
    config: Optional[Dict] = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "epoch": epoch,
            "metrics": _plain(metrics or {}),
            "config": _plain(config or {}),
            "env": environment_report(),
        },
        path,
    )
    return path


def load_checkpoint(
    path: str | Path, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str | torch.device = "cpu",
) -> Dict:
    ck = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ck["model"])
    if optimizer is not None and ck.get("optimizer"):
        optimizer.load_state_dict(ck["optimizer"])
    return ck


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
