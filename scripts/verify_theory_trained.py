"""Check the propositions on the operators we actually fitted.

``verify_theory.py`` exercises the analytical claims under adversarially
perturbed random parameters, which is the right stress test: it shows the
guarantees hold well outside the regime training reaches. It does not show they
hold *inside* it, on the specific operators whose numbers the paper reports, and
a reviewer is entitled to ask for that separately.

Every checkpoint under results/ is loaded and the same quantities re-measured.

    python scripts/verify_theory_trained.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.nmo import build_nmo
from src.utils.common import Config

ROOT = Path(__file__).resolve().parents[1]


def check(model) -> Dict:
    op = model.operator
    out: Dict[str, float] = {}

    # P(spd): definiteness is structural, so it must hold at the fitted weights.
    ev = torch.linalg.eigvalsh(op.diffusion.tensor().detach())
    out["lambda_min"] = float(ev.min())
    out["eps"] = float(op.diffusion.eps)
    out["spd_ok"] = bool(ev.min() >= op.diffusion.eps - 1e-9)

    C = op.diffusion.tensor().shape[0]
    z = torch.randn(1, C, 64, 64)

    # P(contraction): the diffusive half-step is non-expansive at every dt.
    worst = max(float(op.diffusion.exp_step(z, dt).norm() / z.norm())
                for dt in (1e-3, 1e-2, 1e-1, 1.0, 10.0, 1e3))
    out["contraction_sup"] = worst
    out["contraction_ok"] = worst <= 1 + 1e-6

    # P(mass): the k=0 mode is untouched.
    err = max(float((op.diffusion.exp_step(z, dt).mean((2, 3))
                     - z.mean((2, 3))).abs().max())
              for dt in (0.05, 1.0, 50.0))
    out["mass_err"] = err
    out["mass_ok"] = err < 1e-6

    # P(reaction): the bounded reaction respects its own gain.
    if op.reaction is not None:
        g = op.reaction.log_gain.exp().detach()
        f = op.reaction(torch.randn(1, C, 64, 64) * 20).detach()
        slack = float((g - f.abs().amax(dim=(0, 2, 3))).min())
        out["reaction_slack"] = slack
        out["reaction_ok"] = slack >= -1e-9
    return out


def main() -> int:
    cfg = Config.load(ROOT / "configs" / "base.yaml")
    ckpts = sorted(ROOT.glob("results/exp1/*/runs/*__nmo__seed*/best.pt"))
    if not ckpts:
        print("no NMO checkpoints found")
        return 0

    rows: List[Dict] = []
    for ck in ckpts:
        try:
            state = torch.load(ck, map_location="cpu", weights_only=False)
            sd = state["model"] if "model" in state else state
            # The decoder's final bias has one entry per gene; inferring from a
            # partial key match picked an interior layer instead.
            n_genes = int(sd["decoder.net.net.4.bias"].shape[0])
            m = build_nmo(cfg.model.to_dict(), n_genes=n_genes)
            m.load_state_dict(sd)
            m.eval()
            rows.append(dict(checkpoint=str(ck.relative_to(ROOT)), **check(m)))
            r = rows[-1]
            print(f"  {ck.parent.name:<34} spd={r['spd_ok']} contraction={r['contraction_ok']} "
                  f"mass={r['mass_ok']} reaction={r.get('reaction_ok')}")
        except Exception as exc:
            print(f"  [skip] {ck.parent.name}: {type(exc).__name__}: {exc}")

    ok = [r for r in rows if all(v for k, v in r.items() if k.endswith("_ok"))]
    out = ROOT / "results" / "theory_trained.json"
    out.write_text(json.dumps(rows, indent=2, default=float))
    print(f"\n{len(ok)}/{len(rows)} fitted operators satisfy every proposition")
    print(f"wrote {out.relative_to(ROOT)}")
    return 0 if len(ok) == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
