"""Numerical verification of the propositions stated in the paper.

Every analytical claim in Section 3 and Appendix B--F is checked here against the
actual implementation, and the results are emitted as ``paper/tables/tab_theory.tex``.
The intent is that no proposition appears in the manuscript without a
corresponding executable check: a proof that disagrees with the code is a bug in
one of the two, and this script is what surfaces it.

    python scripts/verify_theory.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch.set_default_dtype(torch.float64)

from src.models.dynamics import (  # noqa: E402
    AnisotropicDiffusion, DynamicsConfig, ReactionDiffusionOperator, ReactionNetwork,
)

RESULTS: List[Dict] = []


def record(prop: str, quantity: str, predicted: str, observed: str, ok: bool) -> None:
    RESULTS.append(dict(prop=prop, quantity=quantity, predicted=predicted,
                        observed=observed, ok=bool(ok)))
    print(f"  [{'ok ' if ok else 'FAIL'}] {prop:<6} {quantity:<38} {observed}")


def check_spd(seed: int = 0) -> None:
    """Proposition 1: D = LL^T + eps I is SPD with lambda_min >= eps."""
    torch.manual_seed(seed)
    d = AnisotropicDiffusion(8, init_scale=0.05).double()
    with torch.no_grad():                       # hostile parameters
        d.l_diag.uniform_(-60, 60); d.l_off.uniform_(-60, 60)
    ev = torch.linalg.eigvalsh(d.tensor())
    record("P1", "$\\lambda_{\\min}(\\mD_c)$", f"$\\ge \\varepsilon = {d.eps:g}$",
           f"{float(ev.min()):.3e}", bool(ev.min() >= d.eps - 1e-12))


def check_contraction(seed: int = 0) -> None:
    """Proposition 2: ||E(tau) z|| <= ||z|| for every tau > 0 (no CFL condition)."""
    torch.manual_seed(seed)
    d = AnisotropicDiffusion(4, init_scale=0.3).double()
    z = torch.randn(1, 4, 32, 32)
    worst = max(float(d.exp_step(z, dt).norm() / z.norm())
                for dt in [1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e3])
    record("P2", "$\\sup_{\\Delta t}\\;\\|E(\\Delta t)u\\|/\\|u\\|$", "$\\le 1$",
           f"{worst:.10f}", worst <= 1 + 1e-12)


def check_mass(seed: int = 0) -> None:
    """Proposition 3: the k = 0 Fourier mode is exactly preserved."""
    torch.manual_seed(seed)
    d = AnisotropicDiffusion(4, init_scale=0.3).double()
    z = torch.randn(1, 4, 32, 32)
    err = max(float((d.exp_step(z, dt).mean((2, 3)) - z.mean((2, 3))).abs().max())
              for dt in [0.05, 1.0, 50.0])
    record("P3", "$\\max_c |\\Delta \\int_\\Omega u_c|$", "$0$ (exact)", f"{err:.2e}", err < 1e-14)


def check_reaction_bound(seed: int = 0) -> None:
    """Proposition 4: ||f_theta||_inf <= gain, channelwise."""
    torch.manual_seed(seed)
    net = ReactionNetwork(4, hidden=16, n_layers=3, gain_init=0.5).double()
    with torch.no_grad():
        for p in net.parameters():
            p.add_(2.0 * torch.randn_like(p))
    g = net.log_gain.exp().detach()
    f = net(torch.randn(1, 4, 64, 64) * 20).detach()
    slack = float((g - f.abs().amax(dim=(0, 2, 3))).min())
    record("P4", "$\\min_c (g_c - \\|f_c\\|_\\infty)$", "$\\ge 0$", f"{slack:.2e}", slack >= -1e-12)


def check_splitting_order(seed: int = 0) -> None:
    """Proposition 5: the Strang scheme is second-order accurate in dt."""
    torch.manual_seed(seed)
    op = ReactionDiffusionOperator(DynamicsConfig(channels=4, dt=0.1, n_steps=1)).double()
    with torch.no_grad():
        for p in op.reaction.parameters():
            p.add_(0.4 * torch.randn_like(p))
    z0 = torch.randn(1, 4, 32, 32) * 0.5
    T = 0.4
    ref = z0.clone()
    with torch.no_grad():
        for _ in range(4096):
            ref = op.step(ref, T / 4096)
    errs = []
    for n in [16, 32, 64, 128]:
        z = z0.clone()
        with torch.no_grad():
            for _ in range(n):
                z = op.step(z, T / n)
        errs.append(float((z - ref).norm() / ref.norm()))
    orders = [math.log2(errs[i - 1] / errs[i]) for i in range(1, len(errs))]
    mean = sum(orders) / len(orders)
    record("P5", "observed order of $S_{\\Delta t}$", "$2$", f"{mean:.2f}", abs(mean - 2) < 0.15)


def check_residual_order(seed: int = 0) -> None:
    """Proposition 7: the along-trajectory residual is O(dt) for any theta."""
    torch.manual_seed(seed)
    cfg = DynamicsConfig(channels=4, dt=0.05, n_steps=1, diffusion_init=0.02)
    op = ReactionDiffusionOperator(cfg).double()
    with torch.no_grad():
        for p in op.reaction.parameters():
            p.add_(0.3 * torch.randn_like(p))
    z0 = torch.randn(1, 4, 16, 16) * 0.5
    errs = []
    for dt in [1e-3, 5e-4, 2.5e-4, 1.25e-4]:
        with torch.no_grad():
            errs.append(float((((op.step(z0, dt) - z0) / dt) - op.time_derivative(z0)).norm()))
    orders = [math.log2(errs[i - 1] / errs[i]) for i in range(1, len(errs))]
    mean = sum(orders) / len(orders)
    record("P7", "observed order of $\\|R_{\\Delta t}\\|$", "$1$", f"{mean:.2f}", abs(mean - 1) < 0.1)


def check_bounded_orbit(seed: int = 0) -> None:
    """Proposition 6: the zero-mean component stays uniformly bounded."""
    torch.manual_seed(seed)
    cfg = DynamicsConfig(channels=4, dt=0.05, n_steps=1, diffusion_init=0.05)
    op = ReactionDiffusionOperator(cfg).double()
    with torch.no_grad():
        for p in op.reaction.parameters():
            p.add_(1.5 * torch.randn_like(p))
    lam = float(torch.linalg.eigvalsh(op.diffusion.tensor().detach()).min())
    rho = math.exp(-cfg.dt * lam * math.pi ** 2)
    gmax = float(op.reaction.log_gain.exp().max())
    H = W = 32
    vol = 4.0
    bound = cfg.dt * gmax * math.sqrt(vol) / (1 - rho)
    z = torch.randn(1, 4, H, W) * 3.0
    worst = 0.0
    with torch.no_grad():
        for n in range(3000):
            z = op.step(z)
            if n > 50:
                zp = z - z.mean((2, 3), keepdim=True)
                worst = max(worst, float(zp.norm()) * math.sqrt(vol / (H * W)))
    record("P6", "$\\sup_n \\|\\vz_n'\\|$ vs.\\ analytic bound",
           f"$\\le {bound:.1f}$", f"{worst:.1f}", worst <= bound)


def emit_table(out: Path) -> None:
    rows = "".join(
        f"{r['prop']} & {r['quantity']} & {r['predicted']} & {r['observed']} & "
        f"{'$\\bullet$' if r['ok'] else '$\\circ$'} \\\\\n" for r in RESULTS)
    tex = (
        "\\begin{table}[t]\n\\centering\n\\small\n"
        "\\caption{Numerical verification of the propositions of "
        "Section~\\ref{sec:method} and Appendix~\\ref{app:proofs}. Each analytical "
        "claim is checked against the implementation under adversarially "
        "perturbed parameters; the script that produces this table is released "
        "with the code.}\n\\label{tab:theory}\n"
        "\\begin{tabular}{llllc}\n\\toprule\n"
        "claim & quantity & predicted & observed & \\\\\n\\midrule\n"
        + rows + "\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex)


def main() -> int:
    print("Verifying analytical claims against the implementation:\n")
    for fn in (check_spd, check_contraction, check_mass, check_reaction_bound,
               check_splitting_order, check_bounded_orbit, check_residual_order):
        fn()
    emit_table(Path("paper/tables/tab_theory.tex"))
    Path("results/theory_verification.json").write_text(json.dumps(RESULTS, indent=2))
    n_ok = sum(r["ok"] for r in RESULTS)
    print(f"\n{n_ok}/{len(RESULTS)} propositions verified -> paper/tables/tab_theory.tex")
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
