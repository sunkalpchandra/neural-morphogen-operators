"""Normalize the manuscript and generators to American spelling.

Applied to the .tex sources and to every module that emits LaTeX, so that
regenerating tables cannot reintroduce British forms. Word-boundary anchored and
case-preserving; BibTeX titles are left untouched because they are quotations.
"""
from __future__ import annotations
import re, sys
from pathlib import Path

PAIRS = [
    ("normalise", "normalize"), ("normalisation", "normalization"),
    ("penalise", "penalize"), ("penalisation", "penalization"),
    ("discretise", "discretize"), ("discretisation", "discretization"),
    ("parameterise", "parameterize"), ("parametrise", "parametrize"),
    ("parameterisation", "parameterization"), ("parametrisation", "parametrization"),
    ("characterise", "characterize"), ("characterisation", "characterization"),
    ("summarise", "summarize"), ("minimise", "minimize"), ("maximise", "maximize"),
    ("generalise", "generalize"), ("generalisation", "generalization"),
    ("regularise", "regularize"), ("regularisation", "regularization"),
    ("regulariser", "regularizer"), ("standardise", "standardize"),
    ("standardisation", "standardization"), ("initialise", "initialize"),
    ("initialisation", "initialization"), ("optimise", "optimize"),
    ("optimisation", "optimization"), ("factorise", "factorize"),
    ("rasterise", "rasterize"), ("rasterisation", "rasterization"),
    ("analyse", "analyze"), ("analysed", "analyzed"), ("analysing", "analyzing"),
    ("behaviour", "behavior"), ("colour", "color"), ("coloured", "colored"),
    ("favourable", "favorable"), ("neighbour", "neighbor"),
    ("neighbourhood", "neighborhood"), ("modelling", "modeling"),
    ("modelled", "modeled"), ("labelled", "labeled"), ("labelling", "labeling"),
    ("centre", "center"), ("centred", "centered"), ("artefact", "artifact"),
    ("signalling", "signaling"), ("fibre", "fiber"), ("metre", "meter"),
    ("whilst", "while"), ("amongst", "among"), ("towards", "toward"),
    ("learnt", "learned"), ("cancelled", "canceled"), ("modelt", "modelt"),
]
# expand each stem to its inflections
RULES = []
for br, am in PAIRS:
    for suf in ("", "s", "d", "r", "rs", "ing"):
        b, a = br + suf, am + suf
        if br.endswith("e") and suf in ("d", "r", "rs", "ing"):
            b = br[:-1] + ("ed" if suf == "d" else "er" if suf == "r"
                           else "ers" if suf == "rs" else "ing")
            a = am[:-1] + ("ed" if suf == "d" else "er" if suf == "r"
                           else "ers" if suf == "rs" else "ing")
        RULES.append((b, a))
RULES = sorted(set(RULES), key=lambda x: -len(x[0]))


def convert(text: str) -> tuple[str, int]:
    n = 0
    for b, a in RULES:
        for pat, rep in ((b, a), (b.capitalize(), a.capitalize())):
            text, k = re.subn(rf"\b{re.escape(pat)}\b", rep, text)
            n += k
    return text, n


def main(paths: list[str]) -> int:
    total = 0
    for pth in paths:
        p = Path(pth)
        if not p.exists():
            continue
        new, n = convert(p.read_text())
        if n:
            p.write_text(new)
            print(f"  {n:>4} substitutions  {p}")
        total += n
    print(f"total: {total}")
    return 0


if __name__ == "__main__":
    targets = sys.argv[1:] or [
        "paper/neurips_2026.tex",
        *[str(x) for x in Path("paper/tables").glob("*.tex")],
        "src/evaluation/tables.py", "src/evaluation/numbers.py",
        "src/visualization/figures.py", "scripts/verify_theory.py",
    ]
    raise SystemExit(main(targets))
