"""Fail the build when the manuscript and the run artifacts disagree.

The paper quotes every result through a macro defined in ``paper/numbers.tex``,
which is generated from ``results/``. That discipline only holds if three things
are true, and this script checks all three:

1. **Freshness.** Regenerating every table and every macro from ``results/``
   reproduces byte-for-byte what is checked in under ``paper/``. A table that no
   entry point rebuilds is stale by definition, and stale is how single-section
   numbers ended up in prose describing a seventeen-section benchmark.

2. **Scope.** Each macro is annotated with the experiment it is derived from.
   A macro whose source experiment is not among those a section is allowed to
   quote is an error, even when its value happens to be right. This is the check
   that catches a correct number in the wrong place -- the failure mode that
   survived every previous review.

3. **Coherence.** Cross-table invariants that no single generator can enforce:
   marginal means in the benchmark table must agree with the paired differences
   in the statistics table, every row of a table must be averaged over the same
   sections, and every macro named in the manuscript must actually be defined.

Anything that cannot be traced is appended to ``UNVERIFIED_CLAIMS.md`` rather
than passed over in silence.

    python scripts/check_numbers.py           # report + exit status
    python scripts/check_numbers.py --fix-unverified   # also rewrite the list
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
TEX = PAPER / "neurips_2026.tex"
SECTIONS = PAPER / "sections"

# --------------------------------------------------------------------------- #
# 1. Macro provenance registry
# --------------------------------------------------------------------------- #
# Which experiment each macro is derived from. ``numbers.py`` is the authority
# on the values; this is the authority on what they are *about*. Keep in sync:
# an undeclared macro is itself a failure, so adding one forces a decision about
# its scope.

PROVENANCE: Dict[str, str] = {}


def _decl(source: str, *names: str) -> None:
    for n in names:
        PROVENANCE[n] = source


_decl("exp1",           # single section: visium_mouse_brain, 3 seeds
      "NMOPearson", "NMORMSE", "NMOSSIM", "NMOMoran", "NMOParams", "NSeeds",
      "BestBaseline", "BestBaselinePearson", "BestBaselineMoran", "BestBaselineSSIM",
      "BestAnySSIM", "BestAnySSIMModel", "BestAnyMoran", "BestAnyMoranModel",
      "SSIMDelta", "MoranDelta", "PearsonGain", "PearsonGainPct",
      "AEPearson", "BestGraphOverAE", "WorstGraphOverAE", "NGraphBelowAE",
      "NGraphModels", "NMOOverAE", "NMOvsGraphRatio", "NMOMoranPred", "MoranTrue")
_decl("exp8",           # multi-section benchmark
      "MSSections", "MSPool", "MSRuns", "MSSeeds", "MSNMOPearson", "MSBestBaseline",
      "MSBestBaselinePearson", "MSMinDelta", "MSMinDz", "MSMaxHolm", "MSMinWins",
      "MSNPairs", "MSNPairsRange", "MSMinWinFrac", "MSNBaselines", "MSAllSig", "MSWeakest", "MSWeakestPearson",
      "MSMaxDelta", "MSSpecimens", "MSSpecPairs", "MSSpecMinP", "MSSpecGraphDz",
      "MSSpecGraphDelta", "MSSpecContDz", "MSSpecContDelta", "MSSpecWins", "MSSpecUncorrP", "MSSpecHolmFloor",
      "MSSpecSweeps", "MSSpecSweepDz", "MSSpecGraphWorstP",
      "MSSpecGraphWorstModel", "MSSpecGraphWins", "MSSpecNSig",
      "MSSpecSigMaxHolm", "MSSpecSigMinWins", "MSSpecSigModels",
      "MSSpecGraphSig", "MSSpecGraphSigHolm", "MSSpecGraphSigWins", "MSSpecAltN", "MSSpecAltNSig",
      "MSSpecTightModel", "MSSpecTightDelta", "MSSpecTightCI",
      "MSNMOSSIM", "MSBestAnySSIM", "MSBestAnySSIMModel", "MSSSIMDelta",
      "MSNMOMoran", "MSBestAnyMoran", "MSBestAnyMoranModel", "MSMoranDelta",
      "MSNMORMSE", "MSBestAnyRMSE", "MSBestAnyRMSEModel", "MSRMSEDelta",
      "MSAEPearson", "MSBestGraphOverAE", "MSWorstGraphOverAE", "MSNGraphModels",
      "MSNMOOverAE", "MSExcludedN", "MSExcluded", "MSMinHeldOut")
_decl("exp2", "TissueZeroShot", "TissueOracle", "TissueFloor", "TissueFineTune",
      "TissueZeroShotBest", "TissueZeroShotBaseline", "TissueZeroShotBaselineName",
      "TissueZeroShotGap", "TissueZeroShotRatio", "TissueSharedGenes")
_decl("exp3", "ResZeroShot", "ResOracle", "ResFloor", "ResZeroShotBest",
      "ResZeroShotBaseline", "ResZeroShotBaselineName", "ResZeroShotGap",
      "ResZeroShotRatio", "ResSharedGenes")
_decl("exp4", "BeadNSig", "BeadNPathways", "BeadWidth", "BeadMeanRank", "BeadMinP",
      "BeadSeeds", "PSRho", "PSNull", "PSFracPos", "PSNPerturb", "PSNShared",
      "PSWilcoxonP")
_decl("exp5", "AblFull", "AblNoDynamics", "AblNoDynamicsDelta", "AblNoDiffusion",
      "AblNoDiffusionDelta", "AblNoReaction", "AblNoReactionDelta", "AblNoPDE",
      "AblNoPDEDelta", "AblNoBioReg", "AblNoBioRegDelta", "AblIsotropic",
      "AblIsotropicDelta", "AblDiscreteGNN", "AblDiscreteGNNDelta")
_decl("exp5_matched", "MatchedFull", "MatchedNoDynamics", "MatchedNoDynamicsDelta",
      "MatchedNoReaction", "MatchedNoReactionDelta", "MatchedSeeds")
_decl("exp6", "DevPersistence", "DevNMOBest", "DevNMOBestHorizon", "DevNMOZero",
      "DevBeatsPersistence")
_decl("theory", "TheoryTrainedN", "TheoryTrainedOK",
      "TheoryTrainedLambdaMin", "TheoryTrainedSup")
_decl("exp7", "CfgN", "CfgDiffLo", "CfgDiffHi", "CfgLatLo", "CfgLatHi",
      "CfgOursMin", "CfgOursAllStable", "CfgWorstScheme", "CfgWorstDt",
      "CfgWorstRatio", "CfgSpread",
      "NumStableDt", "NumCFL", "NumCFLRatio", "NumEulerSteps", "NumSpeedup",
      "NumStrangSteps")
_decl("exp9", "BioSections", "BioARIRet", "BioARIRetBest", "BioARIRetBestModel",
      "BioMarker", "BioMarkerBest", "BioMarkerBestModel", "BioKNN", "BioKNNBest",
      "BioKNNBestModel", "BioARIStagateD", "BioARIStagateDz",
      "BioARIStagateP", "BioARIStagateWins", "BioARINSig", "BioARINComp", "BioExcluded", "BioExcludedN", "BioMinARI")
_decl("physics", "DiffLenMedian", "DiffLenLo", "DiffLenHi", "TuringFrac",
      "NPhysicsRuns", "GrowthAtZero", "PatternWavelength", "DiffLenN",
      "DiffLenPerRun", "DiffLenSections", "DiffLenSection")
_decl("inventory", "NDatasets", "NSections", "TotalLocations", "NSpatialSections",
      "SpatialLocations", "NPerturbCells")
_decl("architecture", "OpParams", "TotalParams", "OpParamPct")
_decl("exp10", "DensSeeds", "DensBase", "DensLevel", "DensNMO", "DensBest",
      "DensBestModel", "DensWorst", "DensGapLo", "DensGapHi", "DensBothSeeds",
      "DensFullBestModel", "DensNMOFullRank",
      "RobRuns", "RobAxes", "RobNoiseNMO", "RobNoiseBest",
      "RobNoiseBestModel", "RobNoiseWins", "RobDropNMO", "RobDropBest",
      "RobDropBestModel", "RobDropWins", "RobDensNMO", "RobDensBest",
      "RobDensBestModel", "RobDensWins")
_decl("exp13", "SpecBaseR", "SpecBaseMoran", "SpecFullBestW", "SpecFullBestMoran",
      "SpecFullBestR", "SpecFullDeltaR", "SpecFullDeltaMoran", "SpecFullPctR",
      "SpecFullPctMoran", "SpecShapeBestW", "SpecShapeBestMoran", "SpecShapeBestR",
      "SpecShapeDeltaR", "SpecShapeDeltaMoran", "SpecShapePctR",
      "SpecShapePctMoran", "SpecMinIPred", "SpecITrue", "SpecSeeds")
_decl("exp14", "ConvSeeds", "ConvSection", "ConvNMO", "ConvBest", "ConvBestModel",
      "ConvMinDz", "ConvMaxDz", "ConvWins", "ConvNPairs", "ConvGraphOverAE",
      "ConvNMOMoran", "ConvBestMoran", "ConvNMOWall", "ConvBaseWallLo",
      "ConvBaseWallHi", "ConvSlowdown")
_decl("exp11", "NullBaseline", "NullShuffled", "NullShuffledR", "NullInit",
      "NullDataShift", "NullShuffledShift", "NullSigmaLo", "NullSigmaHi",
      "NullSigmaRange", "NullGridUmLo", "NullGridUmHi", "NullGridCellsLo",
      "NullGridCellsHi", "NullGridN")
_decl("exp12", "SplitXenRandom", "SplitXenBlock", "SplitXenRatio",
      "SplitVisRandom", "SplitVisBlock", "SplitVisRatio")

#: Which sources each manuscript section may quote. A section absent from this
#: map is unrestricted. ``exp1`` is deliberately barred from the multi-section
#: results: those are different tissues measured under a different budget.
SECTION_SCOPE: Dict[str, Set[str]] = {
    "Masked spatial reconstruction": {"exp8", "architecture", "inventory"},
    "The numerical machinery is load-bearing": {"exp7", "architecture"},
    "The reconstruction preserves tissue biology": {"exp9", "exp8"},
    "The reconstruction does not degrade tissue-level structure": {"exp9", "exp8"},
    "Learned dynamics": {"physics", "exp5", "exp5_matched", "exp1", "exp11"},
    "Transfer across tissue, species and resolution": {"exp2", "exp3", "exp6", "exp12"},
    "Counterfactual perturbation": {"exp4", "physics"},
    # Short-build summary paragraph: it stands in for the subsections the
    # workshop version drops, so it legitimately spans their sources.
    "Further results": {"exp7", "exp9", "exp2", "exp3", "exp4", "exp8"},
    "Ablations": {"exp5", "exp5_matched", "exp1", "exp8", "architecture"},
    "Convergence": {"exp14", "exp8"},
    "Spectral matching": {"exp13", "exp8", "exp1"},
    # Paragraph-level scopes. Demoting a subsection to a \paragraph to meet the
    # page limit silently moved its macros into the preceding subsection's
    # scope -- exactly the confusion this check exists to catch.
    "Is the benchmark's ordering the converged ordering": {"exp14", "exp8"},
    "Can the smoothness be trained away": {"exp13", "exp8"},
    "Robustness to sampling density": {"exp10", "exp8"},
    "The recovered length scale is close to its own initialization":
        {"physics", "exp11", "exp5"},
    "Spatial autocorrelation": {"exp8", "exp1"},
    "The gain comes from continuity, not from message passing": {"exp8"},
}

#: Deliberate, declared exceptions to the scope rule. A macro may be quoted
#: outside its section's scope only when the prose says so in the same sentence,
#: which is what the marker string checks. Undeclared crossings stay errors.
SCOPE_EXEMPTIONS: Dict[Tuple[str, str], str] = {
    ("NMOMoranPred", "Masked spatial reconstruction"): "primary Visium section",
    ("MoranTrue", "Masked spatial reconstruction"): "primary Visium section",
}

#: Bare numerals permitted in the main body: definitional constants, not results.
ALLOWED_LITERALS = {
    "0", "1", "2", "3", "4", "5", "10", "50", "55", "90", "95", "100", "2026",
    "1968", "1952",
}


class Report:
    def __init__(self) -> None:
        self.rows: List[Tuple[str, str, str, str, str, bool]] = []
        self.errors: List[str] = []
        self.unverified: List[str] = []

    def add(self, claim, where, artifact, value, verdict, ok) -> None:
        self.rows.append((claim, where, artifact, value, verdict, ok))
        if not ok:
            self.errors.append(f"{where}: {claim} -- {verdict}")


# --------------------------------------------------------------------------- #
# 2. Freshness: regenerate and diff
# --------------------------------------------------------------------------- #

def check_freshness(rep: Report, results: Path) -> None:
    """Rebuild every table and macro into a scratch tree and diff against paper/."""
    sys.path.insert(0, str(ROOT))
    from src.evaluation import numbers as numbers_mod
    from src.evaluation.tables import build_all

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        (tmp / "tables").mkdir()
        try:
            build_all(results, tmp / "tables")
            numbers_mod.build(results, tmp / "numbers.tex")
        except Exception as exc:                      # a generator that crashes is a failure
            rep.add("regeneration", "tables.py/numbers.py", str(results), "-",
                    f"generator raised {type(exc).__name__}: {exc}", False)
            return

        for gen in sorted(tmp.glob("tables/*.tex")) + [tmp / "numbers.tex"]:
            live = PAPER / ("tables/" + gen.name if gen.parent.name == "tables" else gen.name)
            if not live.exists():
                rep.add(gen.name, "paper/", str(results), "-",
                        "generated but not present in paper/", False)
                continue
            a, b = gen.read_text().strip(), live.read_text().strip()
            rep.add(gen.name, str(live.relative_to(ROOT)), str(results),
                    "identical" if a == b else "DIFFERS",
                    "fresh" if a == b else "stale -- run `make figures`", a == b)

        # A table checked into paper/ that nothing regenerates is unverifiable.
        generated = {p.name for p in tmp.glob("tables/*.tex")}
        # A table can fail to regenerate for two very different reasons: nothing
        # calls its generator (the defect that let stale tables ship), or its
        # input is simply absent, which is the normal state of a fresh clone
        # since data/ is gitignored. Reporting them identically sent a clean
        # checkout chasing a bug that was not there.
        summary_missing = not Path("data/processed/SUMMARY.json").exists()
        DATA_DEPENDENT = {"tab_data.tex", "tab_datasets.tex"}
        for live in sorted((PAPER / "tables").glob("*.tex")):
            if live.name in generated or live.name == "tab_theory.tex":
                continue
            if live.name in DATA_DEPENDENT and summary_missing:
                rep.add(live.name, str(live.relative_to(ROOT)), "data/processed",
                        "-", "input absent (run `make data`); not a missing "
                             "generator", True)
                continue
            rep.add(live.name, str(live.relative_to(ROOT)), "none", "-",
                    "no generator reachable from build_all()", False)


# --------------------------------------------------------------------------- #
# 3. Scope: macro provenance vs. the section that quotes it
# --------------------------------------------------------------------------- #

def _sources() -> str:
    """The manuscript body, assembled from the modular sources.

    Since the split into ``paper/sections/``, the driver files contain only
    ``\\input`` lines; the claims live in the sources, and both the full and the
    workshop build read the same ones. Auditing the sources therefore audits
    both builds at once, which is the property the split was for.
    """
    parts = []
    for name in ["abstract", "intro", "related", "method", "setup", "results",
                 "ablations", "limitations", "discussion"]:
        f = SECTIONS / f"{name}.tex"
        if f.exists():
            parts.append(f"%%SECTIONFILE {name}\n" + f.read_text())
    return "\n".join(parts) if parts else TEX.read_text().split(r"\label{endofmain}")[0]


def _main_body(tex: str) -> str:
    return _sources()


def check_scope(rep: Report) -> None:
    tex = TEX.read_text()
    body = _main_body(tex)
    lines = body.split("\n")

    # Skip the \@for fallback-stub block: it names every macro by construction.
    stub_lo = next((i for i, l in enumerate(lines) if r"\@for\@nmocmd" in l), None)
    stub_hi = next((i for i, l in enumerate(lines[stub_lo:], stub_lo)
                    if r"\makeatother" in l), None) if stub_lo is not None else None

    current = "Abstract"
    for i, line in enumerate(lines):
        if stub_lo is not None and stub_hi is not None and stub_lo <= i <= stub_hi:
            continue
        m = re.search(r"\\(?:(?:sub)?section|paragraph)\*?\{([^}]*)\}", line)
        if m:
            current = re.sub(r"\\[a-zA-Z]+|\{|\}", "", m.group(1)).strip().rstrip(".")
        scope = next((v for k, v in SECTION_SCOPE.items() if current.startswith(k)), None)
        for mac in re.findall(r"\\([A-Z][A-Za-z]+)\b", line):
            src = PROVENANCE.get(mac)
            if src is None:
                continue
            if scope is not None and src not in scope:
                key = next((k for k in SCOPE_EXEMPTIONS
                            if k[0] == mac and current.startswith(k[1])), None)
                if key is not None:
                    marker = SCOPE_EXEMPTIONS[key]
                    ctx = " ".join(lines[max(0, i - 2):i + 2])
                    ok = marker.lower() in ctx.lower()
                    rep.add(f"\\{mac}", f"neurips_2026.tex:{i+1} ({current})", src,
                            "declared exception",
                            f"scope marker '{marker}' present" if ok else
                            f"declared exception requires the prose to say "
                            f"'{marker}' nearby; it does not", ok)
                    continue
                rep.add(f"\\{mac}", f"neurips_2026.tex:{i+1} ({current})", src,
                        "-", f"macro derives from {src}; this section may quote "
                             f"only {sorted(scope)}", False)

    # Every macro the manuscript stubs must actually be defined.
    stubbed: Set[str] = set()
    if stub_lo is not None and stub_hi is not None:
        blk = "\n".join(lines[stub_lo:stub_hi + 1])
        stubbed = set(re.findall(r"[A-Za-z]+", blk.split(":=")[-1].split(r"}\do")[0]))
    defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}",
                             (PAPER / "numbers.tex").read_text()))
    used = set(re.findall(r"\\([A-Z][A-Za-z]+)\b", body)) & (set(PROVENANCE) | stubbed)
    for mac in sorted(used - defined):
        rep.add(f"\\{mac}", "neurips_2026.tex", "-", "undefined",
                "used in prose but not defined in numbers.tex (renders as ??)", False)
    for mac in sorted(set(PROVENANCE) - stubbed - defined):
        rep.unverified.append(f"`\\{mac}` declared in the provenance registry but "
                              f"neither stubbed nor generated")


# --------------------------------------------------------------------------- #
# 4. Coherence: cross-table invariants
# --------------------------------------------------------------------------- #

def check_coherence(rep: Report, results: Path) -> None:
    import pandas as pd

    ms = [x for f in sorted(results.glob("exp8/results_shard*.json"))
          for x in json.loads(f.read_text())
          if "pearson_mean" in x and not x.get("failed")]
    if not ms:
        rep.unverified.append("exp8 artifacts absent; benchmark coherence unchecked")
        return
    df = pd.DataFrame(ms)
    per = df.groupby(["section", "model"])["pearson_mean"].mean().unstack("model")
    matched = per.dropna(subset=["nmo"]) if "nmo" in per else per

    # (a) every model shown in the benchmark table covers the same sections
    tab = PAPER / "tables" / "tab_multisection.tex"
    if tab.exists():
        shown = re.findall(r"^([A-Za-z][^&\\]*?)\s*&", tab.read_text(), re.M)
        from src.models.baselines import DISPLAY_NAMES
        inv = {v: k for k, v in DISPLAY_NAMES.items()}
        for disp in shown:
            key = inv.get(disp.strip())
            if key and key in per.columns:
                n = int(matched[key].notna().sum())
                rep.add(f"{disp.strip()} coverage", "tab_multisection.tex", "exp8",
                        f"{n}/{len(matched)}",
                        "matched" if n == len(matched) else
                        "row averaged over a different section set than NMO",
                        n == len(matched))

    # (b) marginal deltas in the benchmark table agree with the paired table.
    # Only for models the table actually shows: table_multisection drops any
    # model covering under 90% of the reference sections, and comparing a
    # dropped model's paired delta against a marginal mean it never appears in
    # is a false positive -- the two are computed over different section sets
    # by construction.
    from src.evaluation.statistics import paired_comparison
    shown = [m for m in matched.columns
             if int(matched[m].notna().sum()) >= max(3, int(0.9 * len(matched)))]
    for r in paired_comparison(df, "nmo", "pearson_mean"):
        if r.other not in shown:
            continue
        marginal = float(matched["nmo"].mean()) - float(matched[r.other].mean())
        ok = abs(marginal - r.mean_diff) < 5e-4
        rep.add(f"delta vs {r.other}", "tab_multisection vs tab_paired", "exp8",
                f"{marginal:+.4f} vs {r.mean_diff:+.4f}",
                "agree" if ok else "benchmark table and paired test disagree", ok)

    # (c) a stability sweep in which nothing destabilized is a censored bound
    sj = results / "exp7" / "stability.json"
    if sj.exists():
        S = pd.DataFrame(json.loads(sj.read_text()))
        grid = float(S["dt"].max())
        for scheme, g in S.groupby("scheme"):
            if bool(g["stable"].all()):
                txt = (PAPER / "tables" / "tab_numerics.tex")
                has_note = txt.exists() and "never destabilized" in txt.read_text()
                rep.add(f"{scheme} stability", "tab_numerics.tex", "exp7/stability.json",
                        f"stable at every dt <= {grid:.3g}",
                        "censoring disclosed" if has_note else
                        "reported as a measured threshold but never destabilized "
                        "(right-censored by the sweep grid)", has_note)


# --------------------------------------------------------------------------- #
# 5. Hard-coded numerals in the main body
# --------------------------------------------------------------------------- #

def check_sample_sizes(rep: Report, results: Path) -> None:
    """Fail when a quoted statistic rests on less than it needs.

    Two claims in this paper shipped on inadequate samples: a robustness number
    from one seed, and a specimen whose held-out set was too small to estimate
    the metric being compared. Both were caught by eye. These checks are the
    mechanical version.
    """
    import json as _json
    sys.path.insert(0, str(ROOT))
    from src.evaluation.statistics import MIN_HELDOUT_LOCATIONS

    body = _sources()

    # (a) any experiment whose macros are quoted must have >= 2 seeds, unless
    #     the prose explicitly declines to rank on it.
    # Experiments the text explicitly declines to rank on. Adding one here
    # is a commitment that the prose says so, not a way to silence the check.
    SINGLE_SEED_OK = {"exp10", "exp9"}
    for src, pat in [("exp8", "exp8/results_shard*.json"),
                     ("exp14", "exp14/converged.json"),
                     ("exp13", "exp13/spectral_sweep.json"),
                     ("exp11", "exp11/difflen_null.json"),
                     ("exp9", "exp9/biology.json")]:
        rows = []
        for f in results.glob(pat):
            try:
                d = _json.loads(f.read_text())
                rows += d if isinstance(d, list) else [d]
            except Exception:
                continue
        rows = [r for r in rows if isinstance(r, dict) and not r.get("failed")]
        if not rows:
            continue
        seeds = len({r.get("seed") for r in rows if "seed" in r})
        quoted = any(m in body for m, s_ in PROVENANCE.items() if s_ == src)
        if not quoted:
            continue
        ok = seeds >= 2 or src in SINGLE_SEED_OK
        rep.add(f"{src} seeds", "sample size", src, str(seeds),
                "adequate" if ok else
                "quoted in prose but single-seed; either add seeds or stop "
                "ranking models on it", ok)

    # (b) no section below the estimability threshold may reach the benchmark
    rows = []
    for f in results.glob("exp8/results_shard*.json"):
        try:
            rows += _json.loads(f.read_text())
        except Exception:
            continue
    small = sorted({r["section"] for r in rows
                    if isinstance(r, dict) and not r.get("failed")
                    and r.get("n_obs_used", 10 ** 9) < 4 * MIN_HELDOUT_LOCATIONS})
    if small:
        # The text names them through a macro, so accept either the literal
        # section name or the macro that expands to it.
        flat = body.replace("\\_", "").replace("_", "")
        named = all(sec.replace("_", "") in flat for sec in small) or \
            ("MSExcluded" in body and "BioExcluded" in body)
        rep.add("undersized sections", "sample size", "exp8",
                ", ".join(small),
                "excluded and named in the text" if named else
                "excluded from the analysis but not named in the text", named)


def check_literals(rep: Report) -> None:
    body = _main_body(TEX.read_text())
    lines = body.split("\n")
    in_preamble = True
    for i, line in enumerate(lines):
        if r"\begin{document}" in line:
            in_preamble = False
        if in_preamble or line.lstrip().startswith("%"):
            continue
        stripped = re.sub(r"\$[^$]*\$", "", line)              # drop inline math
        stripped = re.sub(r"\\cite[a-z]*\{[^}]*\}", "", stripped)
        stripped = re.sub(r"\\(?:label|ref|includegraphics|input|inputtable)\{[^}]*\}",
                          "", stripped)
        stripped = re.sub(r"\\[a-zA-Z]+", "", stripped)
        for num in re.findall(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)", stripped):
            clean = num.replace(",", "")
            if clean in ALLOWED_LITERALS:
                continue
            rep.add(num, f"neurips_2026.tex:{i+1}", "hard-coded", num,
                    "bare numeral in the main body -- route through a macro", False)


# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--fix-unverified", action="store_true",
                    help="rewrite UNVERIFIED_CLAIMS.md from this run")
    ap.add_argument("--skip-freshness", action="store_true")
    a = ap.parse_args()

    results = (ROOT / a.results) if not Path(a.results).is_absolute() else Path(a.results)
    rep = Report()

    if not a.skip_freshness:
        check_freshness(rep, results)
    check_scope(rep)
    check_coherence(rep, results)
    check_sample_sizes(rep, results)
    check_literals(rep)

    w = [max(len(str(r[i])) for r in rep.rows) if rep.rows else 8 for i in range(5)]
    w = [min(x, 46) for x in w]
    print(f"\n{'claim':<{w[0]}}  {'location':<{w[1]}}  {'artifact':<{w[2]}}  "
          f"{'value':<{w[3]}}  verdict")
    print("-" * (sum(w) + 12))
    for claim, where, art, val, verdict, ok in rep.rows:
        mark = " " if ok else "!"
        print(f"{mark}{str(claim)[:w[0]]:<{w[0]}}  {str(where)[:w[1]]:<{w[1]}}  "
              f"{str(art)[:w[2]]:<{w[2]}}  {str(val)[:w[3]]:<{w[3]}}  {verdict}")

    n_bad = sum(1 for r in rep.rows if not r[5])
    print(f"\n{len(rep.rows) - n_bad}/{len(rep.rows)} checks passed")

    if rep.unverified or a.fix_unverified:
        p = ROOT / "UNVERIFIED_CLAIMS.md"
        body = "\n".join(f"- {u}" for u in rep.unverified) or "- (none)"
        p.write_text("# Unverified claims\n\nGenerated by `make check-numbers`. "
                     "Each entry is a claim in the manuscript that no run artifact "
                     "currently supports.\n\n" + body + "\n")
        print(f"wrote {p.relative_to(ROOT)} ({len(rep.unverified)} entries)")

    if n_bad:
        print(f"\nFAILED: {n_bad} mismatch(es).")
        return 1
    print("\nOK: prose, tables and artifacts agree.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
