# Stage 0 — Consistency audit

> **Status: closed.** Every item below has been fixed, and `make check-numbers`
> now gates the classes of defect they belong to. See AUDIT_OUTCOMES.md for what
> each fix did to the claims. Kept as the record of how the defects were found.

Scope: every numeric claim in the main body of `paper/neurips_2026.tex`
(lines 1–683, up to `\label{endofmain}`) traced to the artifact that produces it.
No code or prose was changed.

---

## 1. How numbers actually flow

There are **three independent paths** from artifacts to the PDF, and they do not
agree with each other. This is the root cause of nearly every mismatch below.

```
results/exp1/**/*.json  ─┐                    (visium_mouse_brain ONLY, 3 seeds, 7 models)
results/exp8/results_*   ├─> src/evaluation/numbers.py ──> paper/numbers.tex ──> \macros in prose
results/exp7,9/*.json   ─┘

results/exp1  ──> tables.py::build_all ──> tab_benchmark, tab_datasets, tab_transfer_*,
                                           tab_ablations, tab_bead, tab_perturbseq, tab_development

results/exp8,7,9  ──> tables.py::table_multisection / table_paired_stats /
                      table_numerics / table_biology   ← ** NO CALLER **
```

`experiments/make_figures.py` (`make figures`) drives paths 1 and 2. Path 3 has
no caller at all.

### 1a. The regeneration property is broken for the headline results

| generator | called by | file in `paper/` |
|---|---|---|
| `table_multisection` | **nothing** | `tab_multisection.tex` = **Table 1** |
| `table_paired_stats` | **nothing** | `tab_paired.tex` = Table 3 |
| `table_numerics` | **nothing** | `tab_numerics.tex` = Table 4 |
| `table_biology` | **nothing** | `tab_biology.tex` = Table 5 |
| `table_robustness` | **nothing** | (not in paper) |
| `figure_multisection` | **nothing** | `fig_multisection.pdf` |
| `figure_biology` | **nothing** | `fig_biology.pdf` |
| `figure_numerics` | **nothing** | `fig_numerics.pdf` |
| `figure_datasets` | **nothing** | `fig_datasets.pdf` |
| `figure_results_panel` | **nothing** | `fig_results_panel.pdf` |

Verified: `build_all` (`tables.py:424–468`) returns before those functions are
defined (`:487+`), and `make_figures.py` never imports them. Same class of bug as
the `BASELINES` registry ordering fixed earlier.

**Consequence:** four tables and five figures — including every table carrying a
headline claim — are stale checked-in binaries/`.tex` that `make figures` does not
reproduce. The single-command regeneration property does **not** currently hold
for the main results. `make check-numbers` cannot be trusted until this is fixed,
because it would be checking prose against files nothing regenerates.

### 1b. Confirming your hypothesis

**Confirmed, with a correction to the mechanism.** You suspected Figure 7 values
leaked into 17-section prose. The leak is real but the source is `results/exp1`
directly, not the figure: `numbers.py:48–105` builds `\AEPearson`, `\NMOSSIM`,
`\NMOMoran`, `\BestAnySSIM`, `\BestAnyMoran`, `\NMOMoranPred`, `\MoranTrue`,
`\NSeeds`, `\NGraphModels`, `\{Best,Worst}GraphOverAE`, `\NMOOverAE` from
`exp1/**` — which contains exactly one section (`visium_mouse_brain`), 3 seeds,
7 models. Figure 7 (`fig_results_panel`) draws on the same source, so it agrees
with the prose; **Table 1 is the odd one out** because it came from `exp8`.

When the benchmark moved from 1 section to 17, `numbers.py` gained `MS*` macros
reading `exp8`, but the `exp1`-derived macros were never re-pointed or removed.

---

## 2. Mismatch table

Severity: **A** = wrong number in prose · **B** = defect in the artifact/table
itself · **C** = undeclared scope · **D** = numerics/rendering · **E** = infra.

### A — single-section values in prose describing the 17-section benchmark

All of these sit inside §5.1 ("Masked spatial reconstruction across 17 sections").

| # | .tex line | claim as printed | macro | artifact | artifact value | Table 1 (exp8) | status |
|---|---|---|---|---|---|---|---|
| A1 | 434 | autoencoder "reaches 0.230" | `\AEPearson` | `results/exp1` | 0.2296 | 0.153 | **MISMATCH** |
| A2 | 436–437 | graph-over-AE margin "−0.002 to 0.005" | `\WorstGraphOverAE`, `\BestGraphOverAE` | `results/exp1` | −0.002 / +0.005 | +0.001 / +0.001 | **MISMATCH** |
| A3 | 437 | NMO over AE "0.017" | `\NMOOverAE` | `results/exp1` | +0.0174 | +0.0177 | value coincides; **provenance wrong** |
| A4 | 436 | "across 4 graph models" | `\NGraphModels` | `results/exp1` | 4 | exp8 has 2 (gnn, stagate) | **MISMATCH** |
| A5 | 446 | SSIM "0.404 ± 0.006 against 0.394" | `\NMOSSIM`, `\BestAnySSIM` | `results/exp1` | 0.4036 / 0.3936 (stagate) | 0.316 ± 0.043 / 0.307 (gnn) | **MISMATCH** |
| A6 | 447 | Moran "0.762 ± 0.004 against 0.718" | `\NMOMoran`, `\BestAnyMoran` | `results/exp1` | 0.7615 / 0.7180 (gp) | 0.861 ± 0.039 / **0.625 (SIREN)** | **MISMATCH — and worse than stated** |
| A7 | 448 | "I = 0.936 vs 0.174" | `\NMOMoranPred`, `\MoranTrue` | `results/exp1` | 0.936 / 0.174 | not measured in exp8 | scope undeclared |
| A8 | 395 | "mean ± s.d. over 3 seeds" | `\NSeeds` | `results/exp1` | 3 | exp8 uses **2** | **MISMATCH for §5.1** |

**A6 is the one that moves against the paper.** At 17 sections the best `|ΔI|` is
the SIREN neural field at 0.625, not 0.718. The over-smoothing deficit the paper
concedes is 0.044 is actually **0.236** — 5× larger. Stage 4 should be sized
against 0.236.

| # | .tex line | claim | what's wrong |
|---|---|---|---|
| **A9** | 440 | "An implicit neural field … is the weakest model tested (Δr = `\MSMinDelta` at minimum against NMO)" | `\MSMinDelta` = 0.017 is the **STAGATE** delta (`numbers.py:257` takes `min(mean_diff)` over the family). The neural field's actual Δr is **+0.0652** (Table 3). Wrong number, attached to the wrong model, and "at minimum" inverts the intended sense. **Not on your list.** |

### B — aggregation defects inside the artifacts

| # | where | finding |
|---|---|---|
| **B1** | `tables.py:499` | Table 1 rows are averaged over **different section sets per model**. NMO ran on 15 sections; the graph/SIREN baselines on 17; GP on 1. Nothing restricts to the common set. |
| **B2** | 420–421 | `\MSNMOPearson` (0.161, n=15) is compared against `\MSBestBaselinePearson` (0.154, n=17) in the same sentence. |
| **B3** | Table 1, GP row | GP ran on **one** section (`merfish_allen_01`) → `std` undefined → no `±`; `paired_comparison` requires n ≥ 3 → no stars. **Root cause of your item 8.** GP is effectively unbenchmarked but is presented as a benchmarked row. |
| **B4** | `tables.py:524` | Caption win-note is truncated to `[:4]`, so STAGATE is missing from the caption while starred in the table. |
| **B5** | abstract, 402, 417, 642, 663 | "17 sections" everywhere; **NMO has 15**. Missing: `visium_mouse_brain`, `xenium_mouse_brain`. |
| **B6** | abstract 130 / §7 643 | "5 independent specimens" vs "3 complete pairs" — **root cause found, see below**. |

**B1 quantified** (Pearson *r*, mean over all sections vs. mean over NMO's 15):

| model | all sections | NMO's 15 | bias |
|---|---|---|---|
| STAGATE-style | 0.1539 | 0.1447 | **+0.0092** |
| GNN (GCN) | 0.1530 | 0.1447 | +0.0083 |
| Autoencoder | 0.1526 | 0.1435 | +0.0091 |
| Neural field | 0.1063 | 0.0960 | +0.0103 |
| NMO | 0.1612 | 0.1612 | — |

The two sections NMO is missing are the **two easiest** in the pool (baselines
score 0.21–0.23 there vs ~0.145 elsewhere), so the unequal aggregation
**understates NMO**. Table 1 shows a 0.007 gap; the paired test on matched
sections gives **+0.0165**. Tables 1 and 3 are mutually inconsistent, and a
reviewer will notice. Fixing B1 helps the paper — I flag it as a defect either
way, since the aggregation is wrong regardless of which direction it errs.

**B6 quantified — the highest-value finding in this audit.** Specimen coverage:

| specimen | AE | GNN | STAGATE | SIREN | GP-mb | **NMO** |
|---|---|---|---|---|---|---|
| MERFISH C57BL6J-638850 | 0.157 | 0.159 | 0.158 | 0.105 | 0.126 | **0.178** |
| Stereo-seq embryo series | 0.088 | 0.087 | 0.088 | 0.064 | 0.074 | **0.096** |
| visium_human_breast | 0.096 | 0.093 | 0.095 | 0.057 | 0.058 | **0.087** ← loses |
| visium_mouse_brain | 0.213 | 0.210 | 0.225 | 0.185 | — | **absent** |
| xenium_mouse_brain | 0.229 | 0.222 | 0.222 | 0.182 | — | **absent** |

The specimen-level test has n = 3 **because NMO was never run on two sections**,
each of which is its own specimen — not because of biology. Filling those cells
is **4 runs** (2 sections × 2 seeds). That takes specimen-level n from 3 → 5 and
the smallest attainable Wilcoxon *p* from 0.25 → **0.0625**, directly repairing
the paper's most attackable statistic. This is far cheaper than Stages 2–5 and I
recommend doing it first.

I make no prediction about the outcome. NMO already loses one of its three
specimens (`visium_human_breast`), and the two missing sections are the ones
where baselines do best — the result could go either way.

### C — undeclared scope

| # | line | finding |
|---|---|---|
| C1 | 123, 509–510 | "374 µm median (10–90th pct 295–478)" comes from `results/physics.json`: **3 seeds of one section** (`visium_mouse_brain`), 192 values = 32 latent channels × 2 eigenvalues. The percentile range is spread **across latent channels**, not across sections, seeds or tissues. Prose does not say so. Directly relevant to Stage 2. |
| C2 | 364 | `\NSections` = 18 and `\TotalLocations` = 1,080,453 **include `perturb_norman`** — 111,659 non-spatial cells (10.3%). Spatial only: **17 sections, 968,794 locations**. The sentence does disclose the screen, then folds it into the section and location counts. |
| C3 | 402 | Subsection title hard-codes "17 sections" rather than using `\MSSections`. |
| C4 | 601–602 | "9,536 of 2,198,336 parameters (0.43%)" hard-coded. **9,536 verified exactly**; 0.43% consistent. Not macro-ized, so it will silently rot. |
| C5 | 357–359 | Fig. 2 caption hard-codes "2.7k Visium spots to 104k MERFISH cells", "248–2087", "2–10 mm". 2.7k ✓ (2,691) and 104k ✓ (104,461) verified against `SUMMARY.json`; **gene-panel range and physical extent are not recoverable from `SUMMARY.json`** (`n_genes` is null for every entry) → goes on `UNVERIFIED_CLAIMS.md`. |
| C6 | abstract 141 | "retaining 0.518 of the spatial-domain structure" drops the `± 0.276` that exists in the artifact. (Also Stage 6b.) |

### D — numerics and rendering

| # | line | finding |
|---|---|---|
| **D1** | 461–462 | "remains stable to Δt = 0.501, which is 41× the CFL bound". **strang-spectral never destabilized**: `n_unstable = 0`, and 0.501187 is the top of the sweep grid. The measurement is **right-censored** — the true statement is "stable at every step tested, ≥ 41× the CFL bound". Consistent with the unconditional-stability theory, but reported as a measured threshold when it is a grid ceiling. |
| **D2** | Table 4 "vs CFL" | Format bug, `tables.py:629,631` — `{top/cfl:.0f}×` renders 0.249 → "0×", 1.37 → "1×", 0.77 → "1×". Only the strang-spectral row survives the rounding. **Your item 7: the 0× is a rendering artifact, not a contradicted claim.** The 41× and 32× prose claims both trace correctly to `stability.json`/`cost.json`. |
| **D3** | 463–465 | "At matched accuracy … 4 steps against 128, a 32× difference". Step counts verified. But accuracy is **not** matched: strang rel-err 8.4e-3, euler 2.3e-3 — euler overshoots by 3.6× at its step count, on a coarse doubling grid. Defensible claim: "to reach a 10⁻² tolerance". As written, "at matched accuracy" is an overstatement, and 32× is an upper bound. Appears twice (465, 671) and in the abstract (138). |
| D4 | Table 2 | P1–P7 verified to map to **Propositions 4–10** exactly: P1→4 (spd), P2→5 (contraction), P3→6 (mass), P4→7 (reaction), P5→8 (order), P6→9 (bounded), P7→10 (vacuity). Mapping is **correct but undocumented**; `verify_theory.py` should emit `\ref{prop:...}`. Numbering confirmed against the compiled PDF (`amsthm` shares one counter across theorem/proposition/definition/assumption/remark, which is why the visible numbers start at 4). |

### E — infrastructure

| # | finding |
|---|---|
| E1 | §1a: 4 tables + 5 figures have no caller. `make figures` does not regenerate the main results. |
| E2 | `paper/*.aux` is absent, so `finalize.sh`'s page-count report silently yields `?` on a clean tree. |

---

## 3. What I did **not** find

Checked and clean:

- All `MS*` macros trace correctly to `results/exp8` (given the n=15/17 caveat).
- `\MatchedNoDynamics` (−0.023), `\MatchedNoReaction` (−0.006), `\MatchedSeeds` — trace to `results/exp5_matched`.
- `\NumStableDt`, `\NumCFL`, `\NumEulerSteps`, `\NumStrangSteps`, `\NumSpeedup` — trace to `results/exp7` (subject to D1/D3 wording).
- `\Bio*` — trace to `results/exp9/biology.json`.
- `\Bead*`, `\PS*`, `\Dev*`, `\Abl*`, `\Tissue*`, `\Res*` — trace correctly.
- `\MSSpecContDz` (1.86), `\MSSpecGraphDz` (0.45–0.58) — trace to the specimen-level test.
- Main body is otherwise macro-disciplined: only C3/C4/C5 hard-code numbers.
- Theorem/proposition cross-references resolve; no `??` in the built PDF.

---

## 4. `make check-numbers` — design (not yet implemented, per your hold on code)

`scripts/check_numbers.py`, wired as `check-numbers:` in the Makefile, run in CI
and after every prose edit. Four independent checks:

1. **Regenerate and diff.** Rebuild `numbers.tex` and all tables into a temp dir
   from `results/`; fail on any diff against `paper/`. Catches stale artifacts.
   *Requires E1 fixed first, or it cannot see the four headline tables.*
2. **Macro provenance registry.** A declared map `macro → (artifact glob, section
   scope)`. Every macro used inside a `\section` whose declared scope differs from
   the macro's source is an error. This is the check that would have caught A1–A8
   at the moment the benchmark moved from 1 section to 17.
3. **Cross-table coherence.** Assert Table 1's per-model Δ against NMO agrees with
   Table 3's paired Δ to within a tolerance; assert every model's row is averaged
   over the same section set; assert every macro named in the `\@for` stub list is
   defined in `numbers.tex`. Catches B1, B2, B3, D2.
4. **Hard-coded number sweep.** Flag any bare numeral in the main body not inside
   a macro, math environment or citation, against an explicit allow-list. Catches
   C3–C5 and stops recurrence.

Output: the same claim → location → artifact → value → verdict table as §2, plus
a non-zero exit code. Unresolvable claims append to `UNVERIFIED_CLAIMS.md` rather
than being quietly dropped.

---

## 5. Recommended order (for your decision)

1. **E1** — restore regeneration for the 4 tables + 5 figures. Everything else is
   unverifiable until this is done.
2. **B5/B6** — 4 NMO runs on `visium_mouse_brain` + `xenium_mouse_brain`. Cheapest
   experiment in the whole plan; takes specimen-level n from 3 → 5.
3. **B1** — restrict Table 1 to the common section set (or report per-model n).
4. **A1–A9, C1–C6, D1–D3** — prose diffs, after 1–3 change the numbers.
5. Then Stage 1 onward.

Nothing in 1–4 requires a long job except step 2 (4 runs).
