# What the audit changed

One line per claim: what this revision established, corrected or withdrew,
ordered by how much each moved. Written so the next reader — including the
author — can see at a glance which claims died and which survived, without
reading the commit log.

## Withdrawn

| claim | was | is | evidence |
|---|---|---|---|
| The diffusion length is biologically meaningful | "374 µm, the same order as measured morphogen gradients" | Close to its own initialization. Coordinate-shuffled sections carrying no spatial signal recover 404 µm against 405 µm untrained; real tissue gives 375 µm | `exp11_difflen_null.py` |
| Zero-shot transfer works | "a substantial part of the learned structure survives the domain change" | Cross-tissue zero-shot is −0.005 against a 0.000 floor — at or below predicting the mean — and every baseline transfers better | `exp2` under the block protocol |
| Spectral matching would fix the over-smoothing | deferred to future work as "the natural remedy" | It bounds the problem rather than fixing it: predicted autocorrelation bottoms out at 0.869 against a measured 0.174 at every weight and both forms | `exp13_spectral.py` |

## Corrected

| claim | was | is |
|---|---|---|
| Autoencoder baseline | 0.230 | 0.153 — the 0.230 was a single-section value quoted in 17-section prose |
| Message passing over the autoencoder | 0.005 | 0.001 |
| SSIM | 0.404 vs 0.394 | 0.320 vs 0.307 |
| Moran's *I* deficit | 0.044 | **0.229** — the best comparator across 17 sections is the SIREN field, not the single-section GP |
| Neural-field gap | 0.017 | 0.062 — 0.017 was STAGATE's delta, attached to the wrong model |
| Stability vs CFL | "remains stable to 41×" | "≥ 41×" — the scheme never destabilized, so the sweep grid censors the measurement |
| Integrator cost | "32× at matched accuracy" | 32× to reach a 10⁻² tolerance; the accuracies are not matched |

## Strengthened

| claim | was | is |
|---|---|---|
| Specimen-level analysis | 3 paired specimens, smallest attainable *p* = 0.25 | **5 specimens**, smallest attainable *p* = 0.0625; wins on all five against the continuous baselines |
| Benchmark coverage | NRDO on 15 of 17 sections; rows averaged over different section sets per model | 17/17, every row on a matched set, marginal means agreeing with the paired test |
| "The gain is the operator, not message passing" | held only at the reduced training budget | survives convergence — the non-spatial autoencoder still ties the graph models when every model is trained to early stopping |
| Robustness to sampling density | never ran (OOM-killed at six workers) | 97% retention at ⅛ density against 85% for the best baseline — the axis the design predicted in advance |

## Unresolved

- Separation from **graph** baselines at the specimen level: *d_z* = 0.32–0.81, ahead on 3 of 5. More independent tissues, not more sections of one brain, is what would settle it.
- The converged comparison covers **one** section.
- No GPU available, so no foundation-model baselines (scGPT, Nicheformer).
- Spateo is cited as "closest in spirit" but not benchmarked; its core targets a different task. See `UNVERIFIED_CLAIMS.md`.

## Infrastructure defects found along the way

- **Four tables and five figures had no caller.** `build_all` returned before they were defined and `make_figures.py` never imported them, so `make figures` left every headline table stale. This is the mechanism by which single-section numbers reached 17-section prose.
- `df.get("mode", "full")` returns the *column* when it exists, so records predating a new field kept `NaN` and `groupby` dropped them silently — a figure rendered with half its data and looked plausible.
- Metrics with no `METRIC_LABELS` entry reached LaTeX as raw column names and broke math mode on the underscore, twice. `metric_label()` now escapes unknown keys.
- `finetune_decoder` froze parameters by name prefix; the exact GP has no decoder, so everything froze and `backward()` failed with an opaque autograd error.
- The 9-page overflow was **Table 1's float placement**, not prose volume — and its caption had grown because of two of this audit's own corrections.

`make check-numbers` now gates the first class of defect: freshness by
regenerate-and-diff, macro provenance against a declared registry, cross-table
coherence, and a hard-coded-numeral sweep.

### Held-out scores for three baselines were partly a random draw (found 2026-08-02)

`GPSpatialBaseline`, `MultiScaleGPBaseline`, `TangramStyleBaseline` and
`SpaGEStyleBaseline` subsample their conditioning set with `torch.randperm` inside
`forward`, with no train/eval distinction. Every scoring pass therefore drew a new
subset, and the returned metric depended on global RNG state at the moment of the
call rather than on the checkpoint alone.

Measured on one `gp` checkpoint scored ten times: Pearson r ranged 0.1383--0.1611,
a spread of 0.0227. The across-seed s.d. for the same model is smaller than that,
so the error bars reported for these baselines understate their true variability,
and best-checkpoint selection on validation was partly selecting which subset the
model happened to draw.

Fixed: `_inducing_subset()` resamples in training and uses a fixed subset in eval.
Verified spread is exactly 0 after the change. Pinned by
`test_baseline_scoring_does_not_depend_on_global_rng_state`.

**Affects 99 recorded runs** (gp 19, spage 34, tangram 46), which predate the fix
and are queued for regeneration. This does not change any ranking in the paper --
the GP sits at ~0.15 against NRDO's 0.245, far outside the wobble -- but the
numbers on disk do not yet match the code that would produce them.

**Sensitivity analysis (2026-08-02).** Regenerating the affected runs means
retraining, since the exp8 shards keep no checkpoints. Before spending that
compute we measured whether it could matter: perturbing the four affected
models' per-specimen scores by the measured noise (s.d. 0.0059) and re-running
the full Holm-corrected comparison 2000 times. None of the three reported
significant comparisons changes verdict in any resample. One non-significant
comparison, tangram at p_holm = 0.0781, flips 19.75% of the time.

So the defect cannot have manufactured a result the paper reports; at worst it
masks a fourth. The published numbers are conservative with respect to it, and
regenerating those runs can only add a finding. Reproduced by
`scripts/eval_noise_sensitivity.py`.

### Two audit checks were no-ops (found 2026-08-02)

Splitting the manuscript into `preamble.tex` + `sections/` broke two of the four
checks in `scripts/check_numbers.py`, and both continued to report themselves as
passing.

`check_scope` parsed the `\@for\@nmocmd` stub block out of the section files.
That block is in the preamble, which `_main_body()` deliberately excludes, so the
set of stubbed macros was always empty and the provenance comparison ran against
nothing.

`check_literals` was worse. It initialises `in_preamble = True` and clears the
flag on `\begin{document}`, which never appears in the concatenated section
files, so every line was skipped. The hard-coded-numeral sweep -- the check whose
entire purpose is preventing a number being typed directly into the prose -- had
scanned zero lines since the split.

Both repaired and both now probed by injection in `tests/test_check_numbers.py`:
a bare `0.7391` added to `results.tex` is flagged, and the stub parser is
asserted to recover more than 100 stubs. On the clean tree the sweep finds
nothing, which is the correct answer -- the prose does route every number through
a macro.

One assertion in that new file was itself vacuous on first draft (`rep.rows`
holds tuples, so `getattr(row, "ok", True)` was always `True`). Caught and fixed
before commit, and recorded in the test.

### Gene panel is selected transductively (found 2026-08-02)

HVG selection runs before the split, so the 2000-gene panel is computed from
every location including the held-out blocks. Recomputing it on the training
split alone reproduces 82--84% of it across two sections: up to 365 genes depend
on having seen held-out data. Applied identically to every model, so it cannot
bias the comparison, but it does mean absolute accuracies are transductive in
feature selection. Now stated in the setup section rather than left for a reader
to discover. Measured by `scripts/hvg_leakage.py`.

### Undefined genes leave every mean silently, but do not bias the comparison

`pearson_per_gene` returns NaN for a gene with no variance across the held-out
locations, and `evaluate_prediction` aggregates with `_nanmean`, so those genes
leave the reported mean without being counted anywhere: `n_genes` in the result
dict is `pred.shape[1]`, the total, not the number scored. On the primary section
78 of 2079 genes (3.8%) are undefined.

The condition triggers on either argument, so a model that predicted a constant
for a hard gene would have that gene *excluded* rather than scored near zero,
which would inflate its mean. Checked directly across seven checkpoints on the
primary section: every model excludes the same 78 genes, all from the truth
having no held-out variance. Only the exact GP contributes any of its own, and it
contributes 2 (0.1% of the panel) in the weakest model in the benchmark.

So this is a reporting gap rather than a fairness problem, and no published
comparison is affected. Recording `n_genes_scored` is queued as task 101; it was
deferred rather than applied immediately so that the two halves of an
in-progress regeneration could not disagree about what they recorded.

### Eligibility rules were applied by some consumers and not others (found 2026-08-02)

Two rules govern which records are analysable: a section's held-out set must be
large enough to estimate Pearson r, and a section's measured ARI must be large
enough to divide by. Both were implemented at each point of use rather than at
the point of loading, so every consumer had to remember them independently.

Six did not. `tab_multisection`, `tab_paired`, `fig_multisection` and
`tab_sample_sizes` omitted the size rule; `table_biology` and `figure_biology`
omitted the ARI rule. `numbers.py` applied both, so the prose and the tables it
cites were computed under different rules from the same artifacts.

The size-rule consequence was a count mismatch: the benchmark table and figure
covered 23 sections under captions reading 22, and the section-level Wilcoxon
tests included `visium_human_heart`, which the limitations section names as
excluded.

The ARI consequence was material and flattering. With `visium_human_heart`
included (measured ARI 0.009) the retention column spanned -6.86..+6.90:

| model | table showed | eligible only |
|---|---|---|
| NRDO | 1.127 | 0.485 |
| GNN | -0.256 | 0.478 |
| STAGATE | -0.135 | 0.463 |
| GP, multi-bandwidth | -0.316 | 0.430 |

So Table 8 showed NRDO retaining more structure than the reference it divides by,
with every baseline negative, while the prose citing that table said 0.485
against 0.478 and stated plainly that NRDO does not separate from the graph
baselines. The prose was right; the table was not.

Fixed by `size_eligible_frame()` and `ari_eligible_frame()` in `statistics.py`,
with all six call sites routed through them, and guards that were each verified
to fail with the rule removed.

### Correction: the eval-noise sensitivity analysis used too small a noise scale

The sensitivity analysis (`scripts/eval_noise_sensitivity.py`) perturbed all four
affected models by s.d. 0.0059, measured by scoring one gp checkpoint ten times.
I wrote that applying the gp scale to the other three "likely overstates it for
the others and makes the conclusion conservative."

That was wrong, though not by as much as I first said. Comparing runs before and
after the fix gives a measured s.d. of 0.0087 for tangram (n=46) and 0.0150 for
spage (n=22 so far), against the 0.0059 assumed -- so about 1.5x and 2.5x, not
the eight I initially claimed. That first figure compared the largest single
shift (0.0484) against an s.d., which is not a comparison of like with like;
0.0484 is roughly a 3-sigma draw from s.d. 0.0150 and is what a tail looks like,
not evidence of an eightfold error.

The substantive point survives: the gp measurement was not an upper bound on the
family, and one model's noise does not stand in for another's.

Worse, `gp_multiscale` -- one of the three comparisons the paper reports as
significant, and one of the four affected models -- was deliberately left out of
the regeneration on the grounds that its flip rate was 0% under the assumed
scale. That reasoning is now circular: the 0% came from the assumption the
regeneration has just falsified. There is still no measurement of its noise.

Two caveats on the comparison itself. The old-vs-new difference conflates the
eval-determinism fix with a change in which checkpoint validation selects, since
per-epoch validation also called the nondeterministic subsample and consumed RNG
state that subsequent training draws depended on. So it is an upper bound on the
eval effect alone, not a clean measurement of it. And spage's absolute values are
small (0.00--0.08), so a shift of 0.048 is large relative to the model but small
relative to the 0.19 the benchmark reference reaches.

The conclusion that no reported significant result changes verdict has to be
re-derived from the measured per-model shifts rather than from a single assumed
scale. Queued for once the regeneration completes; the honest position until then
is that the earlier conclusion rested on an assumption now shown to be wrong for
at least one of the four models.
