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
