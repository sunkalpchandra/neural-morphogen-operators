# Unverified claims

Claims that appear, or could appear, in the manuscript without a run artifact
behind them. `make check-numbers` appends machine-detectable cases; the rest are
recorded by hand as they are found. Nothing on this list may be stated in the
paper as established.

## Open

- **Fig. 2 caption, gene-panel range and physical extent.** The per-panel counts
  are drawn at render time from the `.h5ad` objects, but `data/processed/SUMMARY.json`
  carries `n_genes: null` for every entry, so `check_numbers.py` cannot verify
  them independently. Fix: record `n_vars` and the bounding box in the summary at
  build time.

- **Spateo.** The related-work section calls it "closest in spirit". It has not
  been run, and on inspection it should not be: `spateo-release` is not a
  dependency here, and its core contributions -- 3-D reconstruction from serial
  sections, and morphometric vector fields fitted across developmental
  timepoints -- do not define a predictor for held-out locations on a single 2-D
  section. Adapting it would mean inventing a task its authors did not target
  and reporting the result under their name. The related-work claim is therefore
  a statement about intent, not a benchmarked comparison, and the text should
  say so rather than implying an unrun comparison was available.

- **gimVI.** Named in the revision plan as a candidate imputation baseline.
  Requires `scvi-tools`, which is not installed and pulls a substantial
  dependency tree; not attempted. Tangram-style and SpaGE-style cover the
  imputation mechanism (learned soft assignment; aligned-subspace neighbourhood
  regression) without the dependency.

## Resolved

- ~~Transfer caveat distances (19.5 / 116.6 um)~~ -- were hard-coded from an
  ad-hoc measurement; now produced by `experiments/exp12_split_geometry.py` and
  quoted through macros.
- ~~Operator parameter share (9,536 / 2,198,336 / 0.43%)~~ -- were hard-coded;
  now derived from the architecture in `src/evaluation/numbers.py`.
- ~~"374 um, the same order as measured morphogen gradients"~~ -- the
  shuffled-coordinate control in `experiments/exp11_difflen_null.py` recovers
  404.4 um against an untrained value of 404.6, so the figure is close to its
  initialization and the claim is not supported. Pending removal from the text.
