# Audit backups

Not stale clutter. These are result artifacts kept deliberately because a later
analysis needs to compare against them.

## `exp8_pre_evalfix/`

The exp8 shards as they stood before the eval-determinism fix (commit `bf13376`).
Four baselines resampled their conditioning set inside `forward()` with no
train/eval distinction, so a recorded score was one draw rather than a property
of the model; see `AUDIT_OUTCOMES.md`.

`scripts/eval_noise_sensitivity.py` reads these to measure the per-model shift
between the original and regenerated runs. Without them the noise scale has to be
assumed, and assuming it — taking one measurement from `gp` and applying it to
the other three — is the specific mistake these files exist to prevent repeating.

They live outside `results/` on purpose. `numbers.py` uses recursive globs like
`exp1/**/*.json` for several experiments, and 350 duplicate records one directory
below a results glob would silently double every count for whoever writes
`exp8/**/*.json` next.

Safe to delete once the paper is final and no further shift measurement is
needed. Until then, deleting them means the sensitivity analysis falls back to an
assumed scale.
