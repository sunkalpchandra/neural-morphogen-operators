# Further work

Ordered by value, not by area. Items 1–20 would change what the paper can claim.
21–50 harden what it already claims. 51–100 are maintenance and polish, listed
because they were asked for, and honestly labelled: several are low-value and a
few are things I would not do unprompted.

## Tier 1 — changes what the paper can claim (1–20)

- [ ] 1. exp8 with a third seed: every specimen-level p is currently floored by n
- [ ] 2. exp9 biology with 2+ seeds — the seeds guard flags it as single-seed
- [ ] 3. exp10 noise/dropout/knn with 2+ seeds, so the withdrawn ranking can return
- [ ] 4. Two more independent specimens (target 12): STAGATE is 7/10 at p=0.43
- [ ] 5. SpaGCN-style and graph transformer are absent from the 22-section benchmark
- [ ] 6. Repeated splits per section — every result rests on one block split
- [ ] 7. Converged comparison on a second section (currently one)
- [ ] 8. Ablations at converged budget beyond the two decisive variants
- [ ] 9. Hyperparameter sensitivity: latent channels, horizon T, lattice size
- [ ] 10. Learning-curve: accuracy vs training-set fraction, per model
- [x] 11. Per-gene analysis: which genes NRDO wins and loses on, and why — advantage tracks spatial structure (corr 0.22-0.30), vanishes on unstructured genes
- [x] 12. Error-vs-distance computed: NRDO last in the nearest quartile, +50% in the farthest
- [ ] 13. Runtime and peak memory per model, reported alongside accuracy
- [ ] 14. Ablate the occupancy channel (claimed essential, never tested)
- [x] 15. Ablate the aux_z0 term (claimed to keep the encoder conditioned) — inert: 0.0006, below the noise floor
- [x] 16. Verified: decoder reads the field at query coords, not a coordinate map
- [x] 17. Bandwidth trains but moves only ~4% from init; exp15 tests whether it matters
- [x] 18. k-NN size varied properly (edges 23006 -> 12632); changes r by 0.0001
- [x] 19. A permutation test for the specimen-level result, not just Wilcoxon — agrees 7/7
- [x] 20. Power analysis: how many specimens would resolve STAGATE — ~49, a 5x larger study

## Tier 2 — hardens existing claims (21–50)

- [x] 21. Seed-noise floor computed per model; every margin is 1.9-7.3x it, smallest for STAGATE
- [x] 22. Verify every checkpoint loads and reproduces its recorded metrics — 20/21 exact; gp was RNG-dependent at eval, fixed
- [x] 23. Split determinism tested: same section, two loads, identical splits and coords
- [x] 24. Held-out blocks verified contiguous — measurably farther from training data than a random subset of equal size
- [x] 25. Standardisation verified train-only: train mean 0, sd 1; held-out splits differ
- [x] 26. Confirm no gene-selection leakage across the split — found: HVG precedes the split, 16-18% of the panel needs held-out data, now disclosed
- [x] 27. Isotropic normalisation verified on every section
- [x] 28. SPD verified on fitted weights: lambda_min 8.0e-03 against a floor of eps
- [x] 29. Verify the Strang order empirically at more than one dt range — 1.984 and 2.002
- [x] 30. Mass conservation verified at dt up to 50 on fitted operators
- [ ] 31. Check the absorbing-set bound against a longer trajectory
- [x] 32. Propositions re-checked on the 3 fitted operators, not only on random ones
- [x] 33. verify_theory_trained.py: all 3 trained operators satisfy every proposition
- [ ] 34. Confirm the dispersion relation code matches the analytic Jacobian
- [x] 35. Moran's I matches the definition to 2e-17
- [x] 36. Geary's C matches the definition to 3e-15
- [x] 37. ARI/NMI come from sklearn directly
- [x] 38. SSIM matches skimage to 1e-4 and orders degradations identically
- [x] 39. Confirmed percentile, not normal-approximation (asymmetric on skewed data)
- [x] 40. Holm matches statsmodels exactly on the paper's own p-values
- [x] 41. Specimen-collapse unit tested
- [x] 42. MIN_HELDOUT and MIN_REFERENCE_ARI guards unit tested against the sections that motivated them
- [x] 43. Unit test that check_numbers fails on an injected mismatch — found check_literals was dead
- [x] 44. Unit test for the provenance registry — found the stub parser was dead
- [x] 45. Regression pin on the headline macros — a tripwire, not a correctness check
- [x] 46. Gene-permutation equivariance tested
- [x] 47. Location-permutation invariance tested
- [ ] 48. Test the loaders against a truncated/corrupt file
- [ ] 49. Test the build guard fires on a mis-registered dataset
- [ ] 50. Test the figure code on empty and single-row inputs

## Tier 3 — maintenance and polish (51–100)

Lower value. Included because asked for; several are cosmetic and a few
(marked *) I would skip unless someone specifically wants them.

- [ ] 51. Type annotations complete across src/
- [ ] 52. Docstrings on every public function
- [ ] 53. Consistent error messages with actionable text
- [ ] 54. Remove the 169 orphaned macros if numbers.py can drop them cleanly
- [ ] 55. Split results.tex, which is the largest section file
- [ ] 56. Consistent citation style across the bibliography
- [ ] 57. Alphabetise and de-duplicate references.bib
- [ ] 58. Check every URL in the bibliography resolves
- [ ] 59. Add DOIs where missing
- [ ] 60. environment.yml pinned to exact versions
- [ ] 61. requirements.txt matches environment.yml
- [ ] 62. Document minimum RAM and expected wall-clock
- [x] 63. `make test` runs in 16s (slow data tests split out under `make test-data`)
- [ ] 64. Cache-friendly ordering in the download script
- [ ] 65. Resumability check for every long-running experiment
- [ ] 66. Consistent logging levels
- [ ] 67. Progress reporting for the slowest experiments
- [ ] 68. Clean up the ad-hoc shell scripts into one runner
- [ ] 69. Remove logs_*.txt from the working tree
- [ ] 70. .gitignore audit
- [ ] 71. LICENSE headers where required
- [ ] 72. Contribution notes for the repo
- [ ] 73. Architecture diagram of the codebase *
- [ ] 74. Docstring examples that are doctested
- [ ] 75. Consistent naming: nmo vs nrdo in internal identifiers
- [ ] 76. Config validation with clear failures on typos
- [ ] 77. Seed handling audit across numpy/torch/python
- [ ] 78. Device handling audit (cpu/mps/cuda paths)
- [ ] 79. Remove dead code paths
- [ ] 80. Reduce duplication between exp2 and exp3
- [ ] 81. Factor the shared training loop out of the experiment scripts
- [ ] 82. Make figure sizes configurable rather than hard-coded
- [ ] 83. A single style constant for every font size
- [ ] 84. Colour-blind simulation check on the final figures
- [ ] 85. Greyscale legibility check
- [ ] 86. Alt-text for figures *
- [ ] 87. Consistent decimal places across tables
- [ ] 88. Table column alignment audit
- [ ] 89. Caption length consistency
- [ ] 90. Verify no widows or orphans in the built PDF *
- [ ] 91. Check hyphenation in the built PDF *
- [ ] 92. Spell-check the manuscript
- [ ] 93. Grammar pass
- [x] 94. American English enforced by check-numbers, not by a one-off pass
- [x] 95. One British spelling found and fixed; the guard prevents regression
- [ ] 96. Consistent tense across sections
- [ ] 97. Consistent use of "we" vs passive
- [ ] 98. Acronym definition-on-first-use audit
- [ ] 99. Notation table completeness
- [ ] 100. Final read-through against the corpus conventions
