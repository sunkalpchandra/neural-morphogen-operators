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
- [ ] 11. Per-gene analysis: which genes NRDO wins and loses on, and why
- [ ] 12. Error-vs-distance-to-nearest-observation curve, per model
- [ ] 13. Runtime and peak memory per model, reported alongside accuracy
- [ ] 14. Ablate the occupancy channel (claimed essential, never tested)
- [ ] 15. Ablate the aux_z0 term (claimed to keep the encoder conditioned)
- [x] 16. Verified: decoder reads the field at query coords, not a coordinate map
- [x] 17. Bandwidth trains but moves only ~4% from init; exp15 tests whether it matters
- [ ] 18. Sensitivity to the k-NN graph size used by the encoder
- [ ] 19. A permutation test for the specimen-level result, not just Wilcoxon
- [ ] 20. Power analysis: how many specimens would resolve STAGATE

## Tier 2 — hardens existing claims (21–50)

- [ ] 21. Seed-variance table: how much of each margin is optimisation noise
- [ ] 22. Verify every checkpoint loads and reproduces its recorded metrics
- [x] 23. Split determinism tested: same section, two loads, identical splits and coords
- [x] 24. Held-out blocks verified contiguous — measurably farther from training data than a random subset of equal size
- [x] 25. Standardisation verified train-only: train mean 0, sd 1; held-out splits differ
- [ ] 26. Confirm no gene-selection leakage across the split
- [x] 27. Isotropic normalisation verified on every section
- [ ] 28. Verify SPD parametrisation holds after every optimiser step, not just at init
- [ ] 29. Verify the Strang order empirically at more than one dt range
- [ ] 30. Test the mass-conservation claim at extreme dt
- [ ] 31. Check the absorbing-set bound against a longer trajectory
- [ ] 32. Verify Theorem 11's numerical corollary on real fitted operators
- [ ] 33. Test the vacuity proposition on a trained model, not a random one
- [ ] 34. Confirm the dispersion relation code matches the analytic Jacobian
- [x] 35. Moran's I matches the definition to 2e-17
- [x] 36. Geary's C matches the definition to 3e-15
- [x] 37. ARI/NMI come from sklearn directly
- [x] 38. SSIM matches skimage to 1e-4 and orders degradations identically
- [x] 39. Confirmed percentile, not normal-approximation (asymmetric on skewed data)
- [x] 40. Holm matches statsmodels exactly on the paper's own p-values
- [x] 41. Specimen-collapse unit tested
- [x] 42. MIN_HELDOUT and MIN_REFERENCE_ARI guards unit tested against the sections that motivated them
- [ ] 43. Unit test that check_numbers fails on an injected mismatch
- [ ] 44. Unit test for the provenance registry
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
- [ ] 94. American English consistency check
- [ ] 95. Verify no British spellings crept back in
- [ ] 96. Consistent tense across sections
- [ ] 97. Consistent use of "we" vs passive
- [ ] 98. Acronym definition-on-first-use audit
- [ ] 99. Notation table completeness
- [ ] 100. Final read-through against the corpus conventions
