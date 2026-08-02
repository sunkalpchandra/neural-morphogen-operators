# Remaining work

Ordered by whether a reviewer could use it to reject the paper. Checked off as
completed; each batch is pushed.

## A. Claims that may not survive their own sample size

The density claim rested on 70 held-out locations and one seed, and only a plot
caught it. The same question has not been asked of the rest.

- [ ] 1. Density sweep on a large section, 2 seeds — replace or retract the claim
- [x] 2. Audit every remaining exp10 axis — **all four are single-seed**, not just density
- [x] 3. Noise/dropout ranking withdrawn from the text; a 1% gap on one seed is not a result
- [ ] 4. Check exp4 perturbation: how many pathways, seeds, and what n the null uses
- [ ] 5. Check exp6 developmental forecasting: n comparisons, seeds
- [ ] 6. Check exp5 ablation seed counts against what the text claims
- [ ] 7. Check exp7 numerics: is the stability sweep one configuration or many
- [ ] 8. Apply MIN_HELDOUT_LOCATIONS consistently to every quoted result, not just specimens

## B. Statistics

- [ ] 9. Bootstrap CIs on every specimen-level delta, not just point estimates
- [ ] 10. Sensitivity analysis: MOSTA embryos as one specimen vs two
- [ ] 11. Report n and seeds for every experiment in one appendix table
- [ ] 12. State the Holm family explicitly wherever a corrected p is quoted

## C. Figures

- [ ] 13. Rebuild evidence panel (b) from the corrected density data
- [ ] 14. Check the palette is colourblind-safe
- [ ] 15. Unify font sizes across figures
- [ ] 16. Add scale bars to tissue maps
- [ ] 17. Verify legibility at print size (no sub-5pt text)

## D. Manuscript

- [ ] 18. Prose pass for readability against corpus conventions
- [ ] 19. Verify every cross-reference resolves
- [ ] 20. Remove orphaned macros no longer used
- [ ] 21. Bibliography: every citation present, no unused entries
- [ ] 22. Abstract length against corpus distribution
- [ ] 23. Verify the workshop build reads coherently, not just compiles

## E. Infrastructure

- [ ] 24. check_numbers: fail when a quoted statistic has fewer seeds than claimed
- [ ] 25. check_numbers: fail when a quoted statistic is below the size threshold
- [ ] 26. A CI-ready target that runs tests + check-numbers + both builds
- [ ] 27. README: final numbers, and the claims that changed
- [ ] 28. Regenerate the SHA256 manifest
- [ ] 29. Verify single-command reproduction from a clean checkout
- [ ] 30. Final regeneration, both builds, push
