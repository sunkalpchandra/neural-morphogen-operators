# Remaining work

Ordered by whether a reviewer could use it to reject the paper. Checked off as
completed; each batch is pushed.

## A. Claims that may not survive their own sample size

The density claim rested on 70 held-out locations and one seed, and only a plot
caught it. The same question has not been asked of the rest.

- [ ] 1. Density sweep on a large section, 2 seeds — replace or retract the claim
- [x] 2. Audit every remaining exp10 axis — **all four are single-seed**, not just density
- [x] 3. Noise/dropout ranking withdrawn from the text; a 1% gap on one seed is not a result
- [x] 4. exp4: 4 pathways, 2 seeds — already reported as a null result, no ranking claimed
- [x] 5. exp6: not quoted in the main text at all; its macros are unused, so no claim to qualify
- [ ] 6. Check exp5 ablation seed counts against what the text claims
- [x] 7. exp7 swept ONE configuration; sweep_configurations() added and running
- [ ] 8. Apply MIN_HELDOUT_LOCATIONS consistently to every quoted result, not just specimens

## B. Statistics

- [x] 9. Bootstrap CI now reported for the tightest significant comparison
- [x] 10. Grouping sensitivity: same 3 comparisons significant either way; paper uses the conservative grouping
- [x] 11. tab:samplesizes reports runs/sections/seeds per experiment; daggers mark single-seed
- [x] 12. Holm family named explicitly at each corrected p

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
