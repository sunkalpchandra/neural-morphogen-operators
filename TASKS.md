# Remaining work

Ordered by whether a reviewer could use it to reject the paper. Checked off as
completed; each batch is pushed.

## A. Claims that may not survive their own sample size

The density claim rested on 70 held-out locations and one seed, and only a plot
caught it. The same question has not been asked of the rest.

- [x] 1. Density rerun: claim corrected from 97%/85% to 93%/89%, monotone, 2 seeds, both agreeing
- [x] 2. Audit every remaining exp10 axis — **all four are single-seed**, not just density
- [x] 3. Noise/dropout ranking withdrawn from the text; a 1% gap on one seed is not a result
- [x] 4. exp4: 4 pathways, 2 seeds — already reported as a null result, no ranking claimed
- [x] 5. exp6: not quoted in the main text at all; its macros are unused, so no claim to qualify
- [x] 6. exp5 ablations 2 seeds, exp5_matched 3 — matches \MatchedSeeds in the text
- [x] 7. exp7 swept ONE configuration; sweep_configurations() added and running
- [x] 8. Applied to the density sweep (its base is sized from it) and to the biology path

## B. Statistics

- [x] 9. Bootstrap CI now reported for the tightest significant comparison
- [x] 10. Grouping sensitivity: same 3 comparisons significant either way; paper uses the conservative grouping
- [x] 11. tab:samplesizes reports runs/sections/seeds per experiment; daggers mark single-seed
- [x] 12. Holm family named explicitly at each corrected p

## C. Figures

- [x] 13. Evidence panel (b) rebuilt from the corrected sweep; curve is monotone
- [x] 14. Palette is Okabe-Ito; already colourblind-safe
- [x] 15. MODEL_COLORS extended to the four models added since it was written, which were falling through to positional colours
- [x] 16. scale_bar() added and wired into the methods figure
- [x] 17. Six sub-5pt font sizes raised to 5.0; panel labels nudged clear of titles

## D. Manuscript

- [x] 18. Prose pass: abstract 387->315, transfer/benchmark/biology sections rewritten to lead with the claim and let tables carry numbers
- [x] 19. Zero unresolved references or citations in either build
- [x] 20. 169 orphaned macros identified; left in place (unused definitions are harmless, and the provenance checker already blocks stale *use*)
- [x] 21. All 40 citations present in the bib; 12 unused entries retained for the appendix
- [x] 22. Abstract tightened 387 -> 315 words (corpus extraction was unreliable; judged against NeurIPS convention instead)
- [x] 23. Workshop build verified: 6pp main, balanced conditional, Further-results summary present

## E. Infrastructure

- [x] 24. Seed guard added — immediately caught exp9 being quoted while single-seed
- [x] 25. Size guard added; requires excluded sections to be named in the text
- [x] 26. `make ci` runs tests, the audit and both builds
- [x] 27. README claims rewritten to the 10-specimen results, with the withdrawn claims listed
- [x] 28. `make manifest` writes and `--verify` checks 695 artifacts (1.2 GB)
- [x] 29. Clean clone builds the paper from committed artifacts; data-dependent tables correctly report missing input rather than a missing generator
- [x] 30. Final regeneration, both builds, manifest, push
