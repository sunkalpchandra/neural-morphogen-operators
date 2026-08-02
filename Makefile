.PHONY: test-data ci manifest workshop papers help data download build experiments all-experiments exp1 exp2 exp3 exp4 exp5 exp6 figures paper clean-runs test reevaluate finalize check-numbers

# Use the project venv when present, otherwise whatever python is on PATH
# (conda users, CI, etc.). Override with `make PY=python3.11 ...`.
PY      ?= $(shell test -x .venv/bin/python && echo .venv/bin/python || echo python)
SEEDS   ?= 0 1 2
EPOCHS  ?= 500
SECTION ?= visium_mouse_brain
WORKERS ?= 6

help:
	@echo "make data         download every dataset and build processed .h5ad"
	@echo "make all-experiments  run every experiment through a parallel job queue"
	@echo "make experiments      run experiments 1-6 serially (slower)"
	@echo "make figures      regenerate figures, tables and paper/numbers.tex"
	@echo "make paper        figures + compile the PDF"
	@echo "make finalize     re-score, regenerate figures/tables/numbers, build PDF"
	@echo "make test         fast end-to-end smoke test"
	@echo ""
	@echo "variables: SEEDS='$(SEEDS)'  EPOCHS=$(EPOCHS)  SECTION=$(SECTION)"

# ---------------------------------------------------------------- data
data: download build

download:
	$(PY) -m src.data.download --all

build:
	$(PY) -m src.data.build --all

# --------------------------------------------------------- experiments
# Parallel job queue (recommended): one job per ablation variant / transfer
# model, run through a worker pool. Far faster than the serial targets.
all-experiments:
	WORKERS=$(WORKERS) EPOCHS=$(EPOCHS) SEEDS='$(SEEDS)' ./scripts/run_all.sh

experiments: exp1 exp5 exp2 exp3 exp4 exp6

exp1:
	$(PY) experiments/exp1_forecasting.py --section $(SECTION) --seeds $(SEEDS) --epochs $(EPOCHS)

exp2:
	$(PY) experiments/exp2_cross_tissue.py --seeds $(SEEDS) --epochs $(EPOCHS)

exp3:
	$(PY) experiments/exp3_resolution.py --seeds $(SEEDS) --epochs $(EPOCHS)

exp4:
	$(PY) experiments/exp4_perturbation.py --seeds $(SEEDS) --epochs $(EPOCHS)

exp5:
	$(PY) experiments/exp5_ablations.py --section $(SECTION) --seeds $(SEEDS) --epochs $(EPOCHS)

exp6:
	$(PY) experiments/exp6_development.py --seeds $(SEEDS) --epochs $(EPOCHS)

# --------------------------------------------------------------- paper
# Analyses that read existing run artifacts rather than training anything.
# Each writes to results/audit/ and feeds macros the manuscript quotes, so they
# have to be regenerable by name, not only by having been run once by hand.
analysis:
	$(PY) scripts/hvg_leakage.py
	$(PY) scripts/permutation_power.py
	$(PY) scripts/eval_noise_sensitivity.py
	$(PY) scripts/per_gene_analysis.py
	$(PY) scripts/error_vs_distance.py
	$(PY) scripts/verify_theory_trained.py

figures:
	PYTHONPATH=. $(PY) experiments/make_figures.py --section $(SECTION)

# Two builds from one set of sources under paper/sections/. Neither is a fork:
# `workshop` sets \fullpaperfalse, which the sources read to drop material.
paper: figures
	cd paper && (tectonic neurips_2026.tex || latexmk -pdf neurips_2026.tex)

workshop: figures
	cd paper && (tectonic workshop.tex || latexmk -pdf workshop.tex)

# Both builds, with the page count each one lands on.
papers: paper workshop
	@for t in neurips_2026 workshop; do \
	  aux=paper/$$t.aux; \
	  main=$$(grep -o 'endofmain}{{[^}]*}{[0-9]*}' $$aux 2>/dev/null | grep -o '{[0-9]*}$$' | tr -d '{}'); \
	  last=$$(grep -o 'lastpage}{{[^}]*}{[0-9]*}' $$aux 2>/dev/null | grep -o '{[0-9]*}$$' | tr -d '{}'); \
	  echo "$$t: main content ends p$${main:-?}, $${last:-?} pages total"; \
	done

# ---------------------------------------------------------------- misc
test:
	PYTHONPATH=. $(PY) -m pytest tests -q

# The leakage checks load real sections and take a couple of minutes, so they
# are excluded from the default suite and run explicitly.
test-data:
	PYTHONPATH=. $(PY) -m pytest tests -q -m data

# Fail when the manuscript and the run artifacts disagree. Run after every
# prose edit and before every commit; it is also the CI gate.
check-numbers:
	PYTHONPATH=. $(PY) scripts/check_numbers.py --results results

# Everything a reviewer or CI should be able to run in one command: the unit
# tests, the prose-vs-artifact audit, and both PDF builds. Non-zero exit on any
# failure, so it is usable as a gate rather than as a report.
ci: test check-numbers papers
	@echo "CI: tests, number audit and both builds passed"

# Recompute the SHA256 manifest over processed artifacts, so a claim of
# reproducibility can be checked rather than asserted.
manifest:
	PYTHONPATH=. $(PY) scripts/write_manifest.py

# Regenerate every derived artifact from run results, in dependency order.
finalize:
	./scripts/finalize.sh

reevaluate:
	PYTHONPATH=. $(PY) scripts/reevaluate.py --results results/exp1

clean-runs:
	rm -rf results/*/runs checkpoints/*
