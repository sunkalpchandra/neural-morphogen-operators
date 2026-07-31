.PHONY: help data download build experiments all-experiments exp1 exp2 exp3 exp4 exp5 exp6 figures paper clean-runs test reevaluate

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
figures:
	PYTHONPATH=. $(PY) experiments/make_figures.py --section $(SECTION)

paper: figures
	cd paper && (tectonic neurips_2026.tex || latexmk -pdf neurips_2026.tex)

# ---------------------------------------------------------------- misc
test:
	PYTHONPATH=. $(PY) -m pytest tests -q

reevaluate:
	PYTHONPATH=. $(PY) scripts/reevaluate.py --results results/exp1

clean-runs:
	rm -rf results/*/runs checkpoints/*
