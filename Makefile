.PHONY: help data download build experiments exp1 exp2 exp3 exp4 exp5 figures paper clean-runs test

PY      ?= .venv/bin/python
SEEDS   ?= 0 1 2
EPOCHS  ?= 500
SECTION ?= visium_mouse_brain

help:
	@echo "make data         download every dataset and build processed .h5ad"
	@echo "make experiments  run experiments 1-5 (long)"
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
experiments: exp1 exp5 exp2 exp3 exp4

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

# --------------------------------------------------------------- paper
figures:
	$(PY) experiments/make_figures.py --section $(SECTION)

paper: figures
	cd paper && (tectonic neurips_2026.tex || latexmk -pdf neurips_2026.tex)

# ---------------------------------------------------------------- misc
test:
	$(PY) -m pytest tests -q

clean-runs:
	rm -rf results/*/runs checkpoints/*
