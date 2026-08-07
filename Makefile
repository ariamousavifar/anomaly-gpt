.PHONY: install test sweep score eval plot figures clean lint

# Interpreter. Override if you are not running inside an activated venv:
#   make eval PYTHON=.venv/bin/python
PYTHON ?= python

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest tests/ -v --tb=short

# Ablation grid: 9 configs x 3 seeds -> experiments/results.csv
sweep:
	$(PYTHON) -m experiments.run_sweep

# Regenerate anomaly_scores.csv from the trained checkpoint
score:
	$(PYTHON) -m scripts.score_assets

# Every published detection number, at both operating points
eval:
	$(PYTHON) -m scripts.run_eval

# Regenerate the committed figures
figures:
	$(PYTHON) -m scripts.make_figures

plot: figures

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

lint:
	@$(PYTHON) -m py_compile $$(git ls-files '*.py' | while read f; do [ -f "$$f" ] && printf '%s ' "$$f"; done)
	@echo "All files compile OK"
