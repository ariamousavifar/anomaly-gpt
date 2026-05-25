.PHONY: install test sweep train eval plot clean

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --tb=short

sweep:
	python -m experiments.run_sweep

eval:
	python -m eval.harness

plot:
	python -c "from viz.plot import plot_anomaly_timeline; print('Run notebook for plots')"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

lint:
	python -m py_compile gpt/model.py gpt/train.py data/tokenizer.py \
		data/loader.py data/sequences.py anomaly/scorer.py \
		anomaly/baselines.py anomaly/detector.py anomaly/events.py \
		eval/harness.py eval/bootstrap.py eval/vix_correlation.py \
		experiments/grid.py experiments/run_sweep.py \
		viz/plot.py viz/heatmap.py viz/comparison.py
	@echo "All files compile OK"
