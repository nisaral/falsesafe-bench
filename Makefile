PY ?= python
.PHONY: test reproduce install
install:
	$(PY) -m pip install -r requirements.txt
test:
	$(PY) -m pytest -v -s
reproduce: test
	@echo "Only E0 (sanity) exists so far. Experiment entry points (scripts/run_experiment.py) are added in W1 to W3."
	@test -f scripts/run_experiment.py || (echo "scripts/run_experiment.py not written yet"; exit 1)
	$(PY) scripts/run_experiment.py --all
