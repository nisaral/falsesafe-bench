PY ?= python
UV ?= uv
.PHONY: setup doctor test figures reproduce install check list demo fleet data starter
setup:
	$(UV) pip install --python "$(PY)" --cache-dir .cache/uv -r requirements.txt
doctor:
	$(PY) scripts/run_experiment.py doctor
test:
	$(PY) -m pytest -v -s -p no:cacheprovider
figures:
	$(PY) scripts/make_figures.py
check:
	$(PY) scripts/check_scaffold.py
list:
	$(PY) scripts/run_experiment.py list
demo:
	$(PY) scripts/run_experiment.py simulate --config configs/starters/single_robot.json
fleet:
	$(PY) scripts/run_experiment.py simulate --config configs/starters/fleet.json
data:
	$(PY) scripts/run_experiment.py generate-data --config configs/starters/trajectory_data.json
starter: test check demo fleet data
reproduce: test check
	$(PY) scripts/run_experiment.py reproduce
