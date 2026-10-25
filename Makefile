PY ?= python
UV ?= uv
.PHONY: test reproduce install check list demo fleet data starter
install:
	$(UV) pip install --python "$(PY)" --cache-dir .cache/uv -r requirements.txt
test:
	$(PY) -m pytest -v -s -p no:cacheprovider
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
