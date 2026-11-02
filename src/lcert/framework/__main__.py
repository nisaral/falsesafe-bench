"""CLI: python -m lcert.framework run CONFIG.yaml [--out DIR] [--plugin module_or_file.py ...]"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
from pathlib import Path

from . import ENVS, POLICIES, CERTIFIERS, ExperimentSpec, run_experiment


def _load_plugin(ref: str):
    if ref.endswith(".py"):
        spec = importlib.util.spec_from_file_location(Path(ref).stem, ref)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
    else:
        importlib.import_module(ref)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m lcert.framework")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run an experiment config and export viewer JSON")
    r.add_argument("config")
    r.add_argument("--out", default="demos/viewer3d/data")
    r.add_argument("--plugin", action="append", default=[], help="python file/module registering plug-ins")
    l = sub.add_parser("list", help="list registered plug-ins")
    l.add_argument("--plugin", action="append", default=[])
    a = ap.parse_args(argv)
    for p in a.plugin:
        _load_plugin(p)
    if a.cmd == "list":
        print("envs:", sorted(ENVS)); print("policies:", sorted(POLICIES)); print("certifiers:", sorted(CERTIFIERS))
        return 0
    spec = ExperimentSpec.from_yaml(a.config)
    run_experiment(spec, out_dir=a.out)
    print(f"exported {Path(a.out) / (spec.name + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
