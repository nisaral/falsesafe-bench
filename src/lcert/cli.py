from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from .artifacts import RunRecord, library_versions, project_root, write_json
from .config import ConfigurationError, load_config, require_integer, require_positive
from .experiments import EXPERIMENTS, ExperimentUnavailable, require_runnable
from .rl.catalog import ALGORITHMS


def environment_report() -> dict:
    root = project_root()
    installed = library_versions(root)
    packages = []
    for line in (root / 'requirements.txt').read_text(encoding='utf-8').splitlines():
        if line.strip() and not line.startswith('#'):
            name, expected = line.split('==', 1)
            packages.append({'package': name, 'expected': expected, 'installed': installed[name],
                             'matches': installed[name] == expected})
    return {'python': platform.python_version(), 'python_supported': sys.version_info[:2] == (3, 13),
            'packages': packages, 'ready': sys.version_info[:2] == (3, 13) and all(item['matches'] for item in packages)}


def _ground_truth(config: dict, run: RunRecord) -> None:
    grid_config = config.get('grid')
    if not isinstance(grid_config, dict):
        raise ConfigurationError('Ground-truth snapshots need explicit grid configuration')
    grid_size = require_integer(grid_config, 'n', 9)
    horizons = config.get('horizons')
    if not isinstance(horizons, list) or not horizons or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0 for value in horizons):
        raise ConfigurationError('horizons must be nonempty, positive, finite values')
    if len(set(horizons)) != len(horizons):
        raise ConfigurationError('Duplicate horizons would overwrite snapshots')
    system = config.get('system')
    if system == 'double_integrator':
        require_positive(config, 'umax')
        budget = 60
    elif system == 'dubins':
        require_positive(config, 'speed')
        require_positive(config, 'turn_rate')
        budget = 900
    else:
        raise ConfigurationError('Unknown ground-truth system')
    snapshots = []
    for horizon in horizons:
        path = run.directory / f'value_T{str(float(horizon)).replace(".", "p")}.npz'
        command = [sys.executable, str(run.root / 'scripts' / 'solve_snapshot.py'),
                   '--config', str(run.directory / 'config.yaml'), '--horizon', str(horizon),
                   '--output', str(path)]
        environment = os.environ.copy()
        environment['PYTHONIOENCODING'] = 'utf-8'
        print(f'Solving {system}, n={grid_size}, T={horizon}; worker budget {budget}s', flush=True)
        started = time.monotonic()
        try:
            process = subprocess.run(command, cwd=run.root, timeout=budget, capture_output=True,
                                     text=True, encoding='utf-8', env=environment,
                                     creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        except subprocess.TimeoutExpired as timeout_error:
            for stream in (timeout_error.stdout, timeout_error.stderr):
                if stream:
                    print(stream.decode('utf-8') if isinstance(stream, bytes) else stream)
            raise TimeoutError(f'{system} worker was stopped after its {budget}s budget; no smaller grid substituted') from timeout_error
        elapsed = time.monotonic() - started
        print(process.stdout, end='')
        print(process.stderr, end='', file=sys.stderr)
        if process.returncode:
            raise RuntimeError(f'Ground-truth worker failed with exit code {process.returncode}')
        snapshot_path = path.with_suffix('.json')
        snapshot = json.loads(snapshot_path.read_text(encoding='utf-8'))
        snapshot.update(worker_seconds=elapsed, worker_budget_seconds=budget)
        run.record['hardware']['jax_devices'] = snapshot['jax_devices']
        run.track(path)
        run.track(snapshot_path)
        snapshots.append(snapshot)
    manifest = run.directory / 'ground_truth_manifest.json'
    write_json(manifest, {'snapshots': snapshots, 'metric_evaluation': 'not implemented/reviewed'})
    run.track(manifest)
    print('Ground-truth fields saved. No FSV/FUV/IoU or benchmark claim computed.')


def _execute_config(config_path: Path, operation: str, command: list[str]) -> Path:
    config = load_config(config_path)
    if operation == 'run':
        require_runnable(config['experiment'])
    elif config['experiment'] != ('simulation' if operation == 'simulate' else 'data'):
        raise ConfigurationError(f'Config experiment does not match operation {operation}')
    with RunRecord(config, config_path, command) as run:
        if operation == 'simulate':
            from .simulation import simulate
            from .visualization import export_replay

            run.record['hardware']['execution_backend'] = 'python_cpu'
            export_replay(run, simulate(config))
            print('Scripted simulation saved. This is not a trained-policy demonstration or a certificate.')
        elif operation == 'generate-data':
            import numpy as np
            from .data import generate_trajectory_splits

            run.record['hardware']['execution_backend'] = 'numpy_cpu'
            splits, manifest = generate_trajectory_splits(config)
            for role, arrays in splits.items():
                path = run.directory / f'{role}.npz'
                np.savez_compressed(path, **arrays)
                run.track(path)
            manifest_path = run.directory / 'data_manifest.json'
            write_json(manifest_path, manifest)
            run.track(manifest_path)
            print('Fixed trajectory splits written. Test artifacts were generated, not evaluated.')
        else:
            _ground_truth(config, run)
        print(f'Artifacts: {run.directory}')

def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Learn-and-Shield reproducible starter tools')
    subcommands = parser.add_subparsers(dest='operation', required=True)
    for name in ('list', 'algorithms', 'doctor', 'reproduce', 'studio'):
        subcommands.add_parser(name)
    sweep_cmd = subcommands.add_parser('sweep')
    sweep_cmd.add_argument('--config', type=Path, default=Path('configs/e1_di_sweep.yaml'))
    for name in ('validate', 'simulate', 'generate-data', 'run'):
        command = subcommands.add_parser(name)
        command.add_argument('--config', type=Path, required=True)
    options = parser.parse_args(arguments)
    if options.operation != 'doctor' and sys.version_info[:2] != (3, 13):
        parser.error('Use the project Python 3.13 environment, not a silent runtime fallback')
    try:
        if options.operation == 'doctor':
            report = environment_report()
            print(json.dumps(report, indent=2))
            return 0 if report['ready'] else 1
        if options.operation == 'studio':
            import webbrowser
            studio_path = project_root() / 'demos' / 'studio' / 'index.html'
            print(f'Opening Mission Control Studio v3.0: {studio_path}')
            webbrowser.open(studio_path.as_uri())
            return 0
        if options.operation == 'sweep':
            from scripts.run_pipeline import run_sweep
            summary = run_sweep(options.config)
            print(f'Sweep completed successfully: {summary}')
            return 0
        if options.operation == 'list':
            print(json.dumps([specification.to_dict() for specification in EXPERIMENTS], indent=2))
        elif options.operation == 'algorithms':
            print(json.dumps([specification.to_dict() for specification in ALGORITHMS], indent=2))
        elif options.operation == 'validate':
            config = load_config(options.config)
            print(json.dumps(config, indent=2))
        elif options.operation == 'reproduce':
            raise ExperimentUnavailable('Full E1-E12 benchmark reproduction is not implemented. Use the documented starter diagnostics; do not label them a completed research reproduction.')
        else:
            _execute_config(options.config.resolve(), options.operation, [sys.executable, *sys.argv])
    except (ConfigurationError, ExperimentUnavailable, ValueError, RuntimeError, ImportError, OSError) as error:
        print(f'{type(error).__name__}: {error}', file=sys.stderr)
        return 1
    return 0
