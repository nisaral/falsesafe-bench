from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from lcert.artifacts import sha256_file, write_json
from lcert.config import ConfigurationError, load_config, require_integer, require_positive


def main() -> int:
    parser = argparse.ArgumentParser(description='Internal bounded ground-truth snapshot worker')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--horizon', type=float, required=True)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    source, output = options.config.resolve(), options.output.resolve()
    if not source.is_relative_to(ROOT / 'results' / 'runs') or output.parent != source.parent or output.suffix != '.npz':
        raise ConfigurationError('Worker paths must remain in the active parent run')
    if output.exists() or output.with_suffix('.json').exists():
        raise ConfigurationError('Snapshot workers cannot overwrite an earlier artifact')
    record = json.loads((source.parent / 'runlog.json').read_text(encoding='utf-8'))
    if record['experiment'] != 'E0' or record['status'] != 'running' or sha256_file(source) != record['config_sha256']:
        raise ConfigurationError('Worker requires an active, hash-matched E0 parent run')
    config = load_config(source)
    if config['experiment'] != 'E0' or config.get('split_role') == 'test':
        raise ConfigurationError('Only non-test true-dynamics snapshots are supported')
    horizon = options.horizon
    if not math.isfinite(horizon) or horizon <= 0 or horizon not in config['horizons']:
        raise ConfigurationError('Worker horizon must be declared in the run config')

    import jax
    import numpy as np

    grid_config = config['grid']
    size = require_integer(grid_config, 'n', 9)
    dimensions = 2 if config['system'] == 'double_integrator' else 3
    lower, upper = np.asarray(grid_config['lo'], float), np.asarray(grid_config['hi'], float)
    if lower.shape != (dimensions,) or upper.shape != lower.shape or not np.isfinite(lower).all() or not np.isfinite(upper).all() or not np.all(lower < upper):
        raise ConfigurationError('Ground-truth grid bounds must be finite, ordered, and dimension-matched')
    if config['system'] == 'double_integrator':
        from lcert.hj_di import solve_di_brt

        started = time.monotonic()
        grid, value = solve_di_brt(umax=require_positive(config, 'umax'), T=horizon, n=size,
                                   lo=lower, hi=upper, accuracy=config['accuracy'])
    elif config['system'] == 'dubins':
        from lcert.hj_dubins import solve_dubins_brt

        started = time.monotonic()
        grid, value = solve_dubins_brt(speed=require_positive(config, 'speed'),
                                      turn_rate=require_positive(config, 'turn_rate'), horizon=horizon,
                                      n=size, lo=lower, hi=upper, obstacles=config['obstacles'],
                                      accuracy=config['accuracy'])
    else:
        raise ConfigurationError('Unknown true-dynamics system')
    elapsed = time.monotonic() - started
    if not np.isfinite(value).all():
        raise FloatingPointError('Ground-truth solve produced nonfinite values')
    np.savez_compressed(output, value=value, states=np.asarray(grid.states),
                        spacings=np.asarray(grid.spacings), horizon=np.asarray(horizon))
    write_json(output.with_suffix('.json'), {'path': output.name, 'system': config['system'],
               'horizon': horizon, 'grid_shape': list(value.shape), 'solve_seconds': elapsed,
               'jax_devices': [str(device) for device in jax.devices()]})
    print(f'Snapshot saved: {output.name}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
