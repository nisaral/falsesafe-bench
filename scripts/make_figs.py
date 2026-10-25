from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from lcert.artifacts import IST, git_snapshot, sha256_file, validate_runlog, write_json


def verified_artifact(directory: Path, record: dict, filename: str) -> Path:
    path = (directory / filename).resolve()
    if not path.is_relative_to(directory) or not path.is_file():
        raise ValueError('Source artifact must be a real file inside the selected run')
    expected = record.get('artifact_hashes', {}).get(filename)
    if not expected or sha256_file(path) != expected:
        raise ValueError(f'Source artifact hash is absent or mismatched: {filename}')
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description='Plot actual E0 ground-truth artifacts; never invent benchmark metrics')
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--heading', type=float, default=0.0)
    options = parser.parse_args()
    if not math.isfinite(options.heading):
        parser.error('Heading must be finite')
    directory = options.run_dir.resolve()
    if not directory.is_relative_to(ROOT / 'results' / 'runs'):
        parser.error('Choose a project-local results/runs directory')
    log_path = directory / 'runlog.json'
    record = json.loads(log_path.read_text(encoding='utf-8'))
    validate_runlog(record, ROOT)
    if record['status'] != 'ok' or record['experiment'] != 'E0' or record['test_split_used']:
        raise ValueError('Plotting requires a successful, non-test E0 diagnostic run')
    runlog_hash = sha256_file(log_path)
    config_path = (ROOT / record['config_path']).resolve()
    if config_path != directory / 'config.yaml' or sha256_file(config_path) != record['config_sha256']:
        raise ValueError('Copied run configuration path or hash is inconsistent')
    config = json.loads(config_path.read_text(encoding='utf-8'))
    manifest_path = verified_artifact(directory, record, 'ground_truth_manifest.json')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if not manifest.get('snapshots'):
        raise ValueError('No recorded ground-truth snapshots to plot')
    commit, clean = git_snapshot(ROOT)
    cache = ROOT / '.cache' / 'matplotlib'
    cache.mkdir(parents=True, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(cache)

    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as pyplot
    import numpy as np
    from matplotlib.patches import Circle

    stamp = datetime.now(IST)
    output_directory = directory / 'figs' / f'{stamp:%Y%m%d_%H%M%S}_{uuid4().hex[:6]}'
    output_directory.mkdir(parents=True, exist_ok=False)
    figures = []
    sources = [manifest_path]
    for snapshot in manifest['snapshots']:
        source = verified_artifact(directory, record, snapshot['path'])
        sources.append(source)
        with np.load(source, allow_pickle=False) as artifact:
            value = artifact['value']
            states = artifact['states']
            horizon = float(artifact['horizon'])
        if value.ndim not in (2, 3) or states.shape != (*value.shape, value.ndim):
            raise ValueError('Snapshot value/state grids have inconsistent shapes')
        if not np.isfinite(value).all() or not np.isfinite(states).all() or horizon != snapshot['horizon']:
            raise ValueError('Snapshot contains nonfinite values or an inconsistent horizon')
        if list(value.shape) != snapshot['grid_shape'] or snapshot['system'] != config['system']:
            raise ValueError('Snapshot and run configuration disagree')
        heading = None
        if config['system'] == 'double_integrator' and value.ndim == 2:
            displayed, position_x, position_y = value, states[..., 0], states[..., 1]
            labels = ('Position p', 'Velocity v')
            title = 'Double integrator: true-dynamics BRT'
        elif config['system'] == 'dubins' and value.ndim == 3:
            heading_index = int(np.abs(states[0, 0, :, 2] - options.heading).argmin())
            heading = float(states[0, 0, heading_index, 2])
            displayed = value[:, :, heading_index]
            position_x, position_y = states[:, :, heading_index, 0], states[:, :, heading_index, 1]
            labels = ('Position x', 'Position y')
            title = f'Dubins: true-dynamics BRT, heading={heading:.4f} rad'
        else:
            raise ValueError('System and saved state dimension do not match')
        limit = float(np.max(np.abs(displayed)))
        if limit == 0:
            raise ValueError('All-zero field needs explicit investigation before plotting')
        figure, axis = pyplot.subplots(figsize=(8, 6), constrained_layout=True)
        heatmap = axis.pcolormesh(position_x, position_y, displayed, shading='auto',
                                 cmap='RdBu', vmin=-limit, vmax=limit)
        if displayed.min() < 0 < displayed.max():
            axis.contour(position_x, position_y, displayed, levels=[0], colors=['black'], linewidths=1.2)
        if config['system'] == 'dubins':
            for center_x, center_y, radius in config['obstacles']:
                axis.add_patch(Circle((center_x, center_y), radius, fill=False,
                                      edgecolor='white', linewidth=1.2))
            axis.set_aspect('equal')
        axis.set(xlabel=labels[0], ylabel=labels[1], title=f'{title}\nHorizon T={horizon:g} s')
        figure.colorbar(heatmap, ax=axis, label='Signed true-dynamics value')
        axis.text(0.01, 0.01, f'{record["run_id"]}\nSource commit {record["git_commit"][:12]}',
                  transform=axis.transAxes, fontsize=7, backgroundcolor='white')
        filename = f'ground_truth_T{str(horizon).replace(".", "p")}.png'
        output = output_directory / filename
        figure.savefig(output, dpi=160)
        pyplot.close(figure)
        figures.append({'path': output.relative_to(directory).as_posix(), 'sha256': sha256_file(output),
                        'source': snapshot['path'], 'source_sha256': sha256_file(source),
                        'horizon': horizon, 'heading_slice': heading})
        print(output)
    for source in sources:
        verified_artifact(directory, record, source.relative_to(directory).as_posix())
    ending_commit, ending_clean = git_snapshot(ROOT)
    if ending_commit != commit or not ending_clean or sha256_file(log_path) != runlog_hash:
        raise RuntimeError('Source tree or runlog changed while figures were generated')
    write_json(output_directory / 'figures_manifest.json', {
        'source_run_id': record['run_id'], 'source_commit': record['git_commit'],
        'runlog_sha256': runlog_hash, 'plot_commit': commit, 'plot_tree_clean': clean,
        'command': [sys.executable, *sys.argv], 'generated_at_ist': stamp.isoformat(),
        'figures': figures, 'benchmark_metrics': False,
        'limitations': 'True-dynamics numerical diagnostic only. A heading slice is not a 3D safety-volume result.'})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
