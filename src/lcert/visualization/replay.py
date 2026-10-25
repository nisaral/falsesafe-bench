from __future__ import annotations

import json

from ..artifacts import RunRecord, sha256_file, write_json


def export_replay(run: RunRecord, replay: dict) -> None:
    if replay['metadata'].get('learning') or replay['metadata'].get('safety_certificate'):
        raise ValueError('Starter exporter supports explicitly untrained, uncertified diagnostic replays')
    replay['metadata'].update({'run_id': run.record['run_id'], 'git_commit': run.record['git_commit']})
    replay_path = run.directory / 'replay.json'
    write_json(replay_path, replay)
    template_path = run.root / 'demos' / 'replay_template.html'
    template = template_path.read_text(encoding='utf-8')
    if template.count('__REPLAY_JSON__') != 1:
        raise ValueError('Replay template must contain exactly one JSON insertion point')
    serialized = json.dumps(replay, allow_nan=False).replace('<', '\\u003c')
    output = run.directory / 'demo.html'
    output.write_text(template.replace('__REPLAY_JSON__', serialized), encoding='utf-8')
    manifest = run.directory / 'demo_manifest.json'
    write_json(manifest, {'source_run_ids': [run.record['run_id']],
                          'git_commit': run.record['git_commit'],
                          'replay_sha256': sha256_file(replay_path),
                          'template_sha256': sha256_file(template_path),
                          'seed': run.record['seed'], 'split_seed': run.record['split_seed'],
                          'policy_checkpoints': [], 'learning': False, 'safety_certificate': False,
                          'export_command': run.record['command']})
    for path in (replay_path, output, manifest):
        run.track(path)
