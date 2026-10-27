from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .config import ConfigurationError, validate_common


IST = timezone(timedelta(hours=5, minutes=30))


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n',
                    encoding='utf-8')


def library_versions(root: Path) -> dict[str, str]:
    versions = {}
    for line in (root / 'requirements.txt').read_text(encoding='utf-8').splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        package = line.split('==', 1)[0]
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = 'not installed'
    return versions


def git_snapshot(root: Path) -> tuple[str, bool]:
    if not (root / '.git').exists():
        raise RuntimeError('Create a project-local Git repository before recording runs')
    def git(*arguments: str) -> str:
        return subprocess.run(['git', '-C', str(root), *arguments], check=True,
                              capture_output=True, text=True).stdout.strip()
    if Path(git('rev-parse', '--show-toplevel')).resolve() != root.resolve():
        raise RuntimeError('Refusing to use an ancestor Git repository')
    commit = git('rev-parse', 'HEAD')
    clean = not git('status', '--porcelain', '--untracked-files=normal')
    if not clean:
        raise RuntimeError('Runs require a committed, clean project tree; do not hide changes')
    return commit, clean


def check_confirmatory_guard(seeds: list[int] | set[int] | tuple[int, ...], root: Path | None = None) -> None:
    """Automated guard for confirmatory seeds 1000-1029 per PREREGISTRATION.md Section 8."""
    confirmatory_seeds = set(range(1000, 1030))
    if any(s in confirmatory_seeds for s in seeds):
        r = root or project_root()
        tag_check = subprocess.run(["git", "-C", str(r), "tag", "-l", "prereg-v1"], capture_output=True, text=True)
        if "prereg-v1" not in tag_check.stdout:
            raise RuntimeError(
                "CONFIRMATORY GUARD VIOLATION: Pre-registration tag 'prereg-v1' not found. "
                "Per PREREGISTRATION.md Section 8, confirmatory seeds 1000-1029 cannot be executed before tagging."
            )
        commit, clean = git_snapshot(r)
        if not clean:
            raise RuntimeError(
                "CONFIRMATORY GUARD VIOLATION: Working tree is dirty. "
                "Per PREREGISTRATION.md Section 8, confirmatory seeds 1000-1029 require a clean git working tree."
            )
        print(f"[GUARD PASSED] Tag 'prereg-v1' verified at commit {commit[:7]} and working tree clean.", flush=True)


def validate_runlog(record: dict[str, Any], root: Path | None = None) -> None:
    schema = json.loads(((root or project_root()) / 'runlog_schema.json').read_text(encoding='utf-8'))
    missing = set(schema['required']) - record.keys()
    if missing:
        raise ValueError(f'Runlog missing fields: {sorted(missing)}')
    expected_types = {'string': str, 'integer': int, 'boolean': bool,
                      'object': dict, 'array': list, 'null': type(None)}
    for name, specification in schema['properties'].items():
        if name not in record:
            continue
        value = record[name]
        kinds = specification.get('type', [])
        kinds = [kinds] if isinstance(kinds, str) else kinds
        if kinds and not any(type(value) is expected_types[kind] for kind in kinds):
            raise ValueError(f'Invalid type for runlog field {name}')
        if 'enum' in specification and value not in specification['enum']:
            raise ValueError(f'Invalid runlog {name}: {value}')
        if 'pattern' in specification and not re.fullmatch(specification['pattern'], value):
            raise ValueError(f'Invalid runlog format for {name}')
    if record['status'] != 'running' and not record['end_time_ist']:
        raise ValueError('Completed or failed run must have an actual end timestamp')
    if not record['git_tree_clean']:
        raise ValueError('Runlog cannot certify an uncommitted experiment tree')
    if record['status'] == 'ok' and (record.get('git_commit_end') != record['git_commit'] or not record.get('git_tree_clean_end')):
        raise ValueError('Successful runs require the same clean source tree at completion')


class _Tee:
    def __init__(self, terminal, log):
        self.terminal = terminal
        self.log = log

    def write(self, text: str) -> int:
        self.terminal.write(text)
        self.log.write(text)
        return len(text)

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()


class RunRecord:
    def __init__(self, config: dict[str, Any], source: Path, command: list[str]):
        validate_common(config)
        if config.get('split_role') == 'test':
            raise ConfigurationError('Final test access needs reviewed evaluation and ledger logging')
        self.root = project_root()
        schema = json.loads((self.root / 'runlog_schema.json').read_text(encoding='utf-8'))
        if config['experiment'] not in schema['properties']['experiment']['enum']:
            raise ConfigurationError('Experiment identifier is not registered in the runlog schema')
        source = source.resolve()
        if not source.is_relative_to(self.root):
            raise ConfigurationError('Run config must be inside this project')
        commit, clean = git_snapshot(self.root)
        stamp = datetime.now(IST)
        identifier = f'{stamp:%Y%m%d_%H%M%S}_{config["experiment"]}_s{config["seed"]}_{uuid4().hex[:8]}'
        self.directory = self.root / 'results' / 'runs' / identifier
        if not self.directory.resolve().is_relative_to(self.root):
            raise ConfigurationError('Run directory must remain inside the project')
        self.directory.mkdir(parents=True, exist_ok=False)
        copied_config = self.directory / 'config.yaml'
        write_json(copied_config, config)
        self.record = {
            'run_id': identifier, 'experiment': config['experiment'],
            'command': subprocess.list2cmdline(command), 'git_commit': commit,
            'git_tree_clean': clean, 'config_path': copied_config.relative_to(self.root).as_posix(),
            'config_sha256': sha256_file(copied_config),
            'source_config_path': source.relative_to(self.root).as_posix(),
            'source_config_sha256': sha256_file(source),
            'seed': config['seed'], 'split_seed': config['split_seed'],
            'start_time_ist': stamp.isoformat(), 'end_time_ist': None,
            'python_version': platform.python_version(), 'library_versions': library_versions(self.root),
            'hardware': {'platform': platform.platform(), 'processor': platform.processor(),
                         'cpu_count': os.cpu_count(), 'gpu_detection': 'not requested'},
            'status': 'running', 'metrics_path': (self.directory / 'metrics.json').relative_to(self.root).as_posix(),
            'test_split_used': False, 'purpose': 'starter_diagnostic', 'artifact_hashes': {},
            'notes': 'No reviewed benchmark metrics or learned-model safety guarantee.',
        }
        validate_runlog(self.record, self.root)
        write_json(self.directory / 'runlog.json', self.record)
        self.log = None
        self.capture = contextlib.ExitStack()

    def track(self, path: Path) -> None:
        path = path.resolve()
        if not path.is_relative_to(self.directory.resolve()) or not path.is_file():
            raise ValueError('Artifacts must be real files within this run folder')
        self.record['artifact_hashes'][path.relative_to(self.directory).as_posix()] = sha256_file(path)

    def __enter__(self) -> RunRecord:
        self.log = (self.directory / 'stdout.txt').open('w', encoding='utf-8')
        self.capture.enter_context(contextlib.redirect_stdout(_Tee(sys.stdout, self.log)))
        self.capture.enter_context(contextlib.redirect_stderr(_Tee(sys.stderr, self.log)))
        print(f'Run: {self.record["run_id"]}; commit: {self.record["git_commit"]}')
        return self

    def __exit__(self, error_type, error, error_traceback) -> bool:
        original_error = error
        integrity_error = None
        self.record['git_tree_clean_end'] = False
        try:
            commit, clean = git_snapshot(self.root)
            self.record['git_commit_end'] = commit
            self.record['git_tree_clean_end'] = clean
            if commit != self.record['git_commit']:
                raise RuntimeError('Source commit changed during the run')
            source = self.root / self.record['source_config_path']
            copied = self.root / self.record['config_path']
            if sha256_file(source) != self.record['source_config_sha256'] or sha256_file(copied) != self.record['config_sha256']:
                raise RuntimeError('Run configuration changed during execution')
            for filename, expected in self.record['artifact_hashes'].items():
                if sha256_file(self.directory / filename) != expected:
                    raise RuntimeError(f'Registered artifact changed during execution: {filename}')
        except (OSError, RuntimeError, subprocess.SubprocessError) as verification_error:
            integrity_error = verification_error
            self.record['integrity_failure_reason'] = str(verification_error)
            if error is None:
                error_type, error, error_traceback = type(verification_error), verification_error, verification_error.__traceback__
        if error is not None:
            traceback.print_exception(error_type, error, error_traceback)
            self.record['failure_reason'] = f'{error_type.__name__}: {error}'
        self.capture.close()
        self.log.close()
        self.record['status'] = 'failed' if error is not None else 'ok'
        self.record['end_time_ist'] = datetime.now(IST).isoformat()
        metrics = self.directory / 'metrics.json'
        write_json(metrics, {'reportable': False, 'values': {},
                             'reason': 'Protected metric implementation/review is not complete.'})
        self.track(metrics)
        self.track(self.directory / 'stdout.txt')
        validate_runlog(self.record, self.root)
        write_json(self.directory / 'runlog.json', self.record)
        if original_error is None and integrity_error is not None:
            raise integrity_error
        return False
