from __future__ import annotations

import json
import math
import tomllib
from pathlib import Path
from typing import Any


class ConfigurationError(ValueError):
    pass


def require_integer(config: dict[str, Any], name: str, minimum: int = 0) -> int:
    value = config.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigurationError(f'{name} must be an integer >= {minimum}')
    return value


def require_positive(config: dict[str, Any], name: str) -> float:
    value = config.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f'{name} must be a positive finite number')
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ConfigurationError(f'{name} must be a positive finite number')
    return value


def validate_common(config: dict[str, Any]) -> None:
    if not isinstance(config.get('experiment'), str):
        raise ConfigurationError('experiment must be explicitly specified')
    for name in ('seed', 'split_seed'):
        if require_integer(config, name) >= 2**32:
            raise ConfigurationError(f'{name} must be below 2**32')
    role = config.get('split_role', 'validation')
    if role not in ('train', 'validation', 'test'):
        raise ConfigurationError('split_role must be train, validation, or test')
    try:
        json.dumps(config, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ConfigurationError('Config must contain finite JSON-compatible values') from error


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    if path.suffix == '.json':
        config = json.loads(path.read_text(encoding='utf-8'))
    elif path.suffix == '.toml':
        config = tomllib.loads(path.read_text(encoding='utf-8'))
    elif path.suffix in ('.yaml', '.yml'):
        import yaml

        config = yaml.safe_load(path.read_text(encoding='utf-8'))
    else:
        raise ConfigurationError('Use a .json, .toml, .yaml, or .yml config')
    if not isinstance(config, dict):
        raise ConfigurationError('Config must be a mapping')
    validate_common(config)
    return config
