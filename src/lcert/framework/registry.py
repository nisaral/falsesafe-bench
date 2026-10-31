"""Name -> factory registries so YAML configs can refer to plug-ins by name."""
from __future__ import annotations

from typing import Callable

ENVS: dict[str, Callable] = {}
POLICIES: dict[str, Callable] = {}
CERTIFIERS: dict[str, Callable] = {}


def _register(table, name):
    def deco(fn):
        table[name] = fn
        return fn
    return deco


def register_env(name: str):
    return _register(ENVS, name)


def register_policy(name: str):
    return _register(POLICIES, name)


def register_certifier(name: str):
    return _register(CERTIFIERS, name)


def build(table: dict, kind: str, name: str, params: dict | None = None):
    if name not in table:
        raise KeyError(f"Unknown {kind} '{name}'. Registered: {sorted(table)}")
    return table[name](**(params or {}))
