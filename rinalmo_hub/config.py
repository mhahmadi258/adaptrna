"""Configuration: `configs/base.yaml` -> task YAML -> `--set` dotted overrides.

The resolved config is a plain nested dict that also supports attribute access, so both
`cfg["data"]["root"]` and `cfg.data.root` work. Every run writes its resolved config to the
output directory, which is the only reliable record of what was actually trained.
"""

from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Union
import copy

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_CONFIG_PATH = REPO_ROOT / "configs" / "base.yaml"


class Config(dict):
    """Nested dict with attribute access. `cfg.optim.lr` == `cfg["optim"]["lr"]`."""

    def __init__(self, mapping: Optional[Mapping] = None):
        super().__init__()
        for key, value in (mapping or {}).items():
            self[key] = value

    def __setitem__(self, key, value):
        super().__setitem__(key, Config(value) if isinstance(value, Mapping) else value)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(f"No config key '{name}'. Available: {sorted(self)}") from exc

    def __setattr__(self, name, value):
        self[name] = value

    def to_dict(self) -> dict:
        return {k: (v.to_dict() if isinstance(v, Config) else v) for k, v in self.items()}


def deep_merge(base: Mapping, override: Mapping) -> dict:
    """Recursive dict merge; `override` wins on conflicts, mappings merge key by key."""
    merged = copy.deepcopy(dict(base))

    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)

    return merged


def parse_scalar(text: str) -> Any:
    """
    Parse a `--set key=value` right-hand side.

    Deliberately not `yaml.safe_load` alone: YAML 1.1 parses `3e-4` as the *string* "3e-4"
    (it wants `3.0e-4`), and `--set optim.lr=3e-4` is the single most common override there
    is. Numbers are tried first, then YAML for lists and anything else.
    """
    stripped = text.strip()
    lowered = stripped.lower()

    if lowered in ("null", "none", "~", ""):
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False

    try:
        return int(stripped)
    except ValueError:
        pass

    try:
        return float(stripped)
    except ValueError:
        pass

    try:
        return yaml.safe_load(stripped)
    except yaml.YAMLError:
        return stripped


def apply_override(cfg: dict, dotted_key: str, value: Any) -> None:
    """Set `cfg["a"]["b"] = value` for `dotted_key == "a.b"`, creating dicts as needed."""
    parts = dotted_key.split(".")
    node = cfg

    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], Mapping):
            node[part] = {}
        node = node[part]

    node[parts[-1]] = value


def load_yaml(path: Union[str, Path]) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def resolve_config(
    task_config_path: Optional[Union[str, Path]] = None,
    overrides: Optional[Iterable[str]] = None,
    base_config_path: Union[str, Path] = BASE_CONFIG_PATH,
) -> Config:
    """
    Build the effective config: base YAML, then the task YAML, then `--set` overrides.

    `overrides` are `dotted.key=value` strings.
    """
    cfg = load_yaml(base_config_path) if Path(base_config_path).exists() else {}

    if task_config_path is not None:
        cfg = deep_merge(cfg, load_yaml(task_config_path))

    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"--set expects 'dotted.key=value', got '{override}'")

        key, _, raw = override.partition("=")
        apply_override(cfg, key.strip(), parse_scalar(raw))

    return Config(cfg)


def save_config(cfg: Config, path: Union[str, Path]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        yaml.safe_dump(cfg.to_dict() if isinstance(cfg, Config) else dict(cfg), f, sort_keys=False)

    return path
