from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Dict


def load_config(path: str | Path) -> Dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        if config_path.suffix.lower() == ".json":
            config = json.load(handle)
        else:
            try:
                import yaml
            except ImportError as exc:
                raise RuntimeError(
                    "YAML configs require PyYAML; use configs/demo.json or install requirements-core.txt."
                ) from exc
            config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")
    return config


def nested(config: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = config.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{key}' must be a mapping")
    return value
