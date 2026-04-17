from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any
import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"
_TRUE_VALUES = {"1", "true", "yes", "y", "on", "auto"}
_FALSE_VALUES = {"0", "false", "no", "n", "off"}


@lru_cache(maxsize=1)
def _config_path() -> Path:
    return Path(os.getenv("WSI_TUNING_CONFIG_PATH", str(_DEFAULT_CONFIG_PATH))).expanduser()


@lru_cache(maxsize=1)
def _load_tuning_config() -> dict[str, Any]:
    config_path = _config_path()
    if not config_path.is_file():
        raise FileNotFoundError(f"ROI tuning config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise RuntimeError(f"ROI tuning config must be a YAML mapping: {config_path}")
    return data


def _resolve_section(section_path: str) -> dict[str, Any]:
    node: Any = _load_tuning_config()
    for part in section_path.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"Missing tuning section '{section_path}' in {_config_path()}")
        node = node[part]
    if not isinstance(node, dict):
        raise TypeError(f"Tuning section '{section_path}' must map to a YAML mapping")
    return node


def _apply_env_override(key: str, value: Any) -> Any:
    raw = os.getenv(key)
    if raw is None:
        return value

    text = raw.strip()
    if isinstance(value, bool):
        lowered = text.lower()
        if lowered in _TRUE_VALUES:
            return True
        if lowered in _FALSE_VALUES:
            return False
        raise ValueError(f"Invalid boolean override for {key}: {raw}")
    if isinstance(value, int) and not isinstance(value, bool):
        return int(text)
    if isinstance(value, float):
        return float(text)
    if isinstance(value, list):
        return [item.strip() for item in text.split(",") if item.strip()]
    if isinstance(value, tuple):
        return tuple(item.strip() for item in text.split(",") if item.strip())
    return text


def tuning_value(section_path: str, key: str) -> Any:
    section = _resolve_section(section_path)
    if key not in section:
        raise KeyError(f"Missing tuning key '{key}' in section '{section_path}'")
    return _apply_env_override(key, section[key])


def tuning_section(section_path: str) -> dict[str, Any]:
    section = _resolve_section(section_path)
    return {key: _apply_env_override(key, value) for key, value in section.items()}


__all__ = ["tuning_section", "tuning_value"]
