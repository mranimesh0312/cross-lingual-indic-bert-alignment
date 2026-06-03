from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    raw: dict[str, Any]
    path: Path

    @property
    def base_model_name(self) -> str:
        return str(self.raw["model"]["base_model_name"])

    @property
    def languages(self) -> list[str]:
        return list(self.raw["languages"])

    def get(self, *keys: str, default: Any = None) -> Any:
        value: Any = self.raw
        for key in keys:
            if not isinstance(value, dict) or key not in value:
                return default
            value = value[key]
        return value


def load_config(path: str | Path = "config.yaml") -> ProjectConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return ProjectConfig(raw=raw, path=config_path)


def project_root(config: ProjectConfig) -> Path:
    return config.path.resolve().parent

