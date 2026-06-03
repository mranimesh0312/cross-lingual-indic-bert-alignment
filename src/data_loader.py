from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import ProjectConfig, project_root


def load_parallel_corpus(config: ProjectConfig, csv_path: str | Path | None = None) -> pd.DataFrame:
    root = project_root(config)
    path = Path(csv_path or config.get("data", "parallel_csv"))
    if not path.is_absolute():
        path = root / path
    frame = pd.read_csv(path)
    required = {
        config.get("data", "source_column"),
        config.get("data", "target_column"),
        config.get("data", "source_lang_column"),
        config.get("data", "target_lang_column"),
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Parallel CSV missing columns: {sorted(missing)}")
    return frame.dropna(subset=list(required)).reset_index(drop=True)


def load_wikiann(language: str):
    from datasets import load_dataset

    return load_dataset("wikiann", language)


def ner_label_names(dataset) -> list[str]:
    return dataset["train"].features["ner_tags"].feature.names
