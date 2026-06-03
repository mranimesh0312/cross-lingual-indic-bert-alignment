from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification, Trainer

from .config import ProjectConfig, project_root
from .data_loader import load_wikiann, ner_label_names
from .ner_finetuning import build_compute_metrics
from .preprocessing import tokenize_and_align_labels
from .utils import ensure_dir, write_markdown_table

LOGGER = logging.getLogger(__name__)


def evaluate_language(config: ProjectConfig, language: str, model_dir: str | Path | None = None) -> dict[str, object]:
    root = project_root(config)
    path = Path(model_dir) if model_dir else root / str(config.get("outputs", "ner_model_dir")) / language
    dataset = load_wikiann(language)
    label_names = ner_label_names(dataset)
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(path)
    tokenized = dataset.map(
        lambda examples: tokenize_and_align_labels(
            examples,
            tokenizer=tokenizer,
            max_length=int(config.get("training", "max_length")),
        ),
        batched=True,
        remove_columns=dataset["test"].column_names,
    )
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=build_compute_metrics(label_names),
    )
    metrics = trainer.evaluate(tokenized["test"])
    return {
        "language": language,
        "precision": round(float(metrics["eval_precision"]), 4),
        "recall": round(float(metrics["eval_recall"]), 4),
        "f1": round(float(metrics["eval_f1"]), 4),
        "accuracy": round(float(metrics["eval_accuracy"]), 4),
    }


def evaluate_all_languages(config: ProjectConfig, languages: list[str] | None = None) -> pd.DataFrame:
    rows = []
    for language in languages or config.languages:
        try:
            rows.append(evaluate_language(config, language))
        except Exception as exc:
            LOGGER.exception("Evaluation failed for %s", language)
            rows.append({"language": language, "precision": "NA", "recall": "NA", "f1": "NA", "accuracy": "NA", "error": str(exc)})
    results_dir = ensure_dir(project_root(config) / str(config.get("outputs", "results_dir")))
    frame = pd.DataFrame(rows)
    frame.to_csv(results_dir / "ner_results.csv", index=False)
    write_markdown_table(frame.to_dict(orient="records"), results_dir / "ner_results.md")
    return frame

