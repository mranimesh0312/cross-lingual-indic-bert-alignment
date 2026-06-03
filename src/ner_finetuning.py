from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import evaluate
import numpy as np
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from .config import ProjectConfig, project_root
from .data_loader import load_wikiann, ner_label_names
from .preprocessing import tokenize_and_align_labels
from .utils import ensure_dir, set_seed

LOGGER = logging.getLogger(__name__)


def build_compute_metrics(label_names: list[str]):
    seqeval = evaluate.load("seqeval")

    def compute_metrics(prediction: Any) -> dict[str, float]:
        logits, labels = prediction
        predictions = np.argmax(logits, axis=-1)
        true_predictions: list[list[str]] = []
        true_labels: list[list[str]] = []
        for pred_row, label_row in zip(predictions, labels):
            pred_labels: list[str] = []
            gold_labels: list[str] = []
            for pred_id, label_id in zip(pred_row, label_row):
                if label_id == -100:
                    continue
                pred_labels.append(label_names[int(pred_id)])
                gold_labels.append(label_names[int(label_id)])
            true_predictions.append(pred_labels)
            true_labels.append(gold_labels)
        metrics = seqeval.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": float(metrics["overall_precision"]),
            "recall": float(metrics["overall_recall"]),
            "f1": float(metrics["overall_f1"]),
            "accuracy": float(metrics["overall_accuracy"]),
        }

    return compute_metrics


def train_ner(config: ProjectConfig, language: str, model_path: str | None = None) -> Path:
    root = project_root(config)
    set_seed(int(config.get("training", "seed", default=42)))
    base_path = Path(model_path) if model_path else root / str(config.get("outputs", "aligned_model_dir"))
    if not base_path.exists():
        base_path = Path(config.base_model_name)

    dataset = load_wikiann(language)
    label_names = ner_label_names(dataset)
    id2label = {idx: label for idx, label in enumerate(label_names)}
    label2id = {label: idx for idx, label in id2label.items()}

    tokenizer = AutoTokenizer.from_pretrained(base_path, use_fast=True)
    tokenized = dataset.map(
        lambda examples: tokenize_and_align_labels(
            examples,
            tokenizer=tokenizer,
            max_length=int(config.get("training", "max_length")),
        ),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        base_path,
        num_labels=len(label_names),
        id2label=id2label,
        label2id=label2id,
    )
    output_dir = root / str(config.get("outputs", "ner_model_dir")) / language
    ensure_dir(output_dir)
    args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=float(config.get("training", "learning_rate")),
        per_device_train_batch_size=int(config.get("training", "ner_batch_size")),
        per_device_eval_batch_size=int(config.get("training", "ner_batch_size")),
        num_train_epochs=float(config.get("training", "ner_epochs")),
        weight_decay=float(config.get("training", "weight_decay")),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=bool(config.get("training", "fp16", default=False)),
        report_to="none",
        seed=int(config.get("training", "seed", default=42)),
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=build_compute_metrics(label_names),
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    LOGGER.info("Saved %s NER model to %s", language, output_dir)
    return output_dir

