from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
)

from .alignment_loss import cosine_alignment_loss
from .config import ProjectConfig, project_root
from .data_loader import load_parallel_corpus
from .preprocessing import ParallelSentenceDataset
from .utils import ensure_dir, set_seed

LOGGER = logging.getLogger(__name__)


def _collate_parallel(batch: list[dict[str, Any]], collator: DataCollatorForLanguageModeling) -> dict[str, Any]:
    forward = collator([item["forward"] for item in batch])
    reverse = collator([item["reverse"] for item in batch])
    return {"forward": forward, "reverse": reverse}


def train_mlm_alignment(config: ProjectConfig, csv_path: str | None = None) -> Path:
    root = project_root(config)
    set_seed(int(config.get("training", "seed", default=42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_name,
        use_fast=bool(config.get("model", "use_fast_tokenizer", default=True)),
    )
    model = AutoModelForMaskedLM.from_pretrained(config.base_model_name).to(device)

    frame = load_parallel_corpus(config, csv_path)
    dataset = ParallelSentenceDataset(
        frame=frame,
        tokenizer=tokenizer,
        source_column=str(config.get("data", "source_column")),
        target_column=str(config.get("data", "target_column")),
        max_length=int(config.get("training", "max_length")),
    )
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=float(config.get("training", "mlm_probability")),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("training", "batch_size")),
        shuffle=True,
        collate_fn=lambda batch: _collate_parallel(batch, collator),
    )

    epochs = int(config.get("training", "mlm_epochs"))
    total_steps = max(1, epochs * len(loader))
    optimizer = AdamW(
        model.parameters(),
        lr=float(config.get("training", "learning_rate")),
        weight_decay=float(config.get("training", "weight_decay")),
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * float(config.get("training", "warmup_ratio"))),
        num_training_steps=total_steps,
    )
    alpha = float(config.get("training", "alpha_alignment"))

    model.train()
    for epoch in range(epochs):
        progress = tqdm(loader, desc=f"MLM+alignment epoch {epoch + 1}/{epochs}")
        for batch in progress:
            optimizer.zero_grad(set_to_none=True)
            forward = {k: v.to(device) for k, v in batch["forward"].items()}
            reverse = {k: v.to(device) for k, v in batch["reverse"].items()}

            forward_outputs = model(**forward, output_hidden_states=True)
            reverse_outputs = model(**reverse, output_hidden_states=True)
            align_loss = cosine_alignment_loss(
                forward_outputs.hidden_states[-1],
                forward["attention_mask"],
                reverse_outputs.hidden_states[-1],
                reverse["attention_mask"],
            )
            loss = forward_outputs.loss + alpha * align_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            progress.set_postfix({"loss": f"{loss.item():.4f}", "align": f"{align_loss.item():.4f}"})

    output_dir = root / str(config.get("outputs", "aligned_model_dir"))
    ensure_dir(output_dir)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    LOGGER.info("Saved aligned model to %s", output_dir)
    return output_dir

