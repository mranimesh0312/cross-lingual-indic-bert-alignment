from __future__ import annotations

from typing import Any

import pandas as pd
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class ParallelSentenceDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        source_column: str,
        target_column: str,
        max_length: int,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.source_column = source_column
        self.target_column = target_column
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.frame)

    def _encode_pair(self, first: str, second: str) -> dict[str, Any]:
        return self.tokenizer(
            first,
            second,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        source = str(row[self.source_column])
        target = str(row[self.target_column])
        return {
            "forward": self._encode_pair(source, target),
            "reverse": self._encode_pair(target, source),
        }


def tokenize_and_align_labels(
    examples: dict[str, list[Any]],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> dict[str, Any]:
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=max_length,
    )
    labels: list[list[int]] = []
    for batch_index, word_labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=batch_index)
        previous_word_id = None
        label_ids: list[int] = []
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != previous_word_id:
                label_ids.append(word_labels[word_id])
            else:
                # Ignore continuation sub-tokens so seqeval receives word-level labels.
                label_ids.append(-100)
            previous_word_id = word_id
        labels.append(label_ids)
    tokenized["labels"] = labels
    return tokenized

