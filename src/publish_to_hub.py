from __future__ import annotations

import logging
import os
from pathlib import Path

from huggingface_hub import HfApi, login
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

LOGGER = logging.getLogger(__name__)


def publish_model_to_hub(
    model_dir: str | Path,
    repo_name: str,
    private: bool = False,
    model_card_path: str | Path = "model_card.md",
) -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise EnvironmentError("HF_TOKEN environment variable is required for Hugging Face publishing.")
    login(token=token)

    model_path = Path(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    try:
        model = AutoModelForMaskedLM.from_pretrained(model_path)
    except Exception:
        model = AutoModel.from_pretrained(model_path)

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_name, private=private, exist_ok=True)
    model.push_to_hub(repo_name, private=private)
    tokenizer.push_to_hub(repo_name, private=private)

    card = Path(model_card_path)
    if card.exists():
        api.upload_file(
            path_or_fileobj=str(card),
            path_in_repo="README.md",
            repo_id=repo_name,
            repo_type="model",
        )
    LOGGER.info("Published %s to https://huggingface.co/%s", model_path, repo_name)
    return f"https://huggingface.co/{repo_name}"

