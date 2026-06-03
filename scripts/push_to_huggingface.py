from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import load_config, project_root
from src.publish_to_hub import publish_model_to_hub
from src.utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Push saved model and tokenizer to Hugging Face Hub.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--repo-name", default=None)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()
    setup_logging()
    config = load_config(args.config)
    root = project_root(config)
    model_dir = args.model_dir or root / str(config.get("outputs", "aligned_model_dir"))
    repo_name = args.repo_name or str(config.get("hub", "repo_name"))
    url = publish_model_to_hub(
        model_dir=model_dir,
        repo_name=repo_name,
        private=args.private or bool(config.get("hub", "private", default=False)),
        model_card_path=root / "model_card.md",
    )
    print(url)


if __name__ == "__main__":
    main()

