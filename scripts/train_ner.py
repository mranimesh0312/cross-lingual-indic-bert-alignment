from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.ner_finetuning import train_ner
from src.utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune aligned model on WikiANN NER.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--language", default="hi")
    parser.add_argument("--model-path", default=None)
    args = parser.parse_args()
    setup_logging()
    config = load_config(args.config)
    train_ner(config, language=args.language, model_path=args.model_path)


if __name__ == "__main__":
    main()

