from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.evaluate_ner import evaluate_all_languages
from src.utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate NER checkpoints for configured languages.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--languages", nargs="*", default=None)
    args = parser.parse_args()
    setup_logging()
    config = load_config(args.config)
    frame = evaluate_all_languages(config, languages=args.languages)
    print(frame.to_markdown(index=False))


if __name__ == "__main__":
    main()

