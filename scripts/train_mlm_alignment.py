from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.mlm_training import train_mlm_alignment
from src.utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MLM with cross-lingual alignment loss.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--parallel-csv", default=None)
    args = parser.parse_args()
    setup_logging()
    config = load_config(args.config)
    train_mlm_alignment(config, csv_path=args.parallel_csv)


if __name__ == "__main__":
    main()

