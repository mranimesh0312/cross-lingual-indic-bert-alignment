# Experiment Notes

## Suggested Baselines

- Base `bert-base-multilingual-cased` without alignment training.
- MLM-only fine-tuning on concatenated parallel pairs.
- MLM + alignment loss with multiple `alpha` values.
- XLM-R base comparison.

## Suggested Corpora

- Samanantar
- IIT Bombay English-Hindi
- OPUS
- FLORES
- PMIndia

## Reporting

Report each experiment with:

- Base model
- Parallel corpus name and size
- Language pair
- Epochs
- Batch size
- Learning rate
- Alignment loss weight
- WikiANN language
- Precision, recall, F1, accuracy

## Caution

Do not compare tiny-sample runs against published systems. The included CSV exists only to verify that the pipeline executes.

