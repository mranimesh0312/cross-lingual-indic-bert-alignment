# Cross-lingual Indic BERT Alignment

Research reproduction for the M.Tech thesis **"Cross-lingual Alignment of Contextual Word Embedding for Low-resource Indian Languages"**.

This project recreates a simplified, modern implementation of cross-lingual contextual embedding alignment for low-resource Indian languages using Hugging Face Transformers. It fine-tunes a multilingual encoder with masked language modeling on English-Indic parallel sentence pairs and adds an alignment-inspired auxiliary loss between sentence representations.

> Disclaimer: This is an educational research reproduction and GitHub portfolio project. It does not claim state-of-the-art results and should not be used in production without larger experiments, stronger validation, and careful bias/error analysis.

## Background

Low-resource Indian languages often lack large annotated corpora for downstream NLP. Multilingual encoders such as `bert-base-multilingual-cased` and `xlm-roberta-base` already provide shared representation spaces, but targeted fine-tuning with parallel text can encourage better cross-lingual transfer.

The original thesis motivation was explicit alignment of contextual word embeddings for Indian languages. This implementation keeps that idea but makes the code compact, reproducible, and Hugging Face-ready.

## Supported Languages

Hindi (`hi`), Bengali (`bn`), Gujarati (`gu`), Marathi (`mr`), Tamil (`ta`), Telugu (`te`), Kannada (`kn`), Malayalam (`ml`), Punjabi (`pa`), Odia (`or`), Assamese (`as`).

## Architecture

1. Load multilingual BERT or XLM-R.
2. Read parallel sentence pairs with columns `source_text`, `target_text`, `source_lang`, `target_lang`.
3. Encode source-target order and target-source order.
4. Apply masked language modeling to concatenated bilingual inputs.
5. Mean-pool final hidden states for both directions.
6. Compute cosine alignment loss between source-target and target-source representations.
7. Optimize:

```text
total_loss = mlm_loss + alpha * alignment_loss
```

The auxiliary loss is sentence-level and lightweight. It is alignment-inspired rather than a full word-alignment objective.

## Dataset Details

Included sample:

```text
data/sample_parallel/hi_en_sample.csv
```

Real experiments should use larger parallel corpora such as IIT Bombay English-Hindi, PMIndia, Samanantar, FLORES, OPUS, or other curated bilingual resources.

NER evaluation uses WikiANN/PAN-X through Hugging Face Datasets:

```python
load_dataset("wikiann", "hi")
```

The original thesis reported WikiANN/PAN-X experiments over 787,462 examples across 11 Indian languages. See [docs/thesis_results.md](docs/thesis_results.md).

## Training Pipeline

Install:

```bash
pip install -r requirements.txt
```

Train MLM + alignment:

```bash
python scripts/train_mlm_alignment.py
```

Use custom parallel corpus:

```bash
python scripts/train_mlm_alignment.py --parallel-csv path/to/parallel.csv
```

Fine-tune NER for one language:

```bash
python scripts/train_ner.py --language hi
```

Evaluate all configured languages:

```bash
python scripts/evaluate_all_languages.py
```

Push aligned model to Hugging Face:

```bash
set HF_TOKEN=your_token_here
python scripts/push_to_huggingface.py --repo-name cross-lingual-indic-bert-alignment
```

On Linux/macOS:

```bash
export HF_TOKEN=your_token_here
python scripts/push_to_huggingface.py --repo-name cross-lingual-indic-bert-alignment
```

## Configuration

Edit `config.yaml`:

```yaml
model:
  base_model_name: bert-base-multilingual-cased
training:
  batch_size: 4
  mlm_epochs: 1
  ner_epochs: 1
  alpha_alignment: 0.25
```

Switch to XLM-R:

```yaml
model:
  base_model_name: xlm-roberta-base
```

## Evaluation Outputs

Evaluation writes:

```text
outputs/results/ner_results.csv
outputs/results/ner_results.md
```

Historical thesis F1 comparison:

| Language | BERT F1 | Proposed Model F1 | Delta |
| --- | ---: | ---: | ---: |
| pa | 0.5103 | 0.5014 | -0.0089 |
| hi | 0.8656 | 0.8105 | -0.0551 |
| bn | 0.9181 | 0.9036 | -0.0145 |
| or | 0.2105 | 0.4012 | +0.1907 |
| as | 0.9231 | 0.8806 | -0.0425 |
| gu | 0.6804 | 0.6614 | -0.0190 |
| mr | 0.9127 | 0.8753 | -0.0374 |
| kn | 0.5972 | 0.6315 | +0.0343 |
| te | 0.8431 | 0.8512 | +0.0081 |
| ml | 0.8268 | 0.8164 | -0.0104 |
| ta | 0.7990 | 0.8019 | +0.0029 |
| Average | 0.7351 | 0.7395 | +0.0044 |

These are transcribed from the thesis report and are included for continuity with the original work. New runs from this repository will write fresh metrics to `outputs/results/`.

## Project Structure

```text
cross-lingual-indic-bert-alignment/
├── config.yaml
├── data/
├── src/
├── scripts/
├── notebooks/
├── outputs/
└── docs/
```

## Thesis Citation

```text
Ranjan, A. Cross-lingual Alignment of Contextual Word Embedding for
Low-resource Indian Languages. M.Tech Project Report, Department of
Computer Science and Engineering, Indian Institute of Technology Guwahati, 2021.
```

## Notes

This codebase is intentionally simplified for clarity. For thesis-grade experiments, add stronger baselines, multiple random seeds, larger parallel corpora, per-language hyperparameter sweeps, and detailed error analysis.
