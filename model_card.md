# Cross-lingual Indic BERT Alignment

## Model Purpose

This model is intended for research and educational exploration of cross-lingual contextual embedding alignment for low-resource Indian languages. It starts from a multilingual Transformer encoder and fine-tunes it with bilingual masked language modeling plus an auxiliary representation alignment objective.

## Base Model

Default base model:

```text
bert-base-multilingual-cased
```

The project also supports:

```text
xlm-roberta-base
```

## Training Method

Training uses English-Indic parallel sentence pairs. Source-target and target-source inputs are encoded with the same model. The objective combines:

- Masked language modeling loss on bilingual sentence pairs.
- Cosine embedding alignment loss between mean-pooled contextual representations from both directions.

The final loss is:

```text
MLM loss + alpha * alignment loss
```

## Supported Languages

Hindi, Bengali, Gujarati, Marathi, Tamil, Telugu, Kannada, Malayalam, Punjabi, Odia, and Assamese.

## Intended Use

- Cross-lingual NLP research.
- Educational reproduction of thesis-style alignment experiments.
- Starting checkpoint for downstream NER experiments on WikiANN/PAN-X.
- Portfolio demonstration of Hugging Face training and publishing workflows.

## Limitations

- The included sample corpus is tiny and only for smoke testing.
- The alignment loss is sentence-level, not full token-level supervised word alignment.
- Performance depends heavily on parallel corpus size and quality.
- WikiANN evaluation alone is not enough to validate real-world robustness.
- Indian languages have script, morphology, domain, and dialect variation that require careful testing.

## Thesis-Reported Evaluation

The original thesis report compared BERT and the proposed alignment method on WikiANN NER. Reported average F1:

| Model | Average F1 |
| --- | ---: |
| BERT | 0.7351 |
| Proposed aligned model | 0.7395 |

These values are historical thesis results, not model-card benchmark claims for a newly published checkpoint. Re-run the scripts in this repository before reporting current results.

## Not For Production

Do not deploy this model in production or high-stakes settings without additional validation, fairness analysis, security review, and task-specific benchmarking.

## Ethical Considerations

The model may inherit biases from its base model and training data. Low-resource language systems should be evaluated with native speaker review when possible.

## Citation

```text
Animesh Ranjan. Cross-lingual Alignment of Contextual Word Embedding for
Low-resource Indian Languages. M.Tech Project Report, IIT Guwahati, 2021.
```
