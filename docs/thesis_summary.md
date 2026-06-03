# Thesis Summary

The M.Tech thesis studied **Cross-lingual Alignment of Contextual Word Embedding for Low-resource Indian Languages**.

Core motivation:

- Low-resource languages have limited labeled corpora.
- Indian languages share lexical, semantic, and structural signals that can help transfer.
- Contextual embeddings from multilingual models can be improved through explicit cross-lingual alignment.

This repository recreates that theme with current Hugging Face tooling:

- Multilingual BERT or XLM-R as base encoder.
- Parallel English-Indic text for alignment training.
- MLM objective for contextual bilingual learning.
- Auxiliary alignment loss for representation consistency.
- WikiANN NER for downstream evaluation.

The implementation is a clean research reproduction, not an exact archival copy of the original code.

