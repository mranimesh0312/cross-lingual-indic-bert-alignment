# Architecture

This project uses a shared multilingual encoder for both English and Indian-language sentences.

## Inputs

Parallel rows contain:

- `source_text`
- `target_text`
- `source_lang`
- `target_lang`

The tokenizer builds paired sequences:

```text
[source sentence] [SEP] [target sentence]
[target sentence] [SEP] [source sentence]
```

For XLM-R, the tokenizer uses its native special-token format.

## Losses

Masked language modeling trains the model to recover masked tokens in bilingual contexts.

Alignment loss mean-pools final hidden states for source-target and target-source encodings, then minimizes cosine distance:

```text
L = L_mlm + alpha * L_align
```

This approximates bidirectional bilingual consistency. It is intentionally simple and easy to extend.

## Downstream Evaluation

After alignment training, the saved encoder is loaded with `AutoModelForTokenClassification` and fine-tuned on WikiANN/PAN-X NER. Token-label alignment ignores continuation sub-tokens with label `-100`.

