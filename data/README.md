# Data

This repository includes a tiny Hindi-English sample corpus so the training script can run without external files. Real experiments should replace it with a larger parallel corpus containing:

```text
source_text,target_text,source_lang,target_lang
```

NER evaluation uses WikiANN/PAN-X from Hugging Face Datasets. Supported language codes are configured in `config.yaml`.

