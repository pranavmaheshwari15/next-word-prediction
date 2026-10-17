# Data Preparation — Next Word Prediction

## Dataset
- Name: WikiText-2
- Source: Hugging Face `Salesforce/wikitext`
- Configuration: `wikitext-2-raw-v1`
- Official splits: train, validation, and test

## Preprocessing Pipeline
1. Download the dataset using `download_dataset.py`.
2. Clean text using `src/data_preprocessing/preprocess.py`.
3. Split paragraphs into sentences using `src/data_preprocessing/sentence_split.py`.
4. Fit the tokenizer on training sentences only.
5. Generate next-word input–target examples.
6. Pad each input to a maximum context length of 5.
7. Save NumPy arrays and the tokenizer.

## Model Input Configuration
- Vocabulary size: 10,000
- Context length: 5
- Padding: Pre-padding with zeros
- Tokenizer: Keras Tokenizer with an OOV token
- Input format: Integer token IDs
- Target format: Single integer token ID
- Recommended loss: Sparse categorical crossentropy

## Generated Dataset Shapes

| Split | Input shape | Target shape |
|---|---|---|
| Train | (1668886, 5) | (1668886,) |
| Validation | (175071, 5) | (175071,) |
| Test | (196425, 5) | (196425,) |

## How to Reproduce

Run these commands from the project root after installing the required Python dependencies:

```bash
python download_dataset.py
python src/data_preprocessing/preprocess.py
python src/data_preprocessing/sentence_split.py
python src/data_preprocessing/prepare_data.py
python verify_dataset.py
```

## Generated Files

The following files are created in `data/processed/arrays/`:

- `X_train.npy`, `y_train.npy`
- `X_validation.npy`, `y_validation.npy`
- `X_test.npy`, `y_test.npy`
- `tokenizer.json`

These generated files are excluded from Git. They can be reproduced using the pipeline.

## Verification

All three splits passed the dataset verification checks.

## Team Handoff

Pranav should load the saved NumPy arrays and configure the model for a vocabulary size of 10,000 and a context length of 5.

Tushar should reuse the saved tokenizer for converting user-entered text into token IDs and interpreting predictions.
