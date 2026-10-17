from pathlib import Path

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences

DATA_DIR = Path("data/processed/sentences")
OUTPUT_DIR = Path("data/processed/arrays")

VOCAB_SIZE = 10000
CONTEXT_SIZE = 5

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
def load_sentences(split):
    file_path = DATA_DIR / f"{split}.txt"

    with open(file_path, "r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


train_sentences = load_sentences("train")
validation_sentences = load_sentences("validation")
test_sentences = load_sentences("test")

# Fit tokenizer using training data only
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(train_sentences)


def create_examples(sentences):
    inputs = []
    targets = []

    sequences = tokenizer.texts_to_sequences(sentences)

    for sequence in sequences:
        for i in range(1, len(sequence)):
            context = sequence[max(0, i - CONTEXT_SIZE):i]
            target = sequence[i]

            inputs.append(context)
            targets.append(target)

    X = pad_sequences(
        inputs,
        maxlen=CONTEXT_SIZE,
        padding="pre",
        truncating="pre",
        value=0
    )

    y = np.array(targets, dtype=np.int32)

    return X, y


for split, sentences in [
    ("train", train_sentences),
    ("validation", validation_sentences),
    ("test", test_sentences)
]:
    X, y = create_examples(sentences)

    np.save(OUTPUT_DIR / f"X_{split}.npy", X)
    np.save(OUTPUT_DIR / f"y_{split}.npy", y)

    print(f"{split}: X shape = {X.shape}, y shape = {y.shape}")

print("Configured vocabulary size:", VOCAB_SIZE)
tokenizer_path = OUTPUT_DIR / "tokenizer.json"

with open(tokenizer_path, "w", encoding="utf-8") as file:
    file.write(tokenizer.to_json())

print("Tokenizer saved to:", tokenizer_path)
