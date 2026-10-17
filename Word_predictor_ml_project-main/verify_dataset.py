from pathlib import Path
import numpy as np

DATA_DIR = Path("data/processed/arrays")
VOCAB_SIZE = 10000
CONTEXT_SIZE = 5

for split in ["train", "validation", "test"]:
    X = np.load(DATA_DIR / f"X_{split}.npy", mmap_mode="r")
    y = np.load(DATA_DIR / f"y_{split}.npy", mmap_mode="r")

    print(f"\n--- {split.upper()} ---")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("X dtype:", X.dtype)
    print("y dtype:", y.dtype)

    assert X.ndim == 2
    assert X.shape[1] == CONTEXT_SIZE
    assert len(X) == len(y)
    assert np.issubdtype(X.dtype, np.integer)
    assert np.issubdtype(y.dtype, np.integer)

    assert X.min() >= 0
    assert X.max() < VOCAB_SIZE
    assert y.min() >= 1
    assert y.max() < VOCAB_SIZE

    print("Status: PASSED")

assert (DATA_DIR / "tokenizer.json").exists()
print("\nTokenizer file: FOUND")
print("All dataset checks passed!")
