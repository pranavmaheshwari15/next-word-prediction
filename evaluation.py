"""
evaluation.py - Evaluate the trained next-word prediction model on the test set.

Usage (run from anywhere):
    python src/model/evaluation.py
    python src/model/evaluation.py --model models_backup/best_next_word_model.keras
    python src/model/evaluation.py --max-samples 50000 --predict "the history of the"

What it reports:
    * Test loss, top-1 accuracy, and top-k accuracy (k = 1, 3, 5, 10)
    * Perplexity (exp of cross-entropy loss)
    * Accuracy excluding <OOV> targets (a fairer view of real word prediction)
    * Comparison against a "always predict the most frequent word" baseline
    * Sample predictions decoded back to words
    * Plots + a JSON report saved to src/models/saved/evaluation/
    * Optional: predict the next word for your own text with --predict
"""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import tokenizer_from_json


# --------------------------------------------------
# 1. Paths and settings (match train_model.py)
# --------------------------------------------------

def find_project_root():
    """Walk up from this file (or the current folder) until the repo root,
    i.e. the folder containing Word_predictor_ml_project-main, is found.
    This lets evaluation.py live in the repo root or in src/model/."""
    starts = [Path(__file__).resolve().parent, Path.cwd().resolve()]
    for start in starts:
        for folder in [start, *start.parents]:
            if (folder / "Word_predictor_ml_project-main").is_dir():
                return folder
    raise FileNotFoundError(
        "Could not find the 'Word_predictor_ml_project-main' folder. "
        "Place evaluation.py inside the next-word-prediction repo."
    )


PROJECT_ROOT = find_project_root()

DATA_DIR = (
    PROJECT_ROOT
    / "Word_predictor_ml_project-main"
    / "data"
    / "processed"
    / "arrays"
)

MODEL_DIR = PROJECT_ROOT / "src" / "models" / "saved"
OUTPUT_DIR = MODEL_DIR / "evaluation"

VOCAB_SIZE = 10000
OOV_ID = 1  # Keras Tokenizer: index 1 is the <OOV> token, 0 is padding
TOP_K_VALUES = (1, 3, 5, 10)

# Model locations tried in order if --model is not given
DEFAULT_MODEL_CANDIDATES = [
    MODEL_DIR / "best_next_word_model.keras",
    MODEL_DIR / "next_word_model_final.keras",
    PROJECT_ROOT / "models_backup" / "best_next_word_model.keras",
    PROJECT_ROOT / "models_backup" / "next_word_model.keras",
]


# --------------------------------------------------
# 2. Helpers
# --------------------------------------------------

def find_model_path(user_path):
    if user_path:
        path = Path(user_path)
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        return path

    for candidate in DEFAULT_MODEL_CANDIDATES:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "No trained model found. Run train_model.py first or pass --model."
    )


def load_tokenizer():
    path = DATA_DIR / "tokenizer.json"
    with open(path, "r", encoding="utf-8") as f:
        return tokenizer_from_json(f.read())


def build_index_to_word(tokenizer):
    index_to_word = {
        idx: word
        for word, idx in tokenizer.word_index.items()
        if idx < VOCAB_SIZE
    }
    index_to_word[0] = "<PAD>"
    return index_to_word


def decode(ids, index_to_word):
    return " ".join(index_to_word.get(int(i), "<UNK>") for i in ids if int(i) != 0)


def detect_encoding(model, context_size):
    """Pick the input encoding from the model's input size."""
    n_features = model.input_shape[-1]
    if n_features == context_size * VOCAB_SIZE:
        return "flat"       # what the current train_model.py produces
    if n_features == VOCAB_SIZE:
        return "multihot"   # older models in models_backup/ (10,000 inputs)
    raise ValueError(
        f"Model expects {n_features} input features, which matches neither "
        f"flat one-hot ({context_size * VOCAB_SIZE}) nor bag-of-words "
        f"({VOCAB_SIZE}). Check the model/data."
    )


def encode_features(batch_x, context_size, encoding):
    """
    Turn token IDs into model input.
      flat     - flattened one-hot, context_size * vocab features
                 (identical to encode_batch in train_model.py)
      multihot - 1 if the word appears anywhere in the context (vocab features)
      last     - one-hot of only the last context word (vocab features)
    """
    one_hot = tf.one_hot(batch_x, depth=VOCAB_SIZE, dtype=tf.float32)
    if encoding == "flat":
        return tf.reshape(one_hot, [-1, context_size * VOCAB_SIZE])
    if encoding == "multihot":
        return tf.minimum(tf.reduce_sum(one_hot, axis=1), 1.0)
    if encoding == "last":
        return one_hot[:, -1, :]
    raise ValueError(f"Unknown encoding: {encoding}")


def predict_probabilities(model, X, context_size, batch_size, encoding):
    """Yield (start_index, probability matrix) for each batch."""
    for start in range(0, len(X), batch_size):
        batch = X[start:start + batch_size]
        features = encode_features(batch, context_size, encoding)
        yield start, model(features, training=False).numpy()


# --------------------------------------------------
# 3. Core evaluation
# --------------------------------------------------

def evaluate_model(model, X, y, context_size, batch_size, k_values,
                   baseline_id, real_baseline_id, encoding):
    max_k = max(k_values)
    n = len(X)

    total_nll = 0.0
    nll_non_oov = 0.0
    topk_hits = {k: 0 for k in k_values}
    topk_hits_non_oov = {k: 0 for k in k_values}
    baseline_hits = 0
    real_baseline_hits = 0
    non_oov_count = 0
    top1_preds = np.zeros(n, dtype=np.int32)
    top5_preds = np.zeros((n, min(5, max_k)), dtype=np.int32)

    eps = 1e-12

    for start, probs in predict_probabilities(
            model, X, context_size, batch_size, encoding):
        end = start + len(probs)
        targets = y[start:end]
        rows = np.arange(len(targets))

        # negative log-likelihood of the true word
        nll = -np.log(probs[rows, targets] + eps)
        is_oov = targets == OOV_ID
        total_nll += nll.sum()
        nll_non_oov += nll[~is_oov].sum()
        non_oov_count += int((~is_oov).sum())

        # top-k ranking (argpartition then sort the k best)
        part = np.argpartition(-probs, max_k - 1, axis=1)[:, :max_k]
        order = np.argsort(-probs[rows[:, None], part], axis=1)
        ranked = part[rows[:, None], order]

        top1_preds[start:end] = ranked[:, 0]
        top5_preds[start:end] = ranked[:, :top5_preds.shape[1]]

        for k in k_values:
            hit = (ranked[:, :k] == targets[:, None]).any(axis=1)
            topk_hits[k] += int(hit.sum())
            topk_hits_non_oov[k] += int(hit[~is_oov].sum())

        baseline_hits += int((targets == baseline_id).sum())
        real_baseline_hits += int((targets == real_baseline_id).sum())

    results = {
        "num_samples": n,
        "num_non_oov_samples": non_oov_count,
        "loss": total_nll / n,
        "perplexity": float(np.exp(total_nll / n)),
        "topk_accuracy": {k: topk_hits[k] / n for k in k_values},
    }

    if non_oov_count > 0:
        results["loss_excluding_oov"] = nll_non_oov / non_oov_count
        results["perplexity_excluding_oov"] = float(np.exp(nll_non_oov / non_oov_count))
        results["topk_accuracy_excluding_oov"] = {
            k: topk_hits_non_oov[k] / non_oov_count for k in k_values
        }
        results["real_word_baseline_accuracy_excluding_oov"] = (
            real_baseline_hits / non_oov_count)

    results["baseline_accuracy"] = baseline_hits / n
    results["real_word_baseline_accuracy"] = real_baseline_hits / n
    results["oov_target_fraction"] = 1 - non_oov_count / n

    return results, top1_preds, top5_preds


# --------------------------------------------------
# 4. Reporting
# --------------------------------------------------

def print_report(results, baseline_word, real_baseline_word, model_path):
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)
    print(f"Model:            {model_path}")
    print(f"Samples:          {results['num_samples']:,}")
    print(f"<OOV> targets:    {results['oov_target_fraction'] * 100:.2f}%")

    print("\nOverall (all targets)")
    print("-" * 60)
    print(f"Loss:             {results['loss']:.4f}")
    print(f"Perplexity:       {results['perplexity']:.2f}")
    for k, v in results["topk_accuracy"].items():
        print(f"Top-{k:<2} accuracy:  {v * 100:.2f}%")

    if "topk_accuracy_excluding_oov" in results:
        print("\nExcluding <OOV> targets (real words only)")
        print("-" * 60)
        print(f"Loss:             {results['loss_excluding_oov']:.4f}")
        print(f"Perplexity:       {results['perplexity_excluding_oov']:.2f}")
        for k, v in results["topk_accuracy_excluding_oov"].items():
            print(f"Top-{k:<2} accuracy:  {v * 100:.2f}%")

    print("\nBaselines (no learning, always predict one word)")
    print("-" * 60)
    print(f"Most frequent target '{baseline_word}': "
          f"{results['baseline_accuracy'] * 100:.2f}%")
    print(f"Most frequent real word '{real_baseline_word}': "
          f"{results['real_word_baseline_accuracy'] * 100:.2f}% (all targets), "
          f"{results.get('real_word_baseline_accuracy_excluding_oov', 0) * 100:.2f}% (no OOV)")

    top1 = results["topk_accuracy"][1]
    best_base = max(results["baseline_accuracy"], results["real_word_baseline_accuracy"])
    if top1 > best_base:
        print(f"\nModel top-1 beats the best baseline by "
              f"{(top1 - best_base) * 100:.2f} percentage points.")
    else:
        print("\nWARNING: model top-1 does not beat the best single-word baseline.")

    if "topk_accuracy_excluding_oov" in results:
        no_oov_top1 = results["topk_accuracy_excluding_oov"][1]
        real_base = results.get("real_word_baseline_accuracy_excluding_oov", 0)
        print(f"On real words only, top-1 is {no_oov_top1 * 100:.2f}% vs "
              f"{real_base * 100:.2f}% for always guessing '{real_baseline_word}'.")


def show_sample_predictions(X, y, top5, index_to_word, num_samples, seed):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(num_samples, len(X)), replace=False)

    print("\n" + "=" * 60)
    print("SAMPLE PREDICTIONS")
    print("=" * 60)

    for i in idx:
        context = decode(X[i], index_to_word)
        actual = index_to_word.get(int(y[i]), "<UNK>")
        guesses = [index_to_word.get(int(t), "<UNK>") for t in top5[i]]
        mark = "OK  " if int(y[i]) == int(top5[i][0]) else (
            "TOP5" if int(y[i]) in top5[i] else "MISS")
        print(f"[{mark}] ...{context}  ->  actual: {actual:<12} "
              f"predicted: {', '.join(guesses)}")


def save_plots(results, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    ks = list(results["topk_accuracy"].keys())
    overall = [results["topk_accuracy"][k] * 100 for k in ks]

    x = np.arange(len(ks))
    width = 0.38
    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, overall, width, label="All targets")
    if "topk_accuracy_excluding_oov" in results:
        no_oov = [results["topk_accuracy_excluding_oov"][k] * 100 for k in ks]
        plt.bar(x + width / 2, no_oov, width, label="Excluding <OOV>")
    plt.axhline(results["baseline_accuracy"] * 100, color="red",
                linestyle="--", label="Always <OOV> baseline")
    plt.axhline(results["real_word_baseline_accuracy"] * 100, color="gray",
                linestyle=":", label="Most-frequent real word baseline")
    plt.xticks(x, [f"Top-{k}" for k in ks])
    plt.ylabel("Accuracy (%)")
    plt.title("Test Top-k Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "topk_accuracy.png")
    plt.close()


def save_report(results, model_path, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"model": str(model_path), **results}

    def to_native(obj):
        if isinstance(obj, dict):
            return {str(k): to_native(v) for k, v in obj.items()}
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        return obj

    with open(out_dir / "evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(to_native(report), f, indent=2)


# --------------------------------------------------
# 5. Predict next word for custom text
# --------------------------------------------------

def predict_text(model, tokenizer, index_to_word, text, context_size,
                 encoding, top_n=5):
    seq = tokenizer.texts_to_sequences([text])[0]
    seq = [i if i < VOCAB_SIZE else OOV_ID for i in seq]
    context = seq[-context_size:]
    context = [0] * (context_size - len(context)) + context  # pre-pad

    x = np.array([context], dtype=np.int32)
    probs = model(encode_features(x, context_size, encoding),
                  training=False).numpy()[0]

    probs[0] = 0.0  # never suggest padding
    top_ids = np.argsort(-probs)[:top_n]

    print(f"\nInput: \"{text}\"")
    print(f"Context used: {decode(context, index_to_word)}")
    print("Top predictions:")
    for rank, i in enumerate(top_ids, 1):
        print(f"  {rank}. {index_to_word.get(int(i), '<UNK>'):<15} {probs[i] * 100:.2f}%")


# --------------------------------------------------
# 6. Main
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate the next-word model.")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to a .keras model (default: best available)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of test samples (for a quick run)")
    parser.add_argument("--num-examples", type=int, default=15,
                        help="Number of sample predictions to print")
    parser.add_argument("--predict", type=str, default=None,
                        help="Text to predict the next word for")
    parser.add_argument("--encoding", choices=["auto", "flat", "multihot", "last"],
                        default="auto",
                        help="Input encoding (auto: detected from model input size)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(args.seed)

    # ---- load data
    X_test = np.load(DATA_DIR / "X_test.npy")
    y_test = np.load(DATA_DIR / "y_test.npy")
    y_train = np.load(DATA_DIR / "y_train.npy", mmap_mode="r")

    if len(X_test) != len(y_test):
        raise ValueError("X_test and y_test have different lengths.")
    if X_test.max() >= VOCAB_SIZE or y_test.max() >= VOCAB_SIZE:
        raise ValueError("Token ID outside the configured vocabulary range.")

    if args.max_samples and args.max_samples < len(X_test):
        # random subset (test data order follows sentences, so don't just slice)
        rng = np.random.default_rng(args.seed)
        subset = rng.choice(len(X_test), size=args.max_samples, replace=False)
        X_test, y_test = X_test[subset], y_test[subset]

    context_size = X_test.shape[1]

    # ---- load model + tokenizer
    model_path = find_model_path(args.model)
    print("Loading model:", model_path)
    model = tf.keras.models.load_model(model_path)

    encoding = (detect_encoding(model, context_size)
                if args.encoding == "auto" else args.encoding)
    print("Input encoding:", encoding)
    if encoding == "multihot":
        print("NOTE: 10,000-feature models are older backups; their exact "
              "training encoding is not stored in the repo, so multi-hot is "
              "an assumption. Use --encoding last to try another.")

    tokenizer = load_tokenizer()
    index_to_word = build_index_to_word(tokenizer)

    # ---- baselines from the TRAIN set: most frequent target, and most frequent real word
    counts = np.bincount(np.asarray(y_train), minlength=VOCAB_SIZE)
    counts[0] = 0
    baseline_id = int(np.argmax(counts))
    baseline_word = index_to_word.get(baseline_id, "<UNK>")
    counts[OOV_ID] = 0
    real_baseline_id = int(np.argmax(counts))
    real_baseline_word = index_to_word.get(real_baseline_id, "<UNK>")

    # ---- evaluate
    print(f"Evaluating on {len(X_test):,} test samples...")
    results, top1, top5 = evaluate_model(
        model, X_test, y_test, context_size, args.batch_size,
        TOP_K_VALUES, baseline_id, real_baseline_id, encoding
    )

    print_report(results, baseline_word, real_baseline_word, model_path)
    show_sample_predictions(X_test, y_test, top5, index_to_word,
                            args.num_examples, args.seed)

    save_plots(results, OUTPUT_DIR)
    save_report(results, model_path, OUTPUT_DIR)
    print(f"\nPlots and JSON report saved in: {OUTPUT_DIR}")

    if args.predict:
        predict_text(model, tokenizer, index_to_word, args.predict,
                     context_size, encoding)


if __name__ == "__main__":
    main()