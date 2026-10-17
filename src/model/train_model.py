from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from build_model import build_model
from compile_model import compile_model


# --------------------------------------------------
# 1. Set project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    PROJECT_ROOT
    / "Word_predictor_ml_project-main"
    / "data"
    / "processed"
    / "arrays"
)

MODEL_DIR = PROJECT_ROOT / "src" / "models" / "saved"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Training settings
# --------------------------------------------------

TRAIN_LIMIT = 153600
VALIDATION_LIMIT = 20000

EPOCHS = 10
BATCH_SIZE = 256

RANDOM_SEED = 42

tf.keras.utils.set_random_seed(RANDOM_SEED)


# --------------------------------------------------
# 3. Load training and validation data
# --------------------------------------------------

X_train = np.load(DATA_DIR / "X_train.npy")
y_train = np.load(DATA_DIR / "y_train.npy")

X_val = np.load(DATA_DIR / "X_validation.npy")
y_val = np.load(DATA_DIR / "y_validation.npy")

X_train = X_train[:TRAIN_LIMIT]
y_train = y_train[:TRAIN_LIMIT]

X_validation = X_val[:VALIDATION_LIMIT]
y_validation = y_val[:VALIDATION_LIMIT]


print("Training inputs:", X_train.shape)
print("Training targets:", y_train.shape)

print("Validation inputs:", X_validation.shape)
print("Validation targets:", y_validation.shape)


# --------------------------------------------------
# 4. Validate the data
# --------------------------------------------------

if X_train.ndim != 2 or X_validation.ndim != 2:
    raise ValueError("Input arrays must be 2D.")

if X_train.shape[1] != X_validation.shape[1]:
    raise ValueError(
        "Training and validation context sizes do not match."
    )

if len(X_train) != len(y_train):
    raise ValueError(
        "Training inputs and targets have different lengths."
    )

if len(X_validation) != len(y_validation):
    raise ValueError(
        "Validation inputs and targets have different lengths."
    )

if len(X_train) == 0 or len(X_validation) == 0:
    raise ValueError(
        "Training and validation data cannot be empty."
    )

if not np.issubdtype(X_train.dtype, np.integer):
    raise ValueError("X_train must contain integer token IDs.")

if not np.issubdtype(X_validation.dtype, np.integer):
    raise ValueError("X_validation must contain integer token IDs.")

if not np.issubdtype(y_train.dtype, np.integer):
    raise ValueError("y_train must contain integer token IDs.")

if not np.issubdtype(y_validation.dtype, np.integer):
    raise ValueError("y_validation must contain integer token IDs.")


# --------------------------------------------------
# 5. Determine model dimensions
# --------------------------------------------------

CONTEXT_SIZE = X_train.shape[1]
VOCAB_SIZE = 10000

input_size = CONTEXT_SIZE * VOCAB_SIZE


if min(
    X_train.min(),
    X_validation.min(),
    y_train.min(),
    y_validation.min()
) < 0:
    raise ValueError("Token IDs cannot be negative.")

if max(
    X_train.max(),
    X_validation.max(),
    y_train.max(),
    y_validation.max()
) >= VOCAB_SIZE:
    raise ValueError(
        "A token ID is outside the configured vocabulary range."
    )


print("\nContext size:", CONTEXT_SIZE)
print("Vocabulary size:", VOCAB_SIZE)
print("Input feature size:", input_size)


# --------------------------------------------------
# 6. Convert token IDs into flattened one-hot vectors
# --------------------------------------------------

def encode_batch(batch_x, batch_y):

    features = tf.one_hot(
        batch_x,
        depth=VOCAB_SIZE,
        dtype=tf.float32
    )

    features = tf.reshape(
        features,
        [-1, CONTEXT_SIZE * VOCAB_SIZE]
    )

    return features, batch_y


train_data = (
    tf.data.Dataset
    .from_tensor_slices((X_train, y_train))
    .shuffle(
        buffer_size=min(len(X_train), 10000),
        seed=RANDOM_SEED
    )
    .batch(BATCH_SIZE)
    .map(
        encode_batch,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    .prefetch(tf.data.AUTOTUNE)
)


validation_data = (
    tf.data.Dataset
    .from_tensor_slices((X_validation, y_validation))
    .batch(BATCH_SIZE)
    .map(
        encode_batch,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    .prefetch(tf.data.AUTOTUNE)
)


# --------------------------------------------------
# 7. Build and compile the Dense model
# --------------------------------------------------

model = build_model(
    input_size=input_size,
    vocab_size=VOCAB_SIZE
)

model = compile_model(model)

model.summary()


# --------------------------------------------------
# 8. Configure callbacks
# --------------------------------------------------

# Reduce learning rate when validation loss stops improving.
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    verbose=1
)


# Save the model whenever validation loss improves.
# This file is separate from your final model.
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=str(
        MODEL_DIR / "best_next_word_model.keras"
    ),
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)


# --------------------------------------------------
# 9. Train the model
# --------------------------------------------------

print("\nStarting Dense model training...\n")

history = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=validation_data,
    callbacks=[
        reduce_lr,
        checkpoint
    ],
    verbose=1
)


# --------------------------------------------------
# 10. Save the final model separately
# --------------------------------------------------

FINAL_MODEL_PATH = MODEL_DIR / "next_word_model_final.keras"

model.save(FINAL_MODEL_PATH)

print("\nTraining completed.")
print("Final model saved to:", FINAL_MODEL_PATH)

print(
    "Best validation-loss model:",
    MODEL_DIR / "best_next_word_model.keras"
)


# --------------------------------------------------
# 11. Find the best validation results
# --------------------------------------------------

best_loss_epoch = (
    np.argmin(history.history["val_loss"]) + 1
)

best_accuracy_epoch = (
    np.argmax(history.history["val_accuracy"]) + 1
)

best_val_loss = min(history.history["val_loss"])

best_val_accuracy = max(history.history["val_accuracy"])


print("\nTraining Summary")
print("------------------------------")

print(
    f"Best validation loss: {best_val_loss:.4f} "
    f"(Epoch {best_loss_epoch})"
)

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy * 100:.2f}% "
    f"(Epoch {best_accuracy_epoch})"
)

print(
    f"Final training accuracy: "
    f"{history.history['accuracy'][-1] * 100:.2f}%"
)

print(
    f"Final validation accuracy: "
    f"{history.history['val_accuracy'][-1] * 100:.2f}%"
)


# --------------------------------------------------
# 12. Plot training and validation loss
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.tight_layout()

plt.savefig(
    MODEL_DIR / "loss_curve_final.png"
)

plt.close()


# --------------------------------------------------
# 13. Plot training and validation accuracy
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.tight_layout()

plt.savefig(
    MODEL_DIR / "accuracy_curve_final.png"
)

plt.close()


print("\nTraining graphs saved in:", MODEL_DIR)