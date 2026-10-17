from pathlib import Path
from tensorflow.keras.preprocessing.text import Tokenizer

train_file = Path("data/processed/sentences/train.txt")

with open(train_file, "r", encoding="utf-8") as file:
    train_sentences = [line.strip() for line in file if line.strip()]

# Create tokenizer
tokenizer = Tokenizer(oov_token="<OOV>")

# Learn vocabulary from training data only
tokenizer.fit_on_texts(train_sentences)

# Convert sentences to integer sequences
train_sequences = tokenizer.texts_to_sequences(train_sentences)

print("Total training sentences:", len(train_sentences))
print("Vocabulary size:", len(tokenizer.word_index) + 1)

print("\nOriginal sentence:")
print(train_sentences[0])

print("\nToken IDs:")
print(train_sequences[0])

print("\nFirst 10 vocabulary entries:")
for word, index in list(tokenizer.word_index.items())[:10]:
    print(word, ":", index)
