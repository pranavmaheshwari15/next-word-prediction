from pathlib import Path

from datasets import load_dataset
from sklearn.model_selection import train_test_split

# Output directory
OUTPUT_DIR = Path("data/raw/wikitext2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Download WikiText-2
print("Downloading WikiText-2 dataset...")

dataset = load_dataset(
    "Salesforce/wikitext",
    "wikitext-2-raw-v1"
)

# Use the original training split for our custom split
all_lines = [
    record["text"]
    for record in dataset["train"]
    if record["text"].strip()
]

print(f"Total non-empty lines: {len(all_lines)}")

# Split 1: 60% training, 40% remaining
train_lines, remaining_lines = train_test_split(
    all_lines,
    test_size=0.40,
    random_state=42
)

# Split 2: Divide remaining 40% into 20% validation and 20% test
validation_lines, test_lines = train_test_split(
    remaining_lines,
    test_size=0.50,
    random_state=42
)

# Save all three splits
splits = {
    "train": train_lines,
    "validation": validation_lines,
    "test": test_lines
}

for split_name, lines in splits.items():

    output_file = OUTPUT_DIR / f"{split_name}.txt"

    with open(output_file, "w", encoding="utf-8") as file:
        for line in lines:
            file.write(line + "\n")

    print(f"Saved {split_name}: {len(lines)} lines")

print("\nDataset splitting completed!")
print("Target split: 60:20:20")