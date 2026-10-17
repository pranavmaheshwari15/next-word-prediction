from datasets import load_dataset

dataset = load_dataset(
    "Salesforce/wikitext",
    "wikitext-2-raw-v1"
)

print(dataset)

from pathlib import Path

output_dir = Path("data/raw/wikitext2")
output_dir.mkdir(parents=True, exist_ok=True)

for split in ["train", "validation", "test"]:
    output_file = output_dir / f"{split}.txt"

    with open(output_file, "w", encoding="utf-8") as file:
        for record in dataset[split]:
            file.write(record["text"] + "\n")

    print(f"Saved: {output_file}")
from pathlib import Path

output_dir = Path("data/raw/wikitext2")
output_dir.mkdir(parents=True, exist_ok=True)

for split in ["train", "validation", "test"]:
    output_file = output_dir / f"{split}.txt"

    with open(output_file, "w", encoding="utf-8") as file:
        for record in dataset[split]:
            file.write(record["text"] + "\n")

    print(f"Saved: {output_file}")
