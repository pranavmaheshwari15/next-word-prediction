from pathlib import Path

raw_dir = Path("data/raw/wikitext2")
processed_dir = Path("data/processed")

processed_dir.mkdir(parents=True, exist_ok=True)


def clean_text(line):
    line = line.strip()

    if not line:
        return None

    if line.startswith("=") and line.endswith("="):
        return None

    line = line.lower()

    # Normalize WikiText formatting
    line = line.replace(" @-@ ", "-")
    line = line.replace(" @,@ ", ",")
    line = line.replace(" @.@ ", ".")

    # Normalize multiple spaces
    line = " ".join(line.split())

    return line


for split in ["train", "validation", "test"]:

    input_file = raw_dir / f"{split}.txt"
    output_file = processed_dir / f"{split}.txt"

    cleaned_lines = []

    with open(input_file, "r", encoding="utf-8") as file:
        for line in file:
            cleaned_line = clean_text(line)

            if cleaned_line is not None:
                cleaned_lines.append(cleaned_line)

    with open(output_file, "w", encoding="utf-8") as file:
        for line in cleaned_lines:
            file.write(line + "\n")

    print(f"{split}: {len(cleaned_lines)} cleaned lines saved")
