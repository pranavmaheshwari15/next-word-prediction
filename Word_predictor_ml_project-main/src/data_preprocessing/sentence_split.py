from pathlib import Path
from nltk.tokenize import sent_tokenize

input_dir = Path("data/processed")
output_dir = Path("data/processed/sentences")

output_dir.mkdir(parents=True, exist_ok=True)


def split_sentences(paragraph):
    paragraph = paragraph.replace("lit .", "lit<PERIOD>")

    sentences = sent_tokenize(paragraph, language="english")

    return [
        sentence.replace("lit<PERIOD>", "lit .").strip()
        for sentence in sentences
        if sentence.strip()
    ]


for split in ["train", "validation", "test"]:
    input_file = input_dir / f"{split}.txt"
    output_file = output_dir / f"{split}.txt"

    sentence_count = 0

    with open(input_file, "r", encoding="utf-8") as source:
        with open(output_file, "w", encoding="utf-8") as destination:
            for paragraph in source:
                for sentence in split_sentences(paragraph.strip()):
                    destination.write(sentence + "\n")
                    sentence_count += 1

    print(f"{split}: {sentence_count} sentences saved")
