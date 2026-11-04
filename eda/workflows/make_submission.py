"""Create a Kaggle CSV from a saved BiGRU checkpoint and the supplied test file."""

import csv
import io
import json
import tarfile
from pathlib import Path

import torch
from tokenizers import Tokenizer

from eda.colab.colab_extractor import BiGRU, DEVICE, predict
from eda.core.analysis import DATA, OUTPUT


CHECKPOINT = "pilot_pilot_word_mixed"
ARCHIVE = OUTPUT / "eda_pilot_checkpoints.tar.gz"
DESTINATION = OUTPUT.parent / "submissions" / "submission_bigru_pilot_v3.csv"


def load_checkpoint(archive_path: Path, name: str) -> tuple[BiGRU, Tokenizer]:
    with tarfile.open(archive_path, "r:gz") as archive:
        prefix = f"eda_checkpoints/{name}"
        tokenizer_file = archive.extractfile(f"{prefix}.json")
        model_file = archive.extractfile(f"{prefix}.pt")
        if tokenizer_file is None or model_file is None:
            raise FileNotFoundError(f"Missing {name} in {archive_path}")
        tokenizer = Tokenizer.from_str(tokenizer_file.read().decode("utf-8"))
        state = torch.load(io.BytesIO(model_file.read()), map_location="cpu", weights_only=True)
    model = BiGRU(tokenizer.get_vocab_size())
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model, tokenizer


def make_submission() -> Path:
    with (DATA / "test.csv").open(newline="", encoding="utf-8") as file:
        test = list(csv.DictReader(file))
    with (DATA / "submission_example.csv").open(newline="", encoding="utf-8") as file:
        example = list(csv.DictReader(file))
    test_ids = [row["text_id"] for row in test]
    assert test_ids[:len(example)] == [row["text_id"] for row in example]
    assert len(test_ids) == len(set(test_ids))

    model, tokenizer = load_checkpoint(ARCHIVE, CHECKPOINT)
    inputs = [dict(source="test", row_id=row["text_id"], doc_id=None,
                   text=row["text"], markers=[]) for row in test]
    spans_by_id = predict(model, tokenizer, inputs)

    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    with DESTINATION.open("w", newline="", encoding="utf-8") as file:
        file.write("text_id,locations\n")
        for row in test:
            text = row["text"]
            spans = spans_by_id[("test", row["text_id"])]
            locations = list(dict.fromkeys(text[start:end] for start, end in spans
                                          if text[start:end].strip()))
            assert all(location in text for location in locations)
            cell = json.dumps(locations, ensure_ascii=False)
            file.write(f'{row["text_id"]},{cell}\n')

    lines = DESTINATION.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "text_id,locations" and len(lines) == len(test) + 1
    saved = [line.split(",", 1) for line in lines[1:]]
    assert [row_id for row_id, _ in saved] == test_ids
    assert all(isinstance(json.loads(cell), list) for _, cell in saved)
    return DESTINATION


if __name__ == "__main__":
    print(make_submission())
