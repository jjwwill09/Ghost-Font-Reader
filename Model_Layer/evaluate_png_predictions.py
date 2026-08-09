"""Evaluate OCR predictions for the PNG dataset against metadata labels."""

import argparse
import ast
import csv
import sys
from pathlib import Path


MODEL_LAYER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_LAYER_DIR.parent
DEFAULT_PNG_DIR = PROJECT_ROOT / "spooky-data_pngs"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "spooky-data" / "metadata.csv"
DEFAULT_CHECKPOINT_PATH = (
    PROJECT_ROOT / "Model_Files" / "Farne_Back_Models" / "crnn_character_best.pt"
)


def load_ground_truth(metadata_path):
    """Return a mapping from each dataset code to its text label."""
    with metadata_path.open(newline="", encoding="utf-8") as metadata_file:
        rows = csv.DictReader(metadata_file)
        ground_truth = {}
        for row in rows:
            code = Path(row["file_name"]).stem
            label = ast.literal_eval(row["Label"])
            if not isinstance(label, list) or len(label) != 1:
                raise ValueError(f"Expected one label for {code!r}, got {row['Label']!r}")
            ground_truth[code] = label[0]
    return ground_truth


def load_predictor(checkpoint_path, device):
    """Return a predictor backed by the local trained CRNN checkpoint."""
    sys.path.insert(0, str(MODEL_LAYER_DIR))
    from train_crnn_character_splitting import predict as predict_text

    def predict(image_path):
        return predict_text(str(image_path), str(checkpoint_path), device)

    return predict


def evaluate(png_dir, metadata_path, checkpoint_path, device=None):
    ground_truth = load_ground_truth(metadata_path)
    png_paths = sorted(png_dir.glob("*.png"))
    if not png_paths:
        raise FileNotFoundError(f"No PNG files found in {png_dir}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Local CRNN checkpoint not found: {checkpoint_path}. "
            "Train Model_Layer/train_crnn_character_splitting.py or pass "
            "a different path with --checkpoint."
        )
    if device is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    predict = load_predictor(checkpoint_path, device)
    correct = 0
    mismatches = []
    for png_path in png_paths:
        code = png_path.stem
        if code not in ground_truth:
            raise KeyError(f"No metadata row found for PNG code {code!r}")
        expected = ground_truth[code]
        predicted = predict(png_path)
        if predicted == expected:
            correct += 1
        else:
            mismatches.append((code, expected, predicted))

    accuracy = 100 * correct / len(png_paths)
    print(f"Evaluated: {len(png_paths)}")
    print(f"Exact matches: {correct}")
    print(f"Exact-match accuracy: {accuracy:.2f}%")
    if mismatches:
        print("\nMismatches:")
        for code, expected, predicted in mismatches:
            print(f"{code}: expected={expected!r}, predicted={predicted!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--png-dir", type=Path, default=DEFAULT_PNG_DIR)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--device", default=None, help="Torch device, for example cpu or cuda")
    args = parser.parse_args()

    evaluate(args.png_dir, args.metadata, args.checkpoint, args.device)


if __name__ == "__main__":
    main()