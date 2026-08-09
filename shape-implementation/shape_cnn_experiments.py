import csv
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from imagerecognizer import ImageRecognizer, compute_accuracy, train_model
from shape_cnn import evaluate_cnn, train_cnn
from shape_data import ShapeDataset, collect_samples, filter_by_max_sides, label_names
from shape_experiment_utils import write_confusion, write_mistakes
from typeclassifier import get_image_descriptor


MODELS_TO_RUN = ["cnn", "resnet18"]
MAX_TRAIN_SIDES = [3, 4, 5, 6]
MAX_EVAL_SIDES = [3, 4, 5, 6]
INCLUDE_CIRCLES_OPTIONS = [True, False]

TRAIN_DIR = None
TEST_DIR = None
OUT_DIR = "shape_classifier_results"

EPOCHS = 12
BATCH_SIZE = 32
IMAGE_SIZE = 96
LEARNING_RATE = 1e-3
SEED = 42
DEVICE = None


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_resnet18_classifier(samples, epochs, device):
    from torchvision.models import ResNet18_Weights, resnet18

    resnet_model = resnet18(weights=ResNet18_Weights.DEFAULT)
    resnet_model.fc = torch.nn.Identity()
    resnet_model.to(device).eval()

    descriptors, labels = descriptors_from_samples(samples, resnet_model, device)
    classifier = ImageRecognizer(512)
    train_model(descriptors, labels, classifier, epochs=epochs)
    return classifier, resnet_model


def descriptors_from_samples(samples, resnet_model, device):
    descriptors = []
    labels = []

    for sample in samples:
        descriptors.append(get_image_descriptor(sample.path, resnet_model, device))
        labels.append(sample.label)

    return np.asarray(descriptors, dtype=np.float32), np.asarray(labels, dtype=int)


def evaluate_resnet18(classifier, resnet_model, samples, device):
    descriptors, labels = descriptors_from_samples(samples, resnet_model, device)
    scores = classifier(descriptors)
    score_data = scores.data if hasattr(scores, "data") else scores
    predictions = np.argmax(score_data, axis=1)

    names = label_names()
    confusion = np.zeros((len(names), len(names)), dtype=int)
    mistakes = []

    for sample, truth, predicted in zip(samples, labels, predictions):
        confusion[int(truth), int(predicted)] += 1
        if int(truth) != int(predicted):
            mistakes.append((sample.path, names[int(truth)], names[int(predicted)]))

    accuracy = compute_accuracy(scores, labels)
    return {
        "accuracy": accuracy,
        "correct": int(np.sum(predictions == labels)),
        "total": len(labels),
        "confusion": confusion,
        "mistakes": mistakes,
    }


def train_experiment_model(model_name, train_samples, run_dir, device):
    if model_name == "cnn":
        model = train_cnn(
            train_samples,
            image_size=IMAGE_SIZE,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE,
            seed=SEED,
            device=device,
        )
        torch.save({"model_state": model.state_dict(), "shape_names": label_names()}, run_dir / "shape_cnn.pt")
        return model

    if model_name == "resnet18":
        return train_resnet18_classifier(train_samples, EPOCHS, device)

    raise ValueError(f"Unknown model: {model_name}")


def evaluate_experiment_model(model_name, model_bundle, eval_samples, device):
    if model_name == "cnn":
        loader = DataLoader(ShapeDataset(eval_samples, IMAGE_SIZE), batch_size=BATCH_SIZE, shuffle=False)
        return evaluate_cnn(model_bundle, loader, device)

    if model_name == "resnet18":
        classifier, resnet_model = model_bundle
        return evaluate_resnet18(classifier, resnet_model, eval_samples, device)

    raise ValueError(f"Unknown model: {model_name}")


def run_experiments():
    seed_everything(SEED)

    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / "Data_Handling_Layer"
    train_dir = Path(TRAIN_DIR) if TRAIN_DIR else data_root / "shape-pngs"
    test_dir = Path(TEST_DIR) if TEST_DIR else data_root / "shape-test-pngs"
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_all = collect_samples(train_dir)
    test_all = collect_samples(test_dir)
    if not train_all:
        raise FileNotFoundError(f"No labeled shape PNGs found in {train_dir}")
    if not test_all:
        raise FileNotFoundError(f"No labeled shape PNGs found in {test_dir}")

    device = DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")
    summary_rows = []

    for model_name in MODELS_TO_RUN:
        for include_circles in INCLUDE_CIRCLES_OPTIONS:
            circle_tag = "with_circles" if include_circles else "no_circles"

            for max_train_sides in MAX_TRAIN_SIDES:
                train_samples = filter_by_max_sides(train_all, max_train_sides, include_circles=include_circles)
                print(f"\n=== {model_name} {circle_tag} max_train_sides={max_train_sides} device={device} ===")
                print("train distribution:", dict(Counter(sample.label_name for sample in train_samples)))

                train_run_dir = out_dir / f"{model_name}_{circle_tag}_train_{limit_name(max_train_sides)}"
                train_run_dir.mkdir(parents=True, exist_ok=True)
                model_bundle = train_experiment_model(model_name, train_samples, train_run_dir, device)

                for max_eval_sides in MAX_EVAL_SIDES:
                    eval_samples = filter_by_max_sides(test_all, max_eval_sides, include_circles=include_circles)
                    run_dir = train_run_dir / f"eval_{limit_name(max_eval_sides)}"
                    run_dir.mkdir(parents=True, exist_ok=True)

                    results = evaluate_experiment_model(model_name, model_bundle, eval_samples, device)
                    write_confusion(run_dir / "confusion.csv", results["confusion"])
                    mistake_summary = write_mistakes(run_dir / "mistakes.csv", results["mistakes"])

                    summary_rows.append({
                        "model": model_name,
                        "include_circles": include_circles,
                        "max_train_sides": max_train_sides,
                        "max_eval_sides": max_eval_sides,
                        "train_samples": len(train_samples),
                        "eval_samples": len(eval_samples),
                        "accuracy": f"{results['accuracy']:.6f}",
                        "correct": results["correct"],
                        "wrong": results["total"] - results["correct"],
                        "top_mixups": json.dumps(mistake_summary["top_mixups"]),
                        "reason_counts": json.dumps(mistake_summary["reason_counts"], sort_keys=True),
                    })

    write_summary(out_dir / "summary.csv", summary_rows)
    print(f"\nWrote experiment outputs to {out_dir}")


def write_summary(path, rows):
    fieldnames = [
        "model",
        "include_circles",
        "max_train_sides",
        "max_eval_sides",
        "train_samples",
        "eval_samples",
        "accuracy",
        "correct",
        "wrong",
        "top_mixups",
        "reason_counts",
    ]

    with Path(path).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def limit_name(limit):
    return str(limit)


if __name__ == "__main__":
    run_experiments()
