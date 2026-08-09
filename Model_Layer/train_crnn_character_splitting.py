"""Train and run a CRNN that recovers complete text lines from PNGs."""

import argparse
import ast
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz "
CHAR_TO_INDEX = {char: index + 1 for index, char in enumerate(CHARSET)}
INDEX_TO_CHAR = {index + 1: char for index, char in enumerate(CHARSET)}
BLANK_INDEX = 0
IMAGE_HEIGHT = 360
IMAGE_WIDTH = 640
NUM_CLASSES = len(CHARSET) + 1


def read_labels(metadata_path):
    with metadata_path.open(newline="", encoding="utf-8") as metadata_file:
        labels = {}
        for row in csv.DictReader(metadata_file):
            value = ast.literal_eval(row["Label"])
            if not isinstance(value, list) or len(value) != 1:
                raise ValueError(f"Expected one label for {row['file_name']!r}")
            labels[Path(row["file_name"]).stem] = value[0]
    return labels


def _foreground_mask(image):
    threshold_type = cv2.THRESH_BINARY if np.median(image) < 127 else cv2.THRESH_BINARY_INV
    _, mask = cv2.threshold(image, 127, 255, threshold_type)
    components, component_labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    clean_mask = np.zeros_like(mask)
    height, width = image.shape
    for component in range(1, components):
        x, y, component_width, component_height, area = stats[component]
        touches_edge = (
            x == 0
            or y == 0
            or x + component_width == width
            or y + component_height == height
        )
        if area > 20 and not touches_edge:
            clean_mask[component_labels == component] = 255
    return clean_mask


def preprocess_image(image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(image_path)

    foreground = _foreground_mask(image)
    points = cv2.findNonZero(foreground)
    if points is not None:
        x, y, width, height = cv2.boundingRect(points)
        padding = 10
        x1, y1 = max(0, x - padding), max(0, y - padding)
        x2 = min(image.shape[1], x + width + padding)
        y2 = min(image.shape[0], y + height + padding)
        foreground = foreground[y1:y2, x1:x2]

    height, width = foreground.shape
    scale = min(IMAGE_WIDTH / width, IMAGE_HEIGHT / height)
    resized_width = max(1, int(width * scale))
    resized_height = max(1, int(height * scale))
    resized = cv2.resize(foreground, (resized_width, resized_height), interpolation=cv2.INTER_AREA)

    canvas = np.full((IMAGE_HEIGHT, IMAGE_WIDTH), 255, dtype=np.uint8)
    x_offset = (IMAGE_WIDTH - resized_width) // 2
    y_offset = (IMAGE_HEIGHT - resized_height) // 2
    canvas[y_offset:y_offset + resized_height, x_offset:x_offset + resized_width] = 255 - resized
    return torch.from_numpy(canvas).float().unsqueeze(0) / 255.0


class TextDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        image_path, text = self.records[index]
        target = torch.tensor([CHAR_TO_INDEX[char] for char in text], dtype=torch.long)
        return preprocess_image(image_path), target, text


def collate_batch(batch):
    images, targets, texts = zip(*batch)
    target_lengths = torch.tensor([len(target) for target in targets], dtype=torch.long)
    return torch.stack(images), torch.cat(targets), target_lengths, texts


def convolution_block(input_channels, output_channels, pool=True):
    layers = [
        nn.Conv2d(input_channels, output_channels, 3, padding=1),
        nn.BatchNorm2d(output_channels),
        nn.LeakyReLU(0.1, inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2, 2))
    return nn.Sequential(*layers)


class TextCRNN(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()
        self.cnn = nn.Sequential(
            convolution_block(1, 32),
            convolution_block(32, 64),
            convolution_block(64, 128, pool=False),
        )
        self.rnn = nn.LSTM(128 * 16, hidden_size, 2,
                           bidirectional=True, batch_first=True, dropout=0.45)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_size * 2, NUM_CLASSES)

    def forward(self, images):
        features = self.cnn(images)
        features = F.adaptive_avg_pool2d(features, (16, features.shape[-1]))
        batch, channels, height, width = features.shape
        features = features.permute(0, 3, 1, 2).reshape(batch, width, channels * height)
        recurrent, _ = self.rnn(features)
        logits = self.classifier(self.dropout(recurrent))
        return F.log_softmax(logits, dim=2).permute(1, 0, 2)


def decode(log_probs):
    predictions = log_probs.argmax(dim=2).permute(1, 0)
    decoded = []
    for sequence in predictions:
        result = []
        previous = None
        for index in sequence.tolist():
            if index != previous and index != BLANK_INDEX:
                result.append(INDEX_TO_CHAR.get(index, ""))
            previous = index
        decoded.append("".join(result))
    return decoded


def edit_distance(left, right):
    distances = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, 1):
        next_distances = [left_index]
        for right_index, right_char in enumerate(right, 1):
            next_distances.append(min(
                next_distances[-1] + 1,
                distances[right_index] + 1,
                distances[right_index - 1] + (left_char != right_char),
            ))
        distances = next_distances
    return distances[-1]


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_distance = 0
    total_characters = 0
    exact_matches = 0
    with torch.no_grad():
        for images, targets, target_lengths, texts in loader:
            images, targets = images.to(device), targets.to(device)
            log_probs = model(images)
            input_lengths = torch.full((images.size(0),), log_probs.size(0), dtype=torch.long)
            total_loss += criterion(log_probs, targets, input_lengths, target_lengths).item() * images.size(0)
            for prediction, expected in zip(decode(log_probs), texts):
                total_distance += edit_distance(prediction, expected)
                total_characters += len(expected)
                exact_matches += prediction == expected
    count = len(loader.dataset)
    return total_loss / count, total_distance / max(1, total_characters), exact_matches / count


def train(data_dir, metadata_path, checkpoint_path, epochs=200, batch_size=8, learning_rate=1e-3, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    labels = read_labels(metadata_path)
    records = []
    for image_path in sorted(data_dir.glob("*.png")):
        if image_path.stem not in labels:
            raise KeyError(f"No metadata label for {image_path.name}")
        records.append((image_path, labels[image_path.stem]))
    if not records:
        raise FileNotFoundError(f"No PNG files found in {data_dir}")

    indices = np.random.RandomState(42).permutation(len(records))
    validation_count = max(1, int(len(records) * 0.15))
    validation_indices = set(indices[:validation_count])
    train_records = [record for index, record in enumerate(records) if index not in validation_indices]
    validation_records = [record for index, record in enumerate(records) if index in validation_indices]
    train_loader = DataLoader(TextDataset(train_records), batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    validation_loader = DataLoader(TextDataset(validation_records), batch_size=batch_size, collate_fn=collate_batch)

    model = TextCRNN().to(device)
    with torch.no_grad():
        model.classifier.bias[BLANK_INDEX] = -1.0
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.CTCLoss(blank=BLANK_INDEX, zero_infinity=True)
    warmup_steps = 500
    total_steps = max(1, epochs * len(train_loader))

    def learning_rate_schedule(step):
        if step < warmup_steps:
            return max(0.05, step / warmup_steps)
        return max(0.1, 1 - (step - warmup_steps) / max(1, total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_schedule)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_cer = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        for images, targets, target_lengths, _ in train_loader:
            images, targets = images.to(device), targets.to(device)
            log_probs = model(images)
            input_lengths = torch.full((images.size(0),), log_probs.size(0), dtype=torch.long)
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()

        val_loss, cer, exact_accuracy = validate(model, validation_loader, criterion, device)
        print(f"Epoch {epoch:3d}/{epochs} val loss {val_loss:.4f} CER {cer:.3f} exact {exact_accuracy:.2%}")
        if cer < best_cer:
            best_cer = cer
            torch.save({
                "model_state": model.state_dict(),
                "charset": CHARSET,
                "image_height": IMAGE_HEIGHT,
                "image_width": IMAGE_WIDTH,
                "validation_cer": cer,
                "epoch": epoch,
            }, checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")


def predict(image_path, checkpoint_path, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = TextCRNN().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with torch.no_grad():
        return decode(model(preprocess_image(image_path).unsqueeze(0).to(device)))[0]


def main():
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=project_root / "spooky-data_pngs")
    parser.add_argument("--metadata", type=Path, default=project_root / "spooky-data" / "metadata.csv")
    parser.add_argument("--checkpoint", type=Path, default=project_root / "Model_Files" / "Farne_Back_Models" / "crnn_character_best.pt")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    train(args.data_dir, args.metadata, args.checkpoint, args.epochs, args.batch_size, device=args.device)


if __name__ == "__main__":
    main()