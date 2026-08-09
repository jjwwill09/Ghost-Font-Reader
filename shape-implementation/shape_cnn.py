import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from shape_data import ShapeDataset, label_names


class ShapeCNN(nn.Module):
    def __init__(self, num_classes=None):
        super().__init__()
        if num_classes is None:
            num_classes = len(label_names())

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def split_indices(n_items, val_fraction=0.15, seed=42):
    indices = list(range(n_items))
    random.Random(seed).shuffle(indices)
    n_val = max(1, int(n_items * val_fraction))
    return indices[n_val:], indices[:n_val]


def train_cnn(samples, image_size=96, epochs=12, batch_size=32, lr=1e-3, seed=42, device="cpu"):
    dataset = ShapeDataset(samples, image_size=image_size)
    train_idx, val_idx = split_indices(len(dataset), seed=seed)

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False)

    model = ShapeCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    best_state = None
    best_val_accuracy = -1.0

    for epoch in range(epochs):
        model.train()
        losses = []
        for images, labels, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        val_accuracy = evaluate_cnn(model, val_loader, device)["accuracy"]
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Loss: {float(np.mean(losses)):.4f} | "
            f"Val Accuracy: {val_accuracy:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def evaluate_cnn(model, loader, device="cpu"):
    model.eval()
    names = label_names()
    confusion = np.zeros((len(names), len(names)), dtype=int)
    mistakes = []
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels, paths in loader:
            images = images.to(device)
            labels = labels.to(device)
            predicted = model(images).argmax(dim=1)

            total += labels.numel()
            correct += int((predicted == labels).sum().item())

            for truth, pred, path in zip(labels.cpu().tolist(), predicted.cpu().tolist(), paths):
                confusion[truth, pred] += 1
                if truth != pred:
                    mistakes.append((path, names[truth], names[pred]))

    return {
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "confusion": confusion,
        "mistakes": mistakes,
    }
