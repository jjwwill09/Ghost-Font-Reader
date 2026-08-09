from pathlib import Path

import numpy as np
import torch
from torchvision.models import resnet18, ResNet18_Weights

from imagerecognizer import (
    ImageRecognizer,
    LABELS,
    compute_accuracy,
    save_weights,
    train_model,
)
from typeclassifier import get_image_descriptor


current_dir = Path(__file__).resolve().parent

device = "cuda" if torch.cuda.is_available() else "cpu"

resnet_model = resnet18(weights=ResNet18_Weights.DEFAULT)
resnet_model.fc = torch.nn.Identity()
resnet_model.to(device).eval()

shape_dir = Path("/Users/family/Desktop/finalcap/Ghost-Font-Reader/Data_Handling_Layer/shape-pngs")

descriptors = []
labels = []

for image_path in shape_dir.glob("*.png"):
    label_name = image_path.name.rsplit("_", 1)[0]
    if label_name not in LABELS:
        print(f"Skipping {image_path.name}: unknown label")
        continue

    descriptors.append(
        get_image_descriptor(image_path, resnet_model, device)
    )
    labels.append(LABELS[label_name])

descriptors = np.array(descriptors, dtype=np.float32)
labels = np.array(labels, dtype=int)

indices = np.random.permutation(len(descriptors))
descriptors = descriptors[indices]
labels = labels[indices]

split = int(len(descriptors) * 0.8)

classifier = ImageRecognizer(512)

train_model(
    descriptors[:split],
    labels[:split],
    classifier,
    epochs=5,
)

scores = classifier(descriptors[split:])

print("Accuracy:", compute_accuracy(scores, labels[split:]))

save_weights(
    classifier,
    current_dir / "image_recognizer_weights.pkl",
)
