from pathlib import Path

import numpy as np
import torch
from torchvision.models import resnet18, ResNet18_Weights

from imagerecognizer import (
    ImageRecognizer,
    LABELS,
    SHAPE_NAMES,
    compute_accuracy,
    load_weights,
)
from typeclassifier import get_image_descriptor


current_dir = Path(__file__).resolve().parent

device = "cuda" if torch.cuda.is_available() else "cpu"

resnet_model = resnet18(weights=ResNet18_Weights.DEFAULT)
resnet_model.fc = torch.nn.Identity()
resnet_model.to(device).eval()

classifier = ImageRecognizer(512)
load_weights(classifier, current_dir / "image_recognizer_weights.pkl")

shape_dir = Path("/Users/family/Desktop/finalcap/Ghost-Font-Reader/Data_Handling_Layer/shape-test-pngs")

descriptors = []
labels = []
image_ids = []

for image_path in shape_dir.glob("*.png"):
    label_name = image_path.name.rsplit("_", 1)[0]
    if label_name not in LABELS:
        print(f"Skipping {image_path.name}: unknown label")
        continue

    descriptors.append(
        get_image_descriptor(image_path, resnet_model, device)
    )
    labels.append(LABELS[label_name])
    image_ids.append(image_path.name)

descriptors = np.array(descriptors, dtype=np.float32)
labels = np.array(labels, dtype=int)

scores = classifier(descriptors)
predicted_indices = np.argmax(scores.data, axis=1)
accuracy = compute_accuracy(scores, labels)
wrong_indices = np.where(predicted_indices != labels)[0]

print(f"Total test images: {len(labels)}")
print(f"Correct: {len(labels) - len(wrong_indices)}")
print(f"Wrong: {len(wrong_indices)}")
print(f"Accuracy: {accuracy:.4f}")

if len(wrong_indices) > 0:
    print("\nMisclassified:")
    for index in wrong_indices:
        actual = SHAPE_NAMES[labels[index]]
        predicted = SHAPE_NAMES[predicted_indices[index]]
        print(f"{image_ids[index]} | actual: {actual} | predicted: {predicted}")
