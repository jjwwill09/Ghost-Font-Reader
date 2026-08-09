from pathlib import Path

import numpy as np
import torch
from torchvision.models import resnet18, ResNet18_Weights

from typeclassifier import (
    CLASS_NAMES,
    ShapeTextClassifier,
    compute_accuracy,
    get_image_descriptor,
    load_weights,
)


current_dir = Path(__file__).resolve().parent

device = "cuda" if torch.cuda.is_available() else "cpu"

resnet_model = resnet18(weights=ResNet18_Weights.DEFAULT)
resnet_model.fc = torch.nn.Identity()
resnet_model.to(device).eval()

classifier = ShapeTextClassifier(512)
load_weights(classifier, current_dir / "shape_text_classifier_weights.pkl")

descriptors = []
labels = []
image_ids = []

for folder_path, label, label_name in [
    (Path("/Users/family/Desktop/finalcap/Ghost-Font-Reader/Data_Handling_Layer/shape-test-pngs"), 0, "shape"),
    (Path("/Users/family/Desktop/finalcap/Ghost-Font-Reader/Data_Handling_Layer/text-test-pngs"), 1, "text"),
]:
    for image_path in folder_path.glob("*.png"):
        descriptors.append(
            get_image_descriptor(image_path, resnet_model, device)
        )
        labels.append(label)
        image_ids.append(f"{label_name}/{image_path.name}")

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
        actual = CLASS_NAMES[labels[index]]
        predicted = CLASS_NAMES[predicted_indices[index]]
        print(f"{image_ids[index]} | actual: {actual} | predicted: {predicted}")
