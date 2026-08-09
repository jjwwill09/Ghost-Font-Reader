#not part of functional interface, include as optional training for future extra training
from pathlib import Path
import os
import cv2
import numpy as np
import torch
from torchvision.models import resnet18, ResNet18_Weights
from typeclassifier import (get_image_descriptor, compute_accuracy, save_weights, train_model, get_folder_descriptors, ShapeTextClassifier, classify_images)

current_dir = Path(__file__).resolve().parent

device = "cuda" if torch.cuda.is_available() else "cpu"

resnet_model = resnet18(weights=ResNet18_Weights.DEFAULT)
resnet_model.fc = torch.nn.Identity()
resnet_model.to(device).eval()

descriptors = []
labels = []

for folder_path, label in [
    (Path("/Users/family/Desktop/finalcap/Ghost-Font-Reader/Data_Handling_Layer/shape-pngs"), 0),
    (Path("/Users/family/Desktop/finalcap/Ghost-Font-Reader/Data_Handling_Layer/text-pngs"), 1),
]:
    for image_path in folder_path.glob("*.png"):
        descriptors.append(
            get_image_descriptor(image_path, resnet_model, device)
        )
        labels.append(label)

descriptors = np.array(descriptors, dtype=np.float32)
labels = np.array(labels, dtype=int)

indices = np.random.permutation(len(descriptors))
descriptors = descriptors[indices]
labels = labels[indices]

split = int(len(descriptors) * 0.8)

classifier = ShapeTextClassifier(512)

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
    current_dir / "shape_text_classifier_weights.pkl",
)
