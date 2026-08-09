import pickle
from pathlib import Path

import cv2
import numpy as np
import mygrad as mg
import torch

from mynn.optimizers.sgd import SGD
from mygrad.nnet.initializers import glorot_normal
import mynn.layers.dense as dense

from imagerecognizer import classify_images as classify_shape_images


CLASS_NAMES = ["shape", "text"]
LABELS = {"shape": 0, "text": 1}


class ShapeTextClassifier:
    """
    Takes a ResNet-18 image descriptor and produces classification scores (0 is shape, 1 is text)
    """

    def __init__(self, input_dim=512):
        self.dense = dense(input_size=input_dim, output_size=2, weight_initializer=glorot_normal, bias=True,)

    def __call__(self, descriptors):
        descriptors = np.asarray(descriptors, dtype=np.float32)

        # Convert one descriptor of shape (512,) into (1, 512)
        if descriptors.ndim == 1:
            descriptors = descriptors.reshape(1, -1)

        return self.dense(descriptors)

    @property
    def parameters(self):
        return self.dense.parameters

def compute_accuracy(scores, labels):
    if isinstance(scores, mg.Tensor):
        score_data = scores.data
    else:
        score_data = scores

    predicted_indices = np.argmax(score_data, axis=1)
    labels = np.asarray(labels, dtype=int)

    return float(np.mean(predicted_indices == labels))

def compute_loss(scores, labels, margin=1.0):
    labels = np.asarray(labels, dtype=int)
    batch_indices = np.arange(len(labels))
    correct_scores = scores[batch_indices, labels]
    incorrect_labels = 1 - labels
    incorrect_scores = scores[batch_indices, incorrect_labels]
    losses = mg.maximum(
        0.0,
        margin - correct_scores + incorrect_scores,
    )
    return mg.mean(losses)


def train_model(training_descriptors, training_labels, model, batch_size=32, learning_rate=1e-3, momentum=0.9, epochs=5):
    training_descriptors = np.asarray(training_descriptors, dtype=np.float32)
    training_labels = np.asarray(training_labels, dtype=int)
    if len(training_descriptors) != len(training_labels):
        raise ValueError(
            "training_descriptors and training_labels must have "
            "the same number of items."
        )
    optimizer = SGD(model.parameters, learning_rate=learning_rate, momentum=momentum)
    for epoch in range(epochs):
    # Shuffle the training data each epoch
        indices = np.random.permutation(len(training_descriptors))
        shuffled_descriptors = training_descriptors[indices]
        shuffled_labels = training_labels[indices]
        epoch_losses = []
        epoch_accuracies = []
        for start in range(0, len(shuffled_descriptors), batch_size):
            end = start + batch_size
            descriptor_batch = shuffled_descriptors[start:end]
            label_batch = shuffled_labels[start:end]
            scores = model(descriptor_batch)
            loss = compute_loss(scores, label_batch)
            accuracy = compute_accuracy(scores, label_batch)
            loss.backward()
            optimizer.step()
            # Remove gradients before the next batch
            loss.null_gradients()
            epoch_losses.append(float(loss.data))
            epoch_accuracies.append(accuracy)
        mean_loss = float(np.mean(epoch_losses))
        mean_accuracy = float(np.mean(epoch_accuracies))
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Loss: {mean_loss:.4f} | "
            f"Accuracy: {mean_accuracy:.4f}"
        )
    return model



def classify_images(image_ids, descriptors, model):
    image_ids = list(image_ids)
    descriptors = np.asarray(descriptors, dtype=np.float32)
    if len(image_ids) != len(descriptors):
        raise ValueError("image_ids and descriptors must contain the same # of items")
    classified_images = {"shape": [], "text": []}
    scores = model(descriptors)
    score_data = scores.data if isinstance(scores, mg.Tensor) else scores
    predicted_indices = np.argmax(score_data, axis=1)
    for image_id, predicted_index in zip(image_ids, predicted_indices):
        predicted_class = CLASS_NAMES[int(predicted_index)]
        classified_images[predicted_class].append(image_id)
    return classified_images

def classify_text(image_ids):
    #pass IDs into text classifier
    return list(image_ids)

def classify_shape(image_ids, descriptors=None, shape_model=None):
    if shape_model is None or descriptors is None:
        return list(image_ids)
    return classify_shape_images(image_ids, descriptors, shape_model)

def route_images(image_ids, descriptors, model, shape_model=None):
    classified_images = classify_images(image_ids, descriptors, model)
    text_results = classify_text(classified_images["text"])
    shape_indices = [list(image_ids).index(image_id) for image_id in classified_images["shape"]]
    shape_descriptors = np.asarray(descriptors, dtype=np.float32)[shape_indices]
    shape_results = classify_shape(classified_images["shape"], shape_descriptors, shape_model)
    return{"text": text_results,
           "shape": shape_results,}

def save_weights(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    weights = [
        parameter.data.copy()
        for parameter in model.parameters
    ]
    with open(path, "wb") as file:
        pickle.dump(weights, file)

def load_weights(model, path):
    path = Path(path)
    with open(path, "rb") as file:
        weights = pickle.load(file)
    for parameter, weight in zip(model.parameters, weights):
        parameter.data[...] = weight
    return model

def preprocess_image(image_path):
    image_path = Path(image_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(
            f"Could not read image: {image_path}"
        )
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = image.astype(np.float32) / 255.0
    mean = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32,
    )
    std = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32,
    )
    image = (image - mean) / std
    image = np.transpose(image, (2, 0, 1))
    image = image[np.newaxis, ...]
    return image

def get_image_descriptor(image_path, resnet_model, device="cpu"):
    image_array = preprocess_image(image_path)
    image_tensor = torch.from_numpy(image_array).to(device)
    resnet_model = resnet_model.to(device)
    resnet_model.eval()
    with torch.no_grad():
        descriptor = resnet_model(image_tensor)
    descriptor = descriptor.detach().cpu().numpy().reshape(-1)
    if descriptor.size != 512:
        raise ValueError(
            "ResNet returned a descriptor with "
            f"{descriptor.size} values instead of 512. "
            "Make sure its final classification layer was removed."
        )
    return descriptor.astype(np.float32)


def get_folder_descriptors(folder_path, resnet_model, device="cpu"):

    folder_path = Path(folder_path)

    if not folder_path.exists():
        raise FileNotFoundError(
            f"Image folder not found: {folder_path.resolve()}"
        )

    image_paths = sorted(folder_path.glob("*.png"))

    if not image_paths:
        raise ValueError(
            f"No PNG files found in {folder_path.resolve()}"
        )

    image_ids = []
    descriptors = []

    for image_path in image_paths:
        try:
            descriptor = get_image_descriptor(
                image_path,
                resnet_model,
                device=device,
            )

            image_ids.append(image_path.name)
            descriptors.append(descriptor)

        except (ValueError, OSError) as error:
            print(f"Skipping {image_path.name}: {error}")

    if not descriptors:
        raise ValueError(
            "No valid image descriptors were generated."
        )

    return (
        image_ids,
        np.asarray(descriptors, dtype=np.float32),
    )

