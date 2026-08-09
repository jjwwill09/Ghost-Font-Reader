import pickle
from pathlib import Path

import mygrad as mg
import numpy as np

from mygrad.nnet.initializers import glorot_normal
from mynn.optimizers.sgd import SGD
import mynn.layers.dense as dense


SHAPE_NAMES = [
    "circle",
    "rectangle",
    "polygon_3",
    "polygon_4",
    "polygon_5",
    "polygon_6",
]
LABELS = {name: index for index, name in enumerate(SHAPE_NAMES)}


class ImageRecognizer:
    """
    Takes a ResNet-18 image descriptor and produces shape classification scores.
    """

    def __init__(self, input_dim=512):
        self.dense = dense(
            input_size=input_dim,
            output_size=len(SHAPE_NAMES),
            weight_initializer=glorot_normal,
            bias=True,
        )

    def __call__(self, descriptors):
        descriptors = np.asarray(descriptors, dtype=np.float32)
        if descriptors.ndim == 1:
            descriptors = descriptors.reshape(1, -1)
        return self.dense(descriptors)

    @property
    def parameters(self):
        return self.dense.parameters


def compute_accuracy(scores, labels):
    score_data = scores.data if isinstance(scores, mg.Tensor) else scores
    predicted_indices = np.argmax(score_data, axis=1)
    labels = np.asarray(labels, dtype=int)
    return float(np.mean(predicted_indices == labels))


def compute_loss(scores, labels, margin=1.0):
    labels = np.asarray(labels, dtype=int)
    batch_indices = np.arange(len(labels))
    correct_scores = scores[batch_indices, labels]
    losses = mg.maximum(
        0.0,
        margin - correct_scores.reshape(-1, 1) + scores,
    )
    mask = np.ones_like(losses.data)
    mask[batch_indices, labels] = 0.0
    return mg.mean(losses * mask)


def train_model(training_descriptors, training_labels, model, batch_size=32, learning_rate=1e-3, momentum=0.9, epochs=5):
    training_descriptors = np.asarray(training_descriptors, dtype=np.float32)
    training_labels = np.asarray(training_labels, dtype=int)
    optimizer = SGD(model.parameters, learning_rate=learning_rate, momentum=momentum)

    for epoch in range(epochs):
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
            loss.null_grad()

            epoch_losses.append(float(loss.data))
            epoch_accuracies.append(accuracy)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Loss: {float(np.mean(epoch_losses)):.4f} | "
            f"Accuracy: {float(np.mean(epoch_accuracies)):.4f}"
        )

    return model


def classify_images(image_ids, descriptors, model):
    image_ids = list(image_ids)
    descriptors = np.asarray(descriptors, dtype=np.float32)
    classified_images = {name: [] for name in SHAPE_NAMES}
    scores = model(descriptors)
    score_data = scores.data if isinstance(scores, mg.Tensor) else scores
    predicted_indices = np.argmax(score_data, axis=1)

    for image_id, predicted_index in zip(image_ids, predicted_indices):
        predicted_class = SHAPE_NAMES[int(predicted_index)]
        classified_images[predicted_class].append(image_id)

    return classified_images


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
