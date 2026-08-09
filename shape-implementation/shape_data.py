from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from imagerecognizer import LABELS, SHAPE_NAMES


SIDES = {
    "circle": 0,
    "rectangle": 4,
    "polygon_3": 3,
    "polygon_4": 4,
    "polygon_5": 5,
    "polygon_6": 6,
}


@dataclass(frozen=True)
class ShapeSample:
    path: Path
    label_name: str
    label: int
    sides: int


def parse_label(path):
    label_name = Path(path).stem.rsplit("_", 1)[0]
    return label_name if label_name in LABELS else None


def collect_samples(folder_path):
    samples = []
    for image_path in sorted(Path(folder_path).glob("*.png")):
        label_name = parse_label(image_path)
        if label_name is None:
            print(f"Skipping {image_path.name}: unknown label")
            continue
        samples.append(
            ShapeSample(
                path=image_path,
                label_name=label_name,
                label=LABELS[label_name],
                sides=SIDES[label_name],
            )
        )
    return samples


def filter_by_max_sides(samples, max_sides, include_circles=True):
    return [
        sample
        for sample in samples
        if (
            (sample.label_name == "circle" and include_circles)
            or sample.label_name == "rectangle"
            or sample.label_name.startswith("polygon_") and sample.sides <= max_sides
        )
    ]


def load_mask(image_path, image_size=96):
    image = Image.open(image_path).convert("L").resize(
        (image_size, image_size),
        Image.Resampling.BILINEAR,
    )
    image = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(image).unsqueeze(0)


class ShapeDataset(Dataset):
    def __init__(self, samples, image_size=96):
        self.samples = list(samples)
        self.image_size = image_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        return load_mask(sample.path, self.image_size), sample.label, str(sample.path)


def label_names():
    return list(SHAPE_NAMES)
