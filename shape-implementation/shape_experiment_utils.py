import csv
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from shape_data import label_names


def image_stats(image_path):
    image = np.asarray(Image.open(image_path).convert("L"), dtype=np.uint8)
    threshold = otsu_threshold(image)
    mask = image > threshold
    if np.mean(mask) > 0.5:
        mask = ~mask

    area_ratio = float(np.mean(mask))
    ys, xs = np.where(mask)
    if len(xs) == 0:
        bbox_ratio = 0.0
    else:
        bbox_area = (xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)
        bbox_ratio = float(bbox_area / mask.size)

    gy, gx = np.gradient(image.astype(np.float32))
    edge_density = float(np.mean(np.sqrt(gx**2 + gy**2) > 40.0))

    return {
        "area_ratio": area_ratio,
        "bbox_ratio": bbox_ratio,
        "components": connected_components(mask),
        "edge_density": edge_density,
    }


def likely_reason(stats):
    if stats["area_ratio"] < 0.01:
        return "shape too small or mostly lost in extraction"
    if stats["area_ratio"] > 0.45:
        return "mask too large or background leaked into foreground"
    if stats["components"] > 3:
        return "fragmented mask/noise speckles"
    if stats["edge_density"] > 0.18:
        return "high edge noise"
    if stats["bbox_ratio"] < 0.03:
        return "small bounding box"
    return "geometric similarity or ambiguous mask"


def write_confusion(path, confusion):
    with Path(path).open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["actual\\predicted", *label_names()])
        for label_name, row in zip(label_names(), confusion.tolist()):
            writer.writerow([label_name, *row])


def write_mistakes(path, mistakes):
    mixups = Counter()
    reason_counts = Counter()

    with Path(path).open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "image",
            "actual",
            "predicted",
            "area_ratio",
            "bbox_ratio",
            "components",
            "edge_density",
            "likely_reason",
        ])

        for image_path, actual, predicted in mistakes:
            stats = image_stats(image_path)
            reason = likely_reason(stats)
            mixups[(actual, predicted)] += 1
            reason_counts[reason] += 1
            writer.writerow([
                str(image_path),
                actual,
                predicted,
                f"{stats['area_ratio']:.6f}",
                f"{stats['bbox_ratio']:.6f}",
                stats["components"],
                f"{stats['edge_density']:.6f}",
                reason,
            ])

    return {
        "top_mixups": [list(key) + [count] for key, count in mixups.most_common(10)],
        "reason_counts": dict(reason_counts),
    }


def otsu_threshold(image):
    counts = np.bincount(image.ravel(), minlength=256).astype(np.float64)
    total = image.size
    sum_total = np.dot(np.arange(256), counts)
    sum_background = 0.0
    weight_background = 0.0
    max_variance = -1.0
    threshold = 0

    for value in range(256):
        weight_background += counts[value]
        if weight_background == 0:
            continue
        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break
        sum_background += value * counts[value]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        if variance > max_variance:
            max_variance = variance
            threshold = value

    return threshold


def connected_components(mask):
    visited = np.zeros(mask.shape, dtype=bool)
    components = 0
    height, width = mask.shape

    for y in range(height):
        for x in range(width):
            if visited[y, x] or not mask[y, x]:
                continue
            components += 1
            stack = [(y, x)]
            visited[y, x] = True
            while stack:
                cy, cx = stack.pop()
                for ny in range(max(0, cy - 1), min(height, cy + 2)):
                    for nx in range(max(0, cx - 1), min(width, cx + 2)):
                        if not visited[ny, nx] and mask[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))

    return components
