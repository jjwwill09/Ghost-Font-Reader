import os
import glob
import cv2
import numpy as np

SRC_DIR = "text_data_pngs"
OUT_DIR = "char_crops"


def segment_characters(img):
    threshold_type = cv2.THRESH_BINARY if np.median(img) < 127 else cv2.THRESH_BINARY_INV
    _, thresh = cv2.threshold(img, 127, 255, threshold_type)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = img.shape
    boxes = []
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        touches_edge = (
            x == 0
            or y == 0
            or x + box_width == width
            or y + box_height == height
        )
        if cv2.contourArea(contour) > 20 and not touches_edge:
            boxes.append((x, y, box_width, box_height))
    boxes.sort(key=lambda b: b[0])

    # merge components whose x-ranges overlap significantly (dot + stem cases)
    merged = []
    for box in boxes:
        x, y, w, h = box
        merged_into_existing = False
        for i, (mx, my, mw, mh) in enumerate(merged):
            overlap = max(0, min(x + w, mx + mw) - max(x, mx))
            smaller_w = min(w, mw)
            if overlap > 0.5 * smaller_w:  # significant x-overlap -> same character
                nx = min(x, mx)
                ny = min(y, my)
                nx2 = max(x + w, mx + mw)
                ny2 = max(y + h, my + mh)
                merged[i] = (nx, ny, nx2 - nx, ny2 - ny)
                merged_into_existing = True
                break
        if not merged_into_existing:
            merged.append(box)

    merged.sort(key=lambda b: b[0])
    return merged


def extract_all(src_dir=SRC_DIR, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(src_dir, "*.png")))

    kept = 0
    skipped = 0
    char_counts = {}

    for f in files:
        label = os.path.splitext(os.path.basename(f))[0].split("_")[0]
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if img is None:
            skipped += 1
            continue

        boxes = segment_characters(img)

        if len(boxes) != len(label):
            skipped += 1
            continue

        base = os.path.splitext(os.path.basename(f))[0]
        for i, (x, y, w, h) in enumerate(boxes):
            char = label[i]
            pad = 5
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)
            crop = img[y1:y2, x1:x2]

            char_counts[char] = char_counts.get(char, 0) + 1
            safe_char = f"U{ord(char)}"  # avoid filesystem case-collision issues (a vs A on Windows)
            out_path = os.path.join(out_dir, f"{safe_char}_{base}_{i}.png")
            cv2.imwrite(out_path, crop)

        kept += 1

    print(f"Kept {kept} images ({sum(char_counts.values())} character crops), skipped {skipped}")
    print("\nPer-character crop counts:")
    for char in sorted(char_counts):
        print(f"  {char!r}: {char_counts[char]}")


if __name__ == "__main__":
    extract_all()
