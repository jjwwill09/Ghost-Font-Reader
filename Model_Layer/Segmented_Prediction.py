import os
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from evaluate_png_predictions import load_ground_truth
 
from extract_char_crops import segment_characters
 
CHAR_IMG_SIZE = 64
charset = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz "
char_to_idx = {c: i for i, c in enumerate(charset)}
idx_to_char = {i: c for i, c in enumerate(charset)}
num_classes = len(charset)
 
 
def preprocess_char_crop(img):
    h, w = img.shape
    scale = min(CHAR_IMG_SIZE / w, CHAR_IMG_SIZE / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
 
    canvas = np.ones((CHAR_IMG_SIZE, CHAR_IMG_SIZE), dtype=np.uint8) * 255
    x_off = (CHAR_IMG_SIZE - new_w) // 2
    y_off = (CHAR_IMG_SIZE - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
 
    tensor = torch.tensor(canvas, dtype=torch.float32).unsqueeze(0) / 255.0
    tensor = 1 - tensor # Invert colors: white background, black text
    return tensor
 
 
class CharCropDataset(Dataset):
    def __init__(self, crop_dir="char_crops", training=False):
        self.files = sorted(glob.glob(os.path.join(crop_dir, "*.png")))
        if not self.files:
            raise FileNotFoundError(f"No crops found in {crop_dir}")
        self.training = training
 
    def __len__(self):
        return len(self.files)
 
    def __getitem__(self, idx):
        path = self.files[idx]
        base = os.path.basename(path)
        code = int(base.split("_")[0][1:])
        char = chr(code)
 
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(path)
 
        if self.training:
            img = augment(img)
 
        tensor = preprocess_char_crop(img)
        label = char_to_idx[char]
        return tensor, label
 
 
def augment(img):
    h, w = img.shape
    angle = np.random.uniform(-8, 8)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderValue=255)
    alpha = np.random.uniform(0.85, 1.15)
    beta = np.random.uniform(-15, 15)
    img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    return img
 
 
class CharCNN(nn.Module):
    def __init__(self, num_classes=num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64 -> 32
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 -> 16
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16 -> 8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
 
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
 
 
def train_classifier(
    crop_dir="char_crops",
    epochs=40,
    batch_size=32,
    lr=1e-3,
    val_split=0.15,
    checkpoint_path="Model_Files/Farne_Back_Models/char_cnn.pt",
    device=None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    print(f"Device = {device}")

    full_files = sorted(glob.glob(os.path.join(crop_dir, "*.png")))
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(full_files))
    n_val = max(1, int(len(full_files) * val_split))
    val_idx = set(indices[:n_val].tolist())

    train_files = [f for i, f in enumerate(full_files) if i not in val_idx]
    val_files = [f for i, f in enumerate(full_files) if i in val_idx]
    print(f"Train: {len(train_files)} crops, Val: {len(val_files)} crops")

    train_ds = CharCropDataset.__new__(CharCropDataset)
    train_ds.files = train_files
    train_ds.training = True

    val_ds = CharCropDataset.__new__(CharCropDataset)
    val_ds.files = val_files
    val_ds.training = False

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = CharCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        n = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            n += images.size(0)
 
        train_acc = correct / n
        train_loss = total_loss / n
 
        model.eval()
        val_correct, val_n = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_n += images.size(0)
        val_acc = val_correct / val_n
 
        print(f"Epoch {epoch:3d}/{epochs}  train loss {train_loss:.4f}  "
              f"train acc {train_acc:.2%}  val acc {val_acc:.2%}")
 
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model_state": model.state_dict()}, checkpoint_path.replace(".pt", "_best.pt"))
 
    print(f"Best val accuracy: {best_val_acc:.2%}")
    torch.save({"model_state": model.state_dict()}, checkpoint_path)


def predict_by_segmentation(image_path, checkpoint_path="Model_Files/Farne_Back_Models/char_cnn_best.pt", device=None):

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = CharCNN().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
 
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(image_path)
 
    boxes = segment_characters(img)
    if not boxes:
        return ""
 
    result = []
    with torch.no_grad():
        for (x, y, w, h) in boxes:
            pad = 5
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)
            crop = img[y1:y2, x1:x2]
            tensor = preprocess_char_crop(crop).unsqueeze(0).to(device)
            logits = model(tensor)
            pred_idx = logits.argmax(1).item()
            result.append(idx_to_char.get(pred_idx, ""))
 
    return "".join(result)


if __name__ == "__main__":
    train_classifier(epochs=40, batch_size=32, lr=1e-3)
 
    folder = "/workspaces/Ghost-Font-Reader/spooky-data_pngs"
    for i, filename in enumerate(os.listdir(folder)):
        truth = filename.split("_")[0]
        pred = predict_by_segmentation(os.path.join(folder, filename))
        print(f"{i}  Truth: {truth}  Prediction: {pred}")
 