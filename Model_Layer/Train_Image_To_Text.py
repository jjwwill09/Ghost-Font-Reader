import os
import cv2
import glob
import numpy as np
import torch
import torchvision.io as tvio
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR


#Create a character set
charset = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz "
img_height = 360
img_width = 640 

char_to_idx = {c: i + 1 for i, c in enumerate(charset)}
idx_to_char = {i + 1: c for i, c in enumerate(charset)}
blank_idx = 0
num_classes = len(charset) + 1

def encode_text(text):
    return [char_to_idx[c] for c in text if c in char_to_idx]

def decode_text(code):
    return [idx_to_char[i] for i in code if i in idx_to_char]
def resize_with_padding(img, target_w=640, target_h=360):

    h, w = img.shape

    scale = min(
        target_w / w,
        target_h / h
    )

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(
        img,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    canvas = np.ones(
        (target_h, target_w),
        dtype=np.uint8
    ) * 255

    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2

    canvas[
        y_offset:y_offset+new_h,
        x_offset:x_offset+new_w
    ] = resized

    return canvas


def preprocess_frame(img):
    _, thresh = cv2.threshold(
        img,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        x,y,w,h = cv2.boundingRect(
            max(contours, key=cv2.contourArea)
        )

        img = img[y:y+h, x:x+w]

    # resize with aspect-ratio-preserving padding instead of stretching.
    img = resize_with_padding(img, target_w=img_width, target_h=img_height)

    tensor = torch.tensor(
        img,
        dtype=torch.float32
    ).unsqueeze(0) / 255.0

    # invert
    tensor = 1 - tensor

    return tensor


def load_and_preprocess(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return preprocess_frame(img)



class TextLineDataset(Dataset):
    def __init__(self, data_dir, label_dir="data"):
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.png")))
        if not self.files:
            raise FileNotFoundError(f"No .png files found in {data_dir}")
        self.label_dir = label_dir
 
    def __len__(self):
        return len(self.files)
 
    def __getitem__(self, idx):
        path = self.files[idx]
        base = os.path.splitext(os.path.basename(path))[0]
        npz_path = os.path.join(self.label_dir, base + ".npz")

        label_text = base.rsplit("_", 1)[0]
        #print(base, label_text)
        image = load_and_preprocess(path)
        label = torch.tensor(encode_text(label_text), dtype=torch.long)
        return image, label, label_text
def collate_fn(batch):
    images, labels, texts = zip(*batch)
    images = torch.stack(images, dim=0)
    label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    targets = torch.cat(labels)
    return images, targets, label_lengths, texts

def conv_block(in_ch, out_ch, pool=True):
    layers = [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.1, inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)

class CRNN(nn.Module):
    def __init__(self, num_classes=num_classes, base_ch=32, hidden=128, rnn_dropout=0.45, fc_dropout=0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            conv_block(1, base_ch),          
            conv_block(base_ch, base_ch * 2), 
            conv_block(base_ch * 2, base_ch * 4, pool=False),
        )
        cnn_out_h = img_height // 4
        self.rnn_input_size = base_ch * 4 * cnn_out_h

        self.rnn = nn.LSTM(
            input_size=self.rnn_input_size,
            hidden_size=hidden,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=rnn_dropout  # regularization between the 2 layers
        )
        self.dropout = nn.Dropout(fc_dropout)
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, x):
        feat = self.cnn(x)
        b, c, h, w = feat.shape
        feat = feat.permute(0, 3, 1, 2) 
        feat = feat.reshape(b, w, c * h) 

        rnn_out, _ = self.rnn(feat)
        logits = self.fc(rnn_out)
 
        
        log_probs = F.log_softmax(logits, dim=2).permute(1, 0, 2)
        return log_probs

def greedy_decode(log_probs):
    preds = log_probs.argmax(dim=2).permute(1,0)

    res = []
    for seq in preds:
        chars = []
        prev = None
        for idx in seq.tolist():
            if idx != prev and idx != blank_idx:
                chars.append(idx_to_char.get(idx, ""))
            prev = idx
        res.append("".join(chars))
    return res

def edit_distance(a, b):
    """Levenshtein distance, used for a more informative val metric than raw loss."""
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        dp[i][0] = i
    for j in range(len(b) + 1):
        dp[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    return dp[-1][-1]
def evaluate(model, loader, ctc_loss, device):
    model.eval()
    total_loss = 0.0
    n = 0
    total_edit_dist = 0
    total_chars = 0
    exact_matches = 0
    with torch.no_grad():
        for images, targets, target_lengths, texts in loader:
            images = images.to(device)
            targets = targets.to(device)
            log_probs = model(images)
            input_lengths = torch.full(
                size=(images.size(0),), fill_value=log_probs.size(0), dtype=torch.long
            )
            loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
            total_loss += loss.item() * images.size(0)
            n += images.size(0)

            preds = greedy_decode(log_probs)
            for pred, true in zip(preds, texts):
                total_edit_dist += edit_distance(pred, true)
                total_chars += len(true)
                if pred == true:
                    exact_matches += 1
    avg_loss = total_loss / max(n, 1)
    char_error_rate = total_edit_dist / max(total_chars, 1)
    exact_acc = exact_matches / max(n, 1)
    return avg_loss, char_error_rate, exact_acc

def train(
        data_dir = "test_text-pngs",
        label_dir = "data",
        epochs = 50,
        batch_size = 8,
        lr = 1e-3,
        checkpoint_path = "Model_Files/Farne_Back_Models/crnn_ocr.pt",
        val_split = 0.15,
        weight_decay = 1e-4,
        device = None
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)#Can throw an error if there is no parent folder. NO 
    print (f"Device = {device}")
 
    full_files = sorted(glob.glob(os.path.join(data_dir, "*.png")))
    if not full_files:
        raise FileNotFoundError(f"No .png files found in {data_dir}")

    # deterministic split so train/val membership doesn't shuffle between runs
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(full_files))
    n_val = max(1, int(len(full_files) * val_split))
    val_idx = set(indices[:n_val].tolist())

    train_files = [f for i, f in enumerate(full_files) if i not in val_idx]
    val_files = [f for i, f in enumerate(full_files) if i in val_idx]
    print(f"Train: {len(train_files)} images, Val: {len(val_files)} images")

    train_dataset = TextLineDataset.__new__(TextLineDataset)
    train_dataset.files = train_files
    train_dataset.label_dir = label_dir
    train_dataset.training = True
 
    val_dataset = TextLineDataset.__new__(TextLineDataset)
    val_dataset.files = val_files
    val_dataset.label_dir = label_dir
    val_dataset.training = False
 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    model = CRNN().to(device)
    with torch.no_grad():
        model.fc.bias[blank_idx] = -1.0
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    ctc_loss = nn.CTCLoss(blank=blank_idx, zero_infinity=True)

    warmup_steps = 500
    total_steps = epochs * len(train_loader)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.1, 1 - (step - warmup_steps) / (total_steps - warmup_steps))
    
    scheduler = LambdaLR(optimizer, lr_lambda)

    best_val_cer = float("inf")
    best_epoch = -1

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for i, (images, targets, target_lengths, _) in enumerate(train_loader):
            images = images.to(device) 
            targets = targets.to(device) 
            log_probs = model(images) 
 
            input_lengths = torch.full(
                size=(images.size(0),), fill_value=log_probs.size(0), dtype=torch.long
            )
            loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * images.size(0)
 
        avg_loss = total_loss / len(train_dataset)
 
        if epoch % 5 == 0 or epoch == epochs:
            val_loss, val_cer, val_exact_acc = evaluate(model, val_loader, ctc_loss, device)
            print(f"Epoch {epoch:3d}/{epochs}, train loss {avg_loss:.4f}  "
                  f"val loss {val_loss:.4f}  val CER {val_cer:.3f}  val exact-match {val_exact_acc:.2%}")
            if val_cer < best_val_cer:
                best_val_cer = val_cer
                best_epoch = epoch
                torch.save({"model_state": model.state_dict()}, checkpoint_path.replace(".pt", "_best.pt"))
            model.eval()
            with torch.no_grad():
                test_img = load_and_preprocess("text_testing_pngs/mUid_00001.png").unsqueeze(0).to(device)
                log_probs = model(test_img)
                pred = greedy_decode(log_probs)[0]
            model.train()
            print("Test prediction: " + pred)   
        else:
            print(f"Epoch {epoch:3d}/{epochs}, train loss {avg_loss:.4f}")
 
    print(f"Best val CER {best_val_cer:.3f} at epoch {best_epoch} "
          f"(saved to {checkpoint_path.replace('.pt', '_best.pt')})")
    try:
        torch.save({"model_state": model.state_dict()}, checkpoint_path)
        print(f"Saved final model to {checkpoint_path}")
    except OSError as e:
        print(f"Couldn't save to {checkpoint_path}: {e}")

def locate_text_region(img):
    """
    Input: grayscale numpy array (any resolution), 
    Returns: cropped array containing the text region, or None
    """
    # Ensure text is white and background is black
    _, thresh = cv2.threshold(
        img,
        127,
        255,
        cv2.THRESH_BINARY_INV
    )

    # Find connected components
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Remove tiny noise
        if w*h > 20:
            boxes.append((x, y, w, h))

    if not boxes:
        return None

    # Combine all boxes into one large box
    x1 = min([x for x,y,w,h in boxes])
    y1 = min([y for x,y,w,h in boxes])

    x2 = max([x+w for x,y,w,h in boxes])
    y2 = max([y+h for x,y,w,h in boxes])

    # Add padding
    padding = 10

    x1 = max(0, x1-padding)
    y1 = max(0, y1-padding)
    x2 = min(img.shape[1], x2+padding)
    y2 = min(img.shape[0], y2+padding)

    return img[y1:y2, x1:x2]


def find_text_region(image_path, output_path="cropped_text.png"):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(image_path)

    cropped = locate_text_region(img)

    if cropped is None:
        print("No text detected")
        return None

    cv2.imwrite(output_path, cropped)

    h, w = cropped.shape
    #print(f"Found text region: w={w}, h={h}")

    return output_path


def predict(image_path, checkpoint_path="Model_Files/Farne_Back_Models/crnn_ocr.pt", device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    #print (f"Device = {device}") 
 
    model = CRNN().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only= True)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
 
    localized_path = find_text_region(image_path)

    if localized_path is None:
        return ""

    image = load_and_preprocess(localized_path).unsqueeze(0).to(device)
    with torch.no_grad():
        log_probs = model(image)
    probs = log_probs.exp()
    top2 = probs[:, 0, :].topk(2, dim=1)
    for t in range(0, probs.size(0), 4): 
        idxs = top2.indices[t].tolist()
        vals = top2.values[t].tolist()
        chars = [idx_to_char.get(i, "blank" if i == 0 else "?") for i in idxs]
        #print(f"t={t:3d}  top1={chars[0]}({vals[0]:.2f})  top2={chars[1]}({vals[1]:.2f})")
    #print(log_probs.exp().max(dim=2)[0].mean())
    text = greedy_decode(log_probs)[0]
    return text

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    crnnpath = os.path.join(project_root, "Model_Files", "Farne_Back_Models", "crnn_ocr_v4.pt")
    train(
        epochs=100, 
        batch_size=12, 
        checkpoint_path=crnnpath, 
        lr=1e-3, 
        data_dir=os.path.join(project_root, "Data_Handling_Layer", "text-pngs"),
        weight_decay = 2e-4
    )

 
    folder = os.path.join(project_root, "Data_Handling_Layer", "test_text-pngs")
    best_path = crnnpath.replace(".pt", "_best.pt")
    for i, filename in enumerate(os.listdir(folder)):
        pred = predict(os.path.join(folder, filename), checkpoint_path=best_path)
        truth = filename.rsplit("_", 1)[0]
        print(f"{i}  Truth: {truth}  Prediction: {pred}")
    
