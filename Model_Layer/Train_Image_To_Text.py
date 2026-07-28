import os
import cv2
import glob
import numpy as np
import torch
import torchvision.io as tvio
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

img = cv2.imread('clean_nn.png', cv2.IMREAD_UNCHANGED)

#Create a character set
charset = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz "
img_height = 180
img_width = 320 

char_to_idx = {c: i + 1 for i, c in enumerate(charset)}
idx_to_char = {i + 1: c for i, c in enumerate(charset)}
blank_idx = 0
num_classes = len(charset) + 1

def encode_text(text):
    return [char_to_idx[c] for c in text if c in char_to_idx]

def decode_text(code):
    return [idx_to_char[i] for i in code if i in idx_to_char]

def load_and_preprocess(path):
    tensor = tvio.read_image(path, mode=tvio.ImageReadMode.GRAY)
    tensor = tensor.float() / 255.0
    return tensor

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

        with np.load(npz_path, allow_pickle=True) as data:
            label_text = str(data["text"])

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
    def __init__(self, num_classes=num_classes, base_ch=32, hidden=128):
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
        )
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

def train(
        data_dir = "png_data",
        label_dir = "data",
        epochs = 50,
        batch_size = 8,
        lr = 1e-3,
        checkpoint_path = "Model_Files/Farne_Back_Models/crnn_ocr.pt",
        device = None
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    print (f"Device = {device}")
    dataset = TextLineDataset(data_dir, label_dir=label_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
 
    model = CRNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ctc_loss = nn.CTCLoss(blank=blank_idx, zero_infinity=True)
 
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for images, targets, target_lengths, _ in loader:
            images = images.to(device) 
            targets = targets.to(device) 
 
            log_probs = model(images) 
            input_lengths = torch.full(
                size=(images.size(0),), fill_value=log_probs.size(0), dtype=torch.long
            )
            # --- debug prints ---
            #print(f"input_lengths: {input_lengths.tolist()}")
            #print(f"target_lengths: {target_lengths.tolist()}")
            #print(f"any target_length == 0: {(target_lengths == 0).any().item()}")
            #print(f"any input_length < target_length: {(input_lengths < target_lengths).any().item()}")
            #print(f"targets contain blank_idx(0): {(targets == 0).any().item()}")
            #print(f"log_probs has NaN/Inf: {torch.isnan(log_probs).any().item()}, {torch.isinf(log_probs).any().item()}")
            # ---------------------
            loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
            #print(f"batch loss: {loss.item()}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
 
            total_loss += loss.item() * images.size(0)
 
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch:3d}/{epochs}, CTC loss {avg_loss:.4f}")
    try:
        torch.save({"model_state": model.state_dict()}, checkpoint_path)
        print(f"Saved model to {checkpoint_path}")
    except OSError as e:
        print(f"Couldn't save to {checkpoint_path}: {e}")
 
def predict(image_path, checkpoint_path="Model_Files/Farne_Back_Models/crnn_ocr.pt", device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print (f"Device = {device}") 
 
    model = CRNN().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
 
    image = load_and_preprocess(image_path).unsqueeze(0).to(device) 
    with torch.no_grad():
        log_probs = model(image)
    probs = log_probs.exp()  # (T, B, C) - convert log-probs to probs
    top2 = probs[:, 0, :].topk(2, dim=1)  # top-2 classes at each timestep, batch item 0
    for t in range(0, probs.size(0), 4):  # sample every 4th timestep to keep output short
        idxs = top2.indices[t].tolist()
        vals = top2.values[t].tolist()
        chars = [idx_to_char.get(i, "blank" if i == 0 else "?") for i in idxs]
        print(f"t={t:3d}  top1={chars[0]}({vals[0]:.2f})  top2={chars[1]}({vals[1]:.2f})")

    text = greedy_decode(log_probs)[0]
    return text

if __name__ == "__main__":
    crnnpath = "Model_Files/Farne_Back_Models/crnn_ocr.pt"
    #train(epochs=200, checkpoint_path=crnnpath, lr=1e-4)
    #print(predict("png_data/video_00002.png", checkpoint_path=crnnpath))
    print(predict("clean_nn.png", checkpoint_path=crnnpath))