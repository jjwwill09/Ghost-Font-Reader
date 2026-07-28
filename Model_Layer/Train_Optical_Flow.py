import os
import glob
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Load compressed np files of videos and use Farneback Optical Flow to convert them into training data
class OpticalFlowDataset(Dataset):
    def __init__(self, data_dir=None, files=None, frame_stride=1, farneback_params=None, augment=False):
        if files is not None:
            self.files = list(files)
        elif data_dir is not None:
            self.files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        else:
            raise ValueError("Must provide either data_dir or files")

        if not self.files:
            raise FileNotFoundError(f"No provided .npz files found (data_dir={data_dir})")

        self.frame_stride = frame_stride
        self.augment = augment
        self.farneback_params = farneback_params or dict(
            pyr_scale=0.5, levels=3, winsize=11,
            iterations=5, poly_n=7, poly_sigma=1.5, flags=0,
        )

        self._video_cache = {}
        self.index = []
        for file_idx, f in enumerate(self.files):
            with np.load(f) as data:
                n_frames = data["frames"].shape[0]
            for t in range(0, n_frames - frame_stride):
                self.index.append((file_idx, t))

    # Returns valid number of frame pairs across the video
    def __len__(self):
        return len(self.index)

    # Video memory cache mechanism (Quicker Data Loading)
    def _get_video(self, file_idx):
        if file_idx not in self._video_cache:
            with np.load(self.files[file_idx]) as data:
                self._video_cache[file_idx] = data["frames"]
        return self._video_cache[file_idx]

    # Random horizontal and vertical flip applied consistently to both frame and flow
    def _augment(self, frame_pair, flow):
        if random.random() > 0.5:
            frame_pair = frame_pair[:,:,::-1].copy()
            flow = flow[:, :, ::-1].copy()
            flow[0, :, :] *= -1
        if random.random() < 0.5:
            frame_pair = frame_pair[:, ::-1, :].copy()
            flow = flow[:, ::-1, :].copy()
            flow[1, :, :] *= -1
        return frame_pair, flow

    # Fetches frame pair and Farnback output
    def __getitem__(self, idx):
        file_idx, t = self.index[idx]
        frames = self._get_video(file_idx)
        f0 = frames[t]
        f1 = frames[t + self.frame_stride]

        # Computes Farneback optical flow for training data for our model
        flow = cv2.calcOpticalFlowFarneback(f0, f1, None, **self.farneback_params)

        frame_pair = np.stack([f0, f1], axis=0).astype(np.float32) / 255.0
        flow = flow.astype(np.float32).transpose(2, 0, 1)

        if self.augment:
            frame_pair, flow = self._augment(frame_pair, flow)

        return torch.from_numpy(frame_pair), torch.from_numpy(flow)

# Standard convolutional block
def conv_block(in_ch, out_ch, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1), # Extracts spacial features using a 3x3 grid (A window size of 9)
        nn.BatchNorm2d(out_ch), # Normalizes outputs for faster training
        nn.LeakyReLU(0.1, inplace=True), # Activation function to add non-linearity
    )

# 1,465,762 Parameter U-Net Style Neural Network Architecture
class UNet(nn.Module):
    def __init__(self, base_ch=32):
        super().__init__()
        # Encoder (Down Sampling/Compression): 389,088 Parameters
        c = base_ch
        self.enc1 = conv_block(2, c, stride=1) # 672 Params
        self.enc2 = conv_block(c, c * 2, stride=2) # 18,624 Params
        self.enc3 = conv_block(c * 2, c * 4, stride=2) # 74,112 Params
        self.enc4 = conv_block(c * 4, c * 8, stride=2) # 295,680 Params

        # Decoder (Up sampling/Reconstruction): 1,076,096 Parameters
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, kernel_size=4, stride=2, padding=1) # 524,416 Params
        self.dec3 = conv_block(c * 8, c * 4) # 295,296 Params

        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=4, stride=2, padding=1) # 131,136 Params
        self.dec2 = conv_block(c * 4, c * 2) # 73,920 Params

        self.up1 = nn.ConvTranspose2d(c * 2, c, kernel_size=4, stride=2, padding=1) # 32,800 Params
        self.dec1 = conv_block(c * 2, c) # 18,528 Params

        # Flow Head: 578 Parameters
        self.flow_head = nn.Conv2d(c, 2, kernel_size=3, padding=1) # 578 Params

    # Helps fix size mismatches when pairing layers (Padding)
    def _pad_and_cat(self, upsampled, skip):
        diffY = skip.size()[2] - upsampled.size()[2]
        diffX = skip.size()[3] - upsampled.size()[3]

        upsampled = F.pad(upsampled, [
            diffX // 2, diffX - diffX // 2,
            diffY // 2, diffY - diffY // 2
        ])
        return torch.cat([upsampled, skip], dim=1)

    # Forward feed structure
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        d3 = self.up3(e4)
        d3 = self._pad_and_cat(d3, e3)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = self._pad_and_cat(d2, e2)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = self._pad_and_cat(d1, e1)
        d1 = self.dec1(d1)

        return self.flow_head(d1)

"""
This was originally here to test an MLP model against the U-Net one.
Considerably trivial, but open for teasting still.

class DenseFlowMLP(nn.Module):
    def __init__(self, height, width, hidden=2048):
        super().__init__()
        self.height = height
        self.width = width
        in_dim = 2 * height * width
        out_dim = 2 * height * width
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        b = x.shape[0]
        out = self.net(x.reshape(b, -1))
        return out.reshape(b, 2, self.height, self.width)
"""
        
# Endpoint error loss
def epe_loss(pred_flow, gt_flow):
    diff = pred_flow - gt_flow
    epe = torch.sqrt(torch.sum(diff**2, dim=1) + 1e-8)
    return epe.mean()

# Splits .npz files into train/val before building datasets
def _split_files_by_video(data_dir, val_split, seed=42):
    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")

    rng = random.Random(seed)
    files = files[:]
    rng.shuffle(files)

    n_val = max(1, int(len(files) * val_split))
    val_files = files[:n_val]
    train_files = files[n_val:]
    return train_files, val_files

# Train neural network
def train(
    # CONFIG: HYPERPARAMETERS <----------------------------------------------------------------------------------------------------------------------
    data_dir="data",
    epochs=20,
    batch_size=8,
    lr=1e-4,
    val_split=0.1,
    checkpoint_dir="Model_Files",
    model_type="conv",
    device=None,
):
    # Enviornment and Data setup
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_files, val_files = _split_files_by_video(data_dir, val_split)
    print(f"Videos: {len(train_files)} train / {len(val_files)} val")
    train_set = OpticalFlowDataset(files=train_files, augment=True)
    val_set = OpticalFlowDataset(files=val_files, augment=True)
    print(f"Frame pairs: {len(train_set)} tain / {len(val_set)} val")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)

    sample_frames, _ = train_set[0]
    _, h, w = sample_frames.shape

    if model_type == "mlp":
        # model = DenseFlowMLP(h, w).to(device) # Uncomment this to use MLP Architecture
        print("")
    else:
        model = UNet().to(device)

    # Main training configurations (Behavior)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) # Dynamicaly adjust learning rate
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs) # Slowers the learning rate (half cosine curve)

    best_val_epe = float("inf")
    val_loss = float("inf")

    # Training and validation loop
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for frames, gt_flow in train_loader:
            frames, gt_flow = frames.to(device), gt_flow.to(device)

            optimizer.zero_grad()
            pred_flow = model(frames)
            loss = epe_loss(pred_flow, gt_flow)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * frames.size(0)
        train_loss /= len(train_set)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for frames, gt_flow in val_loader:
                frames, gt_flow = frames.to(device), gt_flow.to(device)
                pred_flow = model(frames)
                loss = epe_loss(pred_flow, gt_flow)
                val_loss += loss.item() * frames.size(0)
        val_loss /= len(val_set)

        scheduler.step()

        print(f"Epoch {epoch:3d}/{epochs} | train EPE {train_loss:.4f} | val EPE {val_loss:.4f}")

        # Keeps track of the best model while training
        if val_loss < best_val_epe:
            best_val_epe = val_loss
            torch.save(
                {"model_state": model.state_dict(), "epoch": epoch, "val_epe": val_loss,
                 "model_type": model_type, "height": h, "width": w},
                 os.path.join(checkpoint_dir, "best_conv_model.pt"),
            )

    # Saves the final model
    torch.save(
        {"model_state": model.state_dict(), "epoch": epochs, "val_epe": val_loss,
         "model_type": model_type, "height": h, "width": w},
         os.path.join(checkpoint_dir, "final_conv_model.pt"),
    )
    print(f"Done. Best val EPE: {best_val_epe:.4f}")

if __name__ == "__main__":
    train()