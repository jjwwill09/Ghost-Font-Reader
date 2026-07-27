import os
import glob
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Load compressed np files of videos and use Farneback Optical Flow to convert them into training data
class OpticalFlowDataset(Dataset):
    def __init__(self, data_dir, frame_stride=1, farneback_params=None):
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if not self.files:
            raise FileNotFoundError(f"No .npz files foundin {data_dir}")

        self.frame_stride = frame_stride
        self.farneback_params = farneback_params or dict(
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
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

        return torch.from_numpy(frame_pair), torch.from_numpy(flow)

# Standard convolutional block
def conv_block(in_ch, out_ch, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1), # Extracts spacial features using a 3x3 grid (A window size of 9)
        nn.BatchNorm2d(out_ch), # Normalizes outputs for faster training
        nn.LeakyReLU(0.1, inplace=True), # Activation function to add non-linearity
    )

class UNet(nn.Module):
    def __init__(self, base_ch=32):
        super().__init__()
        # Encoder (Down Sampling/Compression)
        c = base_ch
        self.enc1 = conv_block(2, c, stride=1)
        self.enc2 = conv_block(c, c * 2, stride=2)
        self.enc3 = conv_block(c * 2, c * 4, stride=2)
        self.enc4 = conv_block(c * 4, c * 8, stride=2)

        # Decoder (Up sampling/Reconstruction)
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, kernel_size=4, stride=2, padding=1)
        self.dec3 = conv_block(c * 8, c * 4)

        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=4, stride=2, padding=1)
        self.dec2 = conv_block(c * 4, c * 2)

        self.up1 = nn.ConvTranspose2d(c * 2, c, kernel_size=4, stride=2, padding=1)
        self.dec1 = conv_block(c * 2, c)

        self.flow_head = nn.Conv2d(c, 2, kernel_size=3, padding=1)

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

    dataset = OpticalFlowDataset(data_dir)
    print(f"Dataset loaded succesfully! Total frame pairs: {len(dataset)}")
    n_val = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)

    sample_frames, _ = dataset[0]
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