import os
import glob
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm import tqdm

# Handle loading pre-processed and chunked data files
class OpticalFlowDataset(Dataset):
    def __init__(self, chunk_files, augment=False):
        self.chunk_files = list(chunk_files)
        if not self.chunk_files:
            raise FileNotFoundError("No chunk files provided")
        self.augment = augment

        self.chunk_lengths = []
        for f in self.chunk_files:
            with np.load(f) as data:
                self.chunk_lengths.append(data["f0"].shape[0])

        self.chunk_offsets = np.cumsum([0] + self.chunk_lengths)
        self.total_pairs = int(self.chunk_offsets[-1])

        self._cached_chunk_idx = None
        self._cached_chunk = None

    # Returns the total pairs
    def __len__(self):
        return self.total_pairs

    # Loads and manages memory cache for active chunks
    def _load_chunk(self, chunk_idx):
        if chunk_idx == self._cached_chunk_idx:
            return self._cached_chunk

        with np.load(self.chunk_files[chunk_idx]) as data:
            self._cached_chunk = {
                "f0": np.array(data["f0"]),
                "f1": np.array(data["f1"]),
                "flow": np.array(data["flow"]),
            }
        self._cached_chunk_idx = chunk_idx # Updates tracked index parameter
        return self._cached_chunk

    # Locates chunk item indexs
    def _locate(self, idx):
        chunk_idx = int(np.searchsorted(self.chunk_offsets, idx, side="right") - 1)
        local_idx = idx - self.chunk_offsets[chunk_idx]
        return chunk_idx, local_idx

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
        chunk_idx, local_idx = self._locate(idx)
        chunk = self._load_chunk(chunk_idx)

        # Read chunked data
        f0 = chunk["f0"][local_idx]
        f1 = chunk["f1"][local_idx]
        flow = chunk["flow"][local_idx]

        # Structure like original tensor configurations
        frame_pair = np.stack([f0, f1], axis=0).astype(np.float32) / 255.0
        flow = flow.astype(np.float32).transpose(2, 0, 1)

        if self.augment:
            frame_pair, flow = self._augment(frame_pair, flow)

        return torch.from_numpy(frame_pair), torch.from_numpy(flow)

# Shuffles data within chunks to keep harddrive reads efficient
class ChunkAwareShuffleSampler(Sampler):
    def __init__(self, chunk_offsets, seed=0):
        self.chunk_offsets = chunk_offsets
        self.num_chunks = len(self.chunk_offsets) - 1
        self.seed = seed
        self.epoch = 0

    # Updates epoch
    def set_epoch(self, epoch):
        self.epoch = epoch

    # Creates a randomized list of indicies
    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        chunk_order = list(range(self.num_chunks))
        rng.shuffle(chunk_order)

        for c in chunk_order:
            start, end = int(self.chunk_offsets[c]), int(self.chunk_offsets[c+1])
            local_indicies = list(range(start, end))
            rng.shuffle(local_indicies)
            yield from local_indicies

    # Returns length
    def __len__(self):
        return int(self.chunk_offsets[-1])

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
        
# Endpoint error loss
def epe_loss(pred_flow, gt_flow):
    diff = pred_flow - gt_flow
    epe = torch.sqrt(torch.sum(diff**2, dim=1) + 1e-8)
    return epe.mean()

# Splits .npz files into train/val before building datasets
def _split_chunks(data_dir, val_split, seed=42):
    chunk_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))

    if not chunk_files:
        raise FileNotFoundError(f"No chunk_*.npz files found in {data_dir}")

    rng = random.Random(seed)
    chunk_files = chunk_files[:]
    rng.shuffle(chunk_files)

    n_val = max(1, int(len(chunk_files) * val_split))
    val_files = chunk_files[:n_val]
    train_files = chunk_files[n_val:]
    return train_files, val_files

# Train neural network
def train(
    # CONFIG: HYPERPARAMETERS <----------------------------------------------------------------------------------------------------------------------
    data_dir="chunked_data",
    epochs=20,
    batch_size=32,
    lr=4e-4,
    val_split=0.1,
    checkpoint_dir="Model_Files",
    device=None,
):
    # Enviornment and Data setup
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_chunks, val_chunks = _split_chunks(data_dir, val_split)
    print(f"Chunks: {len(train_chunks)} train / {len(val_chunks)} val")

    train_set = OpticalFlowDataset(train_chunks, augment=True)
    val_set = OpticalFlowDataset(val_chunks, augment=False)

    print(f"Total Entries: {len(train_set) + len(val_set)} | Train Split: {len(train_set)} | Val Split: {len(val_set)}")

    train_sampler = ChunkAwareShuffleSampler(train_set.chunk_offsets, seed=42)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')

    # Training and validation loop
    print("\n=== Training Init Complete. Starting Optimization Loops ===")
    for epoch in range(1, epochs + 1):
        train_sampler.set_epoch(epoch)
        model.train()
        train_loss = 0.0
        for frames, gt_flow in train_loader:
            frames, gt_flow = frames.to(device), gt_flow.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred_flow = model(frames)
            loss = epe_loss(pred_flow, gt_flow)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * frames.size(0)
        epoch_train_loss = train_loss / len(train_set)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for frames, gt_flow in val_loader:
                frames, gt_flow = frames.to(device), gt_flow.to(device)
                pred_flow = model(frames)
                loss = epe_loss(pred_flow, gt_flow)
                val_loss += loss.item() * frames.size(0)
        epoch_val_loss = val_loss / len(val_set)

        print(f"Epoch {epoch:02d}/{epochs:02d} | train EPE {epoch_train_loss:.4f} | val EPE {epoch_val_loss:.4f}")

        # Keeps track of the best model while training
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            save_path = os.path.join(checkpoint_dir, "model_best.pth")
            torch.save(
                {"model_state": model.state_dict(), 
                 "optimizer_state": optimizer.state_dict(),
                 "epoch": epoch + 1, 
                 "best_loss": best_val_loss,
                }, 
                save_path)

if __name__ == "__main__":
    train()