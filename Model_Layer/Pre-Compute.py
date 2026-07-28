import os
import glob
import numpy as np
import cv2
from tqdm import tqdm

def precompute_flow(raw_data_dir="data", output_dir="processed_data"):
    os.makedirs(output_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(raw_data_dir, "*.npz")))

    fb_params = dict(
        pyr_scale=0.5, levels=3, winsize=11,
        iterations=5, poly_n=7, poly_sigma=1.5, flags=0,
    )

    pair_counter = 0
    print(f"Pre-computing flow for {len(files)} videos...")

    for f in tqdm(files):
        with np.load(f) as data:
            frames = data["frames"]

        n_frames = frames.shape[0]
        video_name = os.path.splitext(os.path.basename(f))[0]

        for t in range(n_frames - 1):
            f0 = frames[t]
            f1 = frames[t+1]

            flow = cv2.calcOpticalFlowFarneback(f0, f1, None, **fb_params)

            out_path = os.path.join(output_dir, f"{video_name}_pair_{t:04d}.npz")
            np.savez_compressed(
                out_path,
                f0=f0,
                f1=f1,
                flow=flow.astype(np.float32)
            )
            pair_counter += 1
    print(f"\nDone! saved {pair_counter} total pairs to '{output_dir}'.")

if __name__ == "__main__":
    precompute_flow()