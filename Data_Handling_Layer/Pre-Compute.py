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

def pack_to_chunks(pack_dir="processed_data", output_dir="chunked_data", chunk_size=1000):
    os.makedirs(output_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(pack_dir, "*.npz")))

    f0_list, f1_list, flow_list = [], [], []
    chunk_count = 0

    print(f"Grouping {len(files)} files into large arrays...")
    for idx, f in enumerate(tqdm(files)):
        with np.load(f) as data:
            f0_list.append(data["f0"])
            f1_list.append(data["f1"])
            flow_list.append(data["flow"])

        if len(f0_list) == chunk_size or idx == len(files) - 1:
            np.savez_compressed(
                os.path.join(output_dir, f"chunk_{chunk_count:03d}.npz"),
                f0=np.array(f0_list),
                f1=np.array(f1_list),
                flow=np.array(flow_list),
            )
            f0_list, f1_list, flow_list = [], [], []
            chunk_count += 1
    print(f"\nSuccess! Compressed files down to {chunk_count} chunks.")
if __name__ == "__main__":
    precompute_flow()
    print("Pre-Compute Complete")
    pack_to_chunks()
    print("Packing Complete")