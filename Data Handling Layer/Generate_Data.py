import os
import string
import random
import argparse

import numpy as np
import cv2

from noise_generator import NoiseAnimator

# Generate random text to be hidden in the ghost font
def random_text(min_len=3, max_len=8):
    length = random.randint(min_len, max_len)
    return "".join(random.choices(string.ascii_uppercase, k=length))

# Generate randomized parameters for each file in the dataset
def random_params(width, height):
    return dict(
        text=random_text(),
        font_size=random.randint(max(20, height // 6), max(40, height // 2)),
        position=(
            random.randint(width // 4, 3 * width // 4),
            random.randint(height // 4, 3 * height // 4),
        ),
        direction=random.choices(["horizontal", "vertical"]),
        animation_speed=random.uniform(0.5, 0.6),
        bg_noise_density=random.uniform(0.3, 0.7),
        fg_noise_density=random.uniform(0.3, 0.7),
        use_same_noise=random.random() < 0.5,
        speckle_size=random.choice([1, 1, 1, 2, 3]), 
    )

# Generate an mp4 video alongside a numpy file
def generate(video_idx, out_dir, width, height, fps, duration_seconds, save_mp4=True):
    animator = NoiseAnimator(width=width, height=height, fps=fps)
    params = random_params(width, height)

    animator.direction = params["direction"]
    animator.animation_speed = params["animation_speed"]
    animator.bg_noise_density = params["bg_noise_density"]
    animator.fg_noise_density = params["fg_noise_density"]
    animator.use_same_noise = params["use_same_noise"]
    animator.speckle_size = params["speckle_size"]
    animator.refresh_noise()

    total_frames = int(fps * duration_seconds)
    bg_offset = 0.0
    fg_offset = 0.0

    frames = np.zeros((total_frames, height, width), dtype=np.uint8)

    mask = animator.create_image_mask(
        params["text"], params["position"], font_size=params["font_size"]
    )

    writer=None
    if save_mp4:
        fourcc = cv2.VideoWriter_fourcc(*"mp4")
        mp4_path = os.path.join(out_dir, f"video_{video_idx:05d}.mp4")
        writer = cv2.VideoWriter(mp4_path, fourcc, fps, (width, height))

    for t in range(total_frames):
        bg_offset += params["animation_speed"]
        fg_offset += params["animation_speed"]
        frame_bgr = animator.animate_frame_vectorized(mask, bg_offset, fg_offset)

        frames[t] = frame_bgr[:,:,0]
        if writer is not None:
            writer.write(frame_bgr)

    if writer is None:
        writer.release()

    npz_path = os.path.join(out_dir, f"video_{video_idx:05d}.npz")
    np.savez_compressed(
        npz_path,
        frames=frames,
        text=params["text"],
        font_size=params["font_size"],
        position=np.array(params["position"]),
        direction=params["direction"],
        animation_speed=params["animation_speed"],
        bg_noise_density=params["bg_noise_density"],
        fg_noise_density=params["fg_noise_density"],
        use_same_noise=params["use_same_noise"],
        speckle_size=params["speckle_size"],
    )

    return npz_path