import os
import string
import random

import numpy as np
import cv2

from noise_generator import NoiseAnimator

# Generate random text to be hidden in the ghost font
def random_text(min_len=3, max_len=8):
    length = random.randint(min_len, max_len)
    return "".join(random.choices(string.ascii_letters, k=length))
def random_shape():
    options = ["circle", "rectangle"] + [f"polygon_{sides}" for sides in range(3, 11)]
    return random.choice(options)

# Random text size at explicit ratios so it isnt under represented
def sample_font_size(height):
    r = random.random()
    if r < 0.45:
        lo, hi = 0.10, 0.18
    elif r <0.8:
        lo, hi = 0.18, 0.32
    else:
        lo, hi = 0.32, 0.55
    return(max(14, int(height * random.uniform(lo, hi))))

def random_content(content_type, content=None):
    if content:
        return content
    if content_type == "text":
        return random_text()
    return random_shape()

# Generate randomized parameters for each file in the dataset
def random_params(width, height, content_type="shape", content=None, text_variation=True):
    if text_variation:
        chosen_size = sample_font_size(height)
    else:
        chosen_size = int(height * 0.32)

    return dict(
        content_type=content_type,
        content=random_content(content_type, content),
        size=chosen_size,
        #font_size=random.randint(max(20, height // 6), max(40, height // 2)),
        #text=random_text(),
        position=(
            random.randint(width // 4, 3 * width // 4),
            random.randint(height // 4, 3 * height // 4),
        ),
        direction="horizontal",
        animation_speed=random.uniform(0.5, 0.6),
        bg_noise_density=random.uniform(0.3, 0.7),
        fg_noise_density=random.uniform(0.3, 0.7),
        use_same_noise=random.random() < 0.5,
        speckle_size=random.choice([1, 1, 1, 2, 3]), 
)

# Generate an mp4 video alongside a numpy file
def generate(video_idx, out_dir, width, height, content_type, content, fps, duration_seconds, save_mp4=True, text_variation=True):
    animator = NoiseAnimator(width=width, height=height, fps=fps)
    params = random_params(width, height, content_type, content, text_variation)

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

    if params["content_type"] == "text":
        mask = animator.create_text_mask(
            params["content"],
            params["position"],
            font_size=params["size"],
        )
    else:
        mask = animator.create_shape_mask(
            params["content"],
            params["position"],
            size=params["size"],
        )

    writer=None
    mp4_path = None 
    if save_mp4:
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        mp4_path = os.path.join(out_dir, f"{params['content']}_{video_idx:05d}.mp4")
        writer = cv2.VideoWriter(mp4_path, fourcc, fps, (width, height))

    for t in range(total_frames):
        bg_offset += params["animation_speed"]
        fg_offset -= params["animation_speed"]
        frame_bgr = animator.animate_frame_vectorized(mask, bg_offset, fg_offset)


        frames[t] = frame_bgr[:,:,0]
        if writer is not None:
            writer.write(frame_bgr)

    if writer is not None:
        writer.release()

    return mp4_path if save_mp4 else None

    npz_path = os.path.join(out_dir, f"video_{video_idx:05d}.npz")
    np.savez_compressed(
        npz_path,
        frames=frames,
        shape=params["shape"],
        size=params["size"],
        position=np.array(params["position"]),
        direction=params["direction"],
        animation_speed=params["animation_speed"],
        bg_noise_density=params["bg_noise_density"],
        fg_noise_density=params["fg_noise_density"],
        use_same_noise=params["use_same_noise"],
        speckle_size=params["speckle_size"],
    )

    return mp4_path if save_mp4 else npz_path

# Install given amount of data
def main(
    # Hyper Parameter Config
    num_videos=6,
    out_dir="test_shape_data", #rename, to train make one shape and one text and repeat for testing
    width=640,
    height=360,
    content_type="text", #rename as described above
    content=None,
    fps=30,
    duration_seconds=2.0,
    save_mp4=True,
    seed=None,
    text_variation=True,
):
    if seed is None:
        random.seed(seed)
        np.random.seed(seed)

    os.makedirs(out_dir, exist_ok=True)

    for i in range(num_videos):
        path = generate(i, out_dir, width, height, content_type, content, fps, duration_seconds, save_mp4=save_mp4, text_variation=text_variation)
        last_path = path
        print(f"[{i + 1}/{num_videos} saved path {path}]")

    print(f"Done. {num_videos} clips written to '{out_dir}'.")
    return last_path

if __name__ == "__main__":
    main()