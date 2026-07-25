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