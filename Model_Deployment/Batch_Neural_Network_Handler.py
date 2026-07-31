"""
Batch version of Neural_Network_Handler.py.
Use this to convert generated dataset folders into PNG folders.
"""

import importlib.util
import os
import cv2
import numpy as np
import torch
import time
from concurrent.futures import ThreadPoolExecutor

current_dir = os.path.dirname(__file__)
file_path = os.path.abspath(os.path.join(current_dir, "..", "Model_Layer", "Train_Optical_Flow.py"))
spec = importlib.util.spec_from_file_location("Train_Optical_Flow", file_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

UNet = module.UNet

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running Neural Network math on: {device}")

# Load the model from the saved PyTorch weights
checkpoint_path = "Model_Files/Farne_Back_Models/model_best_9.pth"
checkpoint = torch.load(checkpoint_path, map_location=device)

model = UNet().to(device)
model.load_state_dict(checkpoint["model_state"])
model.eval()

h_target = checkpoint.get("height", 360)
w_target = checkpoint.get("width", 640)

# --- DEBUG: confirm what resolution the checkpoint actually specifies ---
print(f"h_target={h_target} w_target={w_target}, checkpoint keys: {list(checkpoint.keys())}")
# -------------------------------------------------------------------------

script_dir = os.path.dirname(os.path.abspath(__file__))
data_root = os.path.abspath(os.path.join(script_dir, "..", ""))
datasets = [
    #(os.path.join(data_root, "shape_data"), os.path.join(data_root, "shape-pngs")),
    #(os.path.join(data_root, "test_shape_data"), os.path.join(data_root, "test_shape_pngs")),
    (os.path.join(data_root, "text_data"), os.path.join(data_root, "text_data_pngs"))
    #(os.path.join(data_root, "test_text_data"), os.path.join(data_root, "test_text-pngs")),
]

for input_dir, output_dir in datasets:
    if not os.path.isdir(input_dir):
        print(f"Skipping missing input folder: {input_dir}")
        continue

    os.makedirs(output_dir, exist_ok=True)
    video_paths = sorted(
        os.path.join(input_dir, filename)
        for filename in os.listdir(input_dir)
        if filename.endswith(".mp4")
    )

    print(f"Found {len(video_paths)} videos in {input_dir}")

def process_video(video_path):
    video_filename = os.path.basename(video_path)
    video_name = os.path.splitext(video_filename)[0]
    output_filename = os.path.join(output_dir, f"{video_name}.png")

    if os.path.exists(output_filename):
        print(f"Skipping {video_name}, already processed.")
        return

    print(f"Processing {video_name}...")

    cap = cv2.VideoCapture(video_path)

    ret, first_frame = cap.read()
    if not ret:
        print("Failed to grab video source.")
        cap.release()
        return

    prev_frame_resized = cv2.resize(first_frame, (w_target, h_target))
    prev_gray = cv2.cvtColor(prev_frame_resized, cv2.COLOR_BGR2GRAY)

    hsv_mask_nn = np.zeros_like(prev_frame_resized)

    BUFFER_DURATION = 3.0
    buffer_start_time = time.time()
    accumulated_mask = np.zeros((h_target, w_target), dtype=np.float32)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break

        frame_resized = cv2.resize(frame, (w_target, h_target))
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

        frame_pair = np.stack([prev_gray, gray], axis=0).astype(np.float32) / 255.0
        input_tensor = torch.from_numpy(frame_pair).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_flow = model(input_tensor)
            flow_nn = pred_flow.squeeze(0).cpu().numpy().transpose(1, 2, 0)

        dx_nn = flow_nn[..., 0]
        mag_nn = np.sqrt(flow_nn[..., 0]**2 + flow_nn[..., 1]**2)

        NN_MOTION_THRESHOLD = 2 / 10.0

        moving_right_nn = (dx_nn > NN_MOTION_THRESHOLD) & (mag_nn > NN_MOTION_THRESHOLD)
        moving_left_nn = (dx_nn < -NN_MOTION_THRESHOLD) & (mag_nn > NN_MOTION_THRESHOLD)

        hsv_mask_nn.fill(0)
        hsv_mask_nn[..., 1] = 255
        hsv_mask_nn[moving_right_nn, 0] = 60
        hsv_mask_nn[moving_right_nn, 2] = 255
        hsv_mask_nn[moving_left_nn, 0] = 0
        hsv_mask_nn[moving_left_nn, 2] = 255

        segment_motion_nn = cv2.cvtColor(hsv_mask_nn, cv2.COLOR_HSV2BGR)

        elapsed = time.time() - buffer_start_time
        if elapsed < BUFFER_DURATION:
            eval_grey = cv2.cvtColor(segment_motion_nn, cv2.COLOR_BGR2GRAY)
            accumulated_mask += eval_grey
            frame_count += 1
        else:
            break

        prev_gray = gray

    cap.release()

    if frame_count > 0:
        average_frame = (accumulated_mask / frame_count).astype(np.uint8)
        otsu_val, thresh = cv2.threshold(average_frame, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        window_size = 3
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (window_size, window_size))
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=window_size)

        final_for_nn = cv2.bitwise_not(opened)

        saved = cv2.imwrite(output_filename, final_for_nn)
        if saved:
            print(f"Extracted clean text: {os.path.basename(output_filename)}")
        else:
            print(f"Failed to save output for {video_path}")
    else:
        print(f"No frames processed for {video_path}")

with ThreadPoolExecutor(max_workers=3) as executor:
    executor.map(process_video, video_paths)