"""Creates pngs(to png_data) by using the nn while looping through all of the mp4s in data"""

import importlib.util
import os
import glob
import time
import cv2
import numpy as np
import torch
 

script_dir = os.path.dirname(os.path.abspath(__file__))

checkpoint_path = os.path.join(script_dir, "..", "Model_Files", "Farne_Back_Models", "final_conv_model.pt")
train_optical_flow_path = os.path.join(script_dir, "..", "Model_Layer", "Train_Optical_Flow.py")

data_dir = os.path.join(script_dir, "..", "test_shape_data")
png_data_dir = os.path.join(script_dir, "..", "png_data")

 
buffer_duration = 3.0
nn_motion_threshold = 0.9
morph_kernel_size = (3, 3)
 
 
def load_model(checkpoint_path, train_optical_flow_path, device=None):
    """
    Dynamically loads the UNet class from Train_Optical_Flow.py and
    restores it from a saved checkpoint. Returns (model, device, h_target, w_target).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Neural Network math on: {device}")
 
    spec = importlib.util.spec_from_file_location("Train_Optical_Flow", train_optical_flow_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    UNet = module.UNet
 
    checkpoint = torch.load(checkpoint_path, map_location=device)
 
    model = UNet().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
 
    h_target = checkpoint.get("height", 180)
    w_target = checkpoint.get("width", 320)
 
    return model, device, h_target, w_target
 
 
def process_video(video_path, model, device, h_target, w_target):
    """
    Runs the trained UNet over a single video, accumulates a motion mask
    over buffer_duration seconds, then cleans it up with Otsu threshold
    + morphological opening.
 
    Returns a (H, W) uint8 numpy array (the cleaned text mask), or None
    if the video couldn't be read or produced no usable frames.
    """
    cap = cv2.VideoCapture(video_path)
 
    ret, first_frame = cap.read()
    if not ret:
        print(f"Failed to grab video source: {video_path}")
        cap.release()
        return None
 
    prev_frame_resized = cv2.resize(first_frame, (w_target, h_target))
    prev_gray = cv2.cvtColor(prev_frame_resized, cv2.COLOR_BGR2GRAY)
 
    hsv_mask_nn = np.zeros_like(prev_frame_resized)
 
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
        mag_nn = np.sqrt(flow_nn[..., 0] ** 2 + flow_nn[..., 1] ** 2)
 
        moving_right_nn = (dx_nn > nn_motion_threshold) & (mag_nn > nn_motion_threshold)
        moving_left_nn = (dx_nn < -nn_motion_threshold) & (mag_nn > nn_motion_threshold)
 
        hsv_mask_nn.fill(0)
        hsv_mask_nn[..., 1] = 255
        hsv_mask_nn[moving_right_nn, 0] = 60
        hsv_mask_nn[moving_right_nn, 2] = 255
        hsv_mask_nn[moving_left_nn, 0] = 0
        hsv_mask_nn[moving_left_nn, 2] = 255
 
        segment_motion_nn = cv2.cvtColor(hsv_mask_nn, cv2.COLOR_HSV2BGR)
 
        elapsed = time.time() - buffer_start_time
        if elapsed < buffer_duration:
            eval_grey = cv2.cvtColor(segment_motion_nn, cv2.COLOR_BGR2GRAY)
            accumulated_mask += eval_grey
            frame_count += 1
        else:
            break
 
        prev_gray = gray
        cv2.waitKey(1)
 
    cap.release()
 
    if frame_count == 0:
        print(f"No usable frames extracted from: {video_path}")
        return None
 
    average_frame = (accumulated_mask / frame_count).astype(np.uint8)
 
    _, thresh = cv2.threshold(average_frame, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_kernel_size)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
 
    final_for_nn = cv2.bitwise_not(opened)
    return final_for_nn
 
 
def main():
    os.makedirs(png_data_dir, exist_ok=True)
 
    model, device, h_target, w_target = load_model(checkpoint_path, train_optical_flow_path)
 
    video_paths = sorted(glob.glob(os.path.join(data_dir, "*.mp4")))
    if not video_paths:
        print(f"No .mp4 files found in {data_dir}")
        return
 
    print(f"Found {len(video_paths)} videos in {data_dir}")
 
    success_count = 0
    for video_path in video_paths:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(png_data_dir, f"{video_name}.png")
 
        print(f"Processing {video_name}...")
        result = process_video(video_path, model, device, h_target, w_target)
 
        if result is not None:
            cv2.imwrite(output_path, result)
            success_count += 1
        else:
            print(f"  Skipped {video_name} (no output produced)")
 
    print(f"Done. {success_count}/{len(video_paths)} videos converted -> {png_data_dir}")
 
 
if __name__ == "__main__":
    main()
 