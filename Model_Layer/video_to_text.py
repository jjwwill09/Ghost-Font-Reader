import importlib.util
import os
import sys
import cv2
import numpy as np
import torch
import time

_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

from Segmented_Prediction import predict_by_segmentation

# project_root/Model_Layer -> project_root
_project_root = os.path.abspath(os.path.join(_this_dir, ".."))

# --- load the optical-flow UNet model (same as Batch_Optical_Flow.py) ---
file_path = os.path.join(_this_dir, "Train_Optical_Flow.py")
spec = importlib.util.spec_from_file_location("Train_Optical_Flow", file_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
UNet = module.UNet

_device = "cuda" if torch.cuda.is_available() else "cpu"
_flow_checkpoint_path = os.path.join(_project_root, "Model_Files", "Farne_Back_Models", "model_best_9.pth")
_flow_checkpoint = torch.load(_flow_checkpoint_path, map_location=_device)

_flow_model = UNet().to(_device)
_flow_model.load_state_dict(_flow_checkpoint["model_state"])
_flow_model.eval()

_h_target = _flow_checkpoint.get("height", 360)
_w_target = _flow_checkpoint.get("width", 640)

NN_MOTION_THRESHOLD = 2 / 10.0
BUFFER_DURATION = 3.0


def mp4_to_png(video_path, output_path=None):
    """
    Runs the same optical-flow extraction as Batch_Optical_Flow.py on a
    single video and writes the resulting text-silhouette PNG.
    Returns output_path on success, None if extraction failed.
    """
    if output_path is None:
        import tempfile
        output_path = os.path.join(tempfile.gettempdir(), "temp_extracted.png")

    cap = cv2.VideoCapture(video_path)

    ret, first_frame = cap.read()
    if not ret:
        print("Failed to grab video source.")
        cap.release()
        return None

    prev_frame_resized = cv2.resize(first_frame, (_w_target, _h_target))
    prev_gray = cv2.cvtColor(prev_frame_resized, cv2.COLOR_BGR2GRAY)

    hsv_mask_nn = np.zeros_like(prev_frame_resized)
    buffer_start_time = time.time()
    accumulated_mask = np.zeros((_h_target, _w_target), dtype=np.float32)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break

        frame_resized = cv2.resize(frame, (_w_target, _h_target))
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

        frame_pair = np.stack([prev_gray, gray], axis=0).astype(np.float32) / 255.0
        input_tensor = torch.from_numpy(frame_pair).unsqueeze(0).to(_device)

        with torch.no_grad():
            pred_flow = _flow_model(input_tensor)
            flow_nn = pred_flow.squeeze(0).cpu().numpy().transpose(1, 2, 0)

        dx_nn = flow_nn[..., 0]
        mag_nn = np.sqrt(flow_nn[..., 0] ** 2 + flow_nn[..., 1] ** 2)

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

    if frame_count == 0:
        print(f"No frames processed for {video_path}")
        return None

    average_frame = (accumulated_mask / frame_count).astype(np.uint8)
    _, thresh = cv2.threshold(average_frame, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    window_size = 3
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (window_size, window_size))
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=window_size)

    final_for_nn = cv2.bitwise_not(opened)

    saved = cv2.imwrite(output_path, final_for_nn)
    if not saved:
        print(f"Failed to save output for {video_path}")
        return None

    return output_path


_default_char_checkpoint = os.path.join(_project_root, "Model_Files", "Farne_Back_Models", "char_cnn_best.pt")


def predict_video_text(video_path, char_checkpoint_path=None):
    """
    Full pipeline: mp4 -> extracted PNG -> predicted text.
    Returns the predicted string, or None if extraction failed.
    """
    if char_checkpoint_path is None:
        char_checkpoint_path = _default_char_checkpoint

    png_path = mp4_to_png(video_path)
    if png_path is None:
        return None

    text = predict_by_segmentation(png_path, checkpoint_path=char_checkpoint_path)
    return text


if __name__ == "__main__":
    import sys
    video_path = sys.argv[1] if len(sys.argv) > 1 else "example.mp4"
    result = predict_video_text(video_path)
    print(f"{video_path} -> {result!r}")