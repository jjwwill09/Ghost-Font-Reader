import importlib.util
import os
import cv2
import numpy as np
import torch

current_dir = os.path.dirname(__file__)
file_path = os.path.abspath(os.path.join(current_dir, "..", "Model_Layer", "Train_Optical_Flow.py"))
spec = importlib.util.spec_from_file_location("Train_Optical_Flow", file_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

UNet = module.UNet

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running Neural Network math on: {device}")

# Load the model from the saved PyTorch weights
checkpoint_path = "Model_Files/Farne_Back_Models/best_conv_model.pt"
checkpoint = torch.load(checkpoint_path, map_location=device)

model = UNet().to(device)
model.load_state_dict(checkpoint["model_state"])
model.eval()

h_target = checkpoint.get("height", 180)
w_target = checkpoint.get("width", 320)

script_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(script_dir, "Test_Data", "video_00000.mp4") # CONFIG: CHANGE INPUT VIDEO HERE <-----------------------------------------------------------------------------------
cap = cv2.VideoCapture(video_path)

ret, first_frame = cap.read()
if not ret:
    print("Failed to grab video source.")
    cap.release()
    exit()

prev_frame_resized = cv2.resize(first_frame, (w_target, h_target))
prev_gray = cv2.cvtColor(prev_frame_resized, cv2.COLOR_BGR2GRAY)

hsv_mask_nn = np.zeros_like(prev_frame_resized)

cv2.namedWindow('Original Video (Model Scale)', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Original Video (Model Scale)', w_target, h_target)

cv2.namedWindow('Neural Net Isolated Motion', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Neural Net Isolated Motion', w_target, h_target)

def nothing(x): pass
cv2.createTrackbar('NN Threshold x10', 'Neural Net Isolated Motion', 9, 10, nothing) # CONFIG: CHANGE HERE FOR SLIDER VALUES <-------------------------------------------------

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        if not ret: break

    frame_resized = cv2.resize(frame, (w_target, h_target))
    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

    frame_pair = np.stack([prev_gray, gray], axis=0).astype(np.float32) / 255.0
    input_tensor = torch.from_numpy(frame_pair).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_flow = model(input_tensor)
        flow_nn = pred_flow.squeeze(0).cpu().numpy().transpose(1, 2, 0)

    dx_nn = flow_nn[..., 0]
    mag_nn = np.sqrt(flow_nn[..., 0]**2 + flow_nn[..., 1]**2)

    slider_val = cv2.getTrackbarPos('NN Threshold x10', 'Neural Net Isolated Motion')
    NN_MOTION_THRESHOLD = slider_val / 10.0  

    moving_right_nn = (dx_nn > NN_MOTION_THRESHOLD) & (mag_nn > NN_MOTION_THRESHOLD)
    moving_left_nn = (dx_nn < -NN_MOTION_THRESHOLD) & (mag_nn > NN_MOTION_THRESHOLD)

    hsv_mask_nn.fill(0)
    hsv_mask_nn[..., 1] = 255
    hsv_mask_nn[moving_right_nn, 0] = 60
    hsv_mask_nn[moving_right_nn, 2] = 255
    hsv_mask_nn[moving_left_nn, 0] = 0
    hsv_mask_nn[moving_left_nn, 2] = 255

    segment_motion_nn = cv2.cvtColor(hsv_mask_nn, cv2.COLOR_HSV2BGR)
    overlay_nn = cv2.addWeighted(frame_resized, 0.6, segment_motion_nn, 0.4, 0)

    cv2.imshow('Original Video (Model Scale)', frame_resized)
    cv2.imshow('Neural Net Isolated Motion', segment_motion_nn)

    prev_gray = gray

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()