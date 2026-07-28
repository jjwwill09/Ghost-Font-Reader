import sys
from pathlib import Path
import base64
import os

current_dir = Path(__file__).resolve().parent

project_root = current_dir.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
import tempfile
from Data_Handling_Layer.Generate_Data import main as generate_video_clip
from Model_Deployment.data_to_png import process_video, load_model
st.markdown("""
<style>

.stApp {
    background: #F0F4F3;
    font-family: "Roboto", Arial, sans-serif;
}

.stMarkdown,
.stMarkdown p,
.stMarkdown h1,
.stMarkdown h2,
.stMarkdown h3,
.stMarkdown h4,
label {
    color: #1F2926 ;
}

p {
    color: #3c4043;
}

h1 {
    color: #0B0F0E !important;
    font-weight: 500;
    text-align: center;
}

h2, h3 {
    color: #0B0F0E !important;
    font-weight: 500;
}

.stTextInput > div > div > input {
    background-color: white;
    color: #202124;

    height: 50px;
    padding: 0 22px;

    font-size: 18px;
    font-family: "Roboto", Arial, sans-serif;

    transition: all 0.2s ease;
}

.stTextInput > div > div > input::placeholder {
    color: #80868b;
}

.stButton button {
    background: #4285F4;
    color: white;
    border-radius: 20px;
    border: none;
    padding: 10px 24px;
    font-weight: 500;
    transition: 0.2s;
}

.stButton button:hover {
    background: #3367D6;
    color: white;
}

img {
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(60,64,67,.2);
}

.card {
    background: white;
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 1px 6px rgba(60,64,67,.15);
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}


</style>
""", unsafe_allow_html=True)
st.markdown("""
<h1>
Title
</h1>
<p style="text-align:center; color:#5f6368;">
AI to solve time blindness
</p>
""", unsafe_allow_html=True)

script_dir = os.path.dirname(os.path.abspath(__file__))

checkpoint_path = os.path.join(script_dir, "..", "Model_Files", "Farne_Back_Models", "best_conv_model.pt")
train_optical_flow_path = os.path.join(script_dir, "..", "Model_Layer", "Train_Optical_Flow.py")

columns = st.columns(5)
current_path = Path.cwd()
imglist = list(Path(current_path.parent/"png_data").glob("*.png"))
st.markdown("""
    <style>
    .stTextInput input[aria-label="Ghost Font Text:"] {
        color: ##113236;
    }
    </style>
""", unsafe_allow_html=True)
inpt = st.text_input("Ghost Font Text:", placeholder= "Hello")
st.write("---")
if inpt:
    st.write(f"Generating video for text: '{inpt}'...")
    
    model, device, h_target, w_target = load_model(checkpoint_path, train_optical_flow_path)
    with st.spinner("Creating your MP4 clip in server cache..."):
        with tempfile.TemporaryDirectory() as temp_dir:
            
            generated_file_path = generate_video_clip(
                num_videos=1,
                out_dir=temp_dir,
                text=inpt,
                save_mp4=True,
            )
            
            if generated_file_path and Path(generated_file_path).exists():
                with open(generated_file_path, "rb") as video_file:
                    video_bytes = video_file.read()

                predicted_mask = process_video(
                    video_path=generated_file_path, 
                    model=model, 
                    device=device, 
                    h_target=180,
                    w_target=320
                )
    if video_bytes:
        st.success("Pipeline executed successfully!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Input Video (Generated)")
            st.video(video_bytes, format="video/mp4", autoplay=True, loop=True, muted=True)
            
    
        with col2:
            st.subheader("UNet Text Prediction")
            if predicted_mask is not None:
                st.image(predicted_mask, caption="Predicted Text Mask", use_container_width=True)
            else:
                st.error("Model failed to isolate text mask.")

    st.write(f"Model Predicts: {inpt}")