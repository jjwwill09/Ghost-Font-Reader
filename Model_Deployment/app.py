import sys
from pathlib import Path
import os
import subprocess
import ffmpeg

current_dir = Path(__file__).resolve().parent

project_root = current_dir.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

data_handling_dir = str(project_root / "Data_Handling_Layer")
if data_handling_dir not in sys.path:
    sys.path.insert(0, data_handling_dir)
model_layer_dir = str(project_root / "Model_Layer")
if model_layer_dir not in sys.path:
    sys.path.insert(0, model_layer_dir)
import streamlit as st
import tempfile
import cv2
from Data_Handling_Layer.Generate_Data import main as generate_video_clip
from Model_Layer.video_to_text import mp4_to_png, predict_by_segmentation

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
In*Cog*Nito
</h1>
<p style="text-align:center; color:#5f6368;">
Ghost Font Reader: A tool to detect and read hidden text in videos using deep learning.
</p>
""", unsafe_allow_html=True)

script_dir = os.path.dirname(os.path.abspath(__file__))

char_checkpoint_path = os.path.join(
    script_dir, "..", "Model_Files", "Farne_Back_Models", "char_cnn_best.pt"
)
def play_mp4v_video(mp4v_path):
    """Converts an mp4v video to H.264 in the background and plays it in Streamlit."""
    if not os.path.exists(mp4v_path):
        st.error(f"Video file not found at: {mp4v_path}")
        return

    # Define a unique path for the web-ready output video
    dir_name, file_name = os.path.split(mp4v_path)
    h264_path = os.path.join(dir_name, f"playable_{file_name}")

    # Only convert if the playable version doesn't exist yet
    if not os.path.exists(h264_path):
        with st.spinner("Optimizing video codec for web playback..."):
            try:
                # Re-encode video track to H.264 and pixel format to yuv420p
                (
                    ffmpeg
                    .input(mp4v_path)
                    .output(h264_path, vcodec='libx264', pix_fmt='yuv420p', loglevel="quiet")
                    .overwrite_output()
                    .run()
                )
            except ffmpeg.Error as e:
                st.error("Codec conversion failed. Make sure FFmpeg is installed on your system.")
                return

    # Display the converted video seamlessly
    st.video(h264_path)
st.markdown("""
    <style>
    .stTextInput input[aria-label="Ghost Font Text:"] {
        color: ##113236;
    }
    </style>
""", unsafe_allow_html=True)

inpt = st.text_input("Ghost Font Text:", placeholder="Hello")
st.write("---")

if inpt:
    st.write(f"Generating video for text: '{inpt}'...")

    # Initialize variables so they exist outside the temp directory block
    video_bytes = None
    predicted_mask = None
    predicted_text = None

    with st.spinner("Creating your MP4 clip in server cache..."):
        with tempfile.TemporaryDirectory() as temp_dir:

            generated_file_path = generate_video_clip(
                num_videos=1,
                out_dir=temp_dir,
                content=inpt,
                save_mp4=True,
            )

            if generated_file_path and Path(generated_file_path).exists():
                generated_file_path = str(generated_file_path)
                
                import imageio
                try:
                    with st.spinner("Optimizing video codec for web playback..."):
                        h264_path = os.path.join(temp_dir, "web_playable_conversion.mp4")
                        
                        reader = imageio.get_reader(generated_file_path)
                        fps = reader.get_meta_data().get('fps', 30)
                        
                        writer = imageio.get_writer(
                            h264_path, 
                            fps=fps, 
                            codec='libx264', 
                            pixelformat='yuv420p'
                        )
                        
                        for frame in reader:
                            writer.append_data(frame)
                        
                        writer.close()
                        reader.close()

                        with open(h264_path, "rb") as f:
                            video_bytes = f.read()
                            
                except Exception as conversion_error:
                    st.error(f"Fallback converter encountered an issue: {conversion_error}")

                # 3. Handle your UNet masking logic normally
                with st.spinner("Extracting text motion mask..."):
                    png_path = mp4_to_png(generated_file_path)

                if png_path is not None:
                    predicted_mask = cv2.imread(png_path)
                    with st.spinner("Reading extracted text..."):
                        predicted_text = predict_by_segmentation(
                            png_path, checkpoint_path=char_checkpoint_path
                        )

    if video_bytes:
        st.success("Pipeline executed successfully!")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Input Video (Generated)")
            st.video(video_bytes, format="video/mp4", loop=True, autoplay=True, muted=True)

        with col2:
            st.subheader("UNet Text Prediction")
            if predicted_mask is not None:
                st.image(predicted_mask, caption="Predicted Text Mask", use_container_width=True)
            else:
                st.error("Model failed to isolate text mask.")

        if predicted_text is not None:
            st.write(f"Model Predicts: {predicted_text}")
        else:
            st.error("Model failed to read text from the extracted mask.")
    else:
        st.error("Failed to generate or convert video clip successfully.")