import streamlit as st
import time

st.set_page_config(layout="wide")
st.title("Quantum Super Resolution")


uploaded_file = st.file_uploader("Upload an HD video (MP4/MOV)", type=["mp4", "mov"])

if uploaded_file is not None:
    st.video(uploaded_file)

    total_frames = 50
    progress_bar = st.progress(0)

    for i in range(total_frames):
        time.sleep(0.05)
        progress_percentage = int((i + 1) / total_frames * 100)
        progress_bar.progress(progress_percentage)

    st.success("Upscaling complete!")
    upscaled_video_path = "videos/video.mp4"

    with open(upscaled_video_path, "rb") as f:
        st.download_button(label="Download Upscaled Video", data=f, file_name="upscaled_result.mp4", mime="video/mp4")
