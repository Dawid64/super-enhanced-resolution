import tempfile
import streamlit as st
from glob import glob

from qsr.utils import SimpleListener
from qsr.predictor import Upscaler

st.set_page_config(layout="wide")
st.title("Temporal Super Resolution")


class SRListener(SimpleListener):
    def __init__(self, stpbar):
        self.stpbar = stpbar

    def epoch_callback(self, frame):
        self.stpbar.progress(frame)


models = sorted([x.split('/')[1] for x in glob("models/*.pt")])
inputs_ress = [360, 480, 540, 720, 1080, 1440]
outputs_ress = [480, 540, 720, 1080, 1440, 2160]
cols = st.columns(4)
with cols[0]:
    model = st.selectbox("Select an already trained model", models)
with cols[1]:
    input_res = st.selectbox("Input resolution", inputs_ress, index=inputs_ress.index(720))
with cols[2]:
    output_res = st.selectbox("Output resolution", outputs_ress, index=outputs_ress.index(1080))
with cols[3]:
    output_path = st.text_input("Output path", "output.mp4")

uploaded_file = st.file_uploader("Upload a video you want to upscale (MP4/MOV)", type=["mp4", "mov"])

upscaling_button = st.button("Upscale Video")

if upscaling_button and uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    progress_bar = st.progress(0)
    listener = SRListener(progress_bar)
    input_res = (int(16/9*input_res), input_res)
    output_res = (int(16/9*output_res), output_res)
    upscaler = Upscaler(f"models/{model}", listener=listener, original_size=output_res, target_size=input_res)
    upscaler.upscale(tfile.name, num_frames=-1, skip_frames=10, video_path_out=output_path)
    st.success("Upscaling complete!")
    with open(output_path, "rb") as f:
        st.download_button(label="Download Upscaled Video", data=f, file_name="upscaled_result.mp4", mime="video/mp4")
    st.video(output_path)
