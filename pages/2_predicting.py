import tempfile
import pandas as pd
import streamlit as st
from glob import glob

from qsr.utils import SimpleListener
from qsr.predictor import Upscaler, Upscaler2

st.set_page_config(layout="wide")
st.title("Temporal Super Resolution")


class SRListener(SimpleListener):
    def __init__(self, stpbar):
        self.stpbar = stpbar
        self.test_chart = None

    def test_batch_callback(self, progress, history):
        self.stpbar.progress(progress)
        if self.test_chart is None:
            self.df_losses = pd.DataFrame(columns=["test_loss"])
            self.test_chart = st.line_chart(self.df_losses, y="test_loss", x_label="Batches", y_label="Loss")
        test_loss = [history["test_loss"][-1]]
        new_history = pd.DataFrame({"test_loss": test_loss})
        self.test_chart.add_rows(new_history)

    def final_loss_callback(self, loss, linear_loss, cubic_loss):
        st.write(f"Final loss: {loss}")
        st.write(f"Linear loss: {linear_loss}")
        st.write(f"Cubic loss: {cubic_loss}")


models = sorted([x.split('/')[1] for x in glob("models/*.pt")])
inputs_ress = [360, 480, 540, 720, 1080, 1440]
outputs_ress = [480, 540, 720, 1080, 1440, 2160]
loss = ['MSE', 'PNSR', 'DSSIM']

model = st.selectbox("Select an already trained model", models)


cols = st.columns(3)


with cols[0]:
    input_res = st.selectbox("Input resolution", inputs_ress, index=inputs_ress.index(360))
with cols[1]:
    output_res = st.selectbox("Output resolution", outputs_ress, index=outputs_ress.index(720))
with cols[2]:
    output_path = st.text_input("Output path", "output.mp4")

cols2 = st.columns(4)
with cols2[0]:
    frames_backward = st.number_input("Frames backward", min_value=1, max_value=10, value=2)
with cols2[1]:
    frames_forward = st.number_input("Frames forward", min_value=1, max_value=10, value=2)
with cols2[2]:
    mode = st.selectbox("Mode", ["test", "inference"], index=0)
with cols2[3]:
    loss = st.selectbox("Loss", loss, index=loss.index('PNSR'))

uploaded_file = st.file_uploader("Upload a video you want to upscale (MP4/MOV)", type=["mp4", "mov"])

upscaling_button = st.button("Upscale Video")

if upscaling_button and uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    progress_bar = st.progress(0)
    listener = SRListener(progress_bar)
    input_res = (int(16/9*input_res), input_res)
    output_res = (int(16/9*output_res), output_res)
    upscaler = Upscaler2(model_path=f"models/{model}", original_size=output_res, target_size=input_res,
                         listener=listener, frames_backward=frames_backward, frames_forward=frames_forward, mode=mode, loss=loss)
    upscaler.upscale(tfile.name, video_path_out=output_path)
    st.success("Upscaling complete!")
    with open(output_path, "rb") as f:
        st.download_button(label="Download Upscaled Video", data=f, file_name="upscaled_result.mp4", mime="video/mp4")
    st.video(output_path)
