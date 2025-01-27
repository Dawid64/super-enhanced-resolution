import tempfile
import pandas as pd
import streamlit as st
from glob import glob

from qsr.utils import SimpleListener
from qsr.predictor import Upscaler

st.set_page_config(layout="wide")
st.title("Temporal Super Resolution")


class SRListener(SimpleListener):
    def __init__(self, stpbar):
        self.stpbar = stpbar
        self.psnr_chart = None
        self.ssim_chart = None

    def test_batch_callback(self, progress, history):
        self.stpbar.progress(progress)
        if self.psnr_chart is None:
            self.df_losses = pd.DataFrame(columns=["test_psnr"])
            self.psnr_chart = st.line_chart(self.df_losses, y="test_psnr", x_label="Batches", y_label="PSNR")
        test_psnr = [history["test_metrics"]["PSNR"][-1]]
        new_history = pd.DataFrame({"test_psnr": test_psnr})
        self.psnr_chart.add_rows(new_history)
        if self.ssim_chart is None:
            self.df_losses = pd.DataFrame(columns=["test_ssim"])
            self.ssim_chart = st.line_chart(self.df_losses, y="test_ssim", x_label="Batches", y_label="SSIM")
        test_ssim = [history["test_metrics"]["SSIM"][-1]]
        new_history = pd.DataFrame({"test_ssim": test_ssim})
        self.ssim_chart.add_rows(new_history)

    def final_loss_callback(self, final_psnr, final_ssim, cubic_psnr, cubic_ssim):
        st.write(f"Final PSNR: {final_psnr}")
        st.write(f"Final SSIM: {final_ssim}")
        st.write(f"Cubic PSNR: {cubic_psnr}")
        st.write(f"Cubic SSIM: {cubic_ssim}")


models = sorted([x.split('/')[1] for x in glob("models/*final.pt")])
inputs_ress = [360, 480, 540, 720, 1080, 1440]
outputs_ress = [480, 540, 720, 1080, 1440, 2160]

model = st.selectbox("Select an already trained model", models)


cols = st.columns(3)


with cols[0]:
    input_res = st.selectbox("Input resolution", inputs_ress, index=inputs_ress.index(360))
with cols[1]:
    output_res = st.selectbox("Output resolution", outputs_ress, index=outputs_ress.index(720))
with cols[2]:
    output_path = st.text_input("Output path", "output.mp4")

cols2 = st.columns(3)
with cols2[0]:
    frames_backward = st.number_input("Frames backward", min_value=1, max_value=10, value=1)
with cols2[1]:
    frames_forward = st.number_input("Frames forward", min_value=1, max_value=10, value=1)
with cols2[2]:
    mode = st.selectbox("Mode", ["test", "inference"], index=0)

uploaded_file = st.file_uploader("Upload a video you want to upscale (MP4/MOV)", type=["mp4", "mov"])

upscaling_button = st.button("Upscale Video")

if upscaling_button and uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    progress_bar = st.progress(0)
    listener = SRListener(progress_bar)
    input_res = (int(16/9*input_res), input_res)
    output_res = (int(16/9*output_res), output_res)
    upscaler = Upscaler(model_path=f"models/{model}", original_size=output_res, target_size=input_res,
                        listener=listener, frames_backward=frames_backward, frames_forward=frames_forward, mode=mode)
    upscaler.upscale(tfile.name, video_path_out=output_path)
    st.success("Upscaling complete!")
    with open(output_path, "rb") as f:
        st.download_button(label="Download Upscaled Video", data=f, file_name="upscaled_result.mp4", mime="video/mp4")
    st.video(output_path)
