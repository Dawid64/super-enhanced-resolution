import streamlit as st
import pandas as pd
import tempfile
from qsr import model
from qsr.trainer import Trainer, Trainer2
from qsr.utils import SimpleListener


class SLListener(SimpleListener):
    def __init__(self, stpbar):
        self.stpbar = stpbar
        self.chart = None
        self.index = 0

    def callback(self, epoch, history):
        self.stpbar.progress(epoch)
        if self.chart is None:
            self.df_losses = pd.DataFrame(columns=["loss"])
            self.chart = st.line_chart(self.df_losses, y="loss")
        new_history = pd.DataFrame({key: values[self.index:] for key, values in history.items()})
        self.chart.add_rows(new_history)
        self.index = len(list(history.values())[0])


st.set_page_config(layout="wide")
st.title("Quantum Super Resolution")

optimizers = ['AdamW', 'Adagrad', 'SGD']
loss = ['MSE', 'PNSR', 'DSSIM']
inputs_ress = [360, 480, 540, 720, 1080, 1440]
outputs_ress = [480, 540, 720, 1080, 1440, 2160]

row_cols1 = st.columns(3)
with row_cols1[0]:
    model = st.selectbox("Model", ["SrCnn", "SrCNN2"], index=1)
with row_cols1[1]:
    optimizer = st.selectbox("Optimizer", optimizers)
with row_cols1[2]:
    loss = st.selectbox("Loss", loss)

row_cols2 = st.columns(3)
with row_cols2[0]:
    num_epochs = st.number_input("Number of epochs", value=10)
with row_cols2[1]:
    num_frames = st.number_input("Number of frames to train on, -1 for full video", value=-1)
with row_cols2[2]:
    skip_frames = st.number_input("Skip frames", value=10)

row_cols3 = st.columns(2)

with row_cols3[0]:
    low_res = st.selectbox("Low resolution", inputs_ress, index=inputs_ress.index(720))
with row_cols3[1]:
    high_res = st.selectbox("High resolution", outputs_ress, index=outputs_ress.index(1080))

uploaded_file = st.file_uploader(
    "Upload a FHD video (MP4/MOV)", type=["mp4", "mov"])

start_training = st.button("Start Training")

if start_training and uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    st.write("Training started...")
    progress_bar = st.progress(0)
    low_res = (int(16/9*low_res), low_res)
    high_res = (int(16/9*high_res), high_res)
    trainer = Trainer(original_size=high_res, target_size=low_res, optimizer=optimizer, loss=loss) if model == "SrCnn" else Trainer2(
        original_size=high_res, target_size=low_res, optimizer=optimizer, loss=loss)
    trainer.listener = SLListener(progress_bar)
    trainer.train_model(tfile.name, num_epochs=num_epochs, skip_frames=skip_frames,
                        num_frames=num_frames if num_frames != -1 else None)

    st.success("Training Completed!")
