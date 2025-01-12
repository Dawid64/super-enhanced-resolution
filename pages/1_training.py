import streamlit as st
import pandas as pd
import tempfile
from qsr.trainer import Trainer
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


st.title("Quantum Super Resolution")

start_training = st.button("Start Training")
cols = st.columns(4)
with cols[0]:
    num_epochs = st.number_input("Number of epochs", value=10)
with cols[1]:
    skip_frames = st.number_input("Skip frames", value=10)
with cols[2]:
    save_interval = st.number_input("Model save interval", value=1)
with cols[3]:
    num_frames = st.number_input(
        "Number of frames to train on, -1 for full video", value=-1)

uploaded_file = st.file_uploader(
    "Upload a FHD video (MP4/MOV)", type=["mp4", "mov"])

if start_training and uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    st.write("Training started...")
    progress_bar = st.progress(0)
    trainer = Trainer()
    trainer.listener = SLListener(progress_bar)
    trainer.train_model(tfile.name, num_epochs=num_epochs, skip_frames=skip_frames,
                        save_interval=save_interval, num_frames=num_frames if num_frames != -1 else None)

    st.success("Training Completed!")
