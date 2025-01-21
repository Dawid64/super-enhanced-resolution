from textwrap import indent
from sklearn.model_selection import learning_curve
import streamlit as st
import pandas as pd
import tempfile
from qsr import model
from qsr.trainer import MultiTrainer
from qsr.utils import SimpleListener


class SLListener(SimpleListener):
    def __init__(self, epoch_bar, batch_bar, video_loading_bar, val_batch_bar=None):
        self.epoch_bar = epoch_bar
        self.train_batch_bar = batch_bar
        self.val_batch_bar = val_batch_bar
        self.video_loading_bar = video_loading_bar
        self.train_chart = None
        self.val_chart = None
        self.index = 0

    def epoch_callback(self, progress, history):
        self.epoch_bar.progress(progress)
        # if self.train_chart is None:
        #    self.df_losses = pd.DataFrame(columns=["train_loss", "val_loss"])
        #    self.train_chart = st.line_chart(self.df_losses, y="loss")
        # new_history = pd.DataFrame({key: values[self.index:] for key, values in history.items()})
        # self.train_chart.add_rows(new_history)
        # self.index = len(list(history.values())[0])

    def train_batch_callback(self, progress, history):
        self.train_batch_bar.progress(progress)

    def val_batch_callback(self, progress, history):
        self.val_batch_bar.progress(progress)

    def video_loading_callback(self, progress):
        self.video_loading_bar.progress(progress)


st.set_page_config(layout="wide")
st.title("TemporalSuper Resolution")

optimizers = ['AdamW', 'Adagrad', 'SGD']
loss = ['MSE', 'PNSR', 'DSSIM']
inputs_ress = [360, 480, 540, 720, 1080, 1440]
outputs_ress = [480, 540, 720, 1080, 1440, 2160]

row_cols1 = st.columns(3)
with row_cols1[0]:
    learning_rate = st.number_input("Learning rate", value=0.001, step=0.001, format="%.3f")
with row_cols1[1]:
    optimizer = st.selectbox("Optimizer", optimizers)
with row_cols1[2]:
    loss = st.selectbox("Loss", loss, index=loss.index('PNSR'))

row_cols2 = st.columns(3)
with row_cols2[0]:
    bacth_size = st.number_input("Batch size", value=1)
with row_cols2[1]:
    frames_back = st.number_input("Frames back", value=2)
with row_cols2[2]:
    frames_forward = st.number_input("Frames forward", value=2)

row_cols3 = st.columns(3)

with row_cols3[0]:
    low_res = st.selectbox("Low resolution", inputs_ress, index=inputs_ress.index(360))
with row_cols3[1]:
    high_res = st.selectbox("High resolution", outputs_ress, index=outputs_ress.index(720))
with row_cols3[2]:
    num_epochs = st.number_input("Number of epochs", value=3)

uploaded_file = st.file_uploader(
    "Upload a FHD video (MP4/MOV)", type=["mp4", "mov"], accept_multiple_files=True)

start_training = st.button("Start Training")

if start_training and uploaded_file is not None:
    tfiles = [tempfile.NamedTemporaryFile(delete=False) for _ in uploaded_file]
    for tfile, file in zip(tfiles, uploaded_file):
        tfile.write(file.read())
    st.write("Training started...")
    epoch_bar = st.progress(0)
    train_batch_bar = st.progress(0)
    val_batch_bar = st.progress(0)
    video_loading_bar = st.progress(0)
    low_res = (int(16/9*low_res), low_res)
    high_res = (int(16/9*high_res), high_res)
    trainer = MultiTrainer(original_size=high_res, target_size=low_res, learning_rate=learning_rate, optimizer=optimizer,
                           loss=loss, frames_back=frames_back, frames_forward=frames_forward)
    trainer.listener = SLListener(epoch_bar, train_batch_bar, val_batch_bar, video_loading_bar)
    trainer.train_model([tfile.name for tfile in tfiles], batch_size=bacth_size, num_epochs=num_epochs)

    st.success("Training Completed!")
