from glob import glob
import streamlit as st
import pandas as pd
import tempfile
from qsr import model
from qsr.trainer import MultiTrainer
from qsr.utils import SimpleListener


class SLListener(SimpleListener):
    def __init__(self, epoch_bar, train_batch_bar, val_batch_bar, video_loading_bar):
        self.epoch_bar = epoch_bar
        self.train_batch_bar = train_batch_bar
        self.val_batch_bar = val_batch_bar
        self.video_loading_bar = video_loading_bar
        self.train_chart = None
        self.val_chart = None
        self.epoch_chart = None

    def epoch_callback(self, progress, history):
        self.train_batch_bar.progress(0, text="Training Batches")
        self.val_batch_bar.progress(0, text="Validation Batches")
        self.epoch_bar.progress(progress, text="Epochs")
        if self.epoch_chart is None:
            self.df_losses = pd.DataFrame(columns=["epoch_loss"])
            self.epoch_chart = st.line_chart(self.df_losses, y=["epoch_loss"], x_label="Epochs", y_label="Loss")
        epoch_loss = [history["epoch_loss"][-1]]
        new_history = pd.DataFrame({"epoch_loss": epoch_loss})
        self.epoch_chart.add_rows(new_history)

    def train_batch_callback(self, progress, history):
        self.train_batch_bar.progress(progress, text="Training Batches")
        if self.train_chart is None:
            self.df_losses = pd.DataFrame(columns=["train_loss"])
            self.train_chart = st.line_chart(self.df_losses, y="train_loss", x_label="Batches", y_label="Loss")
        train_loss = [history["train_loss"][-1]]
        new_history = pd.DataFrame({"train_loss": train_loss})
        self.train_chart.add_rows(new_history)

    def val_batch_callback(self, progress, history):
        self.val_batch_bar.progress(progress, text="Validation Batches")
        if self.val_chart is None:
            self.df_losses = pd.DataFrame(columns=["val_loss"])
            self.val_chart = st.line_chart(self.df_losses, y="val_loss", x_label="Batches", y_label="Loss")
        val_loss = [history["val_loss"][-1]]
        new_history = pd.DataFrame({"val_loss": val_loss})
        self.val_chart.add_rows(new_history)

    def video_loading_callback(self, progress, ):
        self.video_loading_bar.progress(progress, text="Loading Videos")
        if progress == 1:
            self.video_loading_bar.empty()


st.set_page_config(layout="wide")
st.title("Temporal Super Resolution")

models = sorted([x.split('/')[1] for x in glob("models/*.pt")])+[None]
optimizers = ['AdamW', 'Adagrad', 'SGD']
loss = ['MSE', 'PNSR', 'DSSIM']
inputs_ress = [360, 480, 540, 720, 1080, 1440]
outputs_ress = [480, 540, 720, 1080, 1440, 2160]

model = st.selectbox("Select an already trained model", models)
row_cols1 = st.columns(3)
with row_cols1[0]:
    learning_rate = st.number_input("Learning rate", value=0.001, step=0.001, format="%.3f")
with row_cols1[1]:
    optimizer = st.selectbox("Optimizer", optimizers)
with row_cols1[2]:
    loss = st.selectbox("Loss", loss, index=loss.index('PNSR'))

row_cols2 = st.columns(3)
with row_cols2[0]:
    batch_size = st.number_input("Batch size", value=2)
with row_cols2[1]:
    frames_backward = st.number_input("Frames back", value=2)
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
    epoch_bar = st.progress(0, text="Epochs")
    train_batch_bar = st.progress(0, text="Training Batches")
    val_batch_bar = st.progress(0, text="Validation Batches")
    video_loading_bar = st.progress(0, text="Loading Videos")
    low_res = (int(16/9*low_res), low_res)
    high_res = (int(16/9*high_res), high_res)
    trainer = MultiTrainer(original_size=high_res, target_size=low_res, learning_rate=learning_rate, optimizer=optimizer,
                           loss=loss, frames_backward=frames_backward, frames_forward=frames_forward, model=model)
    trainer.listener = SLListener(epoch_bar, train_batch_bar, val_batch_bar, video_loading_bar)
    trainer.train_model([tfile.name for tfile in tfiles], batch_size=batch_size, num_epochs=num_epochs)

    st.success("Training Completed!")
