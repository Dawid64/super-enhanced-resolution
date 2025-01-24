from glob import glob
import streamlit as st
import pandas as pd
import tempfile
from qsr.trainer import MultiTrainer
from qsr.utils import SimpleListener


class SLListener(SimpleListener):
    def __init__(self, epoch_bar, train_batch_bar, val_batch_bar, video_loading_bar):
        self.epoch_bar = epoch_bar
        self.train_batch_bar = train_batch_bar
        self.val_batch_bar = val_batch_bar
        self.video_loading_bar = video_loading_bar
        self.train_psnr = None
        self.val_psnr = None
        self.train_ssim = None
        self.val_ssim = None
        self.epoch_psnr = None
        self.epoch_ssim = None
        self.epoch_columns = None
        self.train_columns = None
        self.val_columns = None

    def epoch_callback(self, progress, history):
        if self.epoch_columns is None:
            self.epoch_columns = st.columns(2)
        epoch_columns = self.epoch_columns
        self.train_batch_bar.progress(0, text="Training Batches")
        self.val_batch_bar.progress(0, text="Validation Batches")
        self.epoch_bar.progress(progress, text="Epochs")
        with epoch_columns[0]:
            if self.epoch_psnr is None:
                self.df_psnr = pd.DataFrame(columns=["epoch_psnr"])
                self.epoch_psnr = st.line_chart(self.df_psnr, y=["epoch_psnr"], x_label="Epochs", y_label="PSNR")
            epoch_psnr = [history["epoch_metrics"]["PSNR"][-1]]
            new_history = pd.DataFrame({"epoch_psnr": epoch_psnr})
            self.epoch_psnr.add_rows(new_history)
        with epoch_columns[1]:
            if self.epoch_ssim is None:
                self.df_ssim = pd.DataFrame(columns=["epoch_ssim"])
                self.epoch_ssim = st.line_chart(self.df_ssim, y=["epoch_ssim"], x_label="Epochs", y_label="SSIM")
            epoch_ssim = [history["epoch_metrics"]["SSIM"][-1]]
            new_history = pd.DataFrame({"epoch_ssim": epoch_ssim})
            self.epoch_ssim.add_rows(new_history)

    def train_batch_callback(self, progress, history):
        if self.train_columns is None:
            self.train_columns = st.columns(2)
        train_columns = self.train_columns
        self.train_batch_bar.progress(progress, text="Training Batches")
        with train_columns[0]:
            if self.train_psnr is None:
                self.df_psnr = pd.DataFrame(columns=["train_psnr"])
                self.train_psnr = st.line_chart(self.df_psnr, y=["train_psnr"], x_label="Batches", y_label="PSNR")
            train_psnr = [history["train_metrics"]["PSNR"][-1]]
            new_history = pd.DataFrame({"train_psnr": train_psnr})
            self.train_psnr.add_rows(new_history)
        with train_columns[1]:
            if self.train_ssim is None:
                self.df_ssim = pd.DataFrame(columns=["train_ssim"])
                self.train_ssim = st.line_chart(self.df_ssim, y=["train_ssim"], x_label="Batches", y_label="SSIM")
            train_ssim = [history["train_metrics"]["SSIM"][-1]]
            new_history = pd.DataFrame({"train_ssim": train_ssim})
            self.train_ssim.add_rows(new_history)

    def val_batch_callback(self, progress, history):
        if self.val_columns is None:
            self.val_columns = st.columns(2)
        val_columns = self.val_columns
        self.val_batch_bar.progress(progress, text="Validation Batches")
        with val_columns[0]:
            if self.val_psnr is None:
                self.df_psnr = pd.DataFrame(columns=["val_psnr"])
                self.val_psnr = st.line_chart(self.df_psnr, y=["val_psnr"], x_label="Batches", y_label="PSNR")
            val_psnr = [history["val_metrics"]["PSNR"][-1]]
            new_history = pd.DataFrame({"val_psnr": val_psnr})
            self.val_psnr.add_rows(new_history)
        with val_columns[1]:
            if self.val_ssim is None:
                self.df_ssim = pd.DataFrame(columns=["val_ssim"])
                self.val_ssim = st.line_chart(self.df_ssim, y=["val_ssim"], x_label="Batches", y_label="SSIM")
            val_ssim = [history["val_metrics"]["SSIM"][-1]]
            new_history = pd.DataFrame({"val_ssim": val_ssim})
            self.val_ssim.add_rows(new_history)

    def video_loading_callback(self, progress, ):
        self.video_loading_bar.progress(progress, text="Loading Videos")


st.set_page_config(layout="wide")
st.title("Temporal Super Resolution")

models = sorted([x.split('/')[1] for x in glob("models/*final.pt")])+['new TSRCNN_large', 'new TSRCNN_small']
optimizers = ['AdamW', 'Adagrad', 'SGD']
loss = ['MSE', 'PNSR', 'DSSIM']
inputs_ress = [360, 480, 540, 720, 1080, 1440]
outputs_ress = [480, 540, 720, 1080, 1440, 2160]

selected_model = st.selectbox("Select an already trained model", models)
row_cols1 = st.columns(3)
with row_cols1[0]:
    learning_rate = st.number_input("Learning rate", value=0.0001, step=0.0001, format="%.4f")
with row_cols1[1]:
    optimizer = st.selectbox("Optimizer", optimizers)
with row_cols1[2]:
    loss = st.selectbox("Loss", loss, index=loss.index('MSE'))

row_cols2 = st.columns(4)
with row_cols2[0]:
    batch_size = st.number_input("Batch size", value=2)
with row_cols2[1]:
    frames_backward = st.number_input("Frames back", value=1)
with row_cols2[2]:
    frames_forward = st.number_input("Frames forward", value=1)
with row_cols2[3]:
    video_batch_size = st.number_input("Video batch size", value=3)

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


def train(SLListener, loss, learning_rate, optimizer, batch_size, frames_backward, frames_forward, low_res, high_res, num_epochs, uploaded_file, model='new TSRCNN_small'):
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
    path = trainer.train_model([tfile.name for tfile in tfiles], batch_size=batch_size, num_epochs=num_epochs, video_batch_size=video_batch_size)
    st.success("Training Completed!")
    return path


if start_training and uploaded_file is not None:
    train(SLListener, loss, learning_rate, optimizer, batch_size, frames_backward, frames_forward,
          low_res, high_res, num_epochs, uploaded_file, selected_model)
