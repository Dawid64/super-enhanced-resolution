import streamlit as st
import time
import numpy as np
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

uploaded_file = st.file_uploader("Upload an HD video (MP4/MOV)", type=["mp4", "mov"])

if start_training:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    st.write("Training started...")
    progress_bar = st.progress(0)
    trainer = Trainer()
    trainer.listener = SLListener(progress_bar)
    trainer.train_model(tfile.name)
    
    st.success("Training Completed!")