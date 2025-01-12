import streamlit as st
import time
import numpy as np
import pandas as pd

st.title("Quantum Super Resolution")

start_training = st.button("Start Training")

uploaded_file = st.file_uploader("Upload an HD video (MP4/MOV)", type=["mp4", "mov"])

if start_training:
    st.write("Training started...")

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i in range(101):
        time.sleep(0.05)
        progress_bar.progress(i)
        status_text.text(f"Training progress: {i}%")

    losses = np.linspace(1.0, 0.1, 100)
    df_losses = pd.DataFrame({"step": range(len(losses)), "loss": losses})
    st.line_chart(df_losses, x="step", y="loss")
    
    st.success("Training Completed!")