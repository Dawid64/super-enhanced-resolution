import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import mlflow
import mlflow.pytorch

from .dataset_loading import StreamDataset
from .model import SrCnn
from .utils import SimpleListener

class Trainer:
    def __init__(self, device='auto', learning_rate:float=0.001):
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.criterion = nn.MSELoss()
        self.model = SrCnn().to(device)
        self.learning_rate = learning_rate
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.history = {'loss': []}
        self.listener: SimpleListener = None
        self.last_loss = None

    def single_epoch(self, dataset, num_frames, skip_frames, pbar):
        losses = []
        for i, frames in enumerate(dataset):
            if num_frames is not None and i == num_frames:
                break
            if not i % skip_frames:
                prev_high_res_frame, low_res_frame, high_res_frame = [
                    f.unsqueeze(0).to(self.device) for f in frames]
            else:
                low_res_frame, high_res_frame = [
                    f.unsqueeze(0).to(self.device) for f in frames if f is not None]
            self.optimizer.zero_grad()
            pred_high_res_frame = self.model(prev_high_res_frame, low_res_frame)
            loss = self.criterion(pred_high_res_frame, high_res_frame)
            loss.backward()
            self.optimizer.step()
            self.history['loss'].append(float(loss.item()))
            losses.append(float(loss.item()))
            pbar.set_postfix({'loss': loss.item()})
            prev_high_res_frame = pred_high_res_frame.detach()
        self.last_loss = sum(losses) / len(losses)
        

    def train_model(self, video_file='video.mp4', num_epochs=15, skip_frames=10,
                    save_interval=10, num_frames=10, original_size=(1920, 1080),
                    target_size=(1280, 720)) -> SrCnn:
        mlflow.set_experiment("quantum_super_resolution_experiment")
        with mlflow.start_run(run_name="QSR_Training"):
            mlflow.log_param("video_file", video_file)
            mlflow.log_param("num_epochs", num_epochs)
            mlflow.log_param("skip_frames", skip_frames)
            mlflow.log_param("num_frames", num_frames)
            mlflow.log_param("original_size", original_size)
            mlflow.log_param("target_size", target_size)
            mlflow.log_param("optimizer", "AdamW")
            mlflow.log_param("learning_rate", self.learning_rate)
            mlflow.log_param("criterion", self.criterion.__class__.__name__)

            pbar = tqdm(range(1, num_epochs + 1), desc='Training',
                        unit='epoch', postfix={'loss': 'inf'})
            
            for epoch in pbar:
                dataset = StreamDataset(
                    video_file,
                    original_size=original_size,
                    target_size=target_size,
                    skip_frames=skip_frames
                )
                self.model.train()
                self.single_epoch(dataset, num_frames, skip_frames, pbar)
                mlflow.log_metric("train_loss", self.last_loss, step=epoch)

                if self.listener is not None:
                    self.listener.callback(epoch=epoch/num_epochs, history=self.history)

                if epoch % save_interval == 0:
                    self.model.eval()
                    save_path = f'models/model_epoch{epoch}.pt'
                    self.model.save(save_path)
                    mlflow.log_artifact(save_path, artifact_path="models")
                    mlflow.pytorch.log_model(self.model, artifact_path=f"models/model_epoch{epoch}")
                    self.model.to(self.device)

            final_save_path = 'models/model_final.pt'
            self.model.eval()
            self.model.save(final_save_path)
            mlflow.log_artifact(final_save_path, artifact_path="models")
            mlflow.pytorch.log_model(self.model, artifact_path="models/model_final")

    def save(self, path):
        self.model.save(path)
        
if __name__ == '__main__':
    trainer = Trainer()
    trainer.train_model(video_file=r'videos/video.mp4', num_epochs=5)