from typing import Dict, Literal
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import mlflow
import mlflow.pytorch

from .dataset_loading import StreamDataset, NewStreamDataset
from .model import SrCnn, SrCNN2
from .utils import SimpleListener
import torchvision.transforms.functional as F
import cv2
import numpy as np

OPTIMIZER: Dict[Literal['AdamW', 'Adagrad', 'SGD'], optim.Optimizer] = {
    'AdamW': optim.AdamW,
    'Adagrad': optim.Adagrad,
    'SGD': optim.SGD
}

class Trainer:
    def __init__(self, device='auto', learning_rate: float = 0.001, optimizer:Literal['AdamW', 'Adagrad', 'SGD'] = 'AdamW'):
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.criterion = nn.MSELoss()
        self.model = SrCnn().to(device)
        self.learning_rate = learning_rate
        self.optimizer = OPTIMIZER[optimizer](self.model.parameters(), lr=learning_rate)
        self.history = {'loss': []}
        self.listener: SimpleListener = None
        self.last_loss = None
        self.save_interval = 10
        self.dataset_format = StreamDataset
        self.run_name = "QSR_Training"

    def _log_params(self, parameters: Dict):
        for key, value in parameters.items():
            mlflow.log_param(key, value)

    def single_epoch(self, dataset, num_frames, skip_frames, num_epoch):
        losses = []
        pbar = tqdm(range(num_frames), desc=f'Epoch {num_epoch}', unit='frame', leave=True)
        for i, frames in zip(pbar, dataset):
            if num_frames is not None and i == num_frames:
                break
            if not i % skip_frames:
                prev_high_res_frame, low_res_frame, high_res_frame = [f.unsqueeze(0).to(self.device) for f in frames]
            else:
                low_res_frame, high_res_frame = [f.unsqueeze(0).to(self.device) for f in frames if f is not None]
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
                    num_frames=10, original_size=(1920, 1080),
                    target_size=(1280, 720)) -> SrCnn:
        mlflow.set_experiment("quantum_super_resolution_experiment")
        with mlflow.start_run(run_name=self.run_name):
            self._log_params({"video_file": video_file,
                              "num_epochs": num_epochs,
                              "skip_frames": skip_frames,
                              "num_frames": num_frames,
                              "original_size": original_size,
                              "target_size": target_size,
                              "optimizer": self.optimizer.__class__.__name__,
                              "learning_rate": self.learning_rate,
                              "criterion": self.criterion.__class__.__name__})

            pbar = tqdm(range(1, num_epochs + 1), desc='Training',
                        unit='epoch', postfix={'loss': 'inf'})


            for epoch in pbar:
                if self.listener is not None:
                    self.listener.callback(epoch=epoch/num_epochs, history=self.history)
                    
                    if epoch % self.save_interval == 0:
                        save_path = f'models/model_epoch{epoch}.pt'
                        self.model.eval()
                        self.save(save_path)
                        mlflow.log_artifact(save_path, artifact_path="models")
                        mlflow.pytorch.log_model(self.model, artifact_path=f"models/model_epoch{epoch}")
                        self.model.to(self.device)

                additional = {"num_epoch": epoch}
                if skip_frames is not None:
                    additional['skip_frames'] = skip_frames

                dataset = self.dataset_format(video_file, original_size=original_size, target_size=target_size, **additional)
                
                self.model.train()
                self.single_epoch(dataset, num_frames, **additional)
                pbar.set_postfix({'loss': self.last_loss})
                mlflow.log_metric("train_loss", self.last_loss, step=epoch)

    def save(self, path):
        self.model.save(path)
        self.model.to(self.device)

class Trainer2(Trainer):
    def __init__(self, device='auto', learning_rate = 0.001, optimizer = 'AdamW', skip_frames = None):
        super().__init__(device, learning_rate, optimizer)
        self.dataset_format = NewStreamDataset
        self.run_name = "QSR_Training2"
        self.model = SrCNN2().to(self.device)
        self.learning_rate = learning_rate
        self.optimizer = OPTIMIZER[optimizer](self.model.parameters(), lr=learning_rate)

    def single_epoch(self, dataset, num_frames, skip_frames=None, num_epoch = 10):

        losses = []
        pbar = tqdm(range(num_frames), desc=f'Epoch {num_epoch}', unit='frame', leave=True)
        for i, frames in zip(pbar, dataset):
            if num_frames is not None and i == num_frames:
                break
            prev_frame, frame, high_res_frame = [f.unsqueeze(0).to(self.device) for f in frames]
            self.optimizer.zero_grad()
            pred_high_res_frame = self.model(prev_frame, frame)

            loss = self.criterion(pred_high_res_frame, high_res_frame)
            loss.backward()
            self.optimizer.step()
            self.history['loss'].append(float(loss.item()))
            losses.append(float(loss.item()))
            pbar.set_postfix({'loss': loss.item()})

        self.last_loss = sum(losses) / len(losses)


if __name__ == '__main__':
    trainer = Trainer2(optimizer='SGD')
    trainer.train_model(num_epochs=2, video_file='Inter4k/60fps/small/1.mp4', num_frames=120)
    trainer.save('models/model.pt')
