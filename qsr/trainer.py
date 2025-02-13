from typing import Dict, Literal
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import mlflow
import mlflow.pytorch

from .dataset_loading import MultiVideoDataset
from .model import TSRCNN_large, TSRCNN_small
from .utils import SimpleListener
from torch.utils.data import random_split, DataLoader
from piqa.ssim import ssim, gaussian_kernel

OPTIMIZER: Dict[Literal['AdamW', 'Adagrad', 'SGD'], optim.Optimizer] = {
    'AdamW': optim.AdamW,
    'Adagrad': optim.Adagrad,
    'SGD': optim.SGD
}


def PSNR(y_pred, y_gt):
    return 10 * torch.log10(1 / (nn.MSELoss()(y_pred, y_gt)+1e-8))


class PNSR(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, y_pred, y_gt):
        result = 1/PSNR(y_pred, y_gt)
        return result


def SSIM(y_pred, y_gt):
    return ssim(y_pred, y_gt, kernel=gaussian_kernel(7).repeat(3, 1, 1).to(y_pred.device), channel_avg=True)[0].mean()


class DSSIM(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_gt):
        return 1 - SSIM(y_pred, y_gt)


LOSS: Dict[Literal['MSE', 'PNSR', 'DSSIM'], nn.Module] = {
    'MSE': nn.MSELoss,
    'PNSR': PNSR,
    'DSSIM': DSSIM
}


class MultiTrainer:
    def __init__(self, device='auto', original_size=(1920, 1080), target_size=(1280, 720), learning_rate: float = 0.001, optimizer: Literal['AdamW', 'Adagrad', 'SGD'] = 'AdamW', loss: Literal['MSE', 'PNSR', 'DSSIM'] = 'MSE', frames_backward=2, frames_forward=2, model=None):
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.criterion = LOSS[loss]()
        self.original_size = original_size
        self.target_size = target_size
        self.base_videos = int(model.split('_')[3].split('v')[0]) if model[:3] != 'new' else 0
        upscale_factor = original_size[1]/target_size[1]
        if model == 'new TSRCNN_large':
            self.model = TSRCNN_large(upscale_factor=upscale_factor, frames_backward=frames_backward, frames_forward=frames_forward).to(device)
            self.size = 'large'
        elif model == 'new TSRCNN_small':
            self.model = TSRCNN_small(upscale_factor=upscale_factor, frames_backward=frames_backward, frames_forward=frames_forward).to(device)
            self.size = 'small'
        else:
            if model.split('_')[0] == 'models/small' or model.split('_')[0] == 'small':
                if model[:6] != 'models':
                    model = 'models/' + model
                self.model = TSRCNN_small.load(model, frames_backward=frames_backward, frames_forward=frames_forward, upscale_factor=upscale_factor).to(device)
                self.size = 'small'
            else:
                if model[:6] != 'models':
                    model = 'models/' + model
                self.model = TSRCNN_large.load(model, frames_backward=frames_backward, frames_forward=frames_forward, upscale_factor=upscale_factor).to(device)
                self.size = 'large'
        self.learning_rate = learning_rate
        self.optimizer = OPTIMIZER[optimizer](self.model.parameters(), lr=learning_rate)
        self.history = {'train_loss': [], 'val_loss': [], 'epoch_loss': [], 'train_metrics': {'PSNR': [],
                                                                                              'SSIM': []}, 'val_metrics': {'PSNR': [], 'SSIM': []}, 'epoch_metrics': {'PSNR': [], 'SSIM': []}}
        self.listener: SimpleListener = None
        self.last_loss = None
        self.save_interval = 1
        self.dataset_format = MultiVideoDataset
        self.frames_backward = frames_backward
        self.frames_forward = frames_forward

    def _log_params(self, parameters: Dict):
        for key, value in parameters.items():
            mlflow.log_param(key, value)

    def train_model(self, video_files: list[str] = ['video.mp4'], num_epochs=15, batch_size=2, video_batch_size=3) -> str:
        mlflow.set_experiment("temporal_super_resolution_experiment")
        video_batches = [video_files[i:i+video_batch_size] if i+video_batch_size <
                         len(video_files) else video_files[i:] for i in range(0, len(video_files), video_batch_size)]
        global_training = tqdm(enumerate(video_batches), total=len(video_batches), desc='Global Training')
        for i, video_files in global_training:
            dataset = self.dataset_format(video_files, original_size=self.original_size, target_size=self.target_size,
                                          frames_backward=self.frames_backward, frames_forward=self.frames_forward, listener=self.listener)

            train_dataset, val_dataset = random_split(dataset, [0.8, 0.2])

            train_size = len(train_dataset)
            val_size = len(val_dataset)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
            self.run_name = f"{self.size}_{self.target_size[1]}_{self.original_size[1]}_{self.base_videos + i*video_batch_size +
                                                                                         len(video_files)}videos_{self.optimizer.__class__.__name__}opt', f'_{self.criterion.__class__.__name__}loss_{self.frames_backward}fb_{self.frames_forward}ff_{num_epochs}ep"
            with mlflow.start_run(run_name=self.run_name):
                self._log_params({"video_files": video_files,
                                  "already_trained_on_videos": i*video_batch_size,
                                  "num_epochs": num_epochs,
                                  "train_size": train_size,
                                  "val_size": val_size,
                                  "batch_size": batch_size,
                                  "original_size": self.original_size,
                                  "target_size": self.target_size,
                                  "optimizer": self.optimizer.__class__.__name__,
                                  "learning_rate": self.learning_rate,
                                  "criterion": self.criterion.__class__.__name__,
                                  "forward_frames": self.frames_forward,
                                  "backward_frames": self.frames_backward})

                pbar = tqdm(range(1, num_epochs + 1), desc='Training',
                            unit='epoch', postfix={'loss': 'inf'})
                for epoch in pbar:
                    self.single_epoch(train_loader, val_loader, epoch)
                    if self.listener is not None:
                        self.listener.epoch_callback(epoch/num_epochs, history=self.history)
                    if epoch % self.save_interval == 0:
                        save_path = ''.join([f'models/{self.size}_{self.target_size[1]}_{self.original_size[1]}_{self.base_videos + i*video_batch_size +
                                                                                                                 len(video_files)}videos_{self.optimizer.__class__.__name__}opt', f'_{self.criterion.__class__.__name__}loss_{self.frames_backward}fb_{self.frames_forward}ff_{num_epochs}ep_{epoch}.pt'])

                        self.model.eval()
                        self.save(save_path)
                        mlflow.log_artifact(save_path, artifact_path="models")
                        mlflow.pytorch.log_model(self.model, artifact_path=save_path.split('.')[0])
                        self.model.to(self.device)
                    pbar.set_postfix({'loss': self.epoch_loss, 'PSNR': self.epoch_psnr, 'SSIM': self.epoch_ssim})
                    mlflow.log_metric("epoch_loss", self.epoch_loss, step=epoch)
                    mlflow.log_metric("epoch_PSNR", self.epoch_psnr, step=epoch)
                    mlflow.log_metric("epoch_SSIM", self.epoch_ssim, step=epoch)
                self.save('_'.join(save_path.split('_')[:-1]) + '_final.pt')
                self.model.to(self.device)
        return '_'.join(save_path.split('_')[:-1]) + '_final.pt'

    def train_batch(self, prev_frames, low_res_frame, next_frames, high_res_frame):
        prev_frames = prev_frames.to(self.device)
        low_res_frame = low_res_frame.to(self.device)
        next_frames = next_frames.to(self.device)
        high_res_frame = high_res_frame.to(self.device)
        if self.size == 'small':
            high_res_frame = high_res_frame[:, :, 6:-6, 6:-6]
        if self.size == 'large':
            high_res_frame = high_res_frame[:, :, 9:-9, 9:-9]
        self.optimizer.zero_grad()
        pred_high_res_frame = self.model(prev_frames, low_res_frame, next_frames)
        loss = self.criterion(pred_high_res_frame, high_res_frame)
        loss.backward()
        self.optimizer.step()
        loss = loss.item()
        self.history['train_loss'].append(loss)
        psnr = PSNR(pred_high_res_frame, high_res_frame).item()
        ssim = SSIM(pred_high_res_frame, high_res_frame).item()
        self.history['train_metrics']['PSNR'].append(psnr)
        self.history['train_metrics']['SSIM'].append(ssim)
        return loss, (psnr, ssim)

    @torch.no_grad()
    def val_batch(self, prev_frames, low_res_frame, next_frames, high_res_frame):
        prev_frames = prev_frames.to(self.device)
        low_res_frame = low_res_frame.to(self.device)
        next_frames = next_frames.to(self.device)
        high_res_frame = high_res_frame.to(self.device)
        if self.size == 'small':
            high_res_frame = high_res_frame[:, :, 6:-6, 6:-6]
        if self.size == 'large':
            high_res_frame = high_res_frame[:, :, 9:-9, 9:-9]
        pred_high_res_frame = self.model(prev_frames, low_res_frame, next_frames)
        loss = self.criterion(pred_high_res_frame, high_res_frame).item()
        self.history['val_loss'].append(loss)
        psnr = PSNR(pred_high_res_frame, high_res_frame).item()
        ssim = SSIM(pred_high_res_frame, high_res_frame).item()
        self.history['val_metrics']['PSNR'].append(psnr)
        self.history['val_metrics']['SSIM'].append(ssim)
        return loss, (psnr, ssim)

    def single_epoch(self, train_loader, val_loader, epoch):
        self.model.train()
        train_losses = []
        train_psnr = []
        train_ssim = []
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch}', unit='batch', leave=True)
        for i, ((prev_frames, low_res_frame, next_frames), high_res_frame) in enumerate(train_pbar):
            loss, metrics = self.train_batch(prev_frames, low_res_frame, next_frames, high_res_frame)
            if self.listener is not None:
                self.listener.train_batch_callback((i+1)/len(train_loader), self.history)
            train_losses.append(loss)
            train_psnr.append(metrics[0])
            train_ssim.append(metrics[1])
            train_pbar.set_postfix({'train_loss': loss, 'train_PSNR': metrics[0], 'train_SSIM': metrics[1]})

        self.model.eval()
        val_losses = []
        val_psnr = []
        val_ssim = []
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch}', unit='batch', leave=True)
        for i, ((prev_frames, low_res_frame, next_frames), high_res_frame) in enumerate(val_pbar):
            loss, metrics = self.val_batch(prev_frames, low_res_frame, next_frames, high_res_frame)
            if self.listener is not None:
                self.listener.val_batch_callback((i+1)/len(val_loader), self.history)
            val_losses.append(loss)
            val_psnr.append(metrics[0])
            val_ssim.append(metrics[1])
            val_pbar.set_postfix({'val_loss': loss, 'val_PSNR': metrics[0], 'val_SSIM': metrics[1]})
        self.epoch_loss = sum(val_losses) / len(val_losses)
        self.epoch_psnr = sum(val_psnr) / len(val_psnr)
        self.epoch_ssim = sum(val_ssim) / len(val_ssim)
        self.history['epoch_loss'].append(self.epoch_loss)
        self.history['epoch_metrics']['PSNR'].append(self.epoch_psnr)
        self.history['epoch_metrics']['SSIM'].append(self.epoch_ssim)

    def save(self, path):
        self.model.save(path)
        self.model.to(self.device)
