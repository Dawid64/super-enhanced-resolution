import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from .dataset_loading import StreamDataset
from .model import SrCnn
from .utils import SimpleListener


class Trainer:
    def __init__(self, device='auto'):
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.device = device
        self.criterion = nn.MSELoss()
        self.model = SrCnn().to(device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001)
        self.history = {'loss': []}
        self.listener: SimpleListener = None

    def single_epoch(self, dataset, num_frames, skip_frames, pbar):
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
            pred_high_res_frame = self.model(
                prev_high_res_frame, low_res_frame)
            loss = self.criterion(pred_high_res_frame, high_res_frame)
            loss.backward()
            self.optimizer.step()
            self.history['loss'].append(float(loss.item()))
            pbar.set_postfix({'loss': loss.item()})
            prev_high_res_frame = pred_high_res_frame.detach()

    def train_model(self, video_file: str = 'video.mp4', num_epochs=15, skip_frames=10,
                    save_interval=10, num_frames=10, original_size=(1920, 1080),
                    target_size=(1280, 720)) -> SrCnn:

        pbar = tqdm(range(1, num_epochs+1), desc='Training',
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

            if self.listener is not None:
                self.listener.callback(
                    epoch=epoch/num_epochs, history=self.history)

            if epoch % save_interval == 0:
                self.model.eval()
                self.model.save(f'models/model_epoch{epoch}.pt')
                self.model.to(self.device)

    def save(self, path):
        self.model.save(path)
        self.model.to(self.device)


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train_model(num_epochs=2, video_file='videos/video.mp4')
    trainer.save('models/model.pt')
