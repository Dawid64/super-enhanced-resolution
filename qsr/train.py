import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from .model import SrCnn
from .utils import get_bw_difference
from .dataset_loading import StreamDataset

def train_model(model_path='models/model_epoch15.pt', video_file='video.mp4',
    device='auto', num_epochs=15, skip_frames=0, save_interval=10, num_frames=10):

    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = SrCnn.load(model_path).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    pbar = tqdm(range(1, num_epochs+1), desc='Training', unit='epoch', postfix={'loss': 'inf'})
    for epoch in pbar:
        dataset = StreamDataset(
            video_file, 
            original_size=(1920, 1080),
            target_size=(1280, 720),
            skip_frames=skip_frames
        )
        for i, frames in enumerate(dataset):
            if i == num_frames:
                break
            frame_fullHD, frame_hd, frame_hd_gt = [f.unsqueeze(0).to(device) for f in frames]
            optimizer.zero_grad()
            frame_hd_pred = model(frame_fullHD, frame_hd)
            loss = criterion(frame_hd_pred, frame_hd_gt)
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'loss': loss.item()})
        if epoch % save_interval == 0:
            difference = get_bw_difference(model, frame_fullHD, frame_hd, frame_hd_gt)
            cv2.imwrite(f'differences/difference_epoch{epoch}.png', difference)
            model.save(f'models/model_epoch{epoch}.pt')

if __name__ == '__main__':
    train_model(num_epochs=4)