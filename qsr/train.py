import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from .model import SrCnn
from .utils import get_bw_difference
from .dataset_loading import StreamDataset

def train_model(video_file:str ='video.mp4', device='auto', num_epochs=15, skip_frames=10,
                save_interval=10, num_frames=10, original_size=(1920, 1080),
                target_size=(1280, 720)) -> SrCnn:

    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = SrCnn().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    pbar = tqdm(range(1, num_epochs+1), desc='Training',
                unit='epoch', postfix={'loss': 'inf'})
    for epoch in pbar:
        dataset = StreamDataset(
            video_file,
            original_size=original_size,
            target_size=target_size,
            skip_frames=skip_frames
        )
        model.train()
        for i, frames in enumerate(dataset):
            if num_frames is not None and i == num_frames:
                break
            if not i % skip_frames:
                prev_high_res_frame, low_res_frame, high_res_frame = [
                    f.unsqueeze(0).to(device) for f in frames]
            else:
                low_res_frame, high_res_frame = [
                    f.unsqueeze(0).to(device) for f in frames if f is not None]
            optimizer.zero_grad()
            pred_high_res_frame = model(prev_high_res_frame, low_res_frame)
            loss = criterion(pred_high_res_frame, high_res_frame)
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'loss': loss.item()})
            prev_high_res_frame = pred_high_res_frame.detach()
        model.eval()

        if epoch % save_interval == 0:
            difference = get_bw_difference(
                model, prev_high_res_frame, low_res_frame, high_res_frame)
            cv2.imwrite(f'differences/difference_epoch{epoch}.png', difference)
            model.save(f'models/model_epoch{epoch}.pt')
            model.to(device)
    return model


if __name__ == '__main__':
    train_model(num_epochs=10, video_file='videos/video.mp4',
                num_frames=None, skip_frames=10, target_size=(960, 540), save_interval=1)
