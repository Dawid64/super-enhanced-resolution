import cv2
import numpy as np
import torch
import torchvision.transforms as T
from .model import SrCnn
from .dataset_loading import StreamDataset


def get_bw_difference(model, prev_high_res_frame, low_res_frame, high_res_frame):
    with torch.no_grad():
        pred = model(prev_high_res_frame, low_res_frame)
    diff = (pred - high_res_frame).abs().mean(dim=1)
    diff = diff.clamp(0, 1)
    diff_img = diff[0].cpu().numpy()
    diff_img = (diff_img * 255).astype(np.uint8)
    return diff_img


def save_result(
    model,
    video_path_in,
    video_path_out,
    low_res_video_path_out,
    original_size=(1920, 1080),
    target_size=(1280, 720),
    skip_frames=10
):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    low_res = cv2.VideoWriter(low_res_video_path_out,
                              fourcc, 25.0, original_size)
    writer = cv2.VideoWriter(video_path_out, fourcc, 25.0, original_size)
    dataset = StreamDataset(
        video_path_in,
        original_size=original_size,
        target_size=target_size,
        skip_frames=skip_frames
    )
    model.eval()
    for i, frames in enumerate(dataset):
        if not i % skip_frames:
            prev_high_res_frame, low_res_frame, _ = [
                f.unsqueeze(0).to(device) for f in frames]
        else:
            low_res_frame, _ = [
                f.unsqueeze(0).to(device) for f in frames if f is not None]
        prev_high_res_input = prev_high_res_frame.to(device)
        low_res_input = low_res_frame.to(device)
        with torch.no_grad():
            pred = model(prev_high_res_input, low_res_input)
        prev_high_res_frame = pred.detach()
        out = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()
        out = np.clip(out, 0, 1)
        out = (out * 255).astype(np.uint8)
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        low_res_out = low_res_frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
        low_res_out = cv2.resize(
            low_res_out, original_size, interpolation=cv2.INTER_AREA)
        low_res_out = np.clip(low_res_out, 0, 1)
        low_res_out = (low_res_out * 255).astype(np.uint8)
        low_res_out = cv2.cvtColor(low_res_out, cv2.COLOR_RGB2BGR)
        writer.write(out)
        low_res.write(low_res_out)
    writer.release()
    low_res.release()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SrCnn.load("models/model_epoch10.pt").to(device)
    save_result(model, video_path_in="videos/video.mp4", video_path_out="videos/video_out.mp4",
                low_res_video_path_out="videos/low_res_video_out.mp4", target_size=(960, 540))
