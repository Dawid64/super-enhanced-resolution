import cv2
import numpy as np
import torch
import torchvision.transforms as T


def get_bw_difference(model, frame_fullHD, frame_hd, frame_hd_gt):
    with torch.no_grad():
        pred = model(frame_fullHD, frame_hd)
    diff = (pred - frame_hd_gt).abs().mean(dim=1)
    diff = diff.clamp(0, 1)
    diff_img = diff[0].cpu().numpy() 
    diff_img = (diff_img * 255).astype(np.uint8)
    return diff_img

def save_result(model, video_path_in, video_path_out, original_size=(1920, 1080), target_size=(1280, 720)):
    cap = cv2.VideoCapture(video_path_in)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_path_out, fourcc, 25.0, original_size)
    to_tensor = T.ToTensor()
    while True:
        ret1, frame_fullhd = cap.read()
        if not ret1:
            break
        ret2, frame_hd_gt = cap.read()
        if not ret2:
            break
        frame_fullhd = cv2.resize(frame_fullhd, original_size, interpolation=cv2.INTER_AREA)
        frame_fullhd_rgb = cv2.cvtColor(frame_fullhd, cv2.COLOR_BGR2RGB)
        frame_hd_gt = cv2.resize(frame_hd_gt, original_size, interpolation=cv2.INTER_AREA)
        frame_hd = cv2.resize(frame_hd_gt, target_size, interpolation=cv2.INTER_AREA)
        inp_fullhd = to_tensor(frame_fullhd_rgb).unsqueeze(0)
        inp_hd = to_tensor(frame_hd).unsqueeze(0)
        with torch.no_grad():
            pred = model(inp_fullhd, inp_hd)
        out = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()
        out = np.clip(out, 0, 1)
        out = (out * 255).astype(np.uint8)
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        writer.write(out)
    cap.release()
    writer.release()
