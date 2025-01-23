from typing import Dict, Literal
import cv2
import numpy as np
import torch
from tqdm import tqdm
import mlflow
import mlflow.pytorch
from .model import TSRCNN_large, TSRCNN_small
import ffmpegcv
from torch.utils.data import DataLoader

from qsr.utils import SimpleListener

from .dataset_loading import NewStreamDataset, MultiVideoDataset
from torch import nn
from piqa.ssim import ssim, gaussian_kernel


def PSNR(y_pred, y_gt):
    return 10 * torch.log10(1 / (nn.MSELoss()(y_pred, y_gt)+1e-8))


def SSIM(y_pred, y_gt):
    return ssim(y_pred, y_gt, kernel=gaussian_kernel(7).repeat(3, 1, 1).to(y_pred.device), channel_avg=True)[0].mean()


class Upscaler2:
    def __init__(self, model_path, device='auto', original_size=(1920, 1080), target_size=(1280, 720), listener=None, frames_backward=2, frames_forward=2, mode: Literal['test', 'inference'] = 'test'):
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.original_size = original_size
        self.target_size = target_size
        upscale_factor = original_size[0] / target_size[0]
        self.size = model_path.split('_')[0].split('/')[1]
        if self.size == 'large':
            self.model = TSRCNN_large.load(model_path, frames_backward=frames_backward, frames_forward=frames_forward, upscale_factor=upscale_factor).to(device)
        else:
            self.model = TSRCNN_small.load(model_path, frames_backward=frames_backward, frames_forward=frames_forward, upscale_factor=upscale_factor).to(device)
        self.dataset_format = MultiVideoDataset
        self.listener: SimpleListener = listener
        self.run_name = f"{target_size[1]}p -> {original_size[1]}p {frames_backward}fb{frames_forward}ff TSR {mode}"
        self.history = {'test_metrics': {'PSNR': [], 'SSIM': []}, 'cubic_metrics': {'PSNR': [], 'SSIM': []}}
        self.mode = mode
        self.frames_backward = frames_backward
        self.frames_forward = frames_forward

    def _log_params(self, parameters: Dict):
        for key, value in parameters.items():
            mlflow.log_param(key, value)

    def upscale(self, video_file, fps=60.0, video_path_out="output.mp4"):
        writer = ffmpegcv.VideoWriterNV(file=video_path_out, codec='h264_nvenc', fps=fps, preset='p7', pix_fmt='rgb24') if self.device == 'cuda' else ffmpegcv.VideoWriter(
            file=video_path_out, codec='h264', fps=60.0, preset='p7', pix_fmt='rgb24')
        cubic = ffmpegcv.VideoWriterNV(file="bicubic.mp4", codec='h264_nvenc', fps=fps, preset='p7', pix_fmt='rgb24') if self.device == 'cuda' else ffmpegcv.VideoWriter(
            file="bicubic.mp4", codec='h264', fps=60.0, preset='p7', pix_fmt='rgb24')
        test_dataset = self.dataset_format(video_paths=[video_file], original_size=self.original_size,
                                           target_size=self.target_size, frames_backward=self.frames_backward, frames_forward=self.frames_forward, mode='inference' if self.mode == 'inference' else 'training')
        num_frames = len(test_dataset)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        mlflow.set_experiment("temporal_super_resolution_experiment")
        with mlflow.start_run(run_name=self.run_name):
            self._log_params({"video_file": video_file,
                              "test_size": num_frames,
                              "original_size": self.original_size,
                              "target_size": self.target_size})
        self.model.eval()
        test_psnrs = []
        test_ssim = []
        cubic_psnrs = []
        cubic_ssim = []
        test_pbar = tqdm(test_loader, total=num_frames, unit='frame')
        if self.mode == 'test':
            for i, ((prev_frames, low_res_frame, next_frames), high_res_frame) in enumerate(test_pbar):
                pred_frame, metrics = self.test_batch(prev_frames, low_res_frame, next_frames, high_res_frame)
                test_psnrs.append(metrics[0])
                test_ssim.append(metrics[1])
                test_pbar.set_postfix({'test_psnr': metrics[0], 'test_ssim': metrics[1]})
                self.history['test_metrics']['PSNR'].append(metrics[0])
                self.history['test_metrics']['SSIM'].append(metrics[1])
                if self.listener is not None:
                    self.listener.test_batch_callback((i+1)/len(test_loader), self.history)
                cubic_frame = nn.functional.interpolate(low_res_frame, self.original_size[::-1], mode='bicubic')
                cubic_metrics = (PSNR(cubic_frame, high_res_frame).item(), SSIM(cubic_frame, high_res_frame).item())
                cubic_psnrs.append(cubic_metrics[0])
                cubic_ssim.append(cubic_metrics[1])
                self.history['cubic_metrics']['PSNR'].append(cubic_metrics[0])
                self.history['cubic_metrics']['SSIM'].append(cubic_metrics[1])
                cubic_frame = (np.clip(cubic_frame.squeeze(0).permute(1, 2, 0).cpu().numpy(), 0, 1)*255).astype(np.uint8)
                cubic.write(cubic_frame)
                out = pred_frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
                out = np.clip(out, 0, 1)
                out = (out * 255).astype(np.uint8)
                writer.write(out)
        else:
            for i, (prev_frames, low_res_frame, next_frames) in enumerate(test_pbar):
                if self.listener is not None:
                    self.listener.test_batch_callback((i+1)/len(test_loader), None)
                pred_frame = self.upscale_batch(prev_frames, low_res_frame, next_frames, high_res_frame)
                cubic_frame = nn.functional.interpolate(low_res_frame, self.original_size[::-1], mode='bicubic')
                cubic_frame = (np.clip(cubic_frame.squeeze(0).permute(1, 2, 0).cpu().numpy(), 0, 1)*255).astype(np.uint8)
                cubic.write(cubic_frame)
                test_pbar.set_postfix({'frame': i})
                out = pred_frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
                out = np.clip(out, 0, 1)
                out = (out * 255).astype(np.uint8)
                writer.write(out)
        cv2.destroyAllWindows()
        writer.release()
        cubic.release()
        if self.mode == 'test':
            final_psnr = sum(test_psnrs)/len(test_psnrs)
            final_ssim = sum(test_ssim)/len(test_ssim)
            final_cubic_psnr = sum(cubic_psnrs)/len(cubic_psnrs)
            final_cubic_ssim = sum(cubic_ssim)/len(cubic_ssim)
            self.listener.final_loss_callback(final_psnr, final_ssim, final_cubic_psnr, final_cubic_ssim)
            mlflow.log_metric("test_psnr", final_psnr)
            mlflow.log_metric("test_ssim", final_ssim)

    @torch.no_grad()
    def test_batch(self, prev_frames, low_res_frame, next_frames, high_res_frame):
        prev_frames = prev_frames.to(self.device)
        low_res_frame = low_res_frame.to(self.device)
        next_frames = next_frames.to(self.device)
        high_res_frame = high_res_frame.to(self.device)
        if self.size == 'small':
            high_res_frame = high_res_frame[:, :, 6:-6, 6:-6]
        if self.size == 'large':
            high_res_frame = high_res_frame[:, :, 9:-9, 9:-9]
        pred_high_res_frame = self.model(prev_frames, low_res_frame, next_frames)
        psnr = PSNR(pred_high_res_frame, high_res_frame).item()
        ssim = SSIM(pred_high_res_frame, high_res_frame).item()
        return pred_high_res_frame, (psnr, ssim)

    @torch.no_grad()
    def upscale_batch(self, prev_frames, low_res_frame, next_frames):
        prev_frames = prev_frames.to(self.device)
        low_res_frame = low_res_frame.to(self.device)
        next_frames = next_frames.to(self.device)
        pred_high_res_frame = self.model(prev_frames, low_res_frame, next_frames)
        return pred_high_res_frame


class Upscaler:
    def __init__(self, model_path, device='auto', original_size=(1920, 1080), target_size=(1280, 720), listener=None):
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.original_size = original_size
        self.target_size = target_size
        upscale_factor = original_size[0] / target_size[0]
        model = TSRCNN_large(upscale_factor=upscale_factor)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model = model.to(device)
        self.dataset_format = NewStreamDataset
        self.listener: SimpleListener = listener
        self.run_name = "TSR_Upscaling"

    def _log_params(self, parameters: Dict):
        for key, value in parameters.items():
            mlflow.log_param(key, value)

    def upscale(self, video_file,  fps=60.0, video_path_out="output.mp4"):
        writer = ffmpegcv.VideoWriterNV(file=video_path_out, codec='h264_nvenc', fps=60.0, preset='p7', pix_fmt='rgb24') if self.device == 'cuda' else ffmpegcv.VideoWriter(
            file=video_path_out, codec='h264', fps=60.0, preset='p7', pix_fmt='rgb24')
        linear = ffmpegcv.VideoWriterNV(file="baseline.mp4", codec='h264_nvenc', fps=60.0, preset='p7', pix_fmt='rgb24') if self.device == 'cuda' else ffmpegcv.VideoWriter(
            file="baseline.mp4", codec='h264', fps=60.0, preset='p7', pix_fmt='rgb24')
        cubic = ffmpegcv.VideoWriterNV(file="cubic.mp4", codec='h264_nvenc', fps=60.0, preset='p7', pix_fmt='rgb24') if self.device == 'cuda' else ffmpegcv.VideoWriter(
            file="cubic.mp4", codec='h264', fps=60.0, preset='p7', pix_fmt='rgb24')

        dataset = self.dataset_format(video_file=video_file, original_size=self.original_size, target_size=self.target_size)
        num_frames = dataset.get_video_length()-1
        self.model.eval()
        mlflow.set_experiment("temporal_super_resolution_experiment")
        with mlflow.start_run(run_name=self.run_name):
            self._log_params({"video_file": video_file, "input_res": self.target_size, "output_res": self.original_size,
                             "num_frames": num_frames, "fps": fps})
        pbar = tqdm(dataset, total=num_frames, unit='frames')
        for i, frames in enumerate(pbar):
            with torch.no_grad():
                prev_frame, frame, _ = [f.unsqueeze(0).to(self.device) for f in frames]
                pred_frame = self.model(prev_frame, frame)
                linear_frame = cv2.resize(frame.squeeze(0).permute(1, 2, 0).cpu().numpy(), self.original_size, interpolation=cv2.INTER_LINEAR)
                linear_frame = (np.clip(linear_frame, 0, 1)*255).astype(np.uint8)
                cubic_frame = cv2.resize(frame.squeeze(0).permute(1, 2, 0).cpu().numpy(), self.original_size, interpolation=cv2.INTER_CUBIC)
                cubic_frame = (np.clip(cubic_frame, 0, 1)*255).astype(np.uint8)
                linear.write(linear_frame)
                cubic.write(cubic_frame)
                pbar.set_postfix({'frame': i})
                if self.listener is not None:
                    self.listener.epoch_callback(frame=i/num_frames)
                out = pred_frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
                out = np.clip(out, 0, 1)
                out = (out * 255).astype(np.uint8)
                writer.write(out)
        cv2.destroyAllWindows()
        writer.release()
        linear.release()
        cubic.release()
