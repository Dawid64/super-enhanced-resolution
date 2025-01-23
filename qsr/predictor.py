from typing import Dict, Literal, final
import cv2
import numpy as np
import torch
from tqdm import tqdm
import mlflow
import mlflow.pytorch
from .model import SrCNN2, SrCNN
import ffmpegcv
from torch.utils.data import DataLoader

from qsr.utils import SimpleListener

from .dataset_loading import NewStreamDataset, MultiVideoDataset
from torch import nn
from piqa.ssim import ssim, gaussian_kernel


def PSNR(y_pred, y_gt):
    return 10 * torch.log10(1 / (nn.MSELoss()(y_pred, y_gt)+1e-8))


class PNSR(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, y_pred, y_gt):
        result = 1/PSNR(y_pred, y_gt)
        return result


class DSSIM(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_gt):
        result = 1 - ssim(y_pred, y_gt, kernel=gaussian_kernel(7).repeat(3, 1, 1).to(y_pred.device), channel_avg=True)[0].mean()
        return result


LOSS: Dict[Literal['MSE', 'PNSR', 'DSSIM'], nn.Module] = {
    'MSE': nn.MSELoss,
    'PNSR': PNSR,
    'DSSIM': DSSIM
}


class Upscaler2:
    def __init__(self, model_path, device='auto', original_size=(1920, 1080), target_size=(1280, 720), listener=None, frames_backward=2, frames_forward=2, mode: Literal['test', 'inference'] = 'test', loss: Literal['MSE', 'PNSR', 'DSSIM'] = 'MSE'):
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.original_size = original_size
        self.target_size = target_size
        upscale_factor = original_size[0] / target_size[0]
        model = SrCNN(upscale_factor=upscale_factor, frames_backward=frames_backward, frames_forward=frames_forward)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model = model.to(device)
        self.dataset_format = MultiVideoDataset
        self.listener: SimpleListener = listener
        self.run_name = f"{target_size[1]}p -> {original_size[1]}p {frames_backward}fb{frames_forward}ff TSR {mode}"
        self.history = {'test_loss': []}
        self.mode = mode
        self.criterion = LOSS[loss]()

    def _log_params(self, parameters: Dict):
        for key, value in parameters.items():
            mlflow.log_param(key, value)

    def upscale(self, video_file, fps=60.0, video_path_out="output.mp4"):
        writer = ffmpegcv.VideoWriterNV(file=video_path_out, codec='h264_nvenc', fps=fps, preset='p7', pix_fmt='rgb24') if self.device == 'cuda' else ffmpegcv.VideoWriter(
            file=video_path_out, codec='h264', fps=60.0, preset='p7', pix_fmt='rgb24')
        linear = ffmpegcv.VideoWriterNV(file="baseline.mp4", codec='h264_nvenc', fps=fps, preset='p7', pix_fmt='rgb24') if self.device == 'cuda' else ffmpegcv.VideoWriter(
            file="linear.mp4", codec='h264', fps=60.0, preset='p7', pix_fmt='rgb24')
        cubic = ffmpegcv.VideoWriterNV(file="bicubic.mp4", codec='h264_nvenc', fps=fps, preset='p7', pix_fmt='rgb24') if self.device == 'cuda' else ffmpegcv.VideoWriter(
            file="bicubic.mp4", codec='h264', fps=60.0, preset='p7', pix_fmt='rgb24')
        test_dataset = self.dataset_format(video_paths=[video_file], original_size=self.original_size,
                                           target_size=self.target_size, frames_backward=2, frames_forward=2, mode='inference' if self.mode == 'inference' else 'training')
        num_frames = len(test_dataset)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        mlflow.set_experiment("temporal_super_resolution_experiment")
        with mlflow.start_run(run_name=self.run_name):
            self._log_params({"video_file": video_file,
                              "test_size": num_frames,
                              "original_size": self.original_size,
                              "target_size": self.target_size})
        self.model.eval()
        test_losses = []
        linear_lossses = []
        cubic_losses = []
        test_pbar = tqdm(test_loader, total=num_frames, unit='frame')
        if self.mode == 'test':
            for i, ((prev_frames, low_res_frame, next_frames), high_res_frame) in enumerate(test_pbar):
                loss, pred_frame = self.test_batch(prev_frames, low_res_frame, next_frames, high_res_frame)
                test_losses.append(loss)
                test_pbar.set_postfix({'test_loss': loss})
                self.history['test_loss'].append(loss)
                if self.listener is not None:
                    self.listener.test_batch_callback((i+1)/len(test_loader), self.history)
                print(low_res_frame.shape)
                print(self.original_size)
                linear_frame = nn.functional.interpolate(low_res_frame.squeeze(0), self.original_size, mode='linear')
                linear_loss = self.criterion(linear_frame, high_res_frame)
                linear_lossses.append(linear_loss)
                linear_frame = (np.clip(linear_frame.permute(1, 2, 0).cpu().numpy(), 0, 1)*255).astype(np.uint8)
                linear.write(linear_frame)
                cubic_frame = nn.functional.interpolate(low_res_frame.squeeze(0), self.original_size, mode='bicubic')
                cubic_loss = self.criterion(cubic_frame, high_res_frame)
                cubic_losses.append(cubic_loss)
                cubic_frame = (np.clip(cubic_frame.permute(1, 2, 0).cpu().numpy(), 0, 1)*255).astype(np.uint8)
                cubic.write(cubic_frame)
                test_pbar.set_postfix({'frame': i})
                out = pred_frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
                out = np.clip(out, 0, 1)
                out = (out * 255).astype(np.uint8)
                writer.write(out)
        else:
            for i, (prev_frames, low_res_frame, next_frames) in enumerate(test_pbar):
                if self.listener is not None:
                    self.listener.test_batch_callback((i+1)/len(test_loader), None)
                pred_frame = self.upscale_batch(prev_frames, low_res_frame, next_frames, high_res_frame)
                linear_frame = nn.functional.interpolate(low_res_frame.squeeze(0), self.original_size, mode='linear')
                linear_frame = (np.clip(linear_frame.permute(1, 2, 0).cpu().numpy(), 0, 1)*255).astype(np.uint8)
                linear.write(linear_frame)
                cubic_frame = nn.functional.interpolate(low_res_frame.squeeze(0), self.original_size, mode='bicubic')
                cubic_frame = (np.clip(cubic_frame.permute(1, 2, 0).cpu().numpy(), 0, 1)*255).astype(np.uint8)
                cubic.write(cubic_frame)
                test_pbar.set_postfix({'frame': i})
                out = pred_frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
                out = np.clip(out, 0, 1)
                out = (out * 255).astype(np.uint8)
                writer.write(out)
        cv2.destroyAllWindows()
        writer.release()
        linear.release()
        cubic.release()
        if self.mode == 'test':
            final_loss = sum(test_losses)/len(test_losses)
            final_linear_loss = sum(linear_lossses)/len(linear_lossses)
            final_cubic_loss = sum(cubic_losses)/len(cubic_losses)
            self.listener.final_loss_callback(final_loss, final_linear_loss, final_cubic_loss)

    @torch.no_grad()
    def test_batch(self, prev_frames, low_res_frame, next_frames, high_res_frame):
        prev_frames = prev_frames.to(self.device)
        low_res_frame = low_res_frame.to(self.device)
        next_frames = next_frames.to(self.device)
        high_res_frame = high_res_frame.to(self.device)
        pred_high_res_frame = self.model(prev_frames, low_res_frame, next_frames)
        loss = self.criterion(pred_high_res_frame, high_res_frame)
        return loss.item(), pred_high_res_frame

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
        model = SrCNN2(upscale_factor=upscale_factor)
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
