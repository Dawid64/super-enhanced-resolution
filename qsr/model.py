import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TSRCNN_small(nn.Module):
    def __init__(self, frames_backward=2, frames_forward=2, upscale_factor=1.5):
        super(TSRCNN_small, self).__init__()
        self.layers = nn.Sequential(
            nn.Upsample(scale_factor=upscale_factor, mode='bicubic', align_corners=False),
            nn.Conv2d(3 * (frames_backward + 1 + frames_forward), 64, kernel_size=9, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=5, padding=0)
        )

    def forward(self, back_frames: Tensor, low_res_frame: Tensor, forward_frames: Tensor) -> Tensor:
        x = torch.cat([back_frames, low_res_frame, forward_frames], dim=1)
        return self.layers(x)

    def save(self, path):
        torch.save(self.cpu().state_dict(), path)

    @staticmethod
    def load(path, frames_backward=2, frames_forward=2, upscale_factor=1.5):
        model = TSRCNN_small(upscale_factor=upscale_factor, frames_backward=frames_backward, frames_forward=frames_forward)
        model.load_state_dict(torch.load(path, weights_only=True))
        return model


class TSRCNN_large(nn.Module):
    def __init__(self, frames_backward=2, frames_forward=2, upscale_factor=1.5):
        super(TSRCNN_large, self).__init__()
        self.layers = nn.Sequential(
            nn.Upsample(scale_factor=upscale_factor, mode='bicubic', align_corners=False),
            nn.Conv2d(3 * (frames_backward + 1 + frames_forward), 128, kernel_size=11, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=7, padding=0)
        )

    def forward(self, back_frames: Tensor, low_res_frame: Tensor, forward_frames: Tensor) -> Tensor:
        x = torch.cat([back_frames, low_res_frame, forward_frames], dim=1)
        return self.layers(x)

    def save(self, path):
        torch.save(self.cpu().state_dict(), path)

    @staticmethod
    def load(path, frames_backward=2, frames_forward=2, upscale_factor=1.5):
        model = TSRCNN_large(upscale_factor=upscale_factor, frames_backward=frames_backward, frames_forward=frames_forward)
        model.load_state_dict(torch.load(path, weights_only=True))
        return model


class SrCnn(nn.Module):
    def __init__(self):
        super(SrCnn, self).__init__()
        self.prev_high_res = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU()
        )
        self.low_res = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU()
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, 3, padding=1)
        )

    def forward(self, x_prev_high_res, x_low_res):
        if x_prev_high_res.dim() == 3:
            x_prev_high_res = x_prev_high_res.unsqueeze(0)
        if x_low_res.dim() == 3:
            x_low_res = x_low_res.unsqueeze(0)
        x_hd_up = F.interpolate(x_low_res, size=x_prev_high_res.shape[2:], mode='bilinear', align_corners=False)
        x1 = self.prev_high_res(x_prev_high_res)
        x2 = self.low_res(x_hd_up)
        return self.fusion(torch.cat([x1, x2], dim=1))

    def save(self, path):
        torch.save(self.cpu().state_dict(), path)

    @staticmethod
    def load(path):
        model = SrCnn()
        model.load_state_dict(torch.load(path, weights_only=True))
        model.eval()
        return model


if __name__ == "__main__":
    batch_size = 1
    prev_frame = torch.rand(batch_size, 3, 720, 1280)
    curr_frame = torch.rand(batch_size, 3, 720, 1280)

    model = TSRCNN_large(upscale_factor=1.5)
    upscaled_curr_frame = model(prev_frame, curr_frame)

    print("Input shape:", curr_frame.shape)
    print("Output shape:", upscaled_curr_frame.shape)
