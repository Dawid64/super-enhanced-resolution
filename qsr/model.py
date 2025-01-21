import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SrCNN(nn.Module):
    def __init__(self, frames_back=2, frames_forward=2, upscale_factor=1.5):
        super(SrCNN, self).__init__()
        self.layers = nn.Sequential([
            nn.Upsample(scale_factor=upscale_factor, mode='bilinear', align_corners=False),
            nn.Conv2d(3 * (frames_back + 1 + frames_forward), 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1)
        ])

    def forward(self, back_frames: Tensor, low_res_frame: Tensor, forward_frames: Tensor) -> Tensor:
        x = torch.cat([back_frames, low_res_frame, forward_frames], dim=1)
        return self.layers(x)


class SrCNN2(nn.Module):
    def __init__(self, upscale_factor=1.5):
        super(SrCNN2, self).__init__()
        self.conv1 = nn.Conv2d(6, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.up = nn.Upsample(scale_factor=upscale_factor, mode='bilinear', align_corners=False)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 3, kernel_size=3, padding=1)

    def forward(self, prev_frame, curr_frame):
        x = torch.cat([prev_frame, curr_frame], dim=1)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.up(x)

        x = F.relu(self.conv3(x))
        x = self.conv4(x)

        return x

    def save(self, path):
        torch.save(self.cpu().state_dict(), path)

    @staticmethod
    def load(path):
        model = SrCNN2()
        model.load_state_dict(torch.load(path, weights_only=True))
        model.eval()
        return model


if __name__ == "__main__":
    batch_size = 1
    prev_frame = torch.rand(batch_size, 3, 720, 1280)
    curr_frame = torch.rand(batch_size, 3, 720, 1280)

    model = SrCNN2(upscale_factor=1.5)
    upscaled_curr_frame = model(prev_frame, curr_frame)

    print("Input shape:", curr_frame.shape)
    print("Output shape:", upscaled_curr_frame.shape)
