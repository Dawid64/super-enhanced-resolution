import torch
import torch.nn as nn
import torch.nn.functional as F


class SrCnn(nn.Module):
    def __init__(self):
        super(SrCnn, self).__init__()
        self.fullHD_path = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU()
        )
        self.hd_path = nn.Sequential(
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

    def forward(self, x_fullHD, x_hd):
        if x_fullHD.dim() == 3:
            x_fullHD = x_fullHD.unsqueeze(0)
        if x_hd.dim() == 3:
            x_hd = x_hd.unsqueeze(0)
        x_hd_up = F.interpolate(
            x_hd, size=x_fullHD.shape[2:], mode='bilinear', align_corners=False)
        x1 = self.fullHD_path(x_fullHD)
        x2 = self.hd_path(x_hd_up)
        return self.fusion(torch.cat([x1, x2], dim=1))

    def save(self, path):
        torch.save(self.cpu().state_dict(), path)

    @staticmethod
    def load(path):
        model = SrCnn()
        model.load_state_dict(torch.load(path, weights_only=True))
        model.eval()
        return model
