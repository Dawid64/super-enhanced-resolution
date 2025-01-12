import torch
import torch.nn as nn
import torch.nn.functional as F


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
