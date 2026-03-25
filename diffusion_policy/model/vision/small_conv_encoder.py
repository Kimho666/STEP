import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallConvEncoder(nn.Module):
    """轻量级 CNN 编码器，用于 84x84/96x96 RGB 输入.
    输出 (B, embed_dim).
    """
    def __init__(self, in_channels: int = 3, embed_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(128, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W)
        feat = self.net(x)
        # 全局平均池化
        feat = F.adaptive_avg_pool2d(feat, output_size=(1, 1)).flatten(1)
        return self.head(feat)
