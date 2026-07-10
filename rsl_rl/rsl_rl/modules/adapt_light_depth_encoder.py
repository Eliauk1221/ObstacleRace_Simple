import torch.nn as nn


class AdaptLightDepthEncoder(nn.Module):
    """轻量深度编码器，支持任意 H*W*C 的输入图像。

    通过 AdaptiveAvgPool2d 将卷积后的特征图池化到固定空间尺寸，
    从而不依赖具体输入分辨率，仅需指定输入通道数 C。
    """

    # 卷积后的通道数；池化后的固定空间尺寸 (pool_h, pool_w)
    CNN_OUT_CHANNELS = 16
    POOL_H = 6
    POOL_W = 6

    def __init__(
        self,
        latent_dim: int,
        in_channels: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self._flat_size = self.CNN_OUT_CHANNELS * self.POOL_H * self.POOL_W

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 4, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(2, 4),
            nn.ELU(),
            nn.Conv2d(4, self.CNN_OUT_CHANNELS, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, self.CNN_OUT_CHANNELS),
            nn.ELU(),
            # nn.AdaptiveAvgPool2d((self.POOL_H, self.POOL_W)),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._flat_size, 64),
            nn.ELU(),
            nn.Linear(64, latent_dim),
            nn.ELU(),
        )

    def forward(self, depths):
        # B, H, W, C -> B, C, H, W
        depths = depths.permute(0, 3, 1, 2)
        return self.fc(self.cnn(depths))

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """

        super().load_state_dict(state_dict, strict=strict)
        return True  # training resumes
