import torch.nn as nn


class DepthEncoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()

        self.cnn = nn.Sequential(
            # 48*64*1
            nn.Conv2d(1, 16, kernel_size=5, stride=4, padding=1),
            nn.GroupNorm(8, 16),
            nn.ELU(),
            # 12*16*16
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.ELU(),
            # 6*8*32
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.ELU(),
            # 3*4*64
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 3, 128),
            nn.ELU(),
            nn.Linear(128, latent_dim),
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
