import torch.nn as nn


class MlpDepthEncoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()

        # self.cnn = nn.Sequential(
        #     # 16*24*1
        #     nn.Conv2d(1, 4, kernel_size=3, stride=2, padding=1),
        #     nn.GroupNorm(2, 4),
        #     nn.ELU(),
        #     # 8*12*4
        #     nn.Conv2d(4, 16, kernel_size=3, stride=2, padding=1),
        #     nn.GroupNorm(8, 16),
        #     nn.ELU(),
        #     # 4*6*16
        # )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 24, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, latent_dim),
            nn.ELU(),
        )

    def forward(self, depths):
        # B, H, W, C -> B, CHW
        depths = depths.view(depths.shape[0], -1)
        return self.fc(depths)

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
