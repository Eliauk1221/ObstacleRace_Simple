# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
from torch import autograd

DISC_LOGIT_INIT_SCALE = 1.0


class Discriminator(nn.Module):
    def __init__(
        self,
        observation_dim,
        observation_horizon,
        device,
        reward_coef=0.1,
        reward_lerp=0.3,
        shape=[1024, 512],
        style_reward_function="quad_mapping",
        mask_dims=None,  # List of dimension ranges to mask, e.g., [(start1, end1), (start2, end2)]
        mask_joint_names=None,  # List of joint names to mask (both position and velocity)
        joint_names=None,  # List of all joint names in order
        # Optional: encode height_scan (per-frame) to a low-dim latent.
        # If enabled, the discriminator replaces the height_scan slice with the latent slice.
        height_scan_encoder_enable: bool = False,
        height_scan_start_idx: int | None = 148,
        height_scan_input_dim: int | None = 187,
        height_scan_latent_dim: int = 16,
        height_scan_encoder_shape: list[int] | tuple[int, ...] = (64, 32),
        **kwargs,
    ):
        if kwargs:
            print(
                "Discriminator.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        # raw per-frame observation dim (before optional height encoder)
        self.observation_dim = observation_dim
        self.observation_horizon = observation_horizon
        self.input_dim_raw = observation_dim * observation_horizon
        self.device = device
        self.reward_coef = reward_coef
        self.reward_lerp = reward_lerp
        self.style_reward_function = style_reward_function
        self.shape = shape

        # Height scan encoder (optional)
        self.height_scan_encoder_enable = bool(height_scan_encoder_enable)
        self.height_scan_start_idx = height_scan_start_idx
        self.height_scan_input_dim = height_scan_input_dim
        self.height_scan_latent_dim = int(height_scan_latent_dim)
        self.height_scan_encoder_shape = list(height_scan_encoder_shape)
        self.height_scan_encoder = None
        self.no_height_observation_dim = self.observation_dim
        self.encoded_observation_dim = self.observation_dim  # kept for backward naming; now means per-frame dim without height

        if self.height_scan_encoder_enable:
            if self.height_scan_start_idx is None or self.height_scan_input_dim is None:
                raise ValueError(
                    "height_scan_encoder_enable=True requires height_scan_start_idx and height_scan_input_dim."
                )
            if self.height_scan_start_idx < 0 or self.height_scan_input_dim <= 0:
                raise ValueError("Invalid height_scan_start_idx/height_scan_input_dim.")
            if self.height_scan_start_idx + self.height_scan_input_dim > self.observation_dim:
                raise ValueError(
                    f"height_scan slice [{self.height_scan_start_idx},"
                    f" {self.height_scan_start_idx + self.height_scan_input_dim}) exceeds observation_dim"
                    f" ({self.observation_dim})."
                )
            if self.height_scan_latent_dim <= 0:
                raise ValueError("height_scan_latent_dim must be > 0.")

            # Build a small MLP encoder: input_dim -> ... -> latent_dim
            enc_layers = []
            enc_in = self.height_scan_input_dim
            for h in self.height_scan_encoder_shape:
                enc_layers.append(nn.Linear(enc_in, int(h)))
                enc_layers.append(nn.LeakyReLU())
                enc_in = int(h)
            enc_layers.append(nn.Linear(enc_in, self.height_scan_latent_dim))
            self.height_scan_encoder = nn.Sequential(*enc_layers).to(self.device)

            # After removing height from per-frame obs
            self.no_height_observation_dim = self.observation_dim - self.height_scan_input_dim
            self.encoded_observation_dim = self.no_height_observation_dim

        # Final discriminator input dim:
        # - flatten all frames of obs WITHOUT height_scan: horizon * no_height_observation_dim
        # - flatten all frames of encoded height latent: horizon * height_scan_latent_dim
        if self.height_scan_encoder_enable:
            self.input_dim = (self.no_height_observation_dim + self.height_scan_latent_dim) * self.observation_horizon
        else:
            self.input_dim = self.observation_dim * self.observation_horizon

        # Setup mask for specific dimensions
        self.mask_dims = mask_dims
        self.mask_joint_names = mask_joint_names
        self.joint_names = joint_names
        self._setup_mask()

        discriminator_layers = []
        curr_in_dim = self.input_dim
        for hidden_dim in self.shape:
            discriminator_layers.append(nn.Linear(curr_in_dim, hidden_dim))
            discriminator_layers.append(nn.LeakyReLU())
            curr_in_dim = hidden_dim
        self.architecture = nn.Sequential(*discriminator_layers).to(self.device)
        self.discriminator_logits = torch.nn.Linear(hidden_dim, 1)
        self.train()
        if self.height_scan_encoder_enable:
            print(f"Discriminator Height scan encoder: {self.height_scan_encoder}")
        else:
            print("Discriminator Height scan encoder: disabled")
        print(f"Discriminator: {self.architecture}")
        print(f"Discriminator logits: {self.discriminator_logits}")

    def _prepare_disc_input(self, x_flat: torch.Tensor) -> torch.Tensor:
        """Prepare discriminator input.

        If height_scan encoder is enabled:
        - reshape flat input to (B, H, D_raw)
        - per-frame encode height_scan -> (B, H, L)
        - remove height_scan from obs -> (B, H, D_no_height)
        - final input = cat(flatten(obs_no_height), flatten(height_latent)) -> (B, H*D_no_height + H*L)

        Otherwise:
        - return x_flat as-is (expects already flattened (B, H*D_raw))
        """
        if not self.height_scan_encoder_enable:
            return x_flat

        x = x_flat.view(-1, self.observation_horizon, self.observation_dim)  # (B, H, D_raw)
        s = int(self.height_scan_start_idx)
        e = s + int(self.height_scan_input_dim)

        height = x[..., s:e]  # (B, H, D_height)
        obs_no_height = torch.cat([x[..., :s], x[..., e:]], dim=-1)  # (B, H, D_no_height)

        # Encode height per frame; flatten B*H for efficiency
        bh = height.reshape(-1, height.shape[-1])
        latent = self.height_scan_encoder(bh).reshape(x.shape[0], x.shape[1], -1)  # (B, H, L)

        return torch.cat([obs_no_height.flatten(1, 2), latent.flatten(1, 2)], dim=-1)

    def _setup_mask(self):
        """Setup mask tensor for masking specific dimensions in the observation."""
        # Import MotionLoaderE1 indices locally to avoid circular imports
        from rsl_rl.utils.motion_loader import MotionLoaderE1

        # Create a mask tensor (1 = keep, 0 = mask out)
        mask = torch.ones(self.observation_dim, dtype=torch.float32, device=self.device)

        # Track if any masking is applied
        has_mask = False

        # Method 1: Mask by dimension ranges
        if self.mask_dims is not None and len(self.mask_dims) > 0:
            has_mask = True
            for start_idx, end_idx in self.mask_dims:
                if start_idx >= 0 and end_idx <= self.observation_dim:
                    mask[start_idx:end_idx] = 0.0
                else:
                    print(
                        f"Warning: Invalid mask range ({start_idx}, {end_idx}) for observation_dim"
                        f" {self.observation_dim}"
                    )

        # Method 2: Mask by joint names (both position and velocity)
        if self.mask_joint_names is not None and len(self.mask_joint_names) > 0:
            if self.joint_names is None:
                print("Warning: mask_joint_names provided but joint_names is None. Skipping joint-based masking.")
            else:
                has_mask = True
                masked_joints = []

                # Find indices of joints to mask
                joint_indices = []
                for mask_joint in self.mask_joint_names:
                    try:
                        idx = self.joint_names.index(mask_joint)
                        joint_indices.append(idx)
                        masked_joints.append(mask_joint)
                    except ValueError:
                        print(f"Warning: Joint '{mask_joint}' not found in joint_names. Skipping.")

                if joint_indices:
                    # Calculate the relative indices in the AMP observation
                    # AMP observation starts from JOINT_POSE
                    joint_pos_size = MotionLoaderE1.JOINT_POS_SIZE
                    joint_vel_start_relative = MotionLoaderE1.JOINT_VEL_START_IDX - MotionLoaderE1.JOINT_POSE_START_IDX

                    for joint_idx in joint_indices:
                        # Mask joint position
                        pos_idx = joint_idx
                        if pos_idx < joint_pos_size:
                            mask[pos_idx] = 0.0

                        # Mask joint velocity
                        vel_idx = joint_vel_start_relative + joint_idx
                        if vel_idx < self.observation_dim:
                            mask[vel_idx] = 0.0

                    print(f"Discriminator: Masking joints {masked_joints} (pos & vel)")

        # If no masking is applied, set mask to None
        if not has_mask:
            self.mask = None
            return

        # Expand mask to cover all frames in observation horizon
        # Shape: (observation_dim * observation_horizon,)
        self.mask = mask.repeat(self.observation_horizon)

        # Count masked dimensions for logging
        masked_count = (self.mask == 0).sum().item()
        total_dims = self.observation_dim * self.observation_horizon
        print(f"Discriminator: Masking {masked_count}/{total_dims} dimensions total")

    def _apply_mask(self, x):
        """Apply mask to input observations.

        Args:
            x: Input tensor of shape (batch_size, observation_dim * observation_horizon)

        Returns:
            Masked input tensor
        """
        if self.mask is None:
            return x
        return x * self.mask

    def forward(self, x):
        x = self._apply_mask(x)
        x = self._prepare_disc_input(x)
        return self.discriminator_logits(self.architecture(x))

    def get_disc_weights(self):
        weights = []
        for m in self.architecture.modules():
            if isinstance(m, nn.Linear):
                weights.append(torch.flatten(m.weight))
        return weights

    def get_disc_logit_weights(self):
        return torch.flatten(self.discriminator_logits.weight)

    def eval_disc(self, x):
        x = self._apply_mask(x)
        x = self._prepare_disc_input(x)
        return self.discriminator_logits(self.architecture(x))

    def compute_grad_pen(self, expert_data, lambda_=10):
        # Enable gradient computation only for expert_data
        expert_data.requires_grad_(True)
        disc = self.eval_disc(expert_data)
        grad = autograd.grad(
            outputs=disc,
            inputs=expert_data,
            grad_outputs=torch.ones(disc.size(), device=disc.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        # Enforce that the grad norm approaches 0.
        # Use more efficient computation: sum first, then pow
        grad_pen = lambda_ * (grad.pow(2).sum(dim=1).mean())
        return grad_pen

    def compute_wgan_grad_pen(self, expert_data, policy_data, k=2, p=6):
        expert_data.requires_grad_(True)
        policy_data.requires_grad_(True)

        expert_d = self.eval_disc(expert_data)
        expert_grad = autograd.grad(
            outputs=expert_d,
            inputs=expert_data,
            grad_outputs=torch.ones(expert_d.size(), device=expert_d.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        # Optimize: avoid view operation and use direct computation
        expert_grad_norm = expert_grad.pow(2).sum(1).pow(p / 2)

        policy_d = self.eval_disc(policy_data)
        policy_grad = autograd.grad(
            outputs=policy_d,
            inputs=policy_data,
            grad_outputs=torch.ones(policy_d.size(), device=policy_d.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        # Optimize: avoid view operation and use direct computation
        policy_grad_norm = policy_grad.pow(2).sum(1).pow(p / 2)

        grad_pen = (expert_grad_norm.mean() + policy_grad_norm.mean()) * k * 0.5
        return grad_pen

    def compute_weight_decay(self, lambda_=0.0001):
        disc_weights = self.get_disc_weights()
        disc_weights = torch.cat(disc_weights, dim=-1)
        weight_decay = lambda_ * torch.sum(torch.square(disc_weights))
        return weight_decay

    def compute_logit_reg(self, lambda_=0.05):
        logit_weights = self.get_disc_logit_weights()
        disc_logit_loss = lambda_ * torch.sum(torch.square(logit_weights))
        return disc_logit_loss

    def predict_amp_reward(self, state_buf, task_reward, state_normalizer=None, style_reward_normalizer=None):
        with torch.no_grad():
            self.eval()
            # Avoid unnecessary clone when normalizer is not used
            if state_normalizer is not None:
                # Batch normalize: reshape all frames at once for efficiency
                batch_size = state_buf.shape[0]
                state_flat = state_buf.view(-1, state_buf.shape[-1])
                state_flat_norm = state_normalizer.normalize_torch(state_flat, self.device)
                state_buf_norm = state_flat_norm.view(batch_size, -1)
                d = self.eval_disc(state_buf_norm)
            else:
                d = self.eval_disc(state_buf.flatten(1, 2))

            if self.style_reward_function == "quad_mapping":
                style_reward = torch.clamp(1 - (1 / 4) * torch.square(d - 1), min=0)
            elif self.style_reward_function == "log_mapping":
                style_reward = -torch.log(
                    torch.maximum(1 - 1 / (1 + torch.exp(-d)), torch.tensor(0.0001, device=self.device))
                )
            elif self.style_reward_function == "wasserstein_mapping":
                if style_reward_normalizer is not None:
                    d_clone = d.clone()
                    style_reward = style_reward_normalizer.normalize_torch(d_clone, self.device)
                    style_reward_normalizer.update(d.cpu().numpy())
                else:
                    style_reward = torch.exp(torch.tanh(0.3 * d)) - torch.exp(-1 * torch.ones_like(d))
            else:
                raise ValueError("Unexpected style reward mapping specified")
            style_reward *= (1.0 - self.reward_lerp) * self.reward_coef
            task_reward = task_reward.unsqueeze(-1) * self.reward_lerp
            reward = style_reward + task_reward
            self.train()
        return reward.squeeze(), style_reward.squeeze()
