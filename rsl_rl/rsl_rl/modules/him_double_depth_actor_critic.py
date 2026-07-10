# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.networks import MLP, EmpiricalNormalization

from .him_estimator import HimEstimator
from .depth_encoder import DepthEncoder
from .light_depth_encoder import LightDepthEncoder
from .attn_depth_encoder import AttnDepthEncoder
from .cnn_attn_depth_encoder import CnnAttnDepthEncoder
from .him_depth_actor_critic import HimDepthActorCritic


class HimDoubleDepthActorCritic(HimDepthActorCritic):
    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        num_one_step_obs,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        encoder_hidden_dims=[256, 128, 64],  # Encoder hidden dims (output will be encoder_hidden_dims[-1])
        projector_hidden_dims=[256, 256],  # Projector hidden dims for Barlow Twins
        projector_output_dim=256,  # Projector output dim (only used in training)
        command_dim=3,  # default to command linear velocity and angular velocity (x, y, yaw)
        estimate_dim=3,  # default to estimate linear velocity (x, y, z)
        depth_class_name="DepthEncoder",
        depth_latent_dim=32,
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        state_dependent_std=False,
        **kwargs,
    ):
        super().__init__(
            obs,
            obs_groups,
            num_actions,
            num_one_step_obs,
            actor_obs_normalization,
            critic_obs_normalization,
            actor_hidden_dims,
            critic_hidden_dims,
            encoder_hidden_dims,
            projector_hidden_dims,
            projector_output_dim,
            command_dim,
            estimate_dim,
            depth_class_name,
            depth_latent_dim,
            activation,
            init_noise_std,
            noise_std_type,
            state_dependent_std,
            **kwargs,
        )

    def _build_critic(self, num_values=1):
        critic = MLP(self.num_critic_obs + self.depth_latent_dim, num_values, self.critic_hidden_dims, self.activation)
        print(f"Critic MLP: {critic}")
        return critic

    def evaluate(self, obs, **kwargs):
        # critic obs
        critic_obs = self.get_critic_obs(obs)
        critic_obs = self.critic_obs_normalizer(critic_obs)
        # privileged depth obs
        privileged_depth_obs = self.get_privileged_depth_obs(obs)
        if isinstance(self.depth_encoder, AttnDepthEncoder) or isinstance(self.depth_encoder, CnnAttnDepthEncoder):
            # Current proprio obs: [B, num_one_step_obs]
            proprio = critic_obs[:, :self.num_one_step_obs]
            vel = critic_obs[:, self.num_one_step_obs:self.num_one_step_obs + 3]
            privileged_depth_latent = self.depth_encoder(privileged_depth_obs, proprio, vel)
        else:
            privileged_depth_latent = self.depth_encoder(privileged_depth_obs)

        return self.critic(torch.cat([critic_obs, privileged_depth_latent], dim=-1))

    def get_privileged_depth_obs(self, obs):
        return obs["privileged_depth"]
