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

from .depth_encoder import DepthEncoder
from .light_depth_encoder import LightDepthEncoder
from .attn_depth_encoder import AttnDepthEncoder
from .cnn_attn_depth_encoder import CnnAttnDepthEncoder
from .adapt_light_depth_encoder import AdaptLightDepthEncoder
from .pim_actor_critic import PIMActorCritic


class PIMDepthActorCritic(PIMActorCritic):
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
        estimator_class_name="PIMEstimator",
        encoder_hidden_dims=[256, 128, 64],  # Encoder hidden dims (output will be encoder_hidden_dims[-1])
        projector_hidden_dims=[256, 128],  # Projector hidden dims for Barlow Twins
        depth_class_name="DepthEncoder",
        depth_latent_dim=32,
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )

        self.depth_latent_dim = depth_latent_dim

        super().__init__(
            obs,
            obs_groups,
            num_actions,
            num_one_step_obs,
            actor_obs_normalization,
            critic_obs_normalization,
            actor_hidden_dims,
            critic_hidden_dims,
            estimator_class_name,
            encoder_hidden_dims,
            projector_hidden_dims,
            activation,
            init_noise_std,
            noise_std_type,
        )

        self.depth_encoder = self._build_depth_encoder(depth_class_name)

    def _build_actor(self, num_actions):
        actor = MLP(
            self.num_one_step_obs + 3 + self.encoder_latent_dim + self.depth_latent_dim, num_actions, self.actor_hidden_dims, self.activation
        )
        print(f"Actor MLP: {actor}")
        return actor

    def _build_critic(self, num_values=1):
        critic = MLP(self.num_critic_obs, num_values, self.critic_hidden_dims, self.activation)
        print(f"Critic MLP: {critic}")
        return critic

    def _build_depth_encoder(self, depth_class_name):
        depth_encoder_class = eval(depth_class_name)
        if depth_class_name == "AttnDepthEncoder" or depth_class_name == "CnnAttnDepthEncoder":
            depth_encoder : AttnDepthEncoder | CnnAttnDepthEncoder = depth_encoder_class(
                self.depth_latent_dim,
                proprio_dim=self.num_one_step_obs,
                estimator_dim=3)
        else:
            depth_encoder : DepthEncoder | LightDepthEncoder | AdaptLightDepthEncoder = depth_encoder_class(self.depth_latent_dim)
        print(f"Depth Encoder: {depth_encoder}")
        return depth_encoder

    def update_distribution(self, obs, depth):
        with torch.no_grad():
            vel, latent = self.estimator(obs, depth)
        if isinstance(self.depth_encoder, AttnDepthEncoder) or isinstance(self.depth_encoder, CnnAttnDepthEncoder):
            proprio = obs[:, -self.num_one_step_obs :]
            depth_latent = self.depth_encoder(depth, proprio, vel)
        else:
            depth_latent = self.depth_encoder(depth)
        actor_input = torch.cat((obs[:, -self.num_one_step_obs :], vel, latent, depth_latent), dim=-1)
        # compute mean
        mean = self.actor(actor_input)
        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
        self.distribution = Normal(mean, std)

    def act_inference(self, obs):
        proprio_obs = self.get_actor_obs(obs)
        proprio_obs = self.actor_obs_normalizer(proprio_obs)
        depth = self.get_depth_obs(obs)
        vel, latent = self.estimator(proprio_obs, depth)
        if isinstance(self.depth_encoder, AttnDepthEncoder) or isinstance(self.depth_encoder, CnnAttnDepthEncoder):
            proprio = proprio_obs[:, -self.num_one_step_obs :]
            depth_latent = self.depth_encoder(depth, proprio, vel)
        else:
            depth_latent = self.depth_encoder(depth)
        actor_input = torch.cat((proprio_obs[:, -self.num_one_step_obs :], vel, latent, depth_latent), dim=-1)
        return self.actor(actor_input)

    def act(self, obs, **kwargs):
        depth = obs["depth"]
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        self.update_distribution(obs, depth)
        return self.distribution.sample()

    def get_depth_obs(self, obs):
        return obs["depth"]
