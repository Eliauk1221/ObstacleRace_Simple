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

from .gate_pim_estimator import GatePIMEstimator
from .gate_pim_estimator_depth import GatePIMEstimatorDepth
from .gate_pim_estimator_scan_dots import GatePIMEstimatorScanDots
from .actor_critic import ActorCritic
from .cmoe import CMoeLayer
from .moe import MoeLayer


class GatePIMActorCriticCMoE(ActorCritic):
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
        estimator_class_name="GatePIMEstimator",
        encoder_hidden_dims=[256, 64, 16],  # Encoder hidden dims (output will be encoder_hidden_dims[-1])
        projector_hidden_dims=[256, 64],  # Projector hidden dims for Barlow Twins
        depth_class_name="DepthEncoder",
        depth_latent_dim=32,
        num_moe_experts=8,
        moe_gate_hidden_dims=[],
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
        self.num_moe_experts = num_moe_experts
        self.moe_gate_hidden_dims = moe_gate_hidden_dims
        self.encoder_hidden_dims = encoder_hidden_dims
        self.projector_hidden_dims = projector_hidden_dims
        self.encoder_latent_dim = encoder_hidden_dims[-1]
        self.num_one_step_obs = num_one_step_obs
        self.estimator_class_name = estimator_class_name
        self.cur_depth_latent = None
        super().__init__(
            obs,
            obs_groups,
            num_actions,
            actor_obs_normalization,
            critic_obs_normalization,
            actor_hidden_dims,
            critic_hidden_dims,
            activation,
            init_noise_std,
            noise_std_type,
        )

        self.history_size = int(self.num_actor_obs / num_one_step_obs)
        self.estimator = self._build_estimator()
        self.depth_encoder = self._build_depth_encoder(depth_class_name)

    def _build_actor(self, num_actions):
        moe = CMoeLayer(
            self.num_one_step_obs + 3 + self.encoder_latent_dim + self.depth_latent_dim,
            self.num_moe_experts,
            output_dim=num_actions,
            activation=self.activation,
            expert_hidden_dims=self.actor_hidden_dims,
        )
        print(f"MoE Actor: {moe}")
        return moe

    def _build_critic(self, num_values=1):
        moe = MoeLayer(
            self.num_critic_obs,
            self.num_moe_experts,
            output_dim=num_values,
            activation=self.activation,
            expert_hidden_dims=self.critic_hidden_dims,
            gate_hidden_dims=self.moe_gate_hidden_dims,
        )
        print(f"MoE Critic: {moe}")
        return moe

    def _build_estimator(self):
        estimator = eval(self.estimator_class_name)(
            temporal_steps=self.history_size,
            num_one_step_obs=self.num_one_step_obs,
            num_one_step_priveleged_obs=self.num_critic_obs,
            enc_hidden_dims=self.encoder_hidden_dims,
            tar_hidden_dims=self.projector_hidden_dims,
        )
        print(f"Estimator Encoder: {estimator.encoder}")
        print(f"Estimator Targeter: {estimator.target}")
        return estimator

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
            vel, latent, gate = self.estimator(obs, depth)
        if isinstance(self.depth_encoder, AttnDepthEncoder) or isinstance(self.depth_encoder, CnnAttnDepthEncoder):
            proprio = obs[:, -self.num_one_step_obs :]
            depth_latent = self.depth_encoder(depth, proprio, vel)
        else:
            depth_latent = self.depth_encoder(depth)
        self.cur_depth_latent = depth_latent
        actor_input = torch.cat((obs[:, -self.num_one_step_obs :], vel, latent, depth_latent), dim=-1)
        # compute mean
        mean = self.actor(actor_input, gate)
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
        depth = obs["depth"]
        proprio_obs = self.get_actor_obs(obs)
        proprio_obs = self.actor_obs_normalizer(proprio_obs)
        vel, latent, gate = self.estimator(proprio_obs, depth)
        if isinstance(self.depth_encoder, AttnDepthEncoder) or isinstance(self.depth_encoder, CnnAttnDepthEncoder):
            proprio = proprio_obs[:, -self.num_one_step_obs :]
            depth_latent = self.depth_encoder(depth, proprio, vel)
        else:
            depth_latent = self.depth_encoder(depth)
        self.cur_depth_latent = depth_latent
        actor_input = torch.cat((proprio_obs[:, -self.num_one_step_obs :], vel, latent, depth_latent), dim=-1)
        return self.actor(actor_input, gate)

    def act(self, obs, **kwargs):
        depth = obs["depth"]
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        self.update_distribution(obs, depth)
        return self.distribution.sample()

    def get_depth_obs(self, obs):
        return obs["depth"]
