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

from .pim_estimator import PIMEstimator
from .pim_estimator2 import PIMEstimator2
from .actor_critic import ActorCritic


class PIMActorCritic(ActorCritic):
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

        self.encoder_hidden_dims = encoder_hidden_dims
        self.projector_hidden_dims = projector_hidden_dims
        self.encoder_latent_dim = encoder_hidden_dims[-1]
        self.num_one_step_obs = num_one_step_obs
        self.estimator_class_name = estimator_class_name

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

    def _build_actor(self, num_actions):
        actor = MLP(
            self.num_one_step_obs + 3 + self.encoder_latent_dim, num_actions, self.actor_hidden_dims, self.activation
        )
        print(f"Actor MLP: {actor}")
        return actor

    def _build_critic(self, num_values=1):
        critic = MLP(self.num_critic_obs, num_values, self.critic_hidden_dims, self.activation)
        print(f"Critic MLP: {critic}")
        return critic

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

    def update_distribution(self, obs, depth):
        with torch.no_grad():
            vel, latent = self.estimator(obs, depth)
        actor_input = torch.cat((obs[:, -self.num_one_step_obs :], vel, latent), dim=-1)
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
        depth = obs["depth"]
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        vel, latent = self.estimator(obs, depth)
        actor_input = torch.cat((obs[:, -self.num_one_step_obs :], vel, latent), dim=-1)
        return self.actor(actor_input)

    def act(self, obs, **kwargs):
        depth = obs["depth"]
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        self.update_distribution(obs, depth)
        return self.distribution.sample()
