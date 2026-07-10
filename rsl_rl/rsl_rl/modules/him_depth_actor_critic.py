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
from .adapt_light_depth_encoder import AdaptLightDepthEncoder
from .cnn_attn_depth_encoder import CnnAttnDepthEncoder


class HimDepthActorCritic(nn.Module):
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
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        assert command_dim >= 0 and estimate_dim >= 0, "command_dim and estimate_dim must be non-negative."

        # get the observation dimensions
        self.obs_groups = obs_groups
        self.depth_latent_dim = depth_latent_dim
        self.estimate_dim = estimate_dim
        self.actor_hidden_dims = actor_hidden_dims
        self.critic_hidden_dims = critic_hidden_dims
        self.encoder_hidden_dims = encoder_hidden_dims
        self.projector_hidden_dims = projector_hidden_dims
        self.command_dim = command_dim
        self.activation = activation
        num_actor_obs = 0
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_actor_obs += obs[obs_group].shape[-1]
        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_critic_obs += obs[obs_group].shape[-1]

        self.num_one_step_obs = num_one_step_obs
        self.history_size = int(num_actor_obs / num_one_step_obs)
        self.num_critic_obs = num_critic_obs

        # Encoder output dimension (for actor input)
        self.encoder_latent_dim = encoder_hidden_dims[-1]
        self.projector_output_dim = projector_output_dim

        # actor: receives [current_obs, estimate, encoder_representation]
        self.actor = self._build_actor(num_actions)
        # actor observation normalization
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()

        # critic
        self.critic = self._build_critic()
        # critic observation normalization
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()

        # Estimator
        self.him_estimator = self._build_estimator()

        # Depth encoder
        self.depth_encoder = self._build_depth_encoder(depth_class_name)

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

    def _build_actor(self, num_actions):
        actor = MLP(
            self.num_one_step_obs + self.estimate_dim + self.encoder_latent_dim + self.depth_latent_dim, num_actions, self.actor_hidden_dims, self.activation
        )
        print(f"Actor MLP: {actor}")
        return actor

    def _build_critic(self, num_values=1):
        critic = MLP(self.num_critic_obs, num_values, self.critic_hidden_dims, self.activation)
        print(f"Critic MLP: {critic}")
        return critic

    def _build_estimator(self):
        estimator = HimEstimator(
            temporal_steps=self.history_size,
            num_one_step_obs=self.num_one_step_obs,
            num_one_step_priveleged_obs=self.num_critic_obs,
            enc_hidden_dims=self.encoder_hidden_dims,
            proj_hidden_dims=self.projector_hidden_dims,
            command_dim=self.command_dim,
            estimate_dim=self.estimate_dim,
            projector_output_dim=self.projector_output_dim,
        )
        print(f"Estimator Encoder: {estimator.encoder}")
        print(f"Estimator Projector: {estimator.projector}")
        return estimator

    def _build_depth_encoder(self, depth_class_name):
        depth_encoder_class = eval(depth_class_name)
        if depth_class_name == "AttnDepthEncoder" or depth_class_name == "CnnAttnDepthEncoder":
            # We need to pass input dimensions for the attention encoder
            # proprio dim is num_one_step_obs
            # estimator dim is velocity
            depth_encoder = depth_encoder_class(
                self.depth_latent_dim,
                proprio_dim=self.num_one_step_obs,
                estimator_dim=self.estimate_dim)
        else:
            depth_encoder : DepthEncoder | LightDepthEncoder | AdaptLightDepthEncoder = depth_encoder_class(self.depth_latent_dim)
        print(f"Depth Encoder: {depth_encoder}")
        return depth_encoder

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, obs, depth):
        with torch.no_grad():
            vel, latent = self.him_estimator(obs)
        if isinstance(self.depth_encoder, AttnDepthEncoder) or isinstance(self.depth_encoder, CnnAttnDepthEncoder):
            # Current proprio obs: [B, num_one_step_obs]
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

    def act(self, obs, **kwargs):
        proprio_obs = self.get_actor_obs(obs)
        proprio_obs = self.actor_obs_normalizer(proprio_obs)
        depth = self.get_depth_obs(obs)
        self.update_distribution(proprio_obs, depth)
        return self.distribution.sample()

    def act_inference(self, obs):
        proprio_obs = self.get_actor_obs(obs)
        proprio_obs = self.actor_obs_normalizer(proprio_obs)
        depth = self.get_depth_obs(obs)
        vel, latent = self.him_estimator(proprio_obs)
        if isinstance(self.depth_encoder, AttnDepthEncoder) or isinstance(self.depth_encoder, CnnAttnDepthEncoder):
            proprio = proprio_obs[:, -self.num_one_step_obs :]
            depth_latent = self.depth_encoder(depth, proprio, vel)
        else:
            depth_latent = self.depth_encoder(depth)
        actor_input = torch.cat((proprio_obs[:, -self.num_one_step_obs :], vel, latent, depth_latent), dim=-1)
        return self.actor(actor_input)

    def evaluate(self, obs, **kwargs):
        obs = self.get_critic_obs(obs)
        obs = self.critic_obs_normalizer(obs)
        return self.critic(obs)

    def get_actor_obs(self, obs):
        obs_list = []
        for obs_group in self.obs_groups["policy"]:
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1)

    def get_depth_obs(self, obs):
        return obs["depth"]

    def get_critic_obs(self, obs):
        obs_list = []
        for obs_group in self.obs_groups["critic"]:
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs):
        if self.actor_obs_normalization:
            actor_obs = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            critic_obs = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)

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
