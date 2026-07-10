# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from .old_him_estimator import OldHIMEstimator
from .him_double_depth_actor_critic import HimDoubleDepthActorCritic


class OldHimDoubleDepthActorCritic(HimDoubleDepthActorCritic):
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
        encoder_hidden_dims=[256, 64, 16],  # Encoder hidden dims (output will be encoder_hidden_dims[-1])
        projector_hidden_dims=[256, 64],  # Target projector hidden dims for him
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

        super().__init__(
            obs,
            obs_groups,
            num_actions,
            num_one_step_obs,
            actor_obs_normalization,
            critic_obs_normalization,
            actor_hidden_dims,
            critic_hidden_dims,
            encoder_hidden_dims,  # Encoder hidden dims (output will be encoder_hidden_dims[-1])
            projector_hidden_dims,  # Target projector hidden dims for him
            depth_class_name=depth_class_name,
            depth_latent_dim=depth_latent_dim,
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
        )

    def _build_estimator(self):
        estimator = OldHIMEstimator(
            temporal_steps=self.history_size,
            num_one_step_obs=self.num_one_step_obs,
            num_one_step_priveleged_obs=self.num_critic_obs,
            enc_hidden_dims=self.encoder_hidden_dims,
            tar_hidden_dims=self.projector_hidden_dims,
        )
        print(f"Estimator Encoder: {estimator.encoder}")
        print(f"Estimator Target: {estimator.target}")

        return estimator
