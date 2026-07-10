# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from .moe import MoeLayer
from .pim_depth_actor_critic import PIMDepthActorCritic


class PIMDepthMoEActorCritic(PIMDepthActorCritic):
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
        encoder_hidden_dims=[256, 64, 16],  # Encoder hidden dims (output will be encoder_hidden_dims[-1])
        projector_hidden_dims=[256, 64],  # Target projector hidden dims for him
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

        self.num_moe_experts = num_moe_experts
        self.moe_gate_hidden_dims = moe_gate_hidden_dims

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
            depth_class_name,
            depth_latent_dim,
            activation,
            init_noise_std,
            noise_std_type,
        )

    def _build_actor(self, num_actions):
        moe = MoeLayer(
            self.num_one_step_obs + 3 + self.encoder_latent_dim + self.depth_latent_dim,
            self.num_moe_experts,
            output_dim=num_actions,
            activation=self.activation,
            expert_hidden_dims=self.actor_hidden_dims,
            gate_hidden_dims=self.moe_gate_hidden_dims,
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
