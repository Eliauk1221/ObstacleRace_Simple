import torch
import torch.nn as nn
import torch.nn.functional as F

from .old_him_double_depth_actor_critic import OldHimDoubleDepthActorCritic
from .moe import MoeLayer


class OldHimDoubleDepthMoEActorCritic(OldHimDoubleDepthActorCritic):
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
        num_moe_experts=8,
        moe_gate_hidden_dims=[],
        activation="elu",
        **kwargs,
    ):
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
            encoder_hidden_dims,
            projector_hidden_dims,
            depth_class_name,
            depth_latent_dim,
            activation,
            **kwargs,
        )

    def _build_actor(self, num_actions):
        moe = MoeLayer(
            self.num_one_step_obs + self.estimate_dim + self.encoder_latent_dim + self.depth_latent_dim,
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
            self.num_critic_obs + self.depth_latent_dim,
            self.num_moe_experts,
            output_dim=num_values,
            activation=self.activation,
            expert_hidden_dims=self.critic_hidden_dims,
            gate_hidden_dims=self.moe_gate_hidden_dims,
        )
        print(f"MoE Critic: {moe}")
        return moe
