# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import statistics
import time
import warnings
from collections import deque

import torch

import rsl_rl
from rsl_rl.algorithms import AMPPIMPPO, AMPGatePIMPPO
from rsl_rl.env import VecEnv
from rsl_rl.modules import (
    Discriminator,
    PIMActorCritic,
    PIMDepthActorCritic,
    PIMDepthMoEActorCritic,
    GatePIMActorCriticCMoE,
    resolve_rnd_config,
    resolve_symmetry_config,
)
from rsl_rl.utils import Normalizer, resolve_obs_groups, store_code_state
from rsl_rl.utils.motion_loader import *
from rsl_rl.runners.amp_old_him_on_policy_runner import AmpOldHimOnPolicyRunner


class AmpPimOnPolicyRunner(AmpOldHimOnPolicyRunner):
    """On-policy runner for training and evaluation of actor-critic methods."""

    def save(self, path: str, infos=None):
        # -- Save model
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "actor_state_dict": self.alg.policy.actor.state_dict(),
            "estimator_state_dict": self.alg.policy.estimator.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "discriminator_state_dict": self.alg.discriminator.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        # -- Save RND model if used
        if hasattr(self.alg, "rnd") and self.alg.rnd:
            saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
        torch.save(saved_dict, path)

        # upload model to external logging service
        if self.logger_type in ["neptune", "wandb"] and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration)

    def train_mode(self):
        # -- PPO
        self.alg.policy.train()
        # -- AMP
        self.discriminator.train()
        # -- PIM
        self.alg.policy.estimator.train()
        # -- RND
        if hasattr(self.alg, "rnd") and self.alg.rnd:
            self.alg.rnd.train()

    def eval_mode(self):
        # -- PPO
        self.alg.policy.eval()
        # -- AMP
        self.discriminator.eval()
        # -- PIM
        self.alg.policy.estimator.eval()
        # -- RND
        if hasattr(self.alg, "rnd") and self.alg.rnd:
            self.alg.rnd.eval()

    """
    Helper functions.
    """

    def _construct_algorithm(self, obs) -> AMPPIMPPO:
        """Construct the actor-critic algorithm."""
        # resolve RND config
        self.alg_cfg = resolve_rnd_config(self.alg_cfg, obs, self.cfg["obs_groups"], self.env)

        # resolve symmetry config
        self.alg_cfg = resolve_symmetry_config(self.alg_cfg, self.env)

        # resolve deprecated normalization config
        if self.cfg.get("empirical_normalization") is not None:
            warnings.warn(
                "The `empirical_normalization` parameter is deprecated. Please set `actor_obs_normalization` and "
                "`critic_obs_normalization` as part of the `policy` configuration instead.",
                DeprecationWarning,
            )
            if self.policy_cfg.get("actor_obs_normalization") is None:
                self.policy_cfg["actor_obs_normalization"] = self.cfg["empirical_normalization"]
            if self.policy_cfg.get("critic_obs_normalization") is None:
                self.policy_cfg["critic_obs_normalization"] = self.cfg["empirical_normalization"]

        # initialize the actor-critic
        actor_critic_class = eval(self.policy_cfg.pop("class_name"))
        actor_critic: PIMActorCritic | PIMDepthActorCritic | GatePIMActorCriticCMoE = actor_critic_class(
            obs,
            self.cfg["obs_groups"],
            self.env.num_actions,
            self.env.observation_space["policy"].shape[-1],
            **self.policy_cfg,
        ).to(self.device)

        # initialize the amp algorithm
        amp_expert_data: MotionLoaderE1 = self.env.unwrapped.motion_loader
        self.amp_state_normalizer = Normalizer(self.env.unwrapped.motion_loader.observation_dim)

        normalize_style_reward = self.cfg["normalize_style_reward"]
        if normalize_style_reward:
            self.amp_style_reward_normalizer = Normalizer(1)
        else:
            self.amp_style_reward_normalizer = None

        self.discriminator = Discriminator(
            observation_dim=self.env.unwrapped.motion_loader.observation_dim,
            observation_horizon=self.env.unwrapped.motion_loader.reference_observation_horizon,
            device=self.device,
            reward_coef=self.cfg["amp_reward_coef"],
            reward_lerp=self.cfg["amp_reward_lerp"],
            shape=self.cfg["discriminator_shape"],
            style_reward_function=self.cfg["style_reward_function"],
            joint_names=self.cfg["joint_names"],
            mask_joint_names=self.cfg["discriminator_mask_joint_names"],
            mask_dims=self.cfg["discriminator_mask_dims"],
        ).to(self.device)

        # initialize the ppo algorithm
        alg_class = eval(self.alg_cfg.pop("class_name"))
        alg: AMPPIMPPO | AMPGatePIMPPO = alg_class(
            actor_critic,
            self.discriminator,
            amp_expert_data,
            self.amp_state_normalizer,
            self.amp_style_reward_normalizer,
            device=self.device,
            **self.alg_cfg,
            multi_gpu_cfg=self.multi_gpu_cfg,
        )

        # initialize the storage
        alg.init_storage(
            "rl",
            self.env.num_envs,
            self.num_steps_per_env,
            obs,
            [self.env.num_actions],
        )

        return alg
