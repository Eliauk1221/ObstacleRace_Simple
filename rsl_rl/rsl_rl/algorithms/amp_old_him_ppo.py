# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from itertools import chain

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import (
    Discriminator,
    FootholdImagination,
    OldHimDepthActorCritic,
    OldHimDoubleDepthActorCritic,
    OldHimDoubleDepthMoEActorCritic,
)
from rsl_rl.modules.foothold_imagination import world_xy_to_base_yaw
from rsl_rl.modules.rnd import RandomNetworkDistillation
from rsl_rl.storage import FootholdReplayBuffer, OldHimRolloutStorage, ReplayBuffer
from rsl_rl.utils import string_to_callable


class AMPOldHIMPPO:
    """Proximal Policy Optimization algorithm (https://arxiv.org/abs/1707.06347)."""

    policy: OldHimDepthActorCritic | OldHimDoubleDepthActorCritic | OldHimDoubleDepthMoEActorCritic
    discriminator: Discriminator
    """The actor critic module."""

    def __init__(
        self,
        policy,
        discriminator,
        amp_expert_data,
        amp_state_normalizer,
        amp_style_reward_normalizer,
        num_learning_epochs=5,
        num_mini_batches=4,
        clip_param=0.2,
        gamma=0.99,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.01,
        learning_rate=0.001,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="adaptive",
        desired_kl=0.01,
        device="cpu",
        discriminator_learning_rate=1e-3,
        discriminator_gradient_penalty_coef=5,
        discriminator_num_mini_batches=10,
        amp_replay_buffer_size=100000,
        discriminator_loss_function="MSELoss",
        normalize_advantage_per_mini_batch=False,
        foothold_imagination_enabled=False,
        foothold_learning_rate=1e-3,
        foothold_hidden_dims=(512, 256, 128),
        foothold_min_sigma=0.02,
        foothold_max_sigma=0.50,
        foothold_replay_buffer_size=50000,
        foothold_pending_steps=16,
        foothold_batch_size=2048,
        foothold_updates_per_iteration=4,
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
    ):
        # device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # RND components
        if rnd_cfg is not None:
            # Extract parameters used in ppo
            rnd_lr = rnd_cfg.pop("learning_rate", 1e-3)
            # Create RND module
            self.rnd = RandomNetworkDistillation(device=self.device, **rnd_cfg)
            # Create RND optimizer
            params = self.rnd.predictor.parameters()
            self.rnd_optimizer = optim.Adam(params, lr=rnd_lr)
        else:
            self.rnd = None
            self.rnd_optimizer = None

        # Symmetry components
        if symmetry_cfg is not None:
            # Check if symmetry is enabled
            use_symmetry = symmetry_cfg["use_data_augmentation"] or symmetry_cfg["use_mirror_loss"]
            # Print that we are not using symmetry
            if not use_symmetry:
                print("Symmetry not used for learning. We will use it for logging instead.")
            # If function is a string then resolve it to a function
            if isinstance(symmetry_cfg["data_augmentation_func"], str):
                symmetry_cfg["data_augmentation_func"] = string_to_callable(symmetry_cfg["data_augmentation_func"])
            # Check valid configuration
            if symmetry_cfg["use_data_augmentation"] and not callable(symmetry_cfg["data_augmentation_func"]):
                raise ValueError(
                    "Data augmentation enabled but the function is not callable:"
                    f" {symmetry_cfg['data_augmentation_func']}"
                )
            # Store symmetry configuration
            self.symmetry = symmetry_cfg
        else:
            self.symmetry = None

        # PPO components
        self.policy = policy
        self.policy.to(self.device)

        # Create optimizer for PPO (exclude HIM estimator parameters)
        # The HIM estimator has its own optimizer to avoid conflicts
        ppo_params = [param for name, param in self.policy.named_parameters() if not name.endswith("estimator.")]
        self.optimizer = optim.Adam(ppo_params, lr=learning_rate)

        # Create rollout storage
        self.storage: OldHimRolloutStorage = None  # type: ignore
        self.transition = OldHimRolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

        # Training-only privileged model for future foothold distributions.
        self.foothold_imagination_enabled = foothold_imagination_enabled
        self.foothold_learning_rate = foothold_learning_rate
        self.foothold_hidden_dims = tuple(foothold_hidden_dims)
        self.foothold_min_sigma = foothold_min_sigma
        self.foothold_max_sigma = foothold_max_sigma
        self.foothold_replay_buffer_size = foothold_replay_buffer_size
        self.foothold_pending_steps = foothold_pending_steps
        self.foothold_batch_size = foothold_batch_size
        self.foothold_updates_per_iteration = foothold_updates_per_iteration
        self.foothold_imagination: FootholdImagination | None = None
        self.foothold_optimizer: optim.Optimizer | None = None
        self.foothold_replay: FootholdReplayBuffer | None = None

        # Discriminator components
        self.discriminator = discriminator
        self.discriminator.to(self.device)
        self.amp_policy_data = ReplayBuffer(
            discriminator.observation_dim, discriminator.observation_horizon, amp_replay_buffer_size, device
        )
        self.amp_expert_data = amp_expert_data
        self.amp_state_normalizer = amp_state_normalizer
        self.amp_style_reward_normalizer = amp_style_reward_normalizer

        # Discriminator parameters
        self.discriminator_gradient_penalty_coef = discriminator_gradient_penalty_coef
        self.discriminator_loss_function = discriminator_loss_function
        self.discriminator_num_mini_batches = discriminator_num_mini_batches

        # Pre-instantiate loss functions to avoid repeated creation
        if discriminator_loss_function == "BCEWithLogitsLoss":
            self.disc_loss_fn = nn.BCEWithLogitsLoss()
        elif discriminator_loss_function == "MSELoss":
            self.disc_loss_fn = nn.MSELoss()
        else:
            self.disc_loss_fn = None  # For WassersteinLoss, we use custom computation

        # Create optimizer
        params = [
            {"params": self.discriminator.architecture.parameters(), "weight_decay": 10e-4, "name": "amp_trunk"},
            {"params": self.discriminator.discriminator_logits.parameters(), "weight_decay": 10e-2, "name": "amp_head"},
        ]
        self.discriminator_optimizer = optim.Adam(params, lr=discriminator_learning_rate, betas=(0.0, 0.9))

    def init_storage(self, training_type, num_envs, num_transitions_per_env, obs, actions_shape):
        # create rollout storage
        self.storage = OldHimRolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            obs,
            actions_shape,
            self.device,
        )
        if self.foothold_imagination_enabled:
            critic_dim = self.policy.num_critic_obs
            action_dim = actions_shape[0]
            self.foothold_imagination = FootholdImagination(
                critic_dim,
                action_dim,
                hidden_dims=self.foothold_hidden_dims,
                min_sigma=self.foothold_min_sigma,
                max_sigma=self.foothold_max_sigma,
            ).to(self.device)
            self.foothold_optimizer = optim.Adam(
                self.foothold_imagination.parameters(), lr=self.foothold_learning_rate
            )
            self.foothold_replay = FootholdReplayBuffer(
                critic_dim, action_dim, self.foothold_replay_buffer_size, self.device
            )
            pending_shape = (num_envs, 2, self.foothold_pending_steps)
            self._foothold_pending_obs = torch.empty(
                *pending_shape, critic_dim, device=self.device, dtype=torch.float16
            )
            self._foothold_pending_actions = torch.empty(
                *pending_shape, action_dim, device=self.device, dtype=torch.float16
            )
            self._foothold_pending_base_xy = torch.empty(
                *pending_shape, 2, device=self.device, dtype=torch.float
            )
            self._foothold_pending_yaw = torch.empty(*pending_shape, device=self.device)
            self._foothold_pending_terrain = torch.empty(
                *pending_shape, device=self.device, dtype=torch.int8
            )
            self._foothold_pending_count = torch.zeros(num_envs, 2, device=self.device, dtype=torch.long)
            self._foothold_labeled_total = 0
            self._foothold_discarded_total = 0

    def predict_footholds(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Predict from the state/action cached by the latest call to :meth:`act`."""
        if self.foothold_imagination is None:
            return None
        critic_obs = self.policy.get_critic_obs(self.transition.observations)
        return self.foothold_imagination(critic_obs, self.transition.actions)

    def _process_foothold_events(self, extras, dones) -> None:
        if self.foothold_imagination is None or "foothold_events" not in extras:
            return
        events = extras["foothold_events"]
        swing = events["decision_swing"].to(self.device)
        base_xy = events["decision_base_xy"].to(self.device)
        base_yaw = events["decision_base_yaw"].to(self.device)
        stair_direction = events["stair_direction"].to(self.device)
        critic_obs = self.policy.get_critic_obs(self.transition.observations)
        actions = self.transition.actions

        # Append current decision samples to each active swing queue.
        for foot_id in range(2):
            env_ids = swing[:, foot_id].nonzero(as_tuple=False).squeeze(-1)
            if env_ids.numel() == 0:
                continue
            counts = self._foothold_pending_count[env_ids, foot_id]
            overflow = counts >= self.foothold_pending_steps
            if overflow.any():
                overflow_ids = env_ids[overflow]
                self._foothold_pending_count[overflow_ids, foot_id] = 0
                self._foothold_discarded_total += int(overflow_ids.numel())
                counts = self._foothold_pending_count[env_ids, foot_id]
            self._foothold_pending_obs[env_ids, foot_id, counts] = critic_obs[env_ids].to(torch.float16)
            self._foothold_pending_actions[env_ids, foot_id, counts] = actions[env_ids].to(torch.float16)
            self._foothold_pending_base_xy[env_ids, foot_id, counts] = base_xy[env_ids]
            self._foothold_pending_yaw[env_ids, foot_id, counts] = base_yaw[env_ids]
            self._foothold_pending_terrain[env_ids, foot_id, counts] = stair_direction[env_ids]
            self._foothold_pending_count[env_ids, foot_id] = counts + 1

        touchdown = events["touchdown_valid"].to(self.device)
        touchdown_xy = events["touchdown_xy_w"].to(self.device)
        step_ids = torch.arange(self.foothold_pending_steps, device=self.device)
        for foot_id in range(2):
            env_ids = touchdown[:, foot_id].nonzero(as_tuple=False).squeeze(-1)
            if env_ids.numel() == 0:
                continue
            counts = self._foothold_pending_count[env_ids, foot_id]
            valid_steps = step_ids.unsqueeze(0) < counts.unsqueeze(1)
            if not valid_steps.any():
                continue
            obs = self._foothold_pending_obs[env_ids, foot_id][valid_steps]
            queued_actions = self._foothold_pending_actions[env_ids, foot_id][valid_steps]
            queued_xy = self._foothold_pending_base_xy[env_ids, foot_id][valid_steps]
            queued_yaw = self._foothold_pending_yaw[env_ids, foot_id][valid_steps]
            queued_terrain = self._foothold_pending_terrain[env_ids, foot_id][valid_steps]
            repeated_targets = touchdown_xy[env_ids, foot_id].unsqueeze(1).expand(
                -1, self.foothold_pending_steps, -1
            )[valid_steps]
            targets_b = world_xy_to_base_yaw(repeated_targets, queued_xy, queued_yaw)
            foot_ids = torch.full((obs.shape[0],), foot_id, device=self.device, dtype=torch.long)
            self.foothold_replay.insert(obs, queued_actions, targets_b, foot_ids, queued_terrain)
            self._foothold_labeled_total += int(obs.shape[0])
            self._foothold_pending_count[env_ids, foot_id] = 0

        invalid_touchdown = events["touchdown"].to(self.device) & ~touchdown
        invalid_env_ids, invalid_foot_ids = invalid_touchdown.nonzero(as_tuple=True)
        if invalid_env_ids.numel() > 0:
            self._foothold_discarded_total += int(
                self._foothold_pending_count[invalid_env_ids, invalid_foot_ids].sum().item()
            )
            self._foothold_pending_count[invalid_env_ids, invalid_foot_ids] = 0

        done_ids = (dones > 0).nonzero(as_tuple=False).squeeze(-1)
        if done_ids.numel() > 0:
            self._foothold_discarded_total += int(self._foothold_pending_count[done_ids].sum().item())
            self._foothold_pending_count[done_ids] = 0

    def act(self, obs):
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()
        # compute the actions and values
        self.transition.actions = self.policy.act(obs).detach()
        self.transition.values = self.policy.evaluate(obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        # need to record obs before env.step()
        self.transition.observations = obs
        return self.transition.actions

    def process_env_step(self, obs, rewards, dones, extras, next_obs, amp_observation_buf):
        # update the normalizers
        self.policy.update_normalization(obs)
        if self.rnd:
            self.rnd.update_normalization(obs)

        # Record the next critic obs
        self.transition.next_observations = next_obs.clone()
        # Record the rewards and dones
        # Note: we clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Compute the intrinsic rewards and add to extrinsic rewards
        if self.rnd:
            # Compute the intrinsic rewards
            self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
            # Add intrinsic rewards to extrinsic rewards
            self.transition.rewards += self.intrinsic_rewards

        # Bootstrapping on time outs
        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device), 1
            )

        self._process_foothold_events(extras, dones)
        # record the transition
        self.storage.add_transitions(self.transition)
        self.amp_policy_data.insert(amp_observation_buf)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, obs):
        # compute value for the last step
        last_values = self.policy.evaluate(obs).detach()
        self.storage.compute_returns(
            last_values, self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch
        )

    def update(self):  # noqa: C901
        # -- PPO loss
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        # -- HIM loss
        mean_estimate_loss = 0
        mean_swap_loss = 0
        # -- AMP loss
        mean_discriminator_loss = 0
        mean_policy_pred = 0
        mean_expert_pred = 0
        # -- RND loss
        if self.rnd:
            mean_rnd_loss = 0
        else:
            mean_rnd_loss = None
        # -- Symmetry loss
        if self.symmetry:
            mean_symmetry_loss = 0
        else:
            mean_symmetry_loss = None

        # generator for mini batches
        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # generator for amp batches
        amp_policy_generator = self.amp_policy_data.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches,
        )
        amp_expert_generator = self.amp_expert_data.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches,
        )

        for sample, sample_amp_policy, sample_expert_data in zip(generator, amp_policy_generator, amp_expert_generator):

            (
                obs_batch,
                actions_batch,
                next_obs_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
                hid_states_batch,
                masks_batch,
            ) = sample

            # number of augmentations per sample
            # we start with 1 and increase it if we use symmetry augmentation
            num_aug = 1
            # original batch size
            # we assume policy group is always there and needs augmentation
            original_batch_size = obs_batch.batch_size[0]
            obs_batch["next_obs"] = next_obs_batch

            # check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # Perform symmetric augmentation
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                # augmentation using symmetry
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                # returned shape: [batch_size * num_aug, ...]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch,
                    actions=actions_batch,
                    env=self.symmetry["_env"],
                )
                # compute number of augmentations per sample
                # we assume policy group is always there and needs augmentation
                num_aug = int(obs_batch.batch_size[0] / original_batch_size)
                # repeat the rest of the batch
                # -- actor
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                # -- critic
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: we need to do this because we updated the policy with the new parameters
            # -- actor
            self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            # -- critic
            value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            # -- entropy
            # we only keep the entropy of the first augmentation (the original one)
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            # KL
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # Update the learning rate
                    # Perform this adaptation only on the main process
                    # TODO: Is this needed? If KL-divergence is the "same" across all GPUs,
                    #       then the learning rate should be the same across all GPUs.
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # Update the learning rate for all GPUs
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    # Update the learning rate for all parameter groups
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # Discriminator update (before PPO update to avoid gradient conflicts)
            sample_amp_expert, _ = sample_expert_data[0], sample_expert_data[1]
            # Discriminator loss
            if self.amp_state_normalizer is not None:
                with torch.no_grad():
                    # Reshape to [batch_size * horizon, obs_dim] for batch normalization
                    policy_flat = sample_amp_policy.view(-1, sample_amp_policy.shape[-1])
                    expert_flat = sample_amp_expert.view(-1, sample_amp_expert.shape[-1])

                    # Batch normalize all frames at once
                    policy_flat_norm = self.amp_state_normalizer.normalize_torch(policy_flat, self.device)
                    expert_flat_norm = self.amp_state_normalizer.normalize_torch(expert_flat, self.device)

                    # Flatten to final shape
                    policy_data = policy_flat_norm.view(sample_amp_policy.shape[0], -1)
                    expert_data = expert_flat_norm.view(sample_amp_expert.shape[0], -1)
            else:
                policy_data = sample_amp_policy.flatten(1, 2)
                expert_data = sample_amp_expert.flatten(1, 2)

            policy_d = self.discriminator(policy_data)
            expert_d = self.discriminator(expert_data)

            if self.discriminator_loss_function == "BCEWithLogitsLoss":
                expert_loss = self.disc_loss_fn(expert_d, torch.ones_like(expert_d))
                policy_loss = self.disc_loss_fn(policy_d, torch.zeros_like(policy_d))
                grad_pen_loss = self.discriminator.compute_grad_pen(
                    expert_data, lambda_=self.discriminator_gradient_penalty_coef
                )
            elif self.discriminator_loss_function == "MSELoss":
                expert_loss = self.disc_loss_fn(expert_d, torch.ones_like(expert_d))
                policy_loss = self.disc_loss_fn(policy_d, -torch.ones_like(policy_d))
                grad_pen_loss = self.discriminator.compute_grad_pen(
                    expert_data, lambda_=self.discriminator_gradient_penalty_coef
                )
            elif self.discriminator_loss_function == "WassersteinLoss":
                expert_loss = -torch.tanh(0.3 * expert_d).mean()
                policy_loss = torch.tanh(0.3 * policy_d).mean()
                grad_pen_loss = self.discriminator.compute_wgan_grad_pen(expert_data, policy_data)
            else:
                raise ValueError("Unexpected loss function specified")

            amp_loss = 0.5 * (expert_loss + policy_loss)

            discriminator_loss = amp_loss + grad_pen_loss

            self.discriminator_optimizer.zero_grad()
            discriminator_loss.backward()
            self.discriminator_optimizer.step()

            if self.amp_state_normalizer is not None:
                self.amp_state_normalizer.update(sample_amp_policy[:, 0].cpu().numpy())
                self.amp_state_normalizer.update(sample_amp_expert[:, 0].cpu().numpy())

            # Symmetry loss
            if self.symmetry:
                # obtain the symmetric actions
                # if we did augmentation before then we don't need to augment again
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(obs=obs_batch, actions=None, env=self.symmetry["_env"])
                    # compute number of augmentations per sample
                    num_aug = int(obs_batch.shape[0] / original_batch_size)

                # actions predicted by the actor for symmetrically-augmented observations
                mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())

                # compute the symmetrically augmented actions
                # note: we are assuming the first augmentation is the original one.
                #   We do not use the action_batch from earlier since that action was sampled from the distribution.
                #   However, the symmetry loss is computed using the mean of the distribution.
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                )

                # compute the loss (we skip the first augmentation as it is the original one)
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                )
                # add the loss to the total loss
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # Random Network Distillation loss
            # TODO: Move this processing to inside RND module.
            if self.rnd:
                # extract the rnd_state
                # TODO: Check if we still need torch no grad. It is just an affine transformation.
                with torch.no_grad():
                    rnd_state_batch = self.rnd.get_rnd_state(obs_batch[:original_batch_size])
                    rnd_state_batch = self.rnd.state_normalizer(rnd_state_batch)
                # predict the embedding and the target
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                # compute the loss as the mean squared error
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            # Compute the gradients
            # -- For PPO
            self.optimizer.zero_grad()
            loss.backward()
            # -- For RND
            if self.rnd:
                self.rnd_optimizer.zero_grad()  # type: ignore
                rnd_loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients
            # -- For PPO (now excludes HIM estimator)
            nn.utils.clip_grad_norm_(
                [p for n, p in self.policy.named_parameters() if not n.startswith("him_estimator.")], self.max_grad_norm
            )
            self.optimizer.step()
            # -- For RND
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            # HIM Estimator Update (after PPO update, uses its own optimizer)
            actor_obs_batch = self.policy.get_actor_obs(obs_batch)
            actor_obs_batch = self.policy.actor_obs_normalizer(actor_obs_batch)
            # Use unnormalized critic obs for state extraction to avoid normalization mismatch
            critic_obs_batch_raw = self.policy.get_critic_obs(obs_batch)
            next_critic_obs_batch = self.policy.critic_obs_normalizer(obs_batch["next_obs"])
            estimation_loss, swap_loss = self.policy.him_estimator.update(
                actor_obs_batch, critic_obs_batch_raw, next_critic_obs_batch, lr=self.learning_rate
            )

            # Store the losses
            # -- PPO loss
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            # -- HIM loss
            mean_estimate_loss += estimation_loss
            mean_swap_loss += swap_loss
            # -- AMP loss
            mean_discriminator_loss += discriminator_loss.item()
            mean_policy_pred += policy_d.mean().item()
            mean_expert_pred += expert_d.mean().item()
            # -- RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # -- Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        # -- For PPO
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        # -- For RND
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        # -- For Symmetry
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        # -- For HIM
        mean_estimate_loss /= num_updates
        mean_swap_loss /= num_updates
        # -- For AMP
        mean_discriminator_loss /= num_updates
        mean_policy_pred /= num_updates
        mean_expert_pred /= num_updates

        # Delayed touchdown supervision is sampled independently from PPO rollouts.
        mean_foothold_nll = 0.0
        mean_foothold_rmse = 0.0
        mean_foothold_up_rmse = 0.0
        mean_foothold_down_rmse = 0.0
        foothold_up_updates = 0
        foothold_down_updates = 0
        foothold_updates = 0
        if (
            self.foothold_imagination is not None
            and self.foothold_replay is not None
            and self.foothold_replay.num_samples > 0
        ):
            batch_size = min(self.foothold_batch_size, self.foothold_replay.num_samples)
            for _ in range(self.foothold_updates_per_iteration):
                privileged_obs, actions, targets, foot_ids, terrain_ids = self.foothold_replay.sample(
                    batch_size
                )
                mu, sigma = self.foothold_imagination(privileged_obs, actions)
                batch_indices = torch.arange(batch_size, device=self.device)
                selected_mu = mu[batch_indices, foot_ids]
                selected_sigma = sigma[batch_indices, foot_ids]
                foothold_nll = FootholdImagination.gaussian_nll(
                    selected_mu, selected_sigma, targets
                )
                self.foothold_optimizer.zero_grad()
                foothold_nll.backward()
                if self.is_multi_gpu:
                    for parameter in self.foothold_imagination.parameters():
                        if parameter.grad is not None:
                            torch.distributed.all_reduce(
                                parameter.grad, op=torch.distributed.ReduceOp.SUM
                            )
                            parameter.grad /= self.gpu_world_size
                nn.utils.clip_grad_norm_(self.foothold_imagination.parameters(), self.max_grad_norm)
                self.foothold_optimizer.step()
                mean_foothold_nll += foothold_nll.item()
                squared_distance = torch.sum(torch.square(selected_mu.detach() - targets), dim=-1)
                mean_foothold_rmse += torch.sqrt(torch.mean(squared_distance)).item()
                up_mask = terrain_ids == 0
                if up_mask.any():
                    mean_foothold_up_rmse += torch.sqrt(torch.mean(squared_distance[up_mask])).item()
                    foothold_up_updates += 1
                down_mask = terrain_ids == 1
                if down_mask.any():
                    mean_foothold_down_rmse += torch.sqrt(torch.mean(squared_distance[down_mask])).item()
                    foothold_down_updates += 1
                foothold_updates += 1
        if foothold_updates:
            mean_foothold_nll /= foothold_updates
            mean_foothold_rmse /= foothold_updates
        if foothold_up_updates:
            mean_foothold_up_rmse /= foothold_up_updates
        if foothold_down_updates:
            mean_foothold_down_rmse /= foothold_down_updates

        # -- Clear the storage
        self.storage.clear()

        # construct the loss dictionary
        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "estimate": mean_estimate_loss,
            "swap": mean_swap_loss,
            "amp_loss": mean_discriminator_loss,
            "expert_pred": mean_expert_pred,
            "policy_pred": mean_policy_pred,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss
        if self.foothold_imagination is not None:
            loss_dict["foothold_nll"] = mean_foothold_nll
            loss_dict["foothold_rmse"] = mean_foothold_rmse
            loss_dict["foothold_upstairs_rmse"] = mean_foothold_up_rmse
            loss_dict["foothold_downstairs_rmse"] = mean_foothold_down_rmse
            loss_dict["foothold_labeled_samples"] = float(self.foothold_replay.num_samples)
            loss_dict["foothold_pending_samples"] = float(self._foothold_pending_count.sum().item())
            loss_dict["foothold_discarded_total"] = float(self._foothold_discarded_total)

        return loss_dict

    """
    Helper functions
    """

    def broadcast_parameters(self):
        """Broadcast model parameters to all GPUs."""
        # obtain the model parameters on current GPU
        model_params = [self.policy.state_dict(), self.discriminator.state_dict()]
        if self.foothold_imagination is not None:
            model_params.append(self.foothold_imagination.state_dict())
        if self.rnd:
            model_params.append(self.rnd.predictor.state_dict())
        # broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # load the model parameters on all GPUs from source GPU
        self.policy.load_state_dict(model_params[0])
        self.discriminator.load_state_dict(model_params[1])
        next_model_idx = 2
        if self.foothold_imagination is not None:
            self.foothold_imagination.load_state_dict(model_params[next_model_idx])
            next_model_idx += 1
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_params[next_model_idx])

    def reduce_parameters(self):
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        grads += [param.grad.view(-1) for param in self.discriminator.parameters() if param.grad is not None]
        if self.rnd:
            grads += [param.grad.view(-1) for param in self.rnd.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)

        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        # Get all parameters
        all_params = self.policy.parameters()
        all_params = chain(all_params, self.discriminator.parameters())
        if self.rnd:
            all_params = chain(all_params, self.rnd.parameters())

        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                # copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # update the offset for the next parameter
                offset += numel
