# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

"""Functions to specify the symmetry in the observation and action space for E1."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from NoetixE1.tasks.walkrun.noetix_e1.env import ManagerBasedRLAmpEnv


@torch.no_grad()
def data_augmentation_func_e1(
    env: ManagerBasedRLEnv | ManagerBasedRLAmpEnv, obs: TensorDict | None, actions: torch.Tensor | None
):
    """The symmetry data augmentation function.

    The function signature should be as follows:

    Args:

        env (VecEnv): The environment object. This is used to access the environment's properties.
        obs (tensordict.TensorDict | None): The observation tensor. If None, the observation is not used.
        action (torch.Tensor | None): The action tensor. If None, the action is not used.

    Returns:
        A tuple containing the augmented observation and action tensors. The tensors can be None,
        if their respective inputs are None.
    """
    if obs is None:
        obs_aug = None
    else:
        batch_size = obs.batch_size[0]
        obs_aug = obs.repeat(2)

        # ------------------------------------ Policy ------------------------------------
        # check if we have frame stacking
        if len(env.observation_space["policy"].shape) == 3:
            num_frames = env.observation_space["policy"].shape[1]
            single_actor_obs_dim = env.observation_space["policy"].shape[2]
        else:
            num_frames = 1
            single_actor_obs_dim = env.observation_space["policy"].shape[-1]
        # -- original
        obs_aug["policy"][:batch_size] = obs["policy"][:]
        policy_obs = obs["policy"].clone()
        # -- left-right
        # If it's a single frame, use original logic
        if num_frames == 1:
            obs_aug["policy"][batch_size : 2 * batch_size] = flip_e1_obs(policy_obs, env)
        else:
            # Split into frames and process each frame
            frames = torch.split(policy_obs, single_actor_obs_dim, dim=-1)
            flipped_frames = torch.cat([flip_e1_obs(frame, env) for frame in frames], dim=-1)
            obs_aug["policy"][batch_size : 2 * batch_size] = flipped_frames

        # ------------------------------------ Critic ------------------------------------
        # check if we have frame stacking
        if len(env.observation_space["critic"].shape) == 3:
            num_frames = env.observation_space["critic"].shape[1]
            single_critic_obs_dim = env.observation_space["critic"].shape[2]
        else:
            num_frames = 1
            single_critic_obs_dim = env.observation_space["critic"].shape[-1]
        # -- original
        obs_aug["critic"][:batch_size] = obs["critic"][:]
        critic_obs = obs["critic"].clone()
        # -- left-right
        # If it's a single frame, use original logic
        if num_frames == 1:
            obs_aug["critic"][batch_size : 2 * batch_size] = flip_e1_obs(critic_obs, env)
        else:
            # Split into frames and process each frame
            frames = torch.split(critic_obs, single_critic_obs_dim, dim=-1)
            flipped_frames = [flip_e1_obs(frame, env) for frame in frames]
            obs_aug["critic"][batch_size : 2 * batch_size] = torch.cat(flipped_frames, dim=-1)

    if actions is None:
        actions_aug = None
    else:
        batch_size = actions.shape[0]
        actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device)
        # -- original
        actions_aug[:batch_size] = actions[:]
        # -- left-right
        actions_aug[batch_size : 2 * batch_size] = flip_e1_actions(actions)
    return obs_aug, actions_aug

def flip_e1_obs(obs: torch.Tensor, env: ManagerBasedRLEnv | ManagerBasedRLAmpEnv):
    """
    commands: lin_x lin_y ang_yaw       3
    base_ang_vel: roll pitch yaw        3
    euler angle: roll pitch             2
    dof_pos                             num_dofs
    dof_vel                             num_dofs
    actions                             num_dofs
    base_lin_vel                        3
    contacts                            2
    stumble                             2
    left_foot_height_scan               len(foot_points_x) * len(foot_points_y)
    right_foot_height_scan              len(foot_points_x) * len(foot_points_y
    (option)measure_heights             len(measured_points_x) * len(measured_points_y)
    """
    if obs is None:
        return obs

    flipped_obs = torch.zeros_like(obs)
    # commands: lin_x lin_y ang_yaw       3
    flipped_obs[..., :3] = obs[..., 0:3]
    flipped_obs[..., 1] = -flipped_obs[..., 1]  # cmd lin_vel_y
    flipped_obs[..., 2] = -flipped_obs[..., 2]  # cmd ang_vel_z

    # base_ang_vel: roll pitch yaw        3
    flipped_obs[..., 3:6] = obs[..., 3:6]  # ang_vel
    flipped_obs[..., 3] = -flipped_obs[..., 3]  # ang_vel_x
    flipped_obs[..., 5] = -flipped_obs[..., 5]  # ang_vel_z
    # euler angles: roll pitch             2
    flipped_obs[..., 6:8] = obs[..., 6:8] * torch.tensor(
        [-1.0, 1.0], dtype=obs.dtype, device=obs.device
    )
    # dof_pos                             num_dofs
    flipped_obs[..., 8 : 8 + env.num_actions] = flip_e1_dof(
        obs[..., 8 : 8 + env.num_actions]
    )
    # dof_vel                             num_dofs
    flipped_obs[..., 8 + env.num_actions : 8 + 2 * env.num_actions] = flip_e1_dof(
        obs[..., 8 + env.num_actions : 8 + 2 * env.num_actions]
    )
    # actions                             num_dofs
    flipped_obs[..., 8 + 2 * env.num_actions : 8 + 3 * env.num_actions] = flip_e1_dof(
        obs[..., 8 + 2 * env.num_actions : 8 + 3 * env.num_actions]
    )

    # base_lin_vel                        3
    flipped_obs[..., 8 + 3 * env.num_actions : 11 + 3 * env.num_actions] = obs[..., 8 + 3 * env.num_actions : 11 + 3 * env.num_actions]
    flipped_obs[..., 9 + 3 * env.num_actions] = -flipped_obs[..., 9 + 3 * env.num_actions]  # base_lin_vel_y

    # contact mask 2
    flipped_obs[..., 11 + 3 * env.num_actions] = obs[..., 12 + 3 * env.num_actions]
    flipped_obs[..., 12 + 3 * env.num_actions] = obs[..., 11 + 3 * env.num_actions]

    # feet stumble 2
    flipped_obs[..., 13 + 3 * env.num_actions] = obs[..., 14 + 3 * env.num_actions]
    flipped_obs[..., 14 + 3 * env.num_actions] = obs[..., 13 + 3 * env.num_actions]

    # left_foot_height_scan and right_foot_height_scan
    resolution = 1.0 / env.cfg.scene.left_foot_height_scanner.pattern_cfg.resolution
    num_x, num_y = env.cfg.scene.left_foot_height_scanner.pattern_cfg.size
    foot_num_x_size, foot_num_y_size = int((num_x * resolution) + 1), int((num_y * resolution) + 1)
    base_idx = torch.arange(foot_num_x_size * foot_num_y_size, dtype=torch.int, device=flipped_obs.device).view(
        int(foot_num_x_size), int(foot_num_y_size)
    )
    flip_indices = base_idx.flip(dims=[1]).reshape(-1)
    flipped_obs[..., 15 + 3 * env.num_actions : 15 + 3 * env.num_actions + foot_num_x_size * foot_num_y_size] = obs[..., 15 + 3 * env.num_actions + foot_num_x_size * foot_num_y_size : 15 + 3 * env.num_actions + 2 * foot_num_x_size * foot_num_y_size][..., flip_indices]
    flipped_obs[..., 15 + 3 * env.num_actions + foot_num_x_size * foot_num_y_size : 15 + 3 * env.num_actions + 2 * foot_num_x_size * foot_num_y_size] = obs[..., 15 + 3 * env.num_actions : 15 + 3 * env.num_actions + foot_num_x_size * foot_num_y_size][..., flip_indices]

    # measured_heights
    if env.cfg.scene.height_scanner:
        resolution = 1.0 / env.cfg.scene.height_scanner.pattern_cfg.resolution
        num_x, num_y = env.cfg.scene.height_scanner.pattern_cfg.size
        num_x_size, num_y_size = int((num_x * resolution) + 1), int((num_y * resolution) + 1)
        base_idx = torch.arange(num_x_size * num_y_size, dtype=torch.int, device=flipped_obs.device).view(
            int(num_x_size), int(num_y_size)
        )
        flip_indices = base_idx.flip(dims=[1]).reshape(-1)
        flipped_obs[..., 15 + 3 * env.num_actions + 2 * foot_num_x_size * foot_num_y_size :] = obs[..., 15 + 3 * env.num_actions + 2 * foot_num_x_size * foot_num_y_size :][..., flip_indices]
    return flipped_obs


def flip_e1_actions(actions: torch.Tensor | None):
    if actions is None:
        return None
    fliped_actions = flip_e1_dof(actions)
    return fliped_actions


def flip_e1_dof(dof: torch.Tensor):
    flipped_dof = torch.zeros_like(dof)

    flipped_dof[..., 0] = -dof[..., 1]  # hip yaw
    flipped_dof[..., 1] = -dof[..., 0]  # hip yaw

    flipped_dof[..., 2] = -dof[..., 2]  # waist yaw

    flipped_dof[..., 3] = -dof[..., 4]  # hip roll
    flipped_dof[..., 4] = -dof[..., 3]  # hip roll

    flipped_dof[..., 5] = dof[..., 6]  # shoulder pitch
    flipped_dof[..., 6] = dof[..., 5]  # shoulder pitch

    flipped_dof[..., 7] = dof[..., 8]  # hip pitch
    flipped_dof[..., 8] = dof[..., 7]  # hip pitch

    flipped_dof[..., 9] = -dof[..., 10]  # shoulder roll
    flipped_dof[..., 10] = -dof[..., 9]  # shoulder roll

    flipped_dof[..., 11] = dof[..., 12]  # knee
    flipped_dof[..., 12] = dof[..., 11]  # knee

    flipped_dof[..., 13] = -dof[..., 14]  # shoulder yaw
    flipped_dof[..., 14] = -dof[..., 13]  # shoulder yaw

    flipped_dof[..., 15] = dof[..., 16]  # ankle pitch
    flipped_dof[..., 16] = dof[..., 15]  # ankle pitch

    flipped_dof[..., 17] = dof[..., 18]  # elbow pitch
    flipped_dof[..., 18] = dof[..., 17]  # elbow pitch

    flipped_dof[..., 19] = -dof[..., 20]  # ankle roll
    flipped_dof[..., 20] = -dof[..., 19]  # ankle roll

    flipped_dof[..., 21] = -dof[..., 22]  # elbow yaw
    flipped_dof[..., 22] = -dof[..., 21]  # elbow yaw

    return flipped_dof
