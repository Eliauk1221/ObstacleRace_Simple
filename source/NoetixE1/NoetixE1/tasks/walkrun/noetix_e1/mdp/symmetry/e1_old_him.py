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
            obs_aug["policy"][batch_size : 2 * batch_size] = flip_e1_actor_obs(policy_obs, env)
        else:
            # Split into frames and process each frame
            frames = torch.split(policy_obs, single_actor_obs_dim, dim=-1)
            flipped_frames = torch.cat([flip_e1_actor_obs(frame, env) for frame in frames], dim=-1)
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
            obs_aug["critic"][batch_size : 2 * batch_size] = flip_e1_critic_obs(critic_obs, single_actor_obs_dim, env)
        else:
            # Split into frames and process each frame
            frames = torch.split(critic_obs, single_critic_obs_dim, dim=-1)
            flipped_frames = [flip_e1_critic_obs(frame, single_actor_obs_dim, env) for frame in frames]
            obs_aug["critic"][batch_size : 2 * batch_size] = torch.cat(flipped_frames, dim=-1)

        # ------------------------------------ Next Obs ------------------------------------
        # -- original
        obs_aug["next_obs"][:batch_size] = obs["next_obs"][:]
        next_obs = obs["next_obs"].clone()
        # -- left-right
        # If it's a single frame, use original logic
        if num_frames == 1:
            obs_aug["next_obs"][batch_size : 2 * batch_size] = flip_e1_critic_obs(next_obs, single_actor_obs_dim, env)
        else:
            # Split into frames and process each frame
            frames = torch.split(next_obs, single_actor_obs_dim, dim=-1)
            flipped_frames = [flip_e1_critic_obs(frame, single_actor_obs_dim, env) for frame in frames]
            obs_aug["next_obs"][batch_size : 2 * batch_size] = torch.cat(flipped_frames, dim=-1)

        # ------------------------------------ Depth ------------------------------------
        # Handle depth image symmetry if present
        if "depth" in obs.keys():
            # -- original
            obs_aug["depth"][:batch_size] = obs["depth"][:]
            # -- left-right: flip depth image along width dimension (last dimension)
            # Depth shape from raycaster_depth: (batch_size, height, width) = [B, H, W]
            depth = obs["depth"].clone()
            # Flip along the width dimension (last spatial dimension)
            # For [B, H, W]: flip dim=-1
            # For [B, 1, H, W] or [B, C, H, W]: flip dim=-1
            # For [B, H, W, 1]: flip dim=-2
            if depth.ndim == 3:  # [B, H, W]
                obs_aug["depth"][batch_size : 2 * batch_size] = torch.flip(depth, dims=[-1])
            elif depth.ndim == 4 and depth.shape[-1] == 1:  # [B, H, W, 1]
                obs_aug["depth"][batch_size : 2 * batch_size] = torch.flip(depth, dims=[-2])
            else:  # [B, C, H, W] or [B, 1, H, W]
                obs_aug["depth"][batch_size : 2 * batch_size] = torch.flip(depth, dims=[-1])

        # ------------------------------------ Privileged Depth ------------------------------------
        if "privileged_depth" in obs.keys():
            # -- original
            obs_aug["privileged_depth"][:batch_size] = obs["privileged_depth"][:]
            # -- left-right: flip depth image along width dimension (last dimension)
            # Depth shape from raycaster_depth: (batch_size, height, width) = [B, H, W]
            depth = obs["privileged_depth"].clone()
            # Flip along the width dimension (last spatial dimension)
            # For [B, H, W]: flip dim=-1
            # For [B, 1, H, W] or [B, C, H, W]: flip dim=-1
            # For [B, H, W, 1]: flip dim=-2
            if depth.ndim == 3:  # [B, H, W]
                obs_aug["privileged_depth"][batch_size : 2 * batch_size] = torch.flip(depth, dims=[-1])
            elif depth.ndim == 4 and depth.shape[-1] == 1:  # [B, H, W, 1]
                obs_aug["privileged_depth"][batch_size : 2 * batch_size] = torch.flip(depth, dims=[-2])
            else:  # [B, C, H, W] or [B, 1, H, W]
                obs_aug["privileged_depth"][batch_size : 2 * batch_size] = torch.flip(depth, dims=[-1])

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


def flip_e1_actor_obs(one_step_obs: torch.Tensor, env: ManagerBasedRLEnv | ManagerBasedRLAmpEnv):
    """
    commands: lin_x lin_y ang_yaw       3
    base_ang_vel: roll pitch yaw        3
    projected_gravity                   3
    dof_pos                             num_dofs
    dof_vel                             num_dofs
    actions                             num_dofs
    """
    if one_step_obs is None:
        return one_step_obs

    flipped_one_step_obs = torch.zeros_like(one_step_obs)
    flipped_one_step_obs[..., :3] = one_step_obs[..., 0:3]
    flipped_one_step_obs[..., 1] = -flipped_one_step_obs[..., 1]  # cmd lin_vel_y
    flipped_one_step_obs[..., 2] = -flipped_one_step_obs[..., 2]  # cmd ang_vel_z

    flipped_one_step_obs[..., 3:6] = one_step_obs[..., 3:6]  # ang_vel
    flipped_one_step_obs[..., 3] = -flipped_one_step_obs[..., 3]  # ang_vel_x
    flipped_one_step_obs[..., 5] = -flipped_one_step_obs[..., 5]  # ang_vel_z
    # euler angles
    flipped_one_step_obs[..., 6:8] = one_step_obs[..., 6:8] * torch.tensor(
        [-1.0, 1.0], dtype=one_step_obs.dtype, device=one_step_obs.device
    )
    flipped_one_step_obs[..., 8 : 8 + env.num_actions] = flip_e1_dof(
        one_step_obs[..., 8 : 8 + env.num_actions]
    )  # dof_pos
    flipped_one_step_obs[..., 8 + env.num_actions : 8 + 2 * env.num_actions] = flip_e1_dof(
        one_step_obs[..., 8 + env.num_actions : 8 + 2 * env.num_actions]
    )  # dof_vel
    flipped_one_step_obs[..., 8 + 2 * env.num_actions : 8 + 3 * env.num_actions] = flip_e1_dof(
        one_step_obs[..., 8 + 2 * env.num_actions : 8 + 3 * env.num_actions]
    )  # last_actions

    return flipped_one_step_obs


def flip_e1_critic_obs(obs: torch.Tensor, single_actor_obs_dim: int, env: ManagerBasedRLEnv | ManagerBasedRLAmpEnv):
    """
    actor_obs                           num_single_obs
    base_lin_vel                        3
    contacts                            2
    (option)measure_heights             len(measured_points_x) * len(measured_points_y)
    """
    if obs is None:
        return obs

    actor_obs = obs[..., :single_actor_obs_dim]
    num_actor_obs = single_actor_obs_dim

    flipped_critic_obs = torch.zeros_like(obs)
    flipped_critic_obs[..., :num_actor_obs] = flip_e1_actor_obs(actor_obs, env)

    flipped_critic_obs[..., num_actor_obs : num_actor_obs + 3] = obs[..., num_actor_obs : num_actor_obs + 3]
    # base_lin_vel
    flipped_critic_obs[..., num_actor_obs + 1] = -flipped_critic_obs[..., num_actor_obs + 1]  # base_lin_vel_y

    # payload
    flipped_critic_obs[..., num_actor_obs + 3] = obs[..., num_actor_obs + 3]  # payload

    # left_friction
    flipped_critic_obs[..., num_actor_obs + 4: num_actor_obs + 7] = obs[..., num_actor_obs + 7 : num_actor_obs + 10]  # left_friction

    # right_friction
    flipped_critic_obs[..., num_actor_obs + 7 : num_actor_obs + 10] = obs[..., num_actor_obs + 4 : num_actor_obs + 7]  # right_friction

    # stiffness
    flipped_critic_obs[..., num_actor_obs + 10 : num_actor_obs + 33] = flip_e1_dof(obs[..., num_actor_obs + 10 : num_actor_obs + 33])  # stiffness

    # damping
    flipped_critic_obs[..., num_actor_obs + 33 : num_actor_obs + 56] = flip_e1_dof(obs[..., num_actor_obs + 33 : num_actor_obs + 56])  # damping

    # effort_limit
    flipped_critic_obs[..., num_actor_obs + 56 : num_actor_obs + 79] = obs[..., num_actor_obs + 56 : num_actor_obs + 79]  # effort_limit

    # camera_intrinsic_delta: [f_x_delta, f_y_delta, c_x_delta, c_y_delta]
    # 左右翻转时：f_x, f_y 不变，c_x 需要取反（镜像），c_y 不变
    flipped_critic_obs[..., num_actor_obs + 79 : num_actor_obs + 81] = obs[..., num_actor_obs + 79 : num_actor_obs + 81]  # f_x_delta, f_y_delta
    flipped_critic_obs[..., num_actor_obs + 81] = -obs[..., num_actor_obs + 81]  # c_x_delta (取反)
    flipped_critic_obs[..., num_actor_obs + 82] = obs[..., num_actor_obs + 82]  # c_y_delta

    # camera_extrinsic_delta: [x_delta, y_delta, z_delta, roll_delta, pitch_delta, yaw_delta]
    # 左右翻转时：x, z 不变，y 取反；roll, pitch 不变，yaw 取反
    flipped_critic_obs[..., num_actor_obs + 83] = obs[..., num_actor_obs + 83]  # x_delta
    flipped_critic_obs[..., num_actor_obs + 84] = -obs[..., num_actor_obs + 84]  # y_delta (取反)
    flipped_critic_obs[..., num_actor_obs + 85] = obs[..., num_actor_obs + 85]  # z_delta
    flipped_critic_obs[..., num_actor_obs + 86] = obs[..., num_actor_obs + 86]  # roll_delta
    flipped_critic_obs[..., num_actor_obs + 87] = obs[..., num_actor_obs + 87]  # pitch_delta
    flipped_critic_obs[..., num_actor_obs + 88] = -obs[..., num_actor_obs + 88]  # yaw_delta (取反)

    # contact mask
    flipped_critic_obs[..., num_actor_obs + 89 : num_actor_obs + 90] = obs[..., num_actor_obs + 90 : num_actor_obs + 91]
    flipped_critic_obs[..., num_actor_obs + 90 : num_actor_obs + 91] = obs[..., num_actor_obs + 89 : num_actor_obs + 90]

    # feet stumble
    flipped_critic_obs[..., num_actor_obs + 91 : num_actor_obs + 92] = obs[..., num_actor_obs + 92 : num_actor_obs + 93]
    flipped_critic_obs[..., num_actor_obs + 92 : num_actor_obs + 93] = obs[..., num_actor_obs + 91 : num_actor_obs + 92]

    # left_foot_height_scan and right_foot_height_scan
    resolution = 1.0 / env.cfg.scene.left_foot_height_scanner.pattern_cfg.resolution
    num_x, num_y = env.cfg.scene.left_foot_height_scanner.pattern_cfg.size
    foot_num_x_size, foot_num_y_size = int((num_x * resolution) + 1), int((num_y * resolution) + 1)
    base_idx = torch.arange(foot_num_x_size * foot_num_y_size, dtype=torch.int, device=flipped_critic_obs.device).view(
        int(foot_num_x_size), int(foot_num_y_size)
    )
    flip_indices = base_idx.flip(dims=[1]).reshape(-1)
    flipped_critic_obs[..., num_actor_obs + 93 : num_actor_obs + 93 + foot_num_x_size * foot_num_y_size] = obs[..., num_actor_obs + 93 + foot_num_x_size * foot_num_y_size : num_actor_obs + 93 + 2 * foot_num_x_size * foot_num_y_size][..., flip_indices]
    flipped_critic_obs[..., num_actor_obs + 93 + foot_num_x_size * foot_num_y_size : num_actor_obs + 93 + 2 * foot_num_x_size * foot_num_y_size] = obs[..., num_actor_obs + 93 : num_actor_obs + 93 + foot_num_x_size * foot_num_y_size][..., flip_indices]

    # measured_heights
    if env.cfg.scene.height_scanner:
        resolution = 1.0 / env.cfg.scene.height_scanner.pattern_cfg.resolution
        num_x, num_y = env.cfg.scene.height_scanner.pattern_cfg.size
        num_x_size, num_y_size = int((num_x * resolution) + 1), int((num_y * resolution) + 1)
        base_idx = torch.arange(num_x_size * num_y_size, dtype=torch.int, device=flipped_critic_obs.device).view(
            int(num_x_size), int(num_y_size)
        )
        flip_indices = base_idx.flip(dims=[1]).reshape(-1)
        flipped_critic_obs[..., num_actor_obs + 93 + 2 * foot_num_x_size * foot_num_y_size :] = obs[..., num_actor_obs + 93 + 2 * foot_num_x_size * foot_num_y_size :][..., flip_indices]
    return flipped_critic_obs


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
