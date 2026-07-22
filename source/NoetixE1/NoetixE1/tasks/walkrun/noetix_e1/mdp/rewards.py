# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from NoetixE1.sensors.volume_points import VolumePoints

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = math_utils.wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)


def track_lin_vel_xy_yaw_frame_exp(
    env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    vel_yaw = math_utils.quat_apply_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    command = env.command_manager.get_command("base_velocity")
    lin_vel_error = torch.sum(torch.square(command[:, :2] - vel_yaw[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / std**2)


def track_lin_vel_xy_yaw_frame_exp_terrain_full(
    env: ManagerBasedRLEnv,
    std: float,
    terrain_types: tuple[str, ...],
    height_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """与 track_lin_vel_xy_yaw_frame_exp 相同，但在指定地形上当 base 高度 ≥ height_threshold 时直接给满奖励 1.0。

    base 高度为 root_pos_w.z 相对 env_origins.z 的差值（即相对地形的高度）。
    params 示例: {"std": 0.5, "terrain_types": ("frame_platform_terrain",), "height_threshold": 0.3}
    """
    asset: Articulation = env.scene[asset_cfg.name]
    vel_yaw = math_utils.quat_apply_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    command = env.command_manager.get_command("base_velocity")
    lin_vel_error = torch.sum(torch.square(command[:, :2] - vel_yaw[:, :2]), dim=1)
    vel_reward = torch.exp(-lin_vel_error / std**2)

    if not terrain_types:
        return vel_reward

    from NoetixE1.tasks.walkrun.noetix_e1.mdp.curriculums import _check_terrain_types_from_positions

    col_indices = getattr(env.unwrapped, "_current_terrain_col_indices", None)
    if col_indices is None:
        robot_positions_w = asset.data.root_pos_w
        is_on_terrain = _check_terrain_types_from_positions(env, robot_positions_w, terrain_types)
    else:
        is_on_terrain = _check_terrain_types_from_positions(env, None, terrain_types, col_indices=col_indices)

    base_height = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    full_reward_mask = is_on_terrain & (base_height >= height_threshold)
    return torch.where(full_reward_mask, torch.ones_like(vel_reward, device=vel_reward.device), vel_reward)


def track_ang_vel_z_world_exp(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    skip_terrain_types: tuple[str, ...] | None = None,
) -> torch.Tensor:
    """Track yaw rate command. On ``skip_terrain_types`` terrains return 1.0 so the policy may turn freely."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command("base_velocity")
    ang_vel_error = torch.square(command[:, 2] - asset.data.root_ang_vel_w[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    if skip_terrain_types:
        on_terrain = _terrain_type_mask(env, skip_terrain_types, reward.device)
        reward = torch.where(on_terrain, torch.ones_like(reward), reward)
    return reward


def flat_orientation_exp(
    env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward non-flat base orientation using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.exp(-torch.norm(asset.data.projected_gravity_b[:, :2], dim=1) / std**2)


def feet_contact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    net_contact_forces = contact_sensor.data.net_forces_w_history
    feet_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids, :], dim=-1), dim=1)[0] > 0.5

    both_feet_contact = torch.all(feet_contact, dim=1)
    env.unwrapped.feet_both_contact_time[both_feet_contact] += env.step_dt
    env.unwrapped.feet_both_contact_time[~both_feet_contact] = 0

    single_feet_contact = torch.sum(feet_contact, dim=1) == 1
    command = env.command_manager.get_command("base_velocity")
    reward = (
        single_feet_contact
        | (env.feet_both_contact_time < 0.2)
        | ((torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])) < 0.1)
    ).float()
    return reward


def feet_clearance(
    env: ManagerBasedRLEnv,
    asset_feet_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    asset_base_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_feet_height: float = 0.35,
) -> torch.Tensor:

    asset_feet: Articulation = env.scene[asset_feet_cfg.name]
    asset_base: Articulation = env.scene[asset_base_cfg.name]

    feet_positions = asset_feet.data.body_pos_w[:, asset_feet_cfg.body_ids, :]  # (num_envs, num_feet, 3)
    feet_vels = asset_feet.data.body_lin_vel_w[:, asset_feet_cfg.body_ids, :]  # (num_envs, num_feet, 3)

    base_rotation = asset_base.data.root_link_quat_w[:, :]  # (num_envs, 4)
    base_positions = asset_base.data.root_link_pos_w[:, :]  # (num_envs, 3)
    base_vels = asset_base.data.root_link_lin_vel_w[:, :]  # (num_envs, 3)

    num_envs = feet_positions.shape[0]
    num_feet = feet_positions.shape[1]
    cur_footpos_translated = feet_positions - base_positions.unsqueeze(1)
    footpos_in_body_frame = torch.zeros(num_envs, num_feet, 3, device="cuda")
    cur_footvel_translated = feet_vels - base_vels.unsqueeze(1)
    footvel_in_body_frame = torch.zeros(num_envs, num_feet, 3, device="cuda")
    for i in range(num_feet):
        footpos_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(base_rotation, cur_footpos_translated[:, i, :])
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(base_rotation, cur_footvel_translated[:, i, :])

    height_error = torch.square(footpos_in_body_frame[:, :, 2] - target_feet_height).view(num_envs, -1)
    foot_lateral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(num_envs, -1)
    command = env.command_manager.get_command("base_velocity")

    return torch.sum(height_error * foot_lateral_vel, dim=1) * torch.norm(command[:, :2], dim=1) > 1.5


def feet_air_time_positive_biped(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    command = env.command_manager.get_command("base_velocity")
    reward *= (torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])) > 0.1
    return reward


def feet_slide(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: Articulation = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return torch.any(
        torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
        > 5 * torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]),
        dim=1,
    )

def feet_impact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    terrain_types: tuple[str, ...] | None = None,
    skip_terrain_types: tuple[str, ...] | None = None,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    feet_contact = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    horizontal_force = torch.norm(feet_contact[:, :, :2], dim=-1)
    vertical_force = torch.abs(feet_contact[:, :, 2])
    impact = horizontal_force > 0.5 * vertical_force
    result = torch.any(impact, dim=1)
    device = result.device
    if skip_terrain_types:
        on_skip = _terrain_type_mask(env, skip_terrain_types, device)
        result = torch.where(on_skip, torch.zeros_like(result), result)
    if terrain_types:
        on_terrain = _terrain_type_mask(env, terrain_types, device)
        result = result & on_terrain
    return result


def feet_force(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 500, max_reward: float = 400
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    reward = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2].norm(dim=-1)
    reward[reward < threshold] = 0
    reward[reward > threshold] -= threshold
    reward = reward.clamp(min=0, max=max_reward)
    return reward


def feet_too_near_humanoid(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float = 0.2
) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 2
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


def feet_too_near_humanoid_xy(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float = 0.2
) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 2
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    base_pos_w = asset.data.root_pos_w  # [N, 3]
    base_quat_w = asset.data.root_quat_w  # [N, 4] (xyzw)
    feet_pos_rel_w = feet_pos_w - base_pos_w.unsqueeze(1)  # [N, 2, 3]
    feet_pos_b_0 = math_utils.quat_apply_inverse(base_quat_w, feet_pos_rel_w[:, 0])  # [N, 2, 3]
    feet_pos_b_1 = math_utils.quat_apply_inverse(base_quat_w, feet_pos_rel_w[:, 1])  # [N, 2, 3]
    distance = torch.abs(feet_pos_b_0[:, 1] - feet_pos_b_1[:, 1])
    return (threshold - distance).clamp(min=0)


def feet_too_near_in_some_terrain_xy(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"]),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
    threshold: float = 0.2,
    terrain_types: tuple[str, ...] = (),
    min_forward_command: float = 0.3,
) -> torch.Tensor:
    """在指定地形上、x 速度指令 > min_forward_command 且两脚都接触地面时，惩罚两脚过近（鼓励每脚一桩，如梅花桩）。

    使用 body 系 lateral 方向（y）脚距，小于 threshold 时产生惩罚 (threshold - distance)。
    """
    assert len(asset_cfg.body_ids) == 2
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    base_pos_w = asset.data.root_pos_w
    base_quat_w = asset.data.root_quat_w
    feet_pos_rel_w = feet_pos_w - base_pos_w.unsqueeze(1)
    feet_pos_b_0 = math_utils.quat_apply_inverse(base_quat_w, feet_pos_rel_w[:, 0])
    feet_pos_b_1 = math_utils.quat_apply_inverse(base_quat_w, feet_pos_rel_w[:, 1])
    distance = torch.abs(feet_pos_b_0[:, 1] - feet_pos_b_1[:, 1])
    penalty = (threshold - distance).clamp(min=0.0)

    # 仅在两脚都接触时生效
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    feet_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids, :], dim=-1), dim=1)[0] > 0.5
    both_feet_contact = torch.all(feet_contact, dim=1)
    active = both_feet_contact

    if not terrain_types:
        return penalty * active.float()

    from NoetixE1.tasks.walkrun.noetix_e1.mdp.curriculums import _check_terrain_types_from_positions

    col_indices = getattr(env.unwrapped, "_current_terrain_col_indices", None)
    if col_indices is None:
        is_desired_terrain = _check_terrain_types_from_positions(env, asset.data.root_pos_w, terrain_types)
    else:
        is_desired_terrain = _check_terrain_types_from_positions(env, None, terrain_types, col_indices=col_indices)

    command = env.command_manager.get_command("base_velocity")
    cmd_x = command[:, 0] if command.dim() > 1 else command
    active = active & is_desired_terrain & (cmd_x > min_forward_command)
    # print(penalty*active.float())
    return penalty * active.float()


def feet_orientation_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    terrain_types: tuple[str, ...] | None = None,
) -> torch.Tensor:
    """Reward feet being oriented vertically when in contact with the ground.

    Args:
        env: The environment instance.
        sensor_cfg: Configuration for the contact sensor and body IDs to check.
        asset_cfg: Configuration for the robot asset and body IDs.
        terrain_types: Optional tuple of terrain type names. If provided, only applies penalty
            to environments on these terrain types. If None or empty, applies to all terrains.

    Returns:
        Penalty tensor of shape (num_envs,) with the orientation penalty per environment.
    """
    # project gravity onto the foot plane
    asset: RigidObject = env.scene[asset_cfg.name]
    left_quat = asset.data.body_quat_w[:, asset_cfg.body_ids[0], :]
    left_projected_gravity = math_utils.quat_apply_inverse(left_quat, asset.data.GRAVITY_VEC_W)
    right_quat = asset.data.body_quat_w[:, asset_cfg.body_ids[1], :]
    right_projected_gravity = math_utils.quat_apply_inverse(right_quat, asset.data.GRAVITY_VEC_W)

    # contact
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > 1

    penalty = (
        torch.sum(torch.square(left_projected_gravity[:, :2]), dim=-1) ** 0.5 * is_contact[:, 0]
        + torch.sum(torch.square(right_projected_gravity[:, :2]), dim=-1) ** 0.5 * is_contact[:, 1]
    )

    # Apply terrain type filtering if specified
    if terrain_types is not None and len(terrain_types) > 0:
        from NoetixE1.tasks.walkrun.noetix_e1.mdp.curriculums import _check_terrain_types_from_positions

        # Use cached col_indices from step function if available, otherwise compute from positions
        col_indices = getattr(env.unwrapped, "_current_terrain_col_indices", None)
        if col_indices is None:
            # Fallback: compute from positions if cache is not available
            robot_positions_w = asset.data.root_link_pos_w  # (num_envs, 3)
            is_desired_terrain = _check_terrain_types_from_positions(env, robot_positions_w, terrain_types)
        else:
            # Use cached col_indices (batch processing)
            is_desired_terrain = _check_terrain_types_from_positions(env, None, terrain_types, col_indices=col_indices)

        # Only apply penalty to desired terrain types
        penalty = penalty * is_desired_terrain.float()

    return penalty


def joint_deviation_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(angle), dim=1)


def joint_deviation_l1_with_zero_flag(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    command = env.command_manager.get_command("base_velocity")
    zero_flag = torch.logical_or(
        torch.norm(command[:, 1:], dim=1) > 0.1,
        (torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])) < 0.1,
    )
    return torch.sum(torch.abs(angle), dim=1) * ~zero_flag


def joint_deviation_l1_when_standing(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    command = env.command_manager.get_command("base_velocity")
    zero_flag = (torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])) < 0.1
    return torch.sum(torch.abs(angle), dim=1) * zero_flag


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.norm(torch.abs(asset.data.applied_torque * asset.data.joint_vel), dim=-1)
    return reward


def joint_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=1)

def low_speed_reward(
    env: ManagerBasedRLEnv, min_cmd_vel: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    vel_yaw = math_utils.quat_apply_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    command = env.command_manager.get_command("base_velocity")
    # Calculate the absolute value of speed and command for comparison
    absolute_speed = torch.abs(vel_yaw[:, 0])
    absolute_command = torch.abs(command[:, 0])
    # Define speed criteria for desired range
    speed_too_low = absolute_speed < 0.5 * absolute_command
    speed_too_high = absolute_speed > 1.2 * absolute_command
    speed_desired = ~(speed_too_low | speed_too_high)
    # Initialize reward tensor
    reward = torch.zeros_like(absolute_speed)
    # Assign rewards based on conditions
    # Speed too low
    reward[speed_too_low] = -1.0
    # Speed too high
    reward[speed_too_high] = 0.
    # Speed within desired range
    reward[speed_desired] = 1.2
    return reward * (absolute_command > abs(min_cmd_vel))


def delta_yaw_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command("base_velocity")
    speed_yaw = asset.data.root_ang_vel_w[:, 2]
    command_speed_yaw = command[:, 2]
    delta_speed = torch.abs(speed_yaw - command_speed_yaw)
    absolute_command = torch.abs(command_speed_yaw)
    punish = delta_speed * (delta_speed > 0.1) * (delta_speed > 0.5 * absolute_command)
    return punish


def delta_yaw_reward_terrain_rescale(
    env: ManagerBasedRLEnv,
    terrain_types: tuple[str, ...],
    scale_on_terrain: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """``delta_yaw_reward``，在指定子地形上将惩罚乘以 ``scale_on_terrain``（0 即关闭）。"""
    punish = delta_yaw_reward(env, asset_cfg)
    if not terrain_types:
        return punish
    mask = _terrain_type_mask(env, terrain_types, env.device)
    scale = torch.where(mask, torch.full_like(punish, float(scale_on_terrain)), torch.ones_like(punish))
    return punish * scale


def delta_yaw_penalty_exp(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_penalty: float | None = None,
) -> torch.Tensor:
    """角速度误差的指数惩罚：误差越大惩罚越大，penalty = exp(|delta_yaw| / std) - 1。

    std 越小曲线越陡，对误差更敏感。典型参考（penalty = exp(delta/std) - 1）：
        std=0.5:  delta=0.5 -> 1.72,  delta=1.0 -> 6.39,  delta=1.5 -> 19.09
        std=1.0:  delta=0.5 -> 0.65,  delta=1.0 -> 1.72,  delta=1.5 -> 3.48
        std=1.5:  delta=0.5 -> 0.40,  delta=1.0 -> 0.95,  delta=1.5 -> 1.72
    max_penalty: 惩罚上限，超过则截断；None 表示不设上限。
    使用时在 RewTerm 里用 weight 为负，例如 weight=-0.5, params={"std": 1.0, "max_penalty": 5.0}。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command("base_velocity")
    speed_yaw = asset.data.root_ang_vel_w[:, 2]
    command_speed_yaw = command[:, 2]
    delta_speed = torch.abs(speed_yaw - command_speed_yaw)
    penalty = torch.exp(delta_speed / std) - 1.0
    if max_penalty is not None:
        penalty = penalty.clamp(max=float(max_penalty))
    return penalty


def lin_vel_dir_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command("base_velocity")
    vel_yaw = math_utils.quat_apply_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    command = env.command_manager.get_command("base_velocity")[:, :2]
    lin_vel_dir = torch.nn.functional.normalize(vel_yaw[:, :2], dim=-1)
    desired_lin_vel_dir = torch.nn.functional.normalize(command[:, :2], dim=-1)
    punish = 1 - torch.sum(lin_vel_dir * desired_lin_vel_dir, dim=-1)
    punish = punish * (torch.norm(command[:, :2], dim=-1) > 0.1)
    return punish


def feet_unhold_reward(
    env: ManagerBasedRLEnv,
    contact_sensor_cfg: SceneEntityCfg,
    left_foot_height_scanner_cfg: SceneEntityCfg,
    right_foot_height_scanner_cfg: SceneEntityCfg,
    offset: float = 0.06,
    terrain_types: tuple[str, ...] | None = None,
) -> torch.Tensor:
    """Penalize feet not holding onto the ground properly.

    Args:
        env: The environment instance.
        contact_sensor_cfg: Configuration for the contact sensor and body IDs to check.
        left_foot_height_scanner_cfg: Configuration for the left foot height scanner.
        right_foot_height_scanner_cfg: Configuration for the right foot height scanner.
        offset: Offset for height comparison.
        terrain_types: Optional tuple of terrain type names. If provided, only applies penalty
            to environments on these terrain types. If None or empty, applies to all terrains.

    Returns:
        Penalty tensor of shape (num_envs,) with the unhold penalty per environment.
    """
    # contact
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    feet_contact = torch.max(torch.norm(net_contact_forces[:, :, contact_sensor_cfg.body_ids, :], dim=-1), dim=1)[0] > 0.5

    # left
    left_foot_height_scanner: RayCaster = env.scene.sensors[left_foot_height_scanner_cfg.name]
    left_unhold_num = torch.sum(torch.abs(left_foot_height_scanner.data.pos_w[:, 2].unsqueeze(1) - left_foot_height_scanner.data.ray_hits_w[..., 2] - offset) > 0.03, dim=-1)
    left_unhold_punish = left_unhold_num / left_foot_height_scanner.data.ray_hits_w.shape[-2] * feet_contact[:, 0]

    # right
    right_foot_height_scanner: RayCaster = env.scene.sensors[right_foot_height_scanner_cfg.name]
    right_unhold_num = torch.sum(torch.abs(right_foot_height_scanner.data.pos_w[:, 2].unsqueeze(1) - right_foot_height_scanner.data.ray_hits_w[..., 2] - offset) > 0.03, dim=-1)
    right_unhold_punish = right_unhold_num / right_foot_height_scanner.data.ray_hits_w.shape[-2] * feet_contact[:, 1]

    punish = left_unhold_punish + right_unhold_punish

    # Apply terrain type filtering if specified
    if terrain_types is not None and len(terrain_types) > 0:
        from NoetixE1.tasks.walkrun.noetix_e1.mdp.curriculums import _check_terrain_types_from_positions

        # Use cached col_indices from step function if available, otherwise compute from positions
        col_indices = getattr(env.unwrapped, "_current_terrain_col_indices", None)
        if col_indices is None:
            # Fallback: compute from positions if cache is not available
            asset: RigidObject = env.scene["robot"]
            robot_positions_w = asset.data.root_link_pos_w  # (num_envs, 3)
            is_desired_terrain = _check_terrain_types_from_positions(env, robot_positions_w, terrain_types)
        else:
            # Use cached col_indices (batch processing)
            is_desired_terrain = _check_terrain_types_from_positions(env, None, terrain_types, col_indices=col_indices)

        # Only apply penalty to desired terrain types
        punish = punish * is_desired_terrain.float()
    return punish


def undesired_contacts_height(
    env: ManagerBasedRLEnv, threshold: float,
    sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg,
    terrain_types: tuple[str, ...], height_threshold: float
) -> torch.Tensor:
    """Penalize undesired contacts on specified terrain types above a height threshold.
    
    Args:
        env: The environment instance.
        threshold: Force threshold for contact detection.
        sensor_cfg: Configuration for the contact sensor and body IDs to check.
        terrain_types: Terrain types to check.
        height_threshold: Height threshold above which contacts are penalized.
    
    Returns:
        Penalty tensor of shape (num_envs,) with the number of violations per environment.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # Shape: [num_envs, history_length, num_bodies, 3]
    # Get max force over history for each body
    max_forces = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids, :], dim=-1), dim=1)[0]
    # Shape: [num_envs, num_bodies]
    is_contact = max_forces > threshold

    # Get body positions (world coordinates)
    # Assuming sensor_cfg.name refers to the robot asset name
    asset: Articulation = env.scene[asset_cfg.name]
    body_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]  # [num_envs, num_bodies, 3]
    body_heights = body_pos_w[:, :, 2]  # [num_envs, num_bodies] - Z coordinate is height

    # Check if height is above threshold
    is_above_height = body_heights > height_threshold

    # Get terrain type for each environment (batch processing)
    # Use cached col_indices from step function if available, otherwise compute from positions
    from NoetixE1.tasks.walkrun.noetix_e1.mdp.curriculums import _check_terrain_types_from_positions
    
    col_indices = getattr(env.unwrapped, "_current_terrain_col_indices", None)
    if col_indices is None:
        # Fallback: compute from positions if cache is not available
        robot_positions_w = asset.data.root_link_pos_w  # (num_envs, 3)
        is_desired_terrain = _check_terrain_types_from_positions(env, robot_positions_w, terrain_types)
    else:
        # Use cached col_indices (batch processing)
        is_desired_terrain = _check_terrain_types_from_positions(env, None, terrain_types, col_indices=col_indices)
    
    # Expand to match body dimensions for batch processing
    is_desired_terrain_expanded = is_desired_terrain.unsqueeze(1).expand(-1, len(sensor_cfg.body_ids))

    # Violations: contact exists, height is above threshold, and terrain type matches
    violations = is_contact & is_above_height & is_desired_terrain_expanded
    # print(violations)
    # Sum over bodies for each environment
    return torch.sum(violations, dim=1).float()


def applied_torque_limits_by_ratio(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    limit_ratio: float = 0.8,
):
    """Penalize when the applied torque excceed certain ratio of the joint torque limit."""
    asset: Articulation = env.scene[asset_cfg.name]

    joint_effort_limits = asset.data.joint_effort_limits  # (num_envs, num_joints)
    joint_effort_limits = joint_effort_limits[:, asset_cfg.joint_ids]

    applied_torque = asset.data.applied_torque[:, asset_cfg.joint_ids]
    applied_torque = torch.abs(applied_torque)

    out_of_limits = (applied_torque - joint_effort_limits * limit_ratio).clip(min=0)
    out_of_limits_err = torch.sum(torch.square(out_of_limits), dim=-1)  # (num_envs,)

    return out_of_limits_err


def volume_points_penetration(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    tolerance: float = 0.0,
    terrain_types: tuple[str, ...] | None = None,
) -> torch.Tensor:
    """Penalize the penetration of volume points into the environment.

    Args:
        env: The environment instance.
        sensor_cfg: Configuration for the volume points sensor.
        tolerance: Tolerance threshold for penetration detection.
        terrain_types: Optional tuple of terrain type names. If provided, only applies penalty
            to environments on these terrain types. If None or empty, applies to all terrains.

    Returns:
        Penalty tensor of shape (num_envs,) with the penetration penalty per environment.
    """
    # extract the used quantities (to enable type-hinting)
    volume_sensor: VolumePoints = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    penetration = volume_sensor.data.penetration_offset  # (N, B_, P_, 3) where B_ and P_ varies in sensors
    penetration = penetration.flatten(1, 2)  # (N, B_*P_, 3)
    penetration_depth = torch.norm(penetration, dim=-1)  # (N, B_*P_)
    in_obstacle = (penetration_depth > tolerance).float()  # (N, B_*P_)
    points_vel = volume_sensor.data.points_vel_w  # (N, B_, P_, 3) where B_ and P_ varies in sensors
    points_vel = points_vel.flatten(1, 2)  # (N, B_*P_, 3)
    points_vel_norm = torch.norm(points_vel, dim=-1)  # (N, B_*P_)
    velocity_times_penetration = in_obstacle * (points_vel_norm + 1e-6) * penetration_depth  # (N, B_*P_)
    punish = torch.sum(velocity_times_penetration, dim=-1)  # (N,)

    # Apply terrain type filtering if specified
    if terrain_types is not None and len(terrain_types) > 0:
        from NoetixE1.tasks.walkrun.noetix_e1.mdp.curriculums import _check_terrain_types_from_positions

        # Use cached col_indices from step function if available, otherwise compute from positions
        col_indices = getattr(env.unwrapped, "_current_terrain_col_indices", None)
        if col_indices is None:
            # Fallback: compute from positions if cache is not available
            asset: RigidObject = env.scene["robot"]
            robot_positions_w = asset.data.root_link_pos_w  # (num_envs, 3)
            is_desired_terrain = _check_terrain_types_from_positions(env, robot_positions_w, terrain_types)
        else:
            # Use cached col_indices (batch processing)
            is_desired_terrain = _check_terrain_types_from_positions(env, None, terrain_types, col_indices=col_indices)

        # Only apply penalty to desired terrain types
        punish = punish * is_desired_terrain.float()

    return punish


def volume_points_penetration_novel(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    tolerance: float = 0.0,
    terrain_types: tuple[str, ...] | None = None,
) -> torch.Tensor:
    """Penalize volume-point penetration depth only (no velocity weighting).

    Same as ``volume_points_penetration`` but returns ``sum(in_obstacle * depth)``
    instead of ``sum(in_obstacle * ||v|| * depth)``.
    """
    volume_sensor: VolumePoints = env.scene.sensors[sensor_cfg.name]
    penetration = volume_sensor.data.penetration_offset  # (N, B_, P_, 3)
    penetration = penetration.flatten(1, 2)  # (N, B_*P_, 3)
    penetration_depth = torch.norm(penetration, dim=-1)  # (N, B_*P_)
    in_obstacle = (penetration_depth > tolerance).float()
    punish = torch.sum(in_obstacle * penetration_depth, dim=-1)  # (N,)

    if terrain_types is not None and len(terrain_types) > 0:
        from NoetixE1.tasks.walkrun.noetix_e1.mdp.curriculums import _check_terrain_types_from_positions

        col_indices = getattr(env.unwrapped, "_current_terrain_col_indices", None)
        if col_indices is None:
            asset: RigidObject = env.scene["robot"]
            is_desired_terrain = _check_terrain_types_from_positions(
                env, asset.data.root_link_pos_w, terrain_types
            )
        else:
            is_desired_terrain = _check_terrain_types_from_positions(
                env, None, terrain_types, col_indices=col_indices
            )
        punish = punish * is_desired_terrain.float()

    return punish


def corridor_centerline_exp(
    env: ManagerBasedRLEnv,
    std: float,
    terrain_types: tuple[str, ...],
    min_forward_command: float = 0.3,
    local_x_range: tuple[float, float] | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward staying near the corridor centerline (local y=0) on named terrains.

    Uses terrain-cell coordinates: ``local_xy = root_pos_w.xy - env_origins.xy``.
    Active when forward velocity command exceeds ``min_forward_command``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    local_xy = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    local_y = local_xy[:, 1]
    reward = torch.exp(-torch.square(local_y) / (std**2))

    from NoetixE1.tasks.walkrun.noetix_e1.mdp.curriculums import _check_terrain_types_from_positions

    col_indices = getattr(env.unwrapped, "_current_terrain_col_indices", None)
    if col_indices is None:
        is_desired_terrain = _check_terrain_types_from_positions(env, asset.data.root_pos_w, terrain_types)
    else:
        is_desired_terrain = _check_terrain_types_from_positions(env, None, terrain_types, col_indices=col_indices)

    command = env.command_manager.get_command("base_velocity")
    cmd_x = command[:, 0] if command.dim() > 1 else command
    active = is_desired_terrain & (cmd_x > min_forward_command)

    if local_x_range is not None:
        x_lo, x_hi = local_x_range
        local_x = local_xy[:, 0]
        active = active & (local_x >= x_lo) & (local_x <= x_hi)

    return reward * active.float()


def corridor_world_x_progress_reward(
    env: ManagerBasedRLEnv,
    terrain_types: tuple[str, ...],
    safe_half_width_m: float,
    max_forward_speed: float,
    min_forward_command: float = 0.3,
    local_x_range: tuple[float, float] | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward positive world-x progress while the robot remains in the safe corridor band.

    The reward manager integrates this world-frame velocity with ``dt``, so the
    return is proportional to forward x displacement rather than body-frame speed.
    """
    if safe_half_width_m <= 0.0:
        raise ValueError(f"safe_half_width_m must be positive, got {safe_half_width_m}.")
    if max_forward_speed <= 0.0:
        raise ValueError(f"max_forward_speed must be positive, got {max_forward_speed}.")

    asset: Articulation = env.scene[asset_cfg.name]
    local_xy = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    command = env.command_manager.get_command("base_velocity")
    cmd_x = command[:, 0] if command.dim() > 1 else command

    active = (
        _terrain_type_mask(env, terrain_types, asset.device)
        & (cmd_x > min_forward_command)
        & (torch.abs(local_xy[:, 1]) <= float(safe_half_width_m))
    )
    if local_x_range is not None:
        x_lo, x_hi = local_x_range
        active &= (local_xy[:, 0] >= x_lo) & (local_xy[:, 0] <= x_hi)

    forward_speed = asset.data.root_lin_vel_w[:, 0].clamp(min=0.0, max=float(max_forward_speed))
    return forward_speed * active.float()


def corridor_lateral_threshold_penalty(
    env: ManagerBasedRLEnv,
    terrain_types: tuple[str, ...],
    penalty_threshold: float,
    penalty_limit: float,
    min_forward_command: float = 0.3,
    local_x_range: tuple[float, float] | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return a smooth lateral penalty outside a corridor's safe center band.

    The raw penalty is zero up to ``penalty_threshold``, rises quadratically to
    one at ``penalty_limit``, and remains one outside the corridor. The caller
    supplies a negative reward weight.
    """
    if penalty_threshold < 0.0:
        raise ValueError(f"penalty_threshold must be non-negative, got {penalty_threshold}.")
    if penalty_limit <= penalty_threshold:
        raise ValueError(
            f"penalty_limit must exceed penalty_threshold, got {penalty_limit} <= {penalty_threshold}."
        )

    asset: Articulation = env.scene[asset_cfg.name]
    local_xy = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    local_y = local_xy[:, 1]
    command = env.command_manager.get_command("base_velocity")
    cmd_x = command[:, 0] if command.dim() > 1 else command

    active = _terrain_type_mask(env, terrain_types, asset.device) & (cmd_x > min_forward_command)
    if local_x_range is not None:
        x_lo, x_hi = local_x_range
        active &= (local_xy[:, 0] >= x_lo) & (local_xy[:, 0] <= x_hi)

    normalized_excess = (
        (torch.abs(local_y) - float(penalty_threshold))
        / float(penalty_limit - penalty_threshold)
    ).clamp(min=0.0, max=1.0)
    return torch.square(normalized_excess) * active.float()


def interlaced_corridor_ramp_step_up_lift_reward(
    env: ManagerBasedRLEnv,
    contact_sensor_cfg: SceneEntityCfg,
    left_foot_height_scanner_cfg: SceneEntityCfg,
    right_foot_height_scanner_cfg: SceneEntityCfg,
    offset: float = 0.06,
    clearance_threshold: float = 0.15,
    std: float = 0.05,
    x_range: tuple[float, float] = (-3.4, -2.8),
    min_forward_command: float = 0.3,
    min_episode_length: int = 0,
    max_abs_local_y: float | None = None,
    terrain_types: tuple[str, ...] = ("interlaced_corridor_wedges",),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Encourage stepping up by rewarding elevated (swing) feet near the ramp entry.

    Uses left/right foot height scanners to estimate foot clearance above the local ground under each scanner.
    Reward is active only when:
    - robot is on desired terrain types
    - forward command is non-zero (cmd_x > min_forward_command)
    - local_x is within x_range (near the ramp ledge)
    - the corresponding foot is not in contact (swing foot)
    - optional reset-step and lateral-band gates are satisfied
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # Terrain mask
    from NoetixE1.tasks.walkrun.noetix_e1.mdp.curriculums import _check_terrain_types_from_positions

    col_indices = getattr(env.unwrapped, "_current_terrain_col_indices", None)
    if col_indices is None:
        is_desired_terrain = _check_terrain_types_from_positions(env, asset.data.root_pos_w, terrain_types)
    else:
        is_desired_terrain = _check_terrain_types_from_positions(
            env, None, terrain_types, col_indices=col_indices
        )

    # local_x in terrain-cell frame
    local_xy = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    local_x = local_xy[:, 0]
    local_y = local_xy[:, 1]

    command = env.command_manager.get_command("base_velocity")
    cmd_x = command[:, 0] if command.dim() > 1 else command
    x_lo, x_hi = x_range
    active = is_desired_terrain & (cmd_x > min_forward_command) & (local_x >= x_lo) & (local_x <= x_hi)
    if min_episode_length > 0:
        active &= env.episode_length_buf >= int(min_episode_length)
    if max_abs_local_y is not None:
        if max_abs_local_y <= 0.0:
            raise ValueError(f"max_abs_local_y must be positive, got {max_abs_local_y}.")
        active &= torch.abs(local_y) <= float(max_abs_local_y)

    # Foot contact -> swing mask
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    feet_contact = torch.max(
        torch.norm(net_contact_forces[:, :, contact_sensor_cfg.body_ids, :], dim=-1),
        dim=1,
    )[0] > 0.5  # (num_envs, 2)

    left_air = (~feet_contact[:, 0]).float()
    right_air = (~feet_contact[:, 1]).float()

    # Clearance estimation under each foot scanner
    left_scanner: RayCaster = env.scene.sensors[left_foot_height_scanner_cfg.name]
    right_scanner: RayCaster = env.scene.sensors[right_foot_height_scanner_cfg.name]

    # pos_w[:,2] is the scanner origin height; ray_hits_w[...,2] are ground intersections.
    # Positive (pos_z - hit_z - offset) means the scanner is above the local ground under it.
    left_clearance = (
        left_scanner.data.pos_w[:, 2].unsqueeze(1) - left_scanner.data.ray_hits_w[..., 2] - offset
    )
    right_clearance = (
        right_scanner.data.pos_w[:, 2].unsqueeze(1) - right_scanner.data.ray_hits_w[..., 2] - offset
    )

    # Convert clearance tensor to a single value per env.
    # We first mean over points (-1), then over rays (-1).
    left_clear = left_clearance.clamp(min=0.0).mean(dim=-1).mean(dim=-1)
    right_clear = right_clearance.clamp(min=0.0).mean(dim=-1).mean(dim=-1)

    # Reward when clearance is above threshold (bounded in [0,1]).
    left_r = left_air * torch.sigmoid((left_clear - clearance_threshold) / max(std, 1e-6))
    right_r = right_air * torch.sigmoid((right_clear - clearance_threshold) / max(std, 1e-6))

    return torch.max(left_r, right_r) * active.float()

def _terrain_type_mask(
    env: ManagerBasedRLEnv,
    terrain_types: tuple[str, ...],
    device: torch.device,
) -> torch.Tensor:
    """Boolean mask (num_envs,) for envs on named terrain types."""
    if not terrain_types:
        return torch.ones(env.num_envs, dtype=torch.bool, device=device)
    from NoetixE1.tasks.walkrun.noetix_e1.mdp.curriculums import _check_terrain_types_from_positions

    col_indices = getattr(env.unwrapped, "_current_terrain_col_indices", None)
    if col_indices is None:
        asset: Articulation = env.scene["robot"]
        return _check_terrain_types_from_positions(env, asset.data.root_link_pos_w, terrain_types)
    return _check_terrain_types_from_positions(env, None, terrain_types, col_indices=col_indices)

def _get_figure8_polyline(
    env: ManagerBasedRLEnv,
    inner_radius_m: float,
    bridge_width_m: float,
    samples_per_loop: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """缓存双圆 8 字中心线；返回 ``(t_samples[S], pts[S,2])``。"""
    from NoetixE1.terrains.custom_terrain.figure8_geometry import centerline_points_torch

    cache_key = "_figure8_polyline_cache"
    cache = getattr(env.unwrapped, cache_key, None)
    if cache is None:
        cache = {}
        setattr(env.unwrapped, cache_key, cache)
    key = (float(inner_radius_m), float(bridge_width_m), int(samples_per_loop), str(device))
    if key not in cache:
        from NoetixE1.terrains.custom_terrain.figure8_geometry import _circle_angles

        pts = centerline_points_torch(inner_radius_m, bridge_width_m, samples_per_loop, device)
        t_samples = _circle_angles(pts.shape[0], device)
        cache[key] = {"t_samples": t_samples, "pts": pts}
    entry = cache[key]
    return entry["t_samples"], entry["pts"]


def _figure8_path_tangent_lateral_per_env(
    env: ManagerBasedRLEnv,
    foot_xy: torch.Tensor,
    samples_per_loop: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """按 env 桥宽档位取中心线切向/横向距离（与 terrain level 一致）。"""
    device = foot_xy.device
    inner_r, bridge_width_m = _figure8_bridge_geometry_per_env(env)
    t_out = torch.zeros(foot_xy.shape[0], device=device)
    tangent_out = torch.zeros(foot_xy.shape[0], 2, device=device)
    lateral_out = torch.zeros(foot_xy.shape[0], device=device)

    if isinstance(bridge_width_m, (float, int)):
        unique_widths = torch.tensor([float(bridge_width_m)], device=device)
    else:
        unique_widths = torch.unique(bridge_width_m)

    for w_t in unique_widths:
        w = float(w_t.item())
        if isinstance(bridge_width_m, (float, int)):
            mask = torch.ones(foot_xy.shape[0], dtype=torch.bool, device=device)
        else:
            mask = bridge_width_m == w_t
        if not mask.any():
            continue
        t_samples, pts = _get_figure8_polyline(env, inner_r, w, samples_per_loop, device)
        t_param, tangent, lateral = _xy_to_path_tangent_lateral(foot_xy[mask], t_samples, pts)
        t_out[mask] = t_param
        tangent_out[mask] = tangent
        lateral_out[mask] = lateral
    return t_out, tangent_out, lateral_out


def _xy_to_path_tangent_lateral(
    xy: torch.Tensor, t_samples: torch.Tensor, pts: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """点到闭合中心线最近点：路径参数 ``t``、切向单位向量、横向距离（米）。"""
    num_envs = xy.shape[0]
    num_segs = pts.shape[0]
    a = pts
    b = torch.roll(pts, shifts=-1, dims=0)
    ab = b - a
    ab2 = torch.clamp((ab * ab).sum(dim=1), min=1e-12)

    ap = xy[:, None, :] - a[None, :, :]
    seg_t = torch.clamp((ap * ab[None, :, :]).sum(dim=2) / ab2[None, :], 0.0, 1.0)
    closest = a[None, :, :] + seg_t[:, :, None] * ab[None, :, :]
    diff = xy[:, None, :] - closest
    dist2 = (diff * diff).sum(dim=2)
    seg_idx = dist2.argmin(dim=1)

    idx = torch.arange(num_envs, device=xy.device)
    t0 = t_samples[seg_idx]
    t1 = t_samples[(seg_idx + 1) % num_segs]
    dt_seg = t1 - t0
    dt_seg = torch.where(dt_seg < 0.0, dt_seg + 2.0 * math.pi, dt_seg)
    t_param = t0 + seg_t[idx, seg_idx] * dt_seg

    tangent = ab[seg_idx]
    tangent = tangent / (torch.norm(tangent, dim=1, keepdim=True).clamp(min=1e-8))
    lateral = torch.sqrt(dist2.min(dim=1).values)
    return t_param, tangent, lateral


def _figure8_bridge_geometry_per_env(env: ManagerBasedRLEnv) -> tuple[float, torch.Tensor]:
    """从运行时子地形 cfg 读取 figure8_bridge 的内缘半径与逐 env 桥宽（与 terrain 生成一致）。"""
    from NoetixE1.terrains.custom_terrain.figure8_geometry import bridge_width_for_terrain_levels

    terrain_gen_cfg = env.scene.terrain.cfg.terrain_generator
    sub = terrain_gen_cfg.sub_terrains.get("figure8_bridge")
    if sub is None:
        raise KeyError("figure8_bridge sub_terrain not found in terrain_generator.sub_terrains")

    inner_r = float(sub.inner_radius_m)
    widths = tuple(getattr(sub, "bridge_width_by_level", ()) or ())
    if len(widths) == 0:
        raise ValueError("figure8_bridge bridge_width_by_level must not be empty")

    num_rows = int(getattr(terrain_gen_cfg, "num_rows", 1))
    bridge_width_m = bridge_width_for_terrain_levels(env.scene.terrain.terrain_levels, num_rows, widths)
    return inner_r, bridge_width_m


def _figure8_foot_perception(
    env: ManagerBasedRLEnv,
    contact_sensor_cfg: SceneEntityCfg,
    left_foot_height_scanner_cfg: SceneEntityCfg,
    right_foot_height_scanner_cfg: SceneEntityCfg,
    contact_force_threshold: float,
    bridge_height: float,
    bridge_tol: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """脚底射线位置 + 接触：返回左右脚 on_bridge、both_on_bridge、foot_center_xy、feet_contact_any。

    注意：这里的“在桥上”使用 XY 环带几何（与 ``figure8_bridge_terrain`` 一致），
    避免仅用射线高度（bridge_height/tol）在软边/量化下导致 both_on 长期为 False。
    """
    from NoetixE1.terrains.custom_terrain.figure8_geometry import foot_xy_on_bridge_annulus_torch

    device = env.device
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    feet_contact = (
        torch.max(torch.norm(net_contact_forces[:, :, contact_sensor_cfg.body_ids, :], dim=-1), dim=1)[0]
        > contact_force_threshold
    )

    origins_xy = env.scene.env_origins[:, :2]
    left_scanner: RayCaster = env.scene.sensors[left_foot_height_scanner_cfg.name]
    right_scanner: RayCaster = env.scene.sensors[right_foot_height_scanner_cfg.name]

    left_xy = left_scanner.data.pos_w[:, :2] - origins_xy
    right_xy = right_scanner.data.pos_w[:, :2] - origins_xy
    foot_center_xy = 0.5 * (left_xy + right_xy)
    inner_r, bridge_width_m = _figure8_bridge_geometry_per_env(env)
    left_on = foot_xy_on_bridge_annulus_torch(left_xy, inner_r, bridge_width_m)
    right_on = foot_xy_on_bridge_annulus_torch(right_xy, inner_r, bridge_width_m)
    both_on_bridge = left_on & right_on
    return left_on, right_on, both_on_bridge, foot_center_xy, feet_contact


def figure8_bridge_commanded_idle_penalty(
    env: ManagerBasedRLEnv,
    terrain_types: tuple[str, ...],
    min_cmd_speed: float = 0.15,
    min_body_speed: float = 0.08,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """有前进指令但机身几乎不动时惩罚（防在 8 字桥上蹲着刷分）。

    全局 ``low_speed_reward`` 主要奖励跟速；该项为 8 字专用、仅在有指令且速度过低时给正惩罚量。
    """
    device = env.device
    mask = _terrain_type_mask(env, terrain_types, device)
    if not mask.any():
        return torch.zeros(env.num_envs, device=device)

    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command("base_velocity")
    cmd_speed = torch.norm(command[:, :2], dim=1)
    vel_yaw = math_utils.quat_apply_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    body_speed = torch.norm(vel_yaw[:, :2], dim=1)
    idle = (cmd_speed > float(min_cmd_speed)) & (body_speed < float(min_body_speed))
    return idle.float() * mask.float()


def figure8_bridge_on_bridge_reward(
    env: ManagerBasedRLEnv,
    terrain_types: tuple[str, ...],
    contact_sensor_cfg: SceneEntityCfg,
    left_foot_height_scanner_cfg: SceneEntityCfg,
    right_foot_height_scanner_cfg: SceneEntityCfg,
    contact_force_threshold: float = 0.5,
    bridge_height: float = 0.1,
    bridge_tol: float = 0.04,
) -> torch.Tensor:
    """8 字桥：双脚均在桥面（脚底射线）时 1，单脚在桥上 0.5，否则 0。"""
    device = env.device
    mask = _terrain_type_mask(env, terrain_types, device)
    if not mask.any():
        return torch.zeros(env.num_envs, device=device)

    left_on, right_on, both_on, _, feet_contact_lr = _figure8_foot_perception(
        env,
        contact_sensor_cfg,
        left_foot_height_scanner_cfg,
        right_foot_height_scanner_cfg,
        contact_force_threshold,
        bridge_height,
        bridge_tol,
    )
    any_contact = feet_contact_lr.any(dim=1)
    any_on = (left_on | right_on) & any_contact
    one_foot = any_on & ~both_on
    reward = torch.zeros(env.num_envs, device=device)
    reward = torch.where(both_on & any_contact, torch.ones_like(reward), reward)
    reward = torch.where(one_foot, torch.full_like(reward, 0.5), reward)
    return reward * mask.float()


def figure8_bridge_unsafe_foot_penalty(
    env: ManagerBasedRLEnv,
    terrain_types: tuple[str, ...],
    contact_sensor_cfg: SceneEntityCfg,
    left_foot_height_scanner_cfg: SceneEntityCfg,
    right_foot_height_scanner_cfg: SceneEntityCfg,
    contact_force_threshold: float = 0.5,
    bridge_height: float = 0.1,
    bridge_tol: float = 0.04,
) -> torch.Tensor:
    """8 字桥：任一脚有接触且脚底射线不在桥面 → 1。"""
    device = env.device
    mask = _terrain_type_mask(env, terrain_types, device)
    if not mask.any():
        return torch.zeros(env.num_envs, device=device)

    left_on, right_on, _, _, feet_contact_lr = _figure8_foot_perception(
        env,
        contact_sensor_cfg,
        left_foot_height_scanner_cfg,
        right_foot_height_scanner_cfg,
        contact_force_threshold,
        bridge_height,
        bridge_tol,
    )
    left_unsafe = feet_contact_lr[:, 0] & ~left_on
    right_unsafe = feet_contact_lr[:, 1] & ~right_on
    unsafe = left_unsafe | right_unsafe
    return unsafe.float() * mask.float()


def figure8_bridge_path_forward_reward(
    env: ManagerBasedRLEnv,
    terrain_types: tuple[str, ...],
    contact_sensor_cfg: SceneEntityCfg,
    left_foot_height_scanner_cfg: SceneEntityCfg,
    right_foot_height_scanner_cfg: SceneEntityCfg,
    inner_radius_m: float = 2.5,
    bridge_width_m: float = 0.5,
    samples_per_loop: int = 256,
    std: float = 0.25,
    min_cmd_speed: float = 0.1,
    contact_force_threshold: float = 0.5,
    bridge_height: float = 0.1,
    bridge_tol: float = 0.04,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """沿 8 字中心线切向速度奖励（主）；切向/横向几何由脚底中心 XY 估计，仅在双脚感知在桥上生效。"""
    device = env.device
    mask = _terrain_type_mask(env, terrain_types, device)
    if not mask.any():
        return torch.zeros(env.num_envs, device=device)

    _, _, both_on, foot_xy, _ = _figure8_foot_perception(
        env,
        contact_sensor_cfg,
        left_foot_height_scanner_cfg,
        right_foot_height_scanner_cfg,
        contact_force_threshold,
        bridge_height,
        bridge_tol,
    )
    _, tangent, _ = _figure8_path_tangent_lateral_per_env(env, foot_xy, samples_per_loop)

    asset: Articulation = env.scene[asset_cfg.name]
    vel_w = asset.data.root_lin_vel_w[:, :2]
    v_along = (vel_w * tangent).sum(dim=1)

    command = env.command_manager.get_command("base_velocity")
    cmd_speed = torch.norm(command[:, :2], dim=1)
    has_cmd = cmd_speed > min_cmd_speed
    err = torch.square(v_along - cmd_speed)
    reward = torch.exp(-err / (std**2))
    reward = reward * both_on.float() * has_cmd.float() * mask.float()
    return reward


def figure8_bridge_lateral_offset_penalty(
    env: ManagerBasedRLEnv,
    terrain_types: tuple[str, ...],
    contact_sensor_cfg: SceneEntityCfg,
    left_foot_height_scanner_cfg: SceneEntityCfg,
    right_foot_height_scanner_cfg: SceneEntityCfg,
    edge_buffer_m: float = 0.05,
    contact_force_threshold: float = 0.5,
    bridge_height: float = 0.1,
    bridge_tol: float = 0.04,
) -> torch.Tensor:
    """8 字桥“虚拟障碍”边缘惩罚：脚越靠近内/外缘越罚。

    逻辑：
    - 基于 8 字双圆环带几何（与 ``figure8_bridge_terrain`` 一致）；
    - 对每只脚在其所属圆环带内的“到最近边缘的余量” margin 计算惩罚；
    - 当 margin < edge_buffer_m 时惩罚随靠近边缘逐渐增大；
    - 仅在 figure8_bridge 子地形上启用，避免影响其它地形/正常 reset。
    """
    device = env.device
    mask = _terrain_type_mask(env, terrain_types, device)
    if not mask.any():
        return torch.zeros(env.num_envs, device=device)

    from NoetixE1.terrains.custom_terrain.figure8_geometry import centerline_radius_batch

    left_on, right_on, _, _, feet_contact_lr = _figure8_foot_perception(
        env,
        contact_sensor_cfg,
        left_foot_height_scanner_cfg,
        right_foot_height_scanner_cfg,
        contact_force_threshold,
        bridge_height,
        bridge_tol,
    )
    active = feet_contact_lr.any(dim=1) & mask

    inner_r, bridge_width_m = _figure8_bridge_geometry_per_env(env)  # bridge_width per-env
    outer_r = inner_r + bridge_width_m
    r_c = centerline_radius_batch(inner_r, bridge_width_m)  # (num_envs,)

    origins_xy = env.scene.env_origins[:, :2]
    left_scanner: RayCaster = env.scene.sensors[left_foot_height_scanner_cfg.name]
    right_scanner: RayCaster = env.scene.sensors[right_foot_height_scanner_cfg.name]
    feet_xy = torch.stack(
        [
            left_scanner.data.pos_w[:, :2] - origins_xy,
            right_scanner.data.pos_w[:, :2] - origins_xy,
        ],
        dim=1,
    )  # (num_envs, 2, 2)

    # distances to left/right circle centers
    d_left = torch.hypot(feet_xy[..., 0] + r_c.unsqueeze(1), feet_xy[..., 1])
    d_right = torch.hypot(feet_xy[..., 0] - r_c.unsqueeze(1), feet_xy[..., 1])

    # margin to nearest edge within each annulus (only valid if inside that annulus)
    inner_t = torch.full_like(d_left, float(inner_r))
    outer_t = outer_r.unsqueeze(1).expand_as(d_left)
    inside_left = (d_left >= inner_t) & (d_left <= outer_t)
    inside_right = (d_right >= inner_t) & (d_right <= outer_t)
    margin_left = torch.minimum(d_left - inner_t, outer_t - d_left)
    margin_right = torch.minimum(d_right - inner_t, outer_t - d_right)

    # choose the better (larger) margin if a foot is in both rings near the junction
    margin = torch.where(inside_left, margin_left, torch.full_like(margin_left, -1.0))
    margin = torch.where(inside_right, torch.maximum(margin, margin_right), margin)

    # normalize penalty in [0,1] when margin in [0, edge_buffer]
    buf = float(edge_buffer_m)
    buf = max(buf, 1e-6)
    near = torch.clamp(buf - margin, min=0.0) / buf
    per_foot_pen = near * near

    # Only penalize when the corresponding foot is actually on the figure-8 deck (XY annulus).
    # This prevents large penalty spikes during flight / after falling off.
    on_any = torch.stack([left_on, right_on], dim=1)
    per_foot_pen = per_foot_pen * on_any.float()

    penalty = per_foot_pen.sum(dim=1)
    return penalty * active.float()


def _unwrap_path_delta_t(delta_t: torch.Tensor) -> torch.Tensor:
    """路径参数差归一到 ``[-pi, pi]``。"""
    two_pi = 2.0 * math.pi
    delta_t = torch.where(delta_t > math.pi, delta_t - two_pi, delta_t)
    delta_t = torch.where(delta_t < -math.pi, delta_t + two_pi, delta_t)
    return delta_t


def figure8_bridge_path_progress_reward(
    env: ManagerBasedRLEnv,
    terrain_types: tuple[str, ...],
    contact_sensor_cfg: SceneEntityCfg,
    left_foot_height_scanner_cfg: SceneEntityCfg,
    right_foot_height_scanner_cfg: SceneEntityCfg,
    inner_radius_m: float = 2.5,
    bridge_width_m: float = 0.5,
    samples_per_loop: int = 256,
    min_v_along: float = 0.05,
    progress_scale: float = 1.0,
    contact_force_threshold: float = 0.5,
    bridge_height: float = 0.1,
    bridge_tol: float = 0.04,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """沿 8 字弧长正向进度：脚底中心路径参数 ``t`` 的前进量（与切向速度同号）。"""
    device = env.device
    mask = _terrain_type_mask(env, terrain_types, device)
    if not mask.any():
        return torch.zeros(env.num_envs, device=device)

    if not hasattr(env.unwrapped, "figure8_prev_path_t"):
        env.unwrapped.figure8_prev_path_t = torch.zeros(env.num_envs, device=device)
        env.unwrapped.figure8_prev_path_valid = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
        env.unwrapped.figure8_episode_arc_m = torch.zeros(env.num_envs, device=device)

    _, _, both_on, foot_xy, _ = _figure8_foot_perception(
        env,
        contact_sensor_cfg,
        left_foot_height_scanner_cfg,
        right_foot_height_scanner_cfg,
        contact_force_threshold,
        bridge_height,
        bridge_tol,
    )
    t_now, tangent, _ = _figure8_path_tangent_lateral_per_env(env, foot_xy, samples_per_loop)

    asset: Articulation = env.scene[asset_cfg.name]
    v_along = (asset.data.root_lin_vel_w[:, :2] * tangent).sum(dim=1)

    prev_t = env.unwrapped.figure8_prev_path_t
    prev_valid = env.unwrapped.figure8_prev_path_valid
    # step() 先 length+=1 再算 reward，新 episode 首次奖励见 length==1
    reset = env.episode_length_buf <= 1
    prev_t = torch.where(reset, t_now, prev_t)
    prev_valid = torch.where(reset, torch.zeros_like(prev_valid), prev_valid)

    delta_t = _unwrap_path_delta_t(t_now - prev_t)
    signed_progress = torch.where(v_along >= 0.0, delta_t, -delta_t)
    progress = torch.clamp(signed_progress * float(progress_scale), min=0.0)
    moving = torch.abs(v_along) > min_v_along

    reward = progress * both_on.float() * moving.float() * prev_valid.float() * mask.float()

    from NoetixE1.terrains.custom_terrain.figure8_geometry import centerline_radius_batch, path_param_delta_to_arc_m

    inner_r, bridge_width_m = _figure8_bridge_geometry_per_env(env)
    r_c = centerline_radius_batch(inner_r, bridge_width_m)
    arc_inc = path_param_delta_to_arc_m(progress, r_c)
    arc_inc = arc_inc * both_on.float() * moving.float() * prev_valid.float() * mask.float()
    episode_arc = env.unwrapped.figure8_episode_arc_m
    episode_arc = torch.where(reset, torch.zeros_like(episode_arc), episode_arc)
    env.unwrapped.figure8_episode_arc_m = episode_arc + arc_inc

    env.unwrapped.figure8_prev_path_t = torch.where(mask, t_now, prev_t)
    env.unwrapped.figure8_prev_path_valid = torch.where(
        mask, torch.ones_like(prev_valid) | reset, prev_valid
    )
    return reward
