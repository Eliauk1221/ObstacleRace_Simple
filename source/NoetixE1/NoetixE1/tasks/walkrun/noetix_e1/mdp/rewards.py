# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

from __future__ import annotations

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
    env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command("base_velocity")
    ang_vel_error = torch.square(command[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


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

def feet_impact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # 足部的碰撞力
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    feet_contact = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    horizontal_force = torch.norm(feet_contact[:, :,:2], dim=-1)
    vertical_force = torch.abs(feet_contact[:, :, 2])
    impact = horizontal_force > 0.5 * vertical_force
    return torch.any(impact, dim=1)


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
