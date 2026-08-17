# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from NoetixE1.sensors.volume_points import VolumePoints
from .l_railed_platform import (
    l_curve_corner_blend,
    l_curve_path_state,
    l_curve_travel_direction_sign,
)

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


def track_lin_x_speed_linear(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    skip_terrain_types: tuple[str, ...] | None = None,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    vel_yaw = math_utils.quat_apply_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )

    command = env.command_manager.get_command(command_name)
    cmd_x = command[:, 0]
    vel_x = vel_yaw[:, 0]
    abs_cmd_x = torch.abs(cmd_x)
    zero_cmd = abs_cmd_x < 0.1
    rel_error = torch.abs(cmd_x - vel_x) / (abs_cmd_x + 1.0e-5)

    reward = torch.clamp(1.0 - 2.0 * rel_error, min=-1.5, max=1.0)
    reward = torch.where(cmd_x * vel_x < 0.0, reward - 1.0, reward)

    aggressive_mask = abs_cmd_x > 2.5
    undershoot = torch.abs(vel_x) < abs_cmd_x
    undershoot_coef = torch.where(abs_cmd_x > 4.0, 1.0, 0.5)
    undershoot_penalty = torch.where(undershoot, undershoot_coef * rel_error, 0.0)
    reward = reward - undershoot_penalty * aggressive_mask.float()

    command_weight = torch.clamp(abs_cmd_x / 1.5, min=0.2, max=1.0)
    reward = reward * command_weight
    reward = torch.where(zero_cmd, torch.zeros_like(reward), reward)
    if skip_terrain_types:
        reward = torch.where(
            _terrain_type_mask(env, skip_terrain_types, reward.device),
            torch.zeros_like(reward),
            reward,
        )
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    skip_terrain_types: tuple[str, ...] | None = None,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    vel_yaw = math_utils.quat_apply_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    command = env.command_manager.get_command("base_velocity")
    lin_vel_error = torch.sum(torch.square(command[:, :2] - vel_yaw[:, :2]), dim=1)
    reward = torch.exp(-lin_vel_error / std**2)
    if skip_terrain_types:
        reward = torch.where(
            _terrain_type_mask(env, skip_terrain_types, reward.device),
            torch.zeros_like(reward),
            reward,
        )
    return reward


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
    zero_terrain_types: tuple[str, ...] | None = None,
    gate_high_speed_positive: bool = False,
    high_speed_gate_floor: float | None = None,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Track yaw rate, with separate constant-one and zero masks for terrain-specific objectives."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command("base_velocity")
    ang_vel_error = torch.square(command[:, 2] - asset.data.root_ang_vel_w[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    if skip_terrain_types:
        on_terrain = _terrain_type_mask(env, skip_terrain_types, reward.device)
        reward = torch.where(on_terrain, torch.ones_like(reward), reward)
    if zero_terrain_types:
        on_terrain = _terrain_type_mask(env, zero_terrain_types, reward.device)
        reward = torch.where(on_terrain, torch.zeros_like(reward), reward)
    if gate_high_speed_positive:
        reward = reward * _high_speed_positive_reward_gate(env, command_name, high_speed_gate_floor, reward.device)
    return reward


def track_ang_vel_z_world_exp_turn_sequence(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Existing yaw-rate tracking reward gated by the active turning phase."""
    reward = track_ang_vel_z_world_exp(env, std, asset_cfg)
    command_term = env.command_manager.get_term(command_name)
    turn_mask = getattr(command_term, "is_turn_sequence_turning", None)
    if turn_mask is None:
        return torch.zeros_like(reward)
    return reward * turn_mask.float()


def turn_sequence_static_stability_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    contact_sensor_cfg: SceneEntityCfg,
    lin_vel_std: float = 0.15,
    ang_vel_std: float = 0.25,
    orientation_std: float = 0.5,
    lin_vel_weight: float = 0.35,
    ang_vel_weight: float = 0.35,
    orientation_weight: float = 0.2,
    feet_contact_weight: float = 0.1,
    contact_force_threshold: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward static, upright, double-support holding after the turn phase."""
    for name, value in (
        ("lin_vel_std", lin_vel_std),
        ("ang_vel_std", ang_vel_std),
        ("orientation_std", orientation_std),
    ):
        if value <= 0.0:
            raise ValueError(f"{name} must be positive, got {value}.")

    component_weights = (
        float(lin_vel_weight),
        float(ang_vel_weight),
        float(orientation_weight),
        float(feet_contact_weight),
    )
    if any(weight < 0.0 for weight in component_weights):
        raise ValueError(f"Static-stability component weights must be non-negative, got {component_weights}.")
    weight_sum = sum(component_weights)
    if weight_sum <= 0.0:
        raise ValueError("At least one static-stability component weight must be positive.")
    if contact_force_threshold < 0.0:
        raise ValueError(f"contact_force_threshold must be non-negative, got {contact_force_threshold}.")

    asset: Articulation = env.scene[asset_cfg.name]
    lin_vel_reward = torch.exp(
        -torch.sum(torch.square(asset.data.root_lin_vel_w), dim=1) / float(lin_vel_std) ** 2
    )
    ang_vel_reward = torch.exp(
        -torch.sum(torch.square(asset.data.root_ang_vel_w), dim=1) / float(ang_vel_std) ** 2
    )
    orientation_reward = flat_orientation_exp(env, float(orientation_std), asset_cfg)

    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w_history[:, :, contact_sensor_cfg.body_ids, :]
    feet_contact = torch.max(torch.norm(contact_forces, dim=-1), dim=1)[0] > float(contact_force_threshold)
    both_feet_contact = torch.all(feet_contact, dim=1).float()

    reward = (
        component_weights[0] * lin_vel_reward
        + component_weights[1] * ang_vel_reward
        + component_weights[2] * orientation_reward
        + component_weights[3] * both_feet_contact
    ) / weight_sum

    command_term = env.command_manager.get_term(command_name)
    hold_mask = getattr(command_term, "is_turn_sequence_holding", None)
    if hold_mask is None:
        return torch.zeros_like(reward)
    return reward * hold_mask.float()


def flat_orientation_exp(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    gate_high_speed_positive: bool = False,
    high_speed_gate_floor: float | None = None,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Reward non-flat base orientation using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.exp(-torch.norm(asset.data.projected_gravity_b[:, :2], dim=1) / std**2)
    if gate_high_speed_positive:
        reward = reward * _high_speed_positive_reward_gate(env, command_name, high_speed_gate_floor, reward.device)
    return reward


def feet_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    gate_high_speed_positive: bool = False,
    high_speed_gate_floor: float | None = None,
    command_name: str = "base_velocity",
) -> torch.Tensor:
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
    if gate_high_speed_positive:
        reward = reward * _high_speed_positive_reward_gate(env, command_name, high_speed_gate_floor, reward.device)
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


def feet_air_time_positive_biped(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
    gate_high_speed_positive: bool = False,
    high_speed_gate_floor: float | None = None,
    command_name: str = "base_velocity",
) -> torch.Tensor:
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
    if gate_high_speed_positive:
        reward = reward * _high_speed_positive_reward_gate(env, command_name, high_speed_gate_floor, reward.device)
    return reward


def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance between the feet's last air and contact durations."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if not contact_sensor.cfg.track_air_time:
        raise RuntimeError("Activate ContactSensor's track_air_time!")

    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clamp(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clamp(last_contact_time, max=0.5), dim=1
    )


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


def undesired_contacts_on_terrain(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
    terrain_types: tuple[str, ...],
) -> torch.Tensor:
    """Count undesired contacts only on the selected terrain types."""
    contacts = undesired_contacts(env, threshold, sensor_cfg)
    return contacts * _terrain_type_mask(env, terrain_types, contacts.device).float()


def _high_speed_command_mask(
    env: ManagerBasedRLEnv,
    command_name: str,
    device: torch.device,
) -> torch.Tensor:
    """Return the explicit high-speed mode selected by the command generator."""
    command_term = env.command_manager.get_term(command_name)
    high_speed_mask = getattr(command_term, "high_speed_mask", None)
    if high_speed_mask is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=device)
    return high_speed_mask.to(device=device, dtype=torch.bool)


def _high_speed_positive_reward_gate(
    env: ManagerBasedRLEnv,
    command_name: str,
    floor: float | None,
    device: torch.device,
) -> torch.Tensor:
    """Return the command term's high-speed gate, or identity for unsupported terms."""
    command_term = env.command_manager.get_term(command_name)
    gate_fn = getattr(command_term, "high_speed_reward_gate", None)
    if gate_fn is None:
        return torch.ones(env.num_envs, device=device)
    return gate_fn(floor).to(device=device)


def high_speed_tracking_ratio_reward(
    env: ManagerBasedRLEnv,
    terrain_types: tuple[str, ...],
    command_name: str = "base_velocity",
    min_command_speed: float = 0.1,
    eps: float = 1e-6,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Provide a dense relative tracking signal only for selected high-speed commands."""
    if min_command_speed < 0.0:
        raise ValueError(f"min_command_speed must be non-negative, got {min_command_speed}.")
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}.")

    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    cmd_x = command[:, 0]
    command_term = env.command_manager.get_term(command_name)
    tracking_ratio_fn = getattr(command_term, "high_speed_tracking_ratio", None)
    if tracking_ratio_fn is None:
        vel_yaw = math_utils.quat_apply_inverse(
            math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
        )
        relative_error = torch.abs(vel_yaw[:, 0] - cmd_x) / torch.clamp(torch.abs(cmd_x), min=float(eps))
        reward = torch.clamp(1.0 - relative_error, min=0.0, max=1.0)
    else:
        reward = tracking_ratio_fn(float(eps)).to(device=cmd_x.device)
    active = (
        _high_speed_command_mask(env, command_name, reward.device)
        & _terrain_type_mask(env, terrain_types, reward.device)
        & (cmd_x >= float(min_command_speed))
    )
    return reward * active.float()


def low_speed_reward(
    env: ManagerBasedRLEnv,
    min_cmd_vel: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    skip_terrain_types: tuple[str, ...] | None = None,
    skip_high_speed_commands: bool = False,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    vel_yaw = math_utils.quat_apply_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    command = env.command_manager.get_command(command_name)
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
    reward = reward * (absolute_command > abs(min_cmd_vel))
    if skip_terrain_types:
        on_skip = _terrain_type_mask(env, skip_terrain_types, reward.device)
        reward = reward * (~on_skip).float()
    if skip_high_speed_commands:
        high_speed_mask = _high_speed_command_mask(env, command_name, reward.device)
        reward = reward * (~high_speed_mask).float()
    return reward


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
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    skip_terrain_types: tuple[str, ...] | None = None,
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
    if skip_terrain_types:
        punish = torch.where(
            _terrain_type_mask(env, skip_terrain_types, punish.device),
            torch.zeros_like(punish),
            punish,
        )
    return punish


def stairs_outward_progress_reward(
    env: ManagerBasedRLEnv,
    terrain_types: tuple[str, ...],
    min_forward_command: float = 0.1,
    cap_by_command: bool = True,
    max_outward_speed: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward planar velocity directed away from the stair tile's center platform."""
    if min_forward_command < 0.0:
        raise ValueError(f"min_forward_command must be non-negative, got {min_forward_command}.")
    if max_outward_speed <= 0.0:
        raise ValueError(f"max_outward_speed must be positive, got {max_outward_speed}.")

    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command("base_velocity")
    active = _terrain_type_mask(env, terrain_types, asset.device) & (
        command[:, 0] > float(min_forward_command)
    )
    if not active.any():
        return torch.zeros(env.num_envs, device=asset.device)

    local_xy = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    radial_distance = torch.norm(local_xy, dim=1, keepdim=True)
    outward_direction = local_xy / torch.clamp(radial_distance, min=1.0e-6)
    outward_speed = torch.sum(asset.data.root_lin_vel_w[:, :2] * outward_direction, dim=1).clamp(min=0.0)

    speed_cap = torch.full_like(outward_speed, float(max_outward_speed))
    if cap_by_command:
        speed_cap = torch.minimum(speed_cap, command[:, 0].clamp(min=0.0))
    reward = torch.minimum(outward_speed, speed_cap)
    return reward * active.float()


def stair_large_step_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    terrain_types: tuple[str, ...] = ("stairs_down", "spiral_stairs_down"),
    min_forward_command: float = 0.1,
    stable_contact_time: float = 0.04,
    max_radial_step_distance: float = 0.70,
    min_support_height_drop: float = 0.03,
    excess_distance_scale: float = 0.20,
) -> torch.Tensor:
    """Penalize a touchdown far below and outward from the other foot's last support."""
    if len(asset_cfg.body_ids) != 2 or len(sensor_cfg.body_ids) != 2:
        raise ValueError("stair_large_step_penalty requires exactly two foot bodies.")
    if min_forward_command < 0.0:
        raise ValueError(f"min_forward_command must be non-negative, got {min_forward_command}.")
    if stable_contact_time <= 0.0:
        raise ValueError(f"stable_contact_time must be positive, got {stable_contact_time}.")
    if max_radial_step_distance <= 0.0:
        raise ValueError(
            f"max_radial_step_distance must be positive, got {max_radial_step_distance}."
        )
    if min_support_height_drop < 0.0:
        raise ValueError(
            "min_support_height_drop must be non-negative, "
            f"got {min_support_height_drop}."
        )
    if excess_distance_scale <= 0.0:
        raise ValueError(f"excess_distance_scale must be positive, got {excess_distance_scale}.")
    if env.step_dt <= 0.0:
        raise ValueError(f"env.step_dt must be positive, got {env.step_dt}.")

    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    env_u = env.unwrapped
    foot_body_ids = getattr(env_u, "_foothold_body_ids", asset_cfg.body_ids)
    foot_contact_body_ids = getattr(
        env_u, "_foothold_contact_body_ids", sensor_cfg.body_ids
    )
    feet_pos_w = asset.data.body_pos_w[:, foot_body_ids, :]
    foot_radius = torch.norm(
        feet_pos_w[..., :2] - env.scene.env_origins[:, None, :2], dim=-1
    )
    foot_height = feet_pos_w[..., 2]

    cache_radius = getattr(env_u, "_stair_last_support_radius", None)
    cache_height = getattr(env_u, "_stair_last_support_height", None)
    cache_valid = getattr(env_u, "_stair_last_support_valid", None)
    cache_shape = (env.num_envs, 2)
    if (
        cache_radius is None
        or cache_height is None
        or cache_valid is None
        or cache_radius.shape != cache_shape
        or cache_height.shape != cache_shape
        or cache_valid.shape != cache_shape
        or cache_radius.device != foot_radius.device
        or cache_height.device != foot_height.device
        or cache_valid.device != foot_radius.device
    ):
        cache_radius = torch.zeros(cache_shape, device=foot_radius.device, dtype=foot_radius.dtype)
        cache_height = torch.zeros(cache_shape, device=foot_height.device, dtype=foot_height.dtype)
        cache_valid = torch.zeros(cache_shape, device=foot_radius.device, dtype=torch.bool)
        env_u._stair_last_support_radius = cache_radius
        env_u._stair_last_support_height = cache_height
        env_u._stair_last_support_valid = cache_valid

    reset = env.episode_length_buf <= 1
    cache_valid[reset] = False

    # Keep a pre-touchdown snapshot so simultaneous touchdowns cannot update each other's reference.
    previous_radius = cache_radius.clone()
    previous_height = cache_height.clone()
    previous_valid = cache_valid.clone()

    foothold_events = env.extras.get("foothold_events", {})
    touchdown = foothold_events.get("touchdown")
    touchdown_xy_w = foothold_events.get("touchdown_xy_w")
    if touchdown is None or touchdown_xy_w is None:
        touchdown = torch.zeros(cache_shape, dtype=torch.bool, device=foot_radius.device)
        touchdown_radius = foot_radius
    else:
        touchdown = touchdown.to(device=foot_radius.device, dtype=torch.bool)
        if touchdown.shape != cache_shape or touchdown_xy_w.shape != (*cache_shape, 2):
            raise ValueError(
                "foothold touchdown events must have shapes "
                f"{cache_shape} and {(*cache_shape, 2)}, got "
                f"{tuple(touchdown.shape)} and {tuple(touchdown_xy_w.shape)}."
            )
        touchdown_radius = torch.norm(
            touchdown_xy_w.to(device=foot_radius.device) - env.scene.env_origins[:, None, :2], dim=-1
        )

    other = torch.tensor([1, 0], device=foot_radius.device)
    radial_advance = touchdown_radius - previous_radius[:, other]
    support_height_drop = previous_height[:, other] - foot_height
    command = env.command_manager.get_command("base_velocity")
    active_env = _terrain_type_mask(env, terrain_types, asset.device) & (
        command[:, 0] > float(min_forward_command)
    )
    eligible = (
        touchdown
        & previous_valid[:, other]
        & active_env[:, None]
        & (support_height_drop > float(min_support_height_drop))
    )
    severity = torch.clamp(
        (radial_advance - float(max_radial_step_distance)) / float(excess_distance_scale),
        min=0.0,
        max=1.0,
    )
    triggered = eligible & (radial_advance > float(max_radial_step_distance))
    event_penalty = torch.max(severity * triggered.float(), dim=1).values

    stable_support = (
        contact_sensor.data.current_contact_time[:, foot_contact_body_ids]
        >= float(stable_contact_time)
    )
    cache_radius.copy_(torch.where(stable_support, foot_radius, cache_radius))
    cache_height.copy_(torch.where(stable_support, foot_height, cache_height))
    cache_valid |= stable_support

    eligible_count = eligible.sum()
    env.extras.setdefault("log", {})
    env.extras["log"]["Stairs/large_step_trigger_rate"] = (
        triggered.sum().float() / torch.clamp(eligible_count, min=1).float()
    ).detach()
    env.extras["log"]["Stairs/large_step_radial_advance"] = torch.where(
        triggered, radial_advance, torch.zeros_like(radial_advance)
    ).sum().div(torch.clamp(triggered.sum(), min=1)).detach()
    env.extras["log"]["Stairs/large_step_height_drop"] = torch.where(
        triggered, support_height_drop, torch.zeros_like(support_height_drop)
    ).sum().div(torch.clamp(triggered.sum(), min=1)).detach()
    env.extras["log"]["Stairs/large_step_event_penalty"] = event_penalty.mean().detach()

    # RewardManager multiplies by step_dt; cancel it for this one-frame touchdown event.
    return event_penalty / float(env.step_dt)


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


def imagined_foothold_guidance(
    env: ManagerBasedRLEnv,
    left_foot_height_scanner_cfg: SceneEntityCfg,
    patch_size: tuple[float, float] = (0.225, 0.10),
    patch_resolution: float = 0.05,
    support_height_tolerance: float = 0.03,
    reward_sigma: float = 0.25,
    curriculum_level_threshold: int = 3,
    terrain_types: tuple[str, ...] = ("stairs_up", "stairs_down"),
    downstairs_terrain_types: tuple[str, ...] = ("stairs_down",),
    downstairs_scale: float = 1.5,
    cast_height: float = 2.0,
) -> torch.Tensor:
    """Dense stance/swing support reward from actual and imagined sole patches."""
    env_u = env.unwrapped
    if not getattr(env_u, "_foothold_prediction_valid", False):
        return torch.zeros(env.num_envs, device=env.device)

    from NoetixE1.tasks.walkrun.noetix_e1.mdp.curriculums import _check_terrain_types_from_positions

    col_indices = getattr(env_u, "_current_terrain_col_indices", None)
    terrain_mask = _check_terrain_types_from_positions(
        env, None, terrain_types, col_indices=col_indices
    )
    level_mask = env.scene.terrain.terrain_levels >= curriculum_level_threshold
    active_mask = terrain_mask & level_mask
    down_mask_all = _check_terrain_types_from_positions(
        env, None, downstairs_terrain_types, col_indices=col_indices
    )
    if not active_mask.any():
        env.extras.setdefault("log", {})
        env.extras["log"]["Foothold/guidance_active_fraction"] = torch.zeros(
            (), device=env.device
        )
        return torch.zeros(env.num_envs, device=env.device)

    active_ids = active_mask.nonzero(as_tuple=False).squeeze(-1)
    asset: Articulation = env.scene["robot"]
    base_pos_w = env_u._foothold_decision_base_pos_w[active_ids]
    base_quat_yaw = math_utils.yaw_quat(env_u._foothold_decision_base_quat_w[active_ids])
    mu = env_u._imagined_foothold_mu[active_ids]
    sigma = env_u._imagined_foothold_sigma[active_ids]
    contact = env_u._foothold_decision_contact[active_ids]

    # Five deterministic Gaussian sigma points: center and +/- one sigma per axis.
    unit_sigma_points = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
        device=env.device,
        dtype=mu.dtype,
    )
    sample_weights = torch.exp(-0.5 * torch.sum(torch.square(unit_sigma_points), dim=-1))
    sample_weights /= sample_weights.sum()
    candidate_b = mu.unsqueeze(2) + sigma[..., None, None] * unit_sigma_points

    num_envs, num_feet, num_candidates, _ = candidate_b.shape
    candidate_vec_b = torch.nn.functional.pad(candidate_b, (0, 1))
    candidate_quat = base_quat_yaw[:, None, None, :].expand(
        -1, num_feet, num_candidates, -1
    )
    candidate_w = math_utils.quat_apply(
        candidate_quat.reshape(-1, 4), candidate_vec_b.reshape(-1, 3)
    ).reshape(num_envs, num_feet, num_candidates, 3)
    candidate_xy_w = candidate_w[..., :2] + base_pos_w[:, None, None, :2]

    # Stance feet are evaluated at their current physical position.
    foot_pos_w = asset.data.body_pos_w[active_ids][:, env_u._foothold_body_ids]
    candidate_xy_w = torch.where(
        contact[..., None, None],
        foot_pos_w[..., None, :2].expand_as(candidate_xy_w),
        candidate_xy_w,
    )

    nx = max(2, int(round(patch_size[0] / patch_resolution)) + 1)
    ny = max(2, int(round(patch_size[1] / patch_resolution)) + 1)
    patch_x = torch.linspace(-0.5 * patch_size[0], 0.5 * patch_size[0], nx, device=env.device)
    patch_y = torch.linspace(-0.5 * patch_size[1], 0.5 * patch_size[1], ny, device=env.device)
    grid_x, grid_y = torch.meshgrid(patch_x, patch_y, indexing="ij")
    patch_b = torch.stack(
        (grid_x.flatten(), grid_y.flatten(), torch.zeros(nx * ny, device=env.device)), dim=-1
    )
    patch_quat = base_quat_yaw[:, None, :].expand(-1, nx * ny, -1)
    patch_w = math_utils.quat_apply(
        patch_quat.reshape(-1, 4),
        patch_b.unsqueeze(0).expand(num_envs, -1, -1).reshape(-1, 3),
    ).reshape(num_envs, nx * ny, 3)[..., :2]

    ray_xy = candidate_xy_w[..., None, :] + patch_w[:, None, None, :, :]
    ray_starts = torch.empty(*ray_xy.shape[:-1], 3, device=env.device)
    ray_starts[..., :2] = ray_xy
    ray_starts[..., 2] = base_pos_w[:, None, None, None, 2] + cast_height
    ray_directions = torch.zeros_like(ray_starts)
    ray_directions[..., 2] = -1.0

    from NoetixE1.utils.warp.ops import raycast_mesh

    scanner: RayCaster = env.scene.sensors[left_foot_height_scanner_cfg.name]
    mesh = scanner.meshes[scanner.cfg.mesh_prim_paths[0]]
    raycast_start_time = time.perf_counter()
    ray_hits, _, _, _ = raycast_mesh(
        ray_starts.reshape(-1, 3),
        ray_directions.reshape(-1, 3),
        mesh,
        max_dist=2.0 * cast_height,
    )
    raycast_time_ms = (time.perf_counter() - raycast_start_time) * 1000.0
    heights = ray_hits[:, 2].reshape(num_envs, num_feet, num_candidates, nx * ny)
    finite = torch.isfinite(heights)
    imagined_sole_height = torch.where(
        finite, heights, torch.full_like(heights, -torch.inf)
    ).amax(dim=-1, keepdim=True)
    stance_sole_height = foot_pos_w[..., 2, None, None] - 0.06
    sole_height = torch.where(
        contact[..., None, None], stance_sole_height, imagined_sole_height
    )

    from rsl_rl.modules.foothold_imagination import support_deficiency_from_heights

    deficiency = support_deficiency_from_heights(
        heights, sole_height, support_height_tolerance
    )
    expected_deficiency = torch.sum(deficiency * sample_weights, dim=-1)
    stance_deficiency = deficiency[..., 0]
    per_foot_deficiency = torch.where(contact, stance_deficiency, expected_deficiency)
    reward = torch.exp(-torch.square(per_foot_deficiency.sum(dim=-1)) / reward_sigma)

    down_mask = down_mask_all[active_ids]
    reward = reward * torch.where(
        down_mask, torch.as_tensor(downstairs_scale, device=env.device), 1.0
    )
    full_reward = torch.zeros(env.num_envs, device=env.device)
    full_reward[active_ids] = reward

    env.extras.setdefault("log", {})
    env.extras["log"]["Foothold/support_deficiency"] = per_foot_deficiency.mean().detach()
    env.extras["log"]["Foothold/guidance_active_fraction"] = active_mask.float().mean().detach()
    env.extras["log"]["Foothold/predicted_sigma"] = sigma.mean().detach()
    env.extras["log"]["Foothold/raycast_time_ms"] = raycast_time_ms
    env.extras["log"]["Foothold/rays_per_step"] = float(ray_starts.numel() // 3)
    if down_mask.any():
        env.extras["log"]["Foothold/downstairs_deficiency"] = (
            per_foot_deficiency[down_mask].mean().detach()
        )
    return full_reward


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


def _volume_points_penetration_depth(
    volume_sensor: VolumePoints,
    virtual_obstacle_keys: tuple[str, ...] | None,
) -> torch.Tensor:
    """Return per-point penetration depths from all or selected virtual obstacles."""
    if virtual_obstacle_keys is None:
        return torch.norm(volume_sensor.data.penetration_offset.flatten(1, 2), dim=-1)

    points_pos_w = volume_sensor.data.points_pos_w
    num_envs, num_bodies, num_points, _ = points_pos_w.shape
    flat_points = points_pos_w.reshape(num_envs * num_bodies * num_points, 3)
    max_depth = torch.zeros(flat_points.shape[0], device=flat_points.device, dtype=flat_points.dtype)
    for key in virtual_obstacle_keys:
        virtual_obstacle = volume_sensor._virtual_obstacles.get(key)
        if virtual_obstacle is None:
            raise KeyError(f"Volume-point virtual obstacle is not registered: {key!r}.")
        penetration_offset = virtual_obstacle.get_points_penetration_offset(flat_points)
        max_depth = torch.maximum(max_depth, torch.norm(penetration_offset, dim=-1))
    return max_depth.reshape(num_envs, num_bodies * num_points)


def volume_points_penetration_depth_exp(
    env: ManagerBasedRLEnv,
    depth_scale: float = 0.01,
    sensor_cfg: SceneEntityCfg | None = None,
    sensor_cfgs: tuple[SceneEntityCfg, ...] | list[SceneEntityCfg] | None = None,
    sensor_names: tuple[str, ...] | list[str] | str | None = None,
    tolerance: float = 0.0,
    terrain_types: tuple[str, ...] | None = None,
    virtual_obstacle_keys: tuple[str, ...] | None = None,
    max_exponent: float = 20.0,
) -> torch.Tensor:
    """Exponentially penalize selected virtual-obstacle penetration depth without velocity weighting."""
    if sensor_cfgs is None:
        if sensor_names is not None:
            if isinstance(sensor_names, str):
                sensor_names = (sensor_names,)
            sensor_cfgs = tuple(SceneEntityCfg(name) for name in sensor_names)
        elif sensor_cfg is not None:
            sensor_cfgs = (sensor_cfg,)
        else:
            raise ValueError("One of sensor_cfg, sensor_cfgs, or sensor_names must be provided.")
    if len(sensor_cfgs) == 0:
        raise ValueError("At least one volume-point sensor must be provided.")
    if depth_scale <= 0.0:
        raise ValueError(f"depth_scale must be positive, got {depth_scale}.")
    if max_exponent <= 0.0:
        raise ValueError(f"max_exponent must be positive, got {max_exponent}.")

    total_penalty = None
    for sensor_cfg_i in sensor_cfgs:
        volume_sensor: VolumePoints = env.scene.sensors[sensor_cfg_i.name]
        penetration_depth = _volume_points_penetration_depth(volume_sensor, virtual_obstacle_keys)
        in_obstacle = penetration_depth > tolerance
        exponent = torch.clamp(
            penetration_depth / float(depth_scale), max=float(max_exponent)
        )
        penalty = torch.sum(
            (torch.exp(exponent) - 1.0) * in_obstacle.float(), dim=-1
        )
        total_penalty = penalty if total_penalty is None else total_penalty + penalty

    if terrain_types:
        total_penalty = total_penalty * _terrain_type_mask(
            env, terrain_types, total_penalty.device
        ).float()
    return total_penalty


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
    gate_high_speed_positive: bool = False,
    high_speed_gate_floor: float | None = None,
    command_name: str = "base_velocity",
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

    reward = reward * active.float()
    if gate_high_speed_positive:
        reward = reward * _high_speed_positive_reward_gate(env, command_name, high_speed_gate_floor, reward.device)
    return reward


def corridor_centerline_deviation_penalty(
    env: ManagerBasedRLEnv,
    std: float,
    terrain_types: tuple[str, ...],
    min_forward_command: float = 0.3,
    local_x_range: tuple[float, float] | None = None,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return zero on the centerline and smoothly increasing lateral deviation away from it."""
    if std <= 0.0:
        raise ValueError(f"std must be positive, got {std}.")

    asset: Articulation = env.scene[asset_cfg.name]
    local_xy = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    penalty = 1.0 - torch.exp(-torch.square(local_xy[:, 1]) / float(std) ** 2)
    command = env.command_manager.get_command(command_name)
    cmd_x = command[:, 0] if command.dim() > 1 else command
    active = _terrain_type_mask(env, terrain_types, asset.device) & (cmd_x > min_forward_command)
    if local_x_range is not None:
        x_lo, x_hi = local_x_range
        active &= (local_xy[:, 0] >= x_lo) & (local_xy[:, 0] <= x_hi)
    return penalty * active.float()


def corridor_support_foot_lateral_penalty(
    env: ManagerBasedRLEnv,
    terrain_types: tuple[str, ...],
    penalty_threshold: float,
    penalty_limit: float,
    contact_force_threshold: float = 0.5,
    min_forward_command: float = 0.3,
    local_x_range: tuple[float, float] | None = None,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["l_leg_ankle_roll_link", "r_leg_ankle_roll_link"]
    ),
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg(
        "contact_sensor", body_names=".*_ankle_roll_link"
    ),
) -> torch.Tensor:
    """Penalize support-foot contacts that approach a corridor's lateral terrain edge."""
    if penalty_threshold < 0.0:
        raise ValueError(f"penalty_threshold must be non-negative, got {penalty_threshold}.")
    if penalty_limit <= penalty_threshold:
        raise ValueError(
            f"penalty_limit must exceed penalty_threshold, got {penalty_limit} <= {penalty_threshold}."
        )
    if contact_force_threshold < 0.0:
        raise ValueError(
            f"contact_force_threshold must be non-negative, got {contact_force_threshold}."
        )

    asset: Articulation = env.scene[asset_cfg.name]
    local_root_xy = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    foot_y = (
        asset.data.body_pos_w[:, asset_cfg.body_ids, 1]
        - env.scene.env_origins[:, 1].unsqueeze(1)
    )
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w_history
    feet_contact = (
        torch.max(
            torch.norm(contact_forces[:, :, contact_sensor_cfg.body_ids, :], dim=-1), dim=1
        )[0]
        > float(contact_force_threshold)
    )
    normalized_excess = (
        (torch.abs(foot_y) - float(penalty_threshold))
        / float(penalty_limit - penalty_threshold)
    ).clamp(min=0.0, max=1.0)
    penalty = torch.max(torch.square(normalized_excess) * feet_contact.float(), dim=1)[0]

    command = env.command_manager.get_command(command_name)
    cmd_x = command[:, 0] if command.dim() > 1 else command
    active = _terrain_type_mask(env, terrain_types, asset.device) & (cmd_x > min_forward_command)
    if local_x_range is not None:
        x_lo, x_hi = local_x_range
        active &= (local_root_xy[:, 0] >= x_lo) & (local_root_xy[:, 0] <= x_hi)
    return penalty * active.float()


def _corridor_smooth_reference_geometry(
    local_x: torch.Tensor,
    transition_x_range: tuple[float, float],
    path_y_range: tuple[float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the smoothstep lateral reference and its forward unit tangent."""
    x_start, x_end = transition_x_range
    if x_end <= x_start:
        raise ValueError(f"transition_x_range must be increasing, got {transition_x_range}.")

    phase = ((local_x - float(x_start)) / float(x_end - x_start)).clamp(0.0, 1.0)
    y_start, y_end = path_y_range
    lateral_delta = float(y_end) - float(y_start)
    smooth_phase = phase * phase * (3.0 - 2.0 * phase)
    reference_y = float(y_start) + lateral_delta * smooth_phase

    reference_slope = lateral_delta * 6.0 * phase * (1.0 - phase) / float(x_end - x_start)
    tangent = torch.stack((torch.ones_like(reference_slope), reference_slope), dim=1)
    tangent = tangent / torch.clamp(torch.norm(tangent, dim=1, keepdim=True), min=1e-6)
    return reference_y, tangent


def _corridor_smooth_lateral_reference(
    local_x: torch.Tensor,
    transition_x_range: tuple[float, float],
    path_y_range: tuple[float, float],
) -> torch.Tensor:
    """Return a smoothstep lateral reference from ``path_y_range[0]`` to ``path_y_range[1]``."""
    reference_y, _ = _corridor_smooth_reference_geometry(local_x, transition_x_range, path_y_range)
    return reference_y


def corridor_smooth_path_exp(
    env: ManagerBasedRLEnv,
    std: float,
    terrain_types: tuple[str, ...],
    transition_x_range: tuple[float, float],
    path_y_range: tuple[float, float],
    min_forward_command: float = 0.3,
    local_x_range: tuple[float, float] | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking a lateral reference that moves smoothly with corridor-local x."""
    if std <= 0.0:
        raise ValueError(f"std must be positive, got {std}.")

    asset: Articulation = env.scene[asset_cfg.name]
    local_xy = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    reference_y = _corridor_smooth_lateral_reference(
        local_xy[:, 0], transition_x_range, path_y_range
    )
    path_error = local_xy[:, 1] - reference_y
    reward = torch.exp(-torch.square(path_error) / (std**2))

    command = env.command_manager.get_command("base_velocity")
    cmd_x = command[:, 0] if command.dim() > 1 else command
    active = _terrain_type_mask(env, terrain_types, asset.device) & (cmd_x > min_forward_command)
    if local_x_range is not None:
        x_lo, x_hi = local_x_range
        if x_hi <= x_lo:
            raise ValueError(f"local_x_range must be increasing, got {local_x_range}.")
        active &= (local_xy[:, 0] >= x_lo) & (local_xy[:, 0] <= x_hi)

    return reward * active.float()


def corridor_smooth_path_tangent_speed_gated_exp(
    env: ManagerBasedRLEnv,
    std: float,
    terrain_types: tuple[str, ...],
    transition_x_range: tuple[float, float],
    path_y_range: tuple[float, float],
    min_forward_command: float = 0.3,
    local_x_range: tuple[float, float] | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward smooth-path tracking in proportion to forward speed along the path tangent."""
    if std <= 0.0:
        raise ValueError(f"std must be positive, got {std}.")

    asset: Articulation = env.scene[asset_cfg.name]
    local_xy = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    reference_y, tangent = _corridor_smooth_reference_geometry(
        local_xy[:, 0], transition_x_range, path_y_range
    )
    path_error = local_xy[:, 1] - reference_y
    path_reward = torch.exp(-torch.square(path_error) / (std**2))

    command = env.command_manager.get_command("base_velocity")
    cmd_x = command[:, 0] if command.dim() > 1 else command
    tangent_speed = torch.sum(asset.data.root_lin_vel_w[:, :2] * tangent, dim=1)
    speed_factor = torch.clamp(tangent_speed / torch.clamp(cmd_x, min=1e-6), min=0.0, max=1.0)

    active = _terrain_type_mask(env, terrain_types, asset.device) & (cmd_x > min_forward_command)
    if local_x_range is not None:
        x_lo, x_hi = local_x_range
        if x_hi <= x_lo:
            raise ValueError(f"local_x_range must be increasing, got {local_x_range}.")
        active &= (local_xy[:, 0] >= x_lo) & (local_xy[:, 0] <= x_hi)

    return path_reward * speed_factor * active.float()


def corridor_world_x_smooth_path_progress_reward(
    env: ManagerBasedRLEnv,
    terrain_types: tuple[str, ...],
    transition_x_range: tuple[float, float],
    path_y_range: tuple[float, float],
    path_half_width_m: float | tuple[float, float],
    max_forward_speed: float,
    min_forward_command: float = 0.3,
    local_x_range: tuple[float, float] | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward positive world-x progress inside a fixed or front-wide path band."""
    if isinstance(path_half_width_m, tuple):
        if len(path_half_width_m) != 2:
            raise ValueError(
                f"path_half_width_m tuple must contain (early_width, late_width), got {path_half_width_m}."
            )
        early_half_width, late_half_width = map(float, path_half_width_m)
        if early_half_width <= 0.0 or late_half_width <= 0.0:
            raise ValueError(f"path_half_width_m values must be positive, got {path_half_width_m}.")
        if late_half_width > early_half_width:
            raise ValueError(
                f"path_half_width_m must narrow from early to late, got {path_half_width_m}."
            )
    else:
        early_half_width = late_half_width = float(path_half_width_m)
        if early_half_width <= 0.0:
            raise ValueError(f"path_half_width_m must be positive, got {path_half_width_m}.")
    if max_forward_speed <= 0.0:
        raise ValueError(f"max_forward_speed must be positive, got {max_forward_speed}.")

    asset: Articulation = env.scene[asset_cfg.name]
    local_xy = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    reference_y = _corridor_smooth_lateral_reference(
        local_xy[:, 0], transition_x_range, path_y_range
    )
    path_error = local_xy[:, 1] - reference_y
    if early_half_width == late_half_width:
        path_half_width = early_half_width
    else:
        x_start, x_end = transition_x_range
        phase = ((local_xy[:, 0] - float(x_start)) / float(x_end - x_start)).clamp(0.0, 1.0)
        taper_phase = ((phase - 0.5) / 0.5).clamp(0.0, 1.0)
        smooth_taper = taper_phase * taper_phase * (3.0 - 2.0 * taper_phase)
        path_half_width = early_half_width + (late_half_width - early_half_width) * smooth_taper
    command = env.command_manager.get_command("base_velocity")
    cmd_x = command[:, 0] if command.dim() > 1 else command

    active = (
        _terrain_type_mask(env, terrain_types, asset.device)
        & (cmd_x > min_forward_command)
        & (torch.abs(path_error) <= path_half_width)
    )
    if local_x_range is not None:
        x_lo, x_hi = local_x_range
        if x_hi <= x_lo:
            raise ValueError(f"local_x_range must be increasing, got {local_x_range}.")
        active &= (local_xy[:, 0] >= x_lo) & (local_xy[:, 0] <= x_hi)

    forward_speed = asset.data.root_lin_vel_w[:, 0].clamp(min=0.0, max=float(max_forward_speed))
    return forward_speed * active.float()


def corridor_smooth_funnel_shortfall_penalty(
    env: ManagerBasedRLEnv,
    std: float,
    terrain_types: tuple[str, ...],
    transition_x_range: tuple[float, float],
    minimum_y_range: tuple[float, float],
    min_forward_command: float = 0.3,
    local_x_range: tuple[float, float] | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize positions below a smooth, one-sided corridor exit boundary."""
    if std <= 0.0:
        raise ValueError(f"std must be positive, got {std}.")

    asset: Articulation = env.scene[asset_cfg.name]
    local_xy = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    minimum_y = _corridor_smooth_lateral_reference(
        local_xy[:, 0], transition_x_range, minimum_y_range
    )
    shortfall = torch.clamp(minimum_y - local_xy[:, 1], min=0.0)
    penalty = 1.0 - torch.exp(-torch.square(shortfall) / (std**2))

    command = env.command_manager.get_command("base_velocity")
    cmd_x = command[:, 0] if command.dim() > 1 else command
    active = _terrain_type_mask(env, terrain_types, asset.device) & (cmd_x > min_forward_command)
    if local_x_range is not None:
        x_lo, x_hi = local_x_range
        if x_hi <= x_lo:
            raise ValueError(f"local_x_range must be increasing, got {local_x_range}.")
        active &= (local_xy[:, 0] >= x_lo) & (local_xy[:, 0] <= x_hi)

    return penalty * active.float()


def corridor_feedback_direction_progress_reward(
    env: ManagerBasedRLEnv,
    terrain_types: tuple[str, ...],
    target_y: float,
    lateral_gain: float,
    max_lateral_slope: float,
    max_alignment_error: float,
    max_forward_speed: float,
    cap_by_command: bool = False,
    min_forward_command: float = 0.3,
    local_x_range: tuple[float, float] | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward progress along a lateral-error feedback direction with an alignment gate."""
    if lateral_gain <= 0.0:
        raise ValueError(f"lateral_gain must be positive, got {lateral_gain}.")
    if max_lateral_slope <= 0.0:
        raise ValueError(f"max_lateral_slope must be positive, got {max_lateral_slope}.")
    if not 0.0 < max_alignment_error < math.pi:
        raise ValueError(
            f"max_alignment_error must be in (0, pi), got {max_alignment_error}."
        )
    if max_forward_speed <= 0.0:
        raise ValueError(f"max_forward_speed must be positive, got {max_forward_speed}.")

    asset: Articulation = env.scene[asset_cfg.name]
    local_xy = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    lateral_slope = torch.clamp(
        float(lateral_gain) * (float(target_y) - local_xy[:, 1]),
        min=-float(max_lateral_slope),
        max=float(max_lateral_slope),
    )
    desired_direction = torch.stack((torch.ones_like(lateral_slope), lateral_slope), dim=1)
    desired_direction = desired_direction / torch.clamp(
        torch.norm(desired_direction, dim=1, keepdim=True), min=1e-6
    )

    planar_velocity = asset.data.root_lin_vel_w[:, :2]
    planar_speed = torch.norm(planar_velocity, dim=1)
    directional_speed = torch.sum(planar_velocity * desired_direction, dim=1)
    alignment_cos = directional_speed / torch.clamp(planar_speed, min=1e-6)
    minimum_alignment_cos = math.cos(float(max_alignment_error))
    alignment_factor = torch.clamp(
        (alignment_cos - minimum_alignment_cos) / (1.0 - minimum_alignment_cos),
        min=0.0,
        max=1.0,
    )
    command = env.command_manager.get_command("base_velocity")
    cmd_x = command[:, 0] if command.dim() > 1 else command
    reward_speed_cap = torch.full_like(cmd_x, float(max_forward_speed))
    if cap_by_command:
        reward_speed_cap = torch.minimum(reward_speed_cap, torch.clamp(cmd_x, min=0.0))
    reward = torch.minimum(torch.clamp(directional_speed, min=0.0), reward_speed_cap) * alignment_factor

    active = _terrain_type_mask(env, terrain_types, asset.device) & (cmd_x > min_forward_command)
    if local_x_range is not None:
        x_lo, x_hi = local_x_range
        if x_hi <= x_lo:
            raise ValueError(f"local_x_range must be increasing, got {local_x_range}.")
        active &= (local_xy[:, 0] >= x_lo) & (local_xy[:, 0] <= x_hi)

    return reward * active.float()


def corridor_world_x_progress_reward(
    env: ManagerBasedRLEnv,
    terrain_types: tuple[str, ...],
    safe_half_width_m: float,
    max_forward_speed: float,
    cap_by_command: bool = False,
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

    reward_speed_cap = torch.full_like(cmd_x, float(max_forward_speed))
    if cap_by_command:
        reward_speed_cap = torch.minimum(reward_speed_cap, torch.clamp(cmd_x, min=0.0))
    forward_speed = torch.minimum(asset.data.root_lin_vel_w[:, 0].clamp(min=0.0), reward_speed_cap)
    return forward_speed * active.float()


def _braking_slope_x_bounds(
    env: ManagerBasedRLEnv,
    terrain_type: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-environment stair-start, slope-top, and slope-foot local x coordinates."""
    terrain_gen_cfg = env.scene.terrain.cfg.terrain_generator
    cfg = terrain_gen_cfg.sub_terrains.get(terrain_type)
    if cfg is None:
        raise ValueError(f"Missing terrain configuration for {terrain_type!r}.")

    step_heights = torch.as_tensor(cfg.step_height_by_level, dtype=torch.float, device=device)
    tread_depths = torch.as_tensor(cfg.tread_depth_by_level, dtype=torch.float, device=device)
    if step_heights.numel() == 0 or step_heights.shape != tread_depths.shape:
        raise ValueError("braking-slope step-height and tread-depth schedules must match.")

    num_rows = int(getattr(terrain_gen_cfg, "num_rows", 1))
    if num_rows > 1:
        level_indices = torch.round(
            env.scene.terrain.terrain_levels.to(device=device, dtype=torch.float)
            .clamp(0, num_rows - 1)
            / float(num_rows - 1)
            * (step_heights.numel() - 1)
        ).long()
    else:
        difficulty_range = getattr(terrain_gen_cfg, "difficulty_range", (0.0, 0.0))
        difficulty = max(0.0, min(float(difficulty_range[0]), 1.0))
        level = min(int(math.floor(difficulty * step_heights.numel())), step_heights.numel() - 1)
        level_indices = torch.full((env.num_envs,), level, dtype=torch.long, device=device)
    stairs_height = float(cfg.num_steps) * step_heights[level_indices]
    slope_horizontal_length = stairs_height / math.tan(
        math.radians(float(cfg.slope_angle_deg))
    )
    slope_foot_x = torch.full_like(stairs_height, float(cfg.slope_foot_x_m))
    slope_top_x = slope_foot_x - slope_horizontal_length
    stairs_start_x = slope_top_x - float(cfg.num_steps) * tread_depths[level_indices]
    return stairs_start_x, slope_top_x, slope_foot_x


def braking_slope_world_x_progress_reward(
    env: ManagerBasedRLEnv,
    terrain_types: tuple[str, ...],
    section: str,
    safe_half_width_m: float,
    max_forward_speed: float,
    cap_by_command: bool = True,
    min_forward_command: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward world-frame +x progress on the level-specific stair or down-slope section."""
    if len(terrain_types) != 1:
        raise ValueError("braking_slope progress expects exactly one terrain type.")
    if section not in ("stairs", "slope"):
        raise ValueError(f"section must be 'stairs' or 'slope', got {section!r}.")

    asset: Articulation = env.scene[asset_cfg.name]
    terrain_mask = _terrain_type_mask(env, terrain_types, asset.device)
    if not terrain_mask.any():
        return torch.zeros(env.num_envs, device=asset.device)

    local_xy = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    stairs_start_x, slope_top_x, slope_foot_x = _braking_slope_x_bounds(
        env, terrain_types[0], asset.device
    )
    x_min, x_max = (
        (stairs_start_x, slope_top_x) if section == "stairs" else (slope_top_x, slope_foot_x)
    )

    command = env.command_manager.get_command("base_velocity")
    cmd_x = command[:, 0] if command.dim() > 1 else command
    active = (
        terrain_mask
        & (cmd_x > float(min_forward_command))
        & (local_xy[:, 0] >= x_min)
        & (local_xy[:, 0] <= x_max)
        & (torch.abs(local_xy[:, 1]) <= float(safe_half_width_m))
    )
    speed_cap = torch.full_like(cmd_x, float(max_forward_speed))
    if cap_by_command:
        speed_cap = torch.minimum(speed_cap, torch.clamp(cmd_x, min=0.0))
    forward_speed = torch.minimum(asset.data.root_lin_vel_w[:, 0].clamp(min=0.0), speed_cap)
    return forward_speed * active.float()


def corridor_heading_threshold_penalty(
    env: ManagerBasedRLEnv,
    terrain_types: tuple[str, ...],
    penalty_threshold: float,
    penalty_limit: float,
    target_heading: float = 0.0,
    min_forward_command: float = 0.3,
    local_x_range: tuple[float, float] | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize corridor heading error only after it exceeds a dead zone."""
    if penalty_threshold < 0.0:
        raise ValueError(f"penalty_threshold must be non-negative, got {penalty_threshold}.")
    if penalty_limit <= penalty_threshold:
        raise ValueError(
            f"penalty_limit must exceed penalty_threshold, got {penalty_limit} <= {penalty_threshold}."
        )

    asset: Articulation = env.scene[asset_cfg.name]
    local_xy = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    _, _, yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    heading_error = torch.abs(math_utils.wrap_to_pi(yaw - float(target_heading)))

    command = env.command_manager.get_command("base_velocity")
    cmd_x = command[:, 0] if command.dim() > 1 else command
    active = _terrain_type_mask(env, terrain_types, asset.device) & (cmd_x > min_forward_command)
    if local_x_range is not None:
        x_lo, x_hi = local_x_range
        active &= (local_xy[:, 0] >= x_lo) & (local_xy[:, 0] <= x_hi)

    normalized_excess = (
        (heading_error - float(penalty_threshold))
        / float(penalty_limit - penalty_threshold)
    ).clamp(min=0.0, max=1.0)
    return torch.square(normalized_excess) * active.float()


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


def termination_term_indicator(env: ManagerBasedRLEnv, term_name: str) -> torch.Tensor:
    """Return the current value of one termination term as a reward signal."""
    return env.termination_manager.get_term(term_name).float()


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


def _s_curve_width_for_terrain_levels(env: ManagerBasedRLEnv, sub) -> torch.Tensor:
    terrain_gen_cfg = env.scene.terrain.cfg.terrain_generator
    widths = tuple(getattr(sub, "path_width_by_level", ()) or ())
    if len(widths) > 0:
        widths_t = torch.tensor(widths, dtype=torch.float, device=env.device)
        level_max = len(widths) - 1
        num_rows = int(getattr(terrain_gen_cfg, "num_rows", 1))
        if num_rows > 1:
            level_idx = torch.round(
                env.scene.terrain.terrain_levels.float().clamp(0, num_rows - 1)
                / float(num_rows - 1)
                * level_max
            ).long()
        else:
            difficulty_range = getattr(terrain_gen_cfg, "difficulty_range", None)
            difficulty = float(difficulty_range[0]) if difficulty_range is not None else 0.0
            level_idx = torch.full(
                (env.num_envs,),
                int(round(max(0.0, min(1.0, difficulty)) * level_max)),
                dtype=torch.long,
                device=env.device,
            )
        return widths_t[level_idx.clamp(0, level_max)]

    width_lo, width_hi = getattr(sub, "path_width_range", (0.85, 0.4))
    num_rows = int(getattr(terrain_gen_cfg, "num_rows", 1))
    if num_rows > 1:
        difficulty = env.scene.terrain.terrain_levels.float() / float(num_rows - 1)
    else:
        difficulty_range = getattr(terrain_gen_cfg, "difficulty_range", None)
        difficulty_value = float(difficulty_range[0]) if difficulty_range is not None else 0.0
        difficulty = torch.full((env.num_envs,), difficulty_value, dtype=torch.float, device=env.device)
    return float(width_lo) + difficulty.clamp(0.0, 1.0) * float(width_hi - width_lo)


def _s_curve_geometry_per_env(env: ManagerBasedRLEnv) -> tuple[float, torch.Tensor, tuple[float, float]]:
    terrain_gen_cfg = env.scene.terrain.cfg.terrain_generator
    sub = terrain_gen_cfg.sub_terrains.get("s_curve")
    if sub is None:
        raise KeyError("s_curve sub_terrain not found in terrain_generator.sub_terrains")
    radius_m = float(sub.radius_m)
    path_width_m = _s_curve_width_for_terrain_levels(env, sub)
    center_offset_m = tuple(float(v) for v in getattr(sub, "center_offset_m", (0.0, 0.0)))
    return radius_m, path_width_m, center_offset_m


def _has_s_curve_sub_terrain(env: ManagerBasedRLEnv) -> bool:
    terrain_gen_cfg = getattr(env.scene.terrain.cfg, "terrain_generator", None)
    sub_terrains = getattr(terrain_gen_cfg, "sub_terrains", None)
    return sub_terrains is not None and "s_curve" in sub_terrains


def _get_s_curve_centerline(
    env: ManagerBasedRLEnv,
    radius_m: float,
    samples_per_semicircle: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """S 弯中心线采样，正方向为底部起点 -> 顶部终点。

    返回点、切线方向、以及每个点指向所属半圆圆心的“内侧”方向。
    """
    cache_key = "_s_curve_centerline_cache"
    cache = getattr(env.unwrapped, cache_key, None)
    if cache is None:
        cache = {}
        setattr(env.unwrapped, cache_key, cache)

    samples_per_semicircle = max(int(samples_per_semicircle), 8)
    key = (float(radius_m), samples_per_semicircle, str(device))
    if key in cache:
        return cache[key]

    r = float(radius_m)
    lower_theta = torch.linspace(1.5 * math.pi, 0.5 * math.pi, samples_per_semicircle, device=device)
    upper_theta = torch.linspace(-0.5 * math.pi, 0.5 * math.pi, samples_per_semicircle, device=device)

    lower_pts = torch.stack(
        (r * torch.cos(lower_theta), -r + r * torch.sin(lower_theta)),
        dim=1,
    )
    upper_pts = torch.stack(
        (r * torch.cos(upper_theta), r + r * torch.sin(upper_theta)),
        dim=1,
    )
    lower_tangent = torch.stack((torch.sin(lower_theta), -torch.cos(lower_theta)), dim=1)
    upper_tangent = torch.stack((-torch.sin(upper_theta), torch.cos(upper_theta)), dim=1)
    lower_inward = torch.stack((-torch.cos(lower_theta), -torch.sin(lower_theta)), dim=1)
    upper_inward = torch.stack((-torch.cos(upper_theta), -torch.sin(upper_theta)), dim=1)

    points = torch.cat((lower_pts, upper_pts[1:]), dim=0)
    tangents = torch.cat((lower_tangent, upper_tangent[1:]), dim=0)
    inward_dirs = torch.cat((lower_inward, upper_inward[1:]), dim=0)
    tangents = tangents / torch.clamp(torch.norm(tangents, dim=1, keepdim=True), min=1e-6)
    inward_dirs = inward_dirs / torch.clamp(torch.norm(inward_dirs, dim=1, keepdim=True), min=1e-6)
    cache[key] = (points, tangents, inward_dirs)
    return points, tangents, inward_dirs


def _bias_s_curve_tangent_to_inward(
    tangent: torch.Tensor,
    inward_dir: torch.Tensor,
    inward_tangent_bias_deg: float,
) -> torch.Tensor:
    angle = math.radians(float(inward_tangent_bias_deg))
    if abs(angle) <= 1e-6:
        return tangent
    biased_tangent = math.cos(angle) * tangent + math.sin(angle) * inward_dir
    return biased_tangent / torch.clamp(torch.norm(biased_tangent, dim=1, keepdim=True), min=1e-6)


def _s_curve_project_to_centerline(
    env: ManagerBasedRLEnv,
    local_xy: torch.Tensor,
    samples_per_semicircle: int,
    inward_centerline_offset_m: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回最近中心线点、切线、横向距离、桥宽和内侧方向。"""
    radius_m, path_width_m, center_offset_m = _s_curve_geometry_per_env(env)
    points, tangents, inward_dirs = _get_s_curve_centerline(
        env, radius_m, samples_per_semicircle, local_xy.device
    )
    center = torch.tensor(center_offset_m, dtype=local_xy.dtype, device=local_xy.device)
    rel_xy = local_xy - center.view(1, 2)
    delta = rel_xy.unsqueeze(1) - points.unsqueeze(0)
    dist = torch.norm(delta, dim=-1)
    nearest_idx = torch.argmin(dist, dim=1)
    nearest_points = points[nearest_idx]
    nearest_tangent = tangents[nearest_idx]
    nearest_inward = inward_dirs[nearest_idx]
    nearest_points = nearest_points + float(inward_centerline_offset_m) * nearest_inward
    lateral_distance = torch.norm(rel_xy - nearest_points, dim=1)
    nearest_points = nearest_points + center.view(1, 2)
    return nearest_points, nearest_tangent, lateral_distance, path_width_m, nearest_inward


def _s_curve_lookahead_tangent(
    env: ManagerBasedRLEnv,
    local_xy: torch.Tensor,
    samples_per_semicircle: int,
    lookahead_steps: int,
) -> torch.Tensor:
    """Return the path tangent a fixed number of samples ahead of the nearest centerline point."""
    lookahead_steps = int(lookahead_steps)
    if lookahead_steps < 0:
        raise ValueError(f"lookahead_steps must be non-negative, got {lookahead_steps}.")

    radius_m, _, center_offset_m = _s_curve_geometry_per_env(env)
    points, tangents, _ = _get_s_curve_centerline(
        env, radius_m, samples_per_semicircle, local_xy.device
    )
    center = torch.tensor(center_offset_m, dtype=local_xy.dtype, device=local_xy.device)
    rel_xy = local_xy - center.view(1, 2)
    distance = torch.norm(rel_xy.unsqueeze(1) - points.unsqueeze(0), dim=-1)
    nearest_idx = torch.argmin(distance, dim=1)
    lookahead_idx = torch.clamp(nearest_idx + lookahead_steps, max=tangents.shape[0] - 1)
    return tangents[lookahead_idx]


def curve_upgrade_entry_center_progress_reward(
    env: ManagerBasedRLEnv,
    terrain_types: tuple[str, ...] = ("curve_upgrade",),
    local_x_range: tuple[float, float] = (0.5, 1.0),
    local_y_range: tuple[float, float] = (-3.55, -2.85),
    unguided_center_y_range: tuple[float, float] | None = None,
    target_xy: tuple[float, float] = (0.0, -3.2),
    max_progress_speed: float = 2.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward velocity toward the entrance, optionally outside a central strip."""
    x_low, x_high = (float(value) for value in local_x_range)
    y_low, y_high = (float(value) for value in local_y_range)
    if x_low > x_high:
        raise ValueError(f"local_x_range must be ordered, got {local_x_range}.")
    if y_low > y_high:
        raise ValueError(f"local_y_range must be ordered, got {local_y_range}.")
    if unguided_center_y_range is not None:
        center_y_low, center_y_high = (float(value) for value in unguided_center_y_range)
        if center_y_low > center_y_high:
            raise ValueError(
                "unguided_center_y_range must be ordered, got "
                f"{unguided_center_y_range}."
            )
        if center_y_low < y_low or center_y_high > y_high:
            raise ValueError(
                "unguided_center_y_range must lie within local_y_range, got "
                f"{unguided_center_y_range} and {local_y_range}."
            )
    if max_progress_speed <= 0.0:
        raise ValueError(f"max_progress_speed must be positive, got {max_progress_speed}.")

    asset: Articulation = env.scene[asset_cfg.name]
    local_xy = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    target = torch.tensor(target_xy, dtype=local_xy.dtype, device=local_xy.device)
    target_delta = target.unsqueeze(0) - local_xy
    target_direction = target_delta / torch.clamp(
        torch.norm(target_delta, dim=1, keepdim=True), min=1.0e-6
    )
    progress_speed = torch.sum(
        asset.data.root_lin_vel_w[:, :2] * target_direction, dim=1
    ).clamp(min=0.0, max=float(max_progress_speed))

    active = _terrain_type_mask(env, terrain_types, asset.device)
    active &= (local_xy[:, 0] >= x_low) & (local_xy[:, 0] <= x_high)
    active &= (local_xy[:, 1] >= y_low) & (local_xy[:, 1] <= y_high)
    if unguided_center_y_range is not None:
        active &= (local_xy[:, 1] < center_y_low) | (local_xy[:, 1] > center_y_high)
    return progress_speed * active.float()


def s_curve_path_speed_linear_reward(
    env: ManagerBasedRLEnv,
    terrain_types: tuple[str, ...] = ("s_curve",),
    high_speed_threshold: float = 1.0,
    samples_per_semicircle: int = 64,
    inward_tangent_bias_deg: float = 0.0,
    entry_local_x_range: tuple[float, float] = (0.25, 1.25),
    entry_local_y_range: tuple[float, float] = (-3.60, -2.80),
    entry_center_y_range: tuple[float, float] = (-3.45, -2.95),
    entry_target_xy: tuple[float, float] = (0.0, -3.2),
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track signed S-curve route speed, including side-band entry guidance."""
    if high_speed_threshold < 0.0:
        raise ValueError(
            "high_speed_threshold must be non-negative, got "
            f"{high_speed_threshold}."
        )

    x_low, x_high = (float(value) for value in entry_local_x_range)
    y_low, y_high = (float(value) for value in entry_local_y_range)
    center_y_low, center_y_high = (float(value) for value in entry_center_y_range)
    if x_low > x_high:
        raise ValueError(f"entry_local_x_range must be ordered, got {entry_local_x_range}.")
    if y_low > y_high:
        raise ValueError(f"entry_local_y_range must be ordered, got {entry_local_y_range}.")
    if not y_low <= center_y_low <= center_y_high <= y_high:
        raise ValueError(
            "entry_center_y_range must lie within entry_local_y_range, got "
            f"{entry_center_y_range} and {entry_local_y_range}."
        )

    asset: Articulation = env.scene[asset_cfg.name]
    active = _terrain_type_mask(env, terrain_types, asset.device)
    if not active.any() or not _has_s_curve_sub_terrain(env):
        return torch.zeros(env.num_envs, device=asset.device)

    local_xy = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    _, path_tangent, _, _, inward_dir = _s_curve_project_to_centerline(
        env, local_xy, samples_per_semicircle
    )
    path_tangent = _bias_s_curve_tangent_to_inward(
        path_tangent, inward_dir, inward_tangent_bias_deg
    )

    in_entry_region = (
        (local_xy[:, 0] >= x_low)
        & (local_xy[:, 0] <= x_high)
        & (local_xy[:, 1] >= y_low)
        & (local_xy[:, 1] <= y_high)
    )
    in_entry_side_band = in_entry_region & (
        (local_xy[:, 1] < center_y_low) | (local_xy[:, 1] > center_y_high)
    )
    entry_target = torch.tensor(entry_target_xy, dtype=local_xy.dtype, device=local_xy.device)
    entry_direction = entry_target.unsqueeze(0) - local_xy
    entry_direction /= torch.clamp(
        torch.norm(entry_direction, dim=1, keepdim=True), min=1.0e-6
    )
    desired_direction = torch.where(
        in_entry_side_band.unsqueeze(1), entry_direction, path_tangent
    )
    route_speed = torch.sum(asset.data.root_lin_vel_w[:, :2] * desired_direction, dim=1)

    command_x = env.command_manager.get_command(command_name)[:, 0]
    abs_command_x = torch.abs(command_x)
    abs_route_speed = torch.abs(route_speed)
    zero_command = abs_command_x < 0.1
    relative_error = torch.abs(command_x - route_speed) / (abs_command_x + 1.0e-5)
    reward = torch.clamp(1.0 - 2.0 * relative_error, min=-1.5, max=1.0)
    reward = torch.where(command_x * route_speed < 0.0, reward - 1.0, reward)

    aggressive_underspeed = (
        (abs_command_x > float(high_speed_threshold)) & (abs_route_speed < abs_command_x)
    )
    reward -= torch.where(
        aggressive_underspeed,
        0.5 * relative_error,
        torch.zeros_like(relative_error),
    )
    reward *= torch.clamp(abs_command_x / 1.5, min=0.2, max=1.0)
    reward = torch.where(zero_command, torch.zeros_like(reward), reward)
    return reward * active.float()


def s_curve_path_forward_reward(
    env: ManagerBasedRLEnv,
    terrain_types: tuple[str, ...] = ("s_curve",),
    std: float = 0.25,
    min_cmd_speed: float = 0.1,
    samples_per_semicircle: int = 64,
    inward_tangent_bias_deg: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """S 弯切线方向速度跟踪：切线可向所属半圆圆心内侧偏转。"""
    if std <= 0.0:
        raise ValueError(f"std must be positive, got {std}.")

    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command("base_velocity")
    cmd_x = command[:, 0]
    active = _terrain_type_mask(env, terrain_types, asset.device) & (torch.abs(cmd_x) > min_cmd_speed)
    if not active.any() or not _has_s_curve_sub_terrain(env):
        return torch.zeros(env.num_envs, device=asset.device)

    local_xy = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    _, tangent, _, _, inward_dir = _s_curve_project_to_centerline(env, local_xy, samples_per_semicircle)
    tangent = _bias_s_curve_tangent_to_inward(tangent, inward_dir, inward_tangent_bias_deg)

    v_along = (asset.data.root_lin_vel_w[:, :2] * tangent).sum(dim=1)
    reward = torch.exp(-torch.square(v_along - cmd_x) / (std**2))
    return reward * active.float()


def s_curve_heading_tangent_reward(
    env: ManagerBasedRLEnv,
    terrain_types: tuple[str, ...] = ("s_curve",),
    std: float = 0.5,
    min_cmd_speed: float = 0.1,
    samples_per_semicircle: int = 64,
    inward_tangent_bias_deg: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """S 弯航向跟踪：root yaw 对齐可向圆心内侧偏转的路径方向。"""
    if std <= 0.0:
        raise ValueError(f"std must be positive, got {std}.")

    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command("base_velocity")
    active = _terrain_type_mask(env, terrain_types, asset.device) & (torch.abs(command[:, 0]) > min_cmd_speed)
    if not active.any() or not _has_s_curve_sub_terrain(env):
        return torch.zeros(env.num_envs, device=asset.device)

    local_xy = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    _, tangent, _, _, inward_dir = _s_curve_project_to_centerline(env, local_xy, samples_per_semicircle)
    tangent = _bias_s_curve_tangent_to_inward(tangent, inward_dir, inward_tangent_bias_deg)

    quat = asset.data.root_quat_w
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    body_forward = torch.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (w * z + x * y),
        ),
        dim=1,
    )
    body_forward = body_forward / torch.clamp(torch.norm(body_forward, dim=1, keepdim=True), min=1e-6)
    dot = torch.clamp((body_forward * tangent).sum(dim=1), min=-1.0, max=1.0)
    cross_z = body_forward[:, 0] * tangent[:, 1] - body_forward[:, 1] * tangent[:, 0]
    yaw_error = torch.atan2(cross_z, dot)
    reward = torch.exp(-torch.square(yaw_error) / (std**2))
    return reward * active.float()


def s_curve_path_forward_lookahead_reward(
    env: ManagerBasedRLEnv,
    terrain_types: tuple[str, ...] = ("s_curve",),
    std: float = 0.25,
    min_cmd_speed: float = 0.1,
    samples_per_semicircle: int = 64,
    lookahead_steps: int = 4,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track commanded speed along the tangent ahead of the nearest S-curve point."""
    if std <= 0.0:
        raise ValueError(f"std must be positive, got {std}.")

    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command("base_velocity")
    cmd_x = command[:, 0]
    active = _terrain_type_mask(env, terrain_types, asset.device) & (torch.abs(cmd_x) > min_cmd_speed)
    if not active.any() or not _has_s_curve_sub_terrain(env):
        return torch.zeros(env.num_envs, device=asset.device)

    local_xy = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    tangent = _s_curve_lookahead_tangent(
        env,
        local_xy,
        samples_per_semicircle=samples_per_semicircle,
        lookahead_steps=lookahead_steps,
    )
    v_along = (asset.data.root_lin_vel_w[:, :2] * tangent).sum(dim=1)
    reward = torch.exp(-torch.square(v_along - cmd_x) / (std**2))
    return reward * active.float()


def s_curve_heading_lookahead_tangent_reward(
    env: ManagerBasedRLEnv,
    terrain_types: tuple[str, ...] = ("s_curve",),
    std: float = 0.5,
    min_cmd_speed: float = 0.1,
    samples_per_semicircle: int = 64,
    lookahead_steps: int = 4,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Align root heading with the tangent ahead of the nearest S-curve point."""
    if std <= 0.0:
        raise ValueError(f"std must be positive, got {std}.")

    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command("base_velocity")
    active = _terrain_type_mask(env, terrain_types, asset.device) & (torch.abs(command[:, 0]) > min_cmd_speed)
    if not active.any() or not _has_s_curve_sub_terrain(env):
        return torch.zeros(env.num_envs, device=asset.device)

    local_xy = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    tangent = _s_curve_lookahead_tangent(
        env,
        local_xy,
        samples_per_semicircle=samples_per_semicircle,
        lookahead_steps=lookahead_steps,
    )

    quat = asset.data.root_quat_w
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    body_forward = torch.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (w * z + x * y),
        ),
        dim=1,
    )
    body_forward = body_forward / torch.clamp(torch.norm(body_forward, dim=1, keepdim=True), min=1e-6)
    dot = torch.clamp((body_forward * tangent).sum(dim=1), min=-1.0, max=1.0)
    cross_z = body_forward[:, 0] * tangent[:, 1] - body_forward[:, 1] * tangent[:, 0]
    yaw_error = torch.atan2(cross_z, dot)
    reward = torch.exp(-torch.square(yaw_error) / (std**2))
    return reward * active.float()


def s_curve_centerline_reward(
    env: ManagerBasedRLEnv,
    terrain_types: tuple[str, ...] = ("s_curve",),
    std: float = 0.25,
    edge_buffer_m: float = 0.05,
    samples_per_semicircle: int = 64,
    inward_centerline_offset_m: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """S 弯中心保持：参考中心线可向所属半圆圆心内侧偏移。"""
    if std <= 0.0:
        raise ValueError(f"std must be positive, got {std}.")

    asset: Articulation = env.scene[asset_cfg.name]
    active = _terrain_type_mask(env, terrain_types, asset.device)
    if not active.any() or not _has_s_curve_sub_terrain(env):
        return torch.zeros(env.num_envs, device=asset.device)

    local_xy = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    _, _, lateral_distance, path_width_m, _ = _s_curve_project_to_centerline(
        env, local_xy, samples_per_semicircle, inward_centerline_offset_m
    )

    center_reward = torch.exp(-torch.square(lateral_distance) / (std**2))
    half_width = 0.5 * path_width_m
    edge_margin = half_width - lateral_distance
    edge_scale = torch.clamp(edge_margin / max(float(edge_buffer_m), 1e-6), min=0.0, max=1.0)
    reward = center_reward * edge_scale
    return reward * active.float()

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
def l_curve_path_speed_reward(
    env: ManagerBasedRLEnv,
    terrain_types: tuple[str, ...] = ("l_curve",),
    std: float = 0.25,
    min_cmd_speed: float = 0.1,
    turn_radius_ratio: float = 0.4,
    corner_speed_cap_mps: float = 0.4,
    corner_approach_distance_m: float = 0.5,
    corner_recovery_distance_m: float = 0.3,
    sub_terrain_key: str = "l_curve",
    bidirectional: bool = False,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track commanded speed along the rounded L-bridge tangent.

    The target speed smoothly drops to ``corner_speed_cap_mps`` around the
    quarter-circle turn and returns to the sampled command on the exit leg.
    """
    if std <= 0.0:
        raise ValueError(f"std must be positive, got {std}.")
    if corner_speed_cap_mps <= 0.0:
        raise ValueError(
            "corner_speed_cap_mps must be positive, got "
            f"{corner_speed_cap_mps}."
        )

    asset: Articulation = env.scene[asset_cfg.name]
    command_x = env.command_manager.get_command(command_name)[:, 0]
    active = _terrain_type_mask(env, terrain_types, asset.device)
    active &= command_x > float(min_cmd_speed)
    if not active.any():
        return torch.zeros(env.num_envs, device=asset.device)

    local_xy = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    state, _ = l_curve_path_state(
        env,
        local_xy,
        sub_terrain_key=sub_terrain_key,
        turn_radius_ratio=turn_radius_ratio,
        bidirectional=bidirectional,
    )
    corner_blend = l_curve_corner_blend(
        state,
        approach_distance_m=corner_approach_distance_m,
        recovery_distance_m=corner_recovery_distance_m,
    )
    corner_target = torch.minimum(
        command_x,
        torch.full_like(command_x, float(corner_speed_cap_mps)),
    )
    target_speed = torch.lerp(command_x, corner_target, corner_blend)
    velocity_along_path = (
        asset.data.root_lin_vel_w[:, :2] * state.tangent
    ).sum(dim=1)
    reward = torch.exp(
        -torch.square(velocity_along_path - target_speed) / float(std) ** 2
    )
    return reward * active.float()


def l_curve_heading_tangent_reward(
    env: ManagerBasedRLEnv,
    terrain_types: tuple[str, ...] = ("l_curve",),
    std: float = 0.5,
    min_cmd_speed: float = 0.1,
    turn_radius_ratio: float = 0.4,
    lookahead_base_m: float = 0.25,
    lookahead_speed_scale_s: float = 0.25,
    sub_terrain_key: str = "l_curve",
    bidirectional: bool = False,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Align root heading with a speed-dependent preview tangent of the L path."""
    if std <= 0.0:
        raise ValueError(f"std must be positive, got {std}.")
    if lookahead_base_m < 0.0 or lookahead_speed_scale_s < 0.0:
        raise ValueError("L-curve lookahead settings must be non-negative.")

    asset: Articulation = env.scene[asset_cfg.name]
    command_x = env.command_manager.get_command(command_name)[:, 0]
    active = _terrain_type_mask(env, terrain_types, asset.device)
    active &= command_x > float(min_cmd_speed)
    if not active.any():
        return torch.zeros(env.num_envs, device=asset.device)

    local_xy = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    lookahead = float(lookahead_base_m) + float(
        lookahead_speed_scale_s
    ) * torch.abs(command_x)
    state, _ = l_curve_path_state(
        env,
        local_xy,
        sub_terrain_key=sub_terrain_key,
        turn_radius_ratio=turn_radius_ratio,
        lookahead_distance_m=lookahead,
        bidirectional=bidirectional,
    )

    quaternion = asset.data.root_quat_w
    w, x, y, z = (
        quaternion[:, 0],
        quaternion[:, 1],
        quaternion[:, 2],
        quaternion[:, 3],
    )
    body_forward = torch.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (w * z + x * y),
        ),
        dim=1,
    )
    body_forward = body_forward / torch.linalg.vector_norm(
        body_forward, dim=1, keepdim=True
    ).clamp_min(1.0e-6)
    dot = torch.clamp(
        (body_forward * state.preview_tangent).sum(dim=1),
        min=-1.0,
        max=1.0,
    )
    cross_z = (
        body_forward[:, 0] * state.preview_tangent[:, 1]
        - body_forward[:, 1] * state.preview_tangent[:, 0]
    )
    yaw_error = torch.atan2(cross_z, dot)
    reward = torch.exp(-torch.square(yaw_error) / float(std) ** 2)
    return reward * active.float()


def _l_curve_preview_guidance(
    env: ManagerBasedRLEnv,
    asset: Articulation,
    command_x: torch.Tensor,
    *,
    turn_radius_ratio: float,
    lookahead_base_m: float,
    lookahead_speed_scale_s: float,
    lookahead_max_m: float,
    preview_point_weight: float,
    sub_terrain_key: str,
    bidirectional: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return path state, geometry width, preview direction and yaw error."""
    if lookahead_base_m < 0.0 or lookahead_speed_scale_s < 0.0:
        raise ValueError("L-curve lookahead settings must be non-negative.")
    if lookahead_max_m <= 0.0:
        raise ValueError("lookahead_max_m must be positive.")
    if not 0.0 <= preview_point_weight <= 1.0:
        raise ValueError("preview_point_weight must be in [0, 1].")

    local_xy = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    lookahead = (
        float(lookahead_base_m)
        + float(lookahead_speed_scale_s) * torch.abs(command_x)
    ).clamp(max=float(lookahead_max_m))
    state, geometry = l_curve_path_state(
        env,
        local_xy,
        sub_terrain_key=sub_terrain_key,
        turn_radius_ratio=turn_radius_ratio,
        lookahead_distance_m=lookahead,
        bidirectional=bidirectional,
    )

    direction_to_preview = state.preview_point - local_xy
    preview_distance = torch.linalg.vector_norm(
        direction_to_preview, dim=1, keepdim=True
    )
    direction_to_preview = direction_to_preview / preview_distance.clamp_min(
        1.0e-6
    )
    direction_to_preview = torch.where(
        preview_distance > 1.0e-6,
        direction_to_preview,
        state.preview_tangent,
    )
    target_direction = (
        float(preview_point_weight) * direction_to_preview
        + (1.0 - float(preview_point_weight)) * state.preview_tangent
    )
    target_direction = target_direction / torch.linalg.vector_norm(
        target_direction, dim=1, keepdim=True
    ).clamp_min(1.0e-6)

    quaternion = asset.data.root_quat_w
    w, x, y, z = (
        quaternion[:, 0],
        quaternion[:, 1],
        quaternion[:, 2],
        quaternion[:, 3],
    )
    body_forward = torch.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (w * z + x * y),
        ),
        dim=1,
    )
    body_forward = body_forward / torch.linalg.vector_norm(
        body_forward, dim=1, keepdim=True
    ).clamp_min(1.0e-6)
    dot = torch.clamp(
        (body_forward * target_direction).sum(dim=1),
        min=-1.0,
        max=1.0,
    )
    cross_z = (
        body_forward[:, 0] * target_direction[:, 1]
        - body_forward[:, 1] * target_direction[:, 0]
    )
    yaw_error = torch.atan2(cross_z, dot)
    return state, geometry.passage_width_m, target_direction, yaw_error


def l_curve_corner_heading_penalty(
    env: ManagerBasedRLEnv,
    terrain_types: tuple[str, ...] = ("l_curve",),
    min_cmd_speed: float = 0.1,
    turn_radius_ratio: float = 0.4,
    lookahead_base_m: float = 0.55,
    lookahead_speed_scale_s: float = 0.1,
    lookahead_max_m: float = 0.75,
    preview_point_weight: float = 0.7,
    corner_approach_distance_m: float = 0.7,
    corner_recovery_distance_m: float = 0.3,
    sub_terrain_key: str = "l_curve",
    bidirectional: bool = False,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize preview-heading error without Gaussian saturation."""
    asset: Articulation = env.scene[asset_cfg.name]
    command_x = env.command_manager.get_command(command_name)[:, 0]
    active = _terrain_type_mask(env, terrain_types, asset.device)
    active &= command_x > float(min_cmd_speed)
    if not active.any():
        return torch.zeros(env.num_envs, device=asset.device)

    state, _, _, yaw_error = _l_curve_preview_guidance(
        env,
        asset,
        command_x,
        turn_radius_ratio=turn_radius_ratio,
        lookahead_base_m=lookahead_base_m,
        lookahead_speed_scale_s=lookahead_speed_scale_s,
        lookahead_max_m=lookahead_max_m,
        preview_point_weight=preview_point_weight,
        sub_terrain_key=sub_terrain_key,
        bidirectional=bidirectional,
    )
    corner_blend = l_curve_corner_blend(
        state,
        approach_distance_m=corner_approach_distance_m,
        recovery_distance_m=corner_recovery_distance_m,
    )
    penalty = corner_blend * (1.0 - torch.cos(yaw_error))
    return penalty * active.float()


def l_curve_corner_yaw_rate_reward(
    env: ManagerBasedRLEnv,
    terrain_types: tuple[str, ...] = ("l_curve",),
    std: float = 0.35,
    min_cmd_speed: float = 0.1,
    turn_radius_ratio: float = 0.4,
    corner_speed_cap_mps: float = 0.35,
    heading_feedback_gain: float = 0.8,
    max_ref_ang_vel_rps: float = 1.2,
    forward_gate_floor: float = 0.3,
    lookahead_base_m: float = 0.55,
    lookahead_speed_scale_s: float = 0.1,
    lookahead_max_m: float = 0.75,
    preview_point_weight: float = 0.7,
    corner_approach_distance_m: float = 0.7,
    corner_recovery_distance_m: float = 0.3,
    sub_terrain_key: str = "l_curve",
    bidirectional: bool = False,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track a curvature-aware left-turn yaw rate near the L corner."""
    if std <= 0.0:
        raise ValueError(f"std must be positive, got {std}.")
    if corner_speed_cap_mps <= 0.0 or max_ref_ang_vel_rps <= 0.0:
        raise ValueError("Corner speed and maximum yaw rate must be positive.")
    if not 0.0 <= forward_gate_floor <= 1.0:
        raise ValueError("forward_gate_floor must be in [0, 1].")

    asset: Articulation = env.scene[asset_cfg.name]
    command_x = env.command_manager.get_command(command_name)[:, 0]
    active = _terrain_type_mask(env, terrain_types, asset.device)
    active &= command_x > float(min_cmd_speed)
    if not active.any():
        return torch.zeros(env.num_envs, device=asset.device)

    state, passage_width, target_direction, yaw_error = (
        _l_curve_preview_guidance(
            env,
            asset,
            command_x,
            turn_radius_ratio=turn_radius_ratio,
            lookahead_base_m=lookahead_base_m,
            lookahead_speed_scale_s=lookahead_speed_scale_s,
            lookahead_max_m=lookahead_max_m,
            preview_point_weight=preview_point_weight,
            sub_terrain_key=sub_terrain_key,
            bidirectional=bidirectional,
        )
    )
    corner_blend = l_curve_corner_blend(
        state,
        approach_distance_m=corner_approach_distance_m,
        recovery_distance_m=corner_recovery_distance_m,
    )
    corner_speed = torch.minimum(
        command_x,
        torch.full_like(command_x, float(corner_speed_cap_mps)),
    )
    turn_radius = (
        passage_width.to(dtype=command_x.dtype)
        * float(turn_radius_ratio)
    ).clamp_min(1.0e-6)
    # Curvature feedforward is needed only on the circular segment. The
    # preview-heading feedback starts the turn on approach and removes any
    # residual heading error after the arc without forcing continued rotation.
    on_arc = (
        (state.progress_m >= state.arc_start_progress_m)
        & (state.progress_m <= state.arc_end_progress_m)
    ).float()
    travel_direction = (
        l_curve_travel_direction_sign(
            env,
            command_x.device,
            command_x.dtype,
        )
        if bidirectional
        else torch.ones_like(command_x)
    )
    feedforward_yaw_rate = (
        travel_direction * on_arc * corner_speed / turn_radius
    )
    reference_yaw_rate = torch.clamp(
        feedforward_yaw_rate + float(heading_feedback_gain) * yaw_error,
        min=-float(max_ref_ang_vel_rps),
        max=float(max_ref_ang_vel_rps),
    )
    yaw_rate_reward = torch.exp(
        -torch.square(
            asset.data.root_ang_vel_w[:, 2] - reference_yaw_rate
        )
        / float(std) ** 2
    )

    velocity_toward_preview = (
        asset.data.root_lin_vel_w[:, :2] * target_direction
    ).sum(dim=1)
    progress_ratio = torch.clamp(
        velocity_toward_preview / corner_speed.clamp_min(float(min_cmd_speed)),
        min=0.0,
        max=1.0,
    )
    forward_gate = float(forward_gate_floor) + (
        1.0 - float(forward_gate_floor)
    ) * progress_ratio
    return yaw_rate_reward * forward_gate * corner_blend * active.float()


def l_curve_corner_progress_reward(
    env: ManagerBasedRLEnv,
    terrain_types: tuple[str, ...] = ("l_curve",),
    min_cmd_speed: float = 0.1,
    turn_radius_ratio: float = 0.4,
    corner_speed_cap_mps: float = 0.35,
    lookahead_base_m: float = 0.55,
    lookahead_speed_scale_s: float = 0.1,
    lookahead_max_m: float = 0.75,
    preview_point_weight: float = 0.7,
    corner_approach_distance_m: float = 0.7,
    corner_recovery_distance_m: float = 0.3,
    sub_terrain_key: str = "l_curve",
    bidirectional: bool = False,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward velocity toward the preview point around the L corner."""
    if corner_speed_cap_mps <= 0.0:
        raise ValueError("corner_speed_cap_mps must be positive.")

    asset: Articulation = env.scene[asset_cfg.name]
    command_x = env.command_manager.get_command(command_name)[:, 0]
    active = _terrain_type_mask(env, terrain_types, asset.device)
    active &= command_x > float(min_cmd_speed)
    if not active.any():
        return torch.zeros(env.num_envs, device=asset.device)

    state, _, target_direction, _ = _l_curve_preview_guidance(
        env,
        asset,
        command_x,
        turn_radius_ratio=turn_radius_ratio,
        lookahead_base_m=lookahead_base_m,
        lookahead_speed_scale_s=lookahead_speed_scale_s,
        lookahead_max_m=lookahead_max_m,
        preview_point_weight=preview_point_weight,
        sub_terrain_key=sub_terrain_key,
        bidirectional=bidirectional,
    )
    corner_blend = l_curve_corner_blend(
        state,
        approach_distance_m=corner_approach_distance_m,
        recovery_distance_m=corner_recovery_distance_m,
    )
    corner_speed = torch.minimum(
        command_x,
        torch.full_like(command_x, float(corner_speed_cap_mps)),
    )
    velocity_toward_preview = (
        asset.data.root_lin_vel_w[:, :2] * target_direction
    ).sum(dim=1)
    progress_reward = torch.clamp(
        velocity_toward_preview / corner_speed.clamp_min(float(min_cmd_speed)),
        min=0.0,
        max=1.0,
    )
    return progress_reward * corner_blend * active.float()


def l_curve_centerline_reward(
    env: ManagerBasedRLEnv,
    terrain_types: tuple[str, ...] = ("l_curve",),
    std: float = 0.5,
    edge_buffer_m: float = 0.05,
    turn_radius_ratio: float = 0.4,
    sub_terrain_key: str = "l_curve",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward the rounded centerline using the active bridge width as scale."""
    if std <= 0.0:
        raise ValueError(f"std must be positive, got {std}.")
    if edge_buffer_m <= 0.0:
        raise ValueError(f"edge_buffer_m must be positive, got {edge_buffer_m}.")

    asset: Articulation = env.scene[asset_cfg.name]
    active = _terrain_type_mask(env, terrain_types, asset.device)
    if not active.any():
        return torch.zeros(env.num_envs, device=asset.device)

    local_xy = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    state, geometry = l_curve_path_state(
        env,
        local_xy,
        sub_terrain_key=sub_terrain_key,
        turn_radius_ratio=turn_radius_ratio,
    )
    half_width = (0.5 * geometry.passage_width_m).clamp_min(1.0e-6)
    normalized_lateral_error = state.lateral_distance / half_width
    center_reward = torch.exp(
        -torch.square(normalized_lateral_error / float(std))
    )
    edge_margin = half_width - state.lateral_distance
    edge_scale = torch.clamp(
        edge_margin / float(edge_buffer_m),
        min=0.0,
        max=1.0,
    )
    return center_reward * edge_scale * active.float()


def l_curve_lateral_velocity_penalty(
    env: ManagerBasedRLEnv,
    terrain_types: tuple[str, ...] = ("l_curve",),
    turn_radius_ratio: float = 0.4,
    sub_terrain_key: str = "l_curve",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize squared velocity normal to the rounded L reference path."""
    asset: Articulation = env.scene[asset_cfg.name]
    active = _terrain_type_mask(env, terrain_types, asset.device)
    if not active.any():
        return torch.zeros(env.num_envs, device=asset.device)

    local_xy = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    state, _ = l_curve_path_state(
        env,
        local_xy,
        sub_terrain_key=sub_terrain_key,
        turn_radius_ratio=turn_radius_ratio,
    )
    path_normal = torch.stack(
        (-state.tangent[:, 1], state.tangent[:, 0]), dim=1
    )
    lateral_velocity = (
        asset.data.root_lin_vel_w[:, :2] * path_normal
    ).sum(dim=1)
    return torch.square(lateral_velocity) * active.float()
