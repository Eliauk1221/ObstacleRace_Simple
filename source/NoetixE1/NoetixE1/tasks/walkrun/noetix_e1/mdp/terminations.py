# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

"""Termination functions for the MDP."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch
from isaaclab.assets import Articulation, RigidObject

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.managers import ManagerTermBase, ManagerTermBaseCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.terrains import TerrainImporter

def fall_down_in_some_terrain(
    env: ManagerBasedRLEnv,
    minimum_height: float = 0.5,
    terrain_types: list[str] | tuple[str, ...] | None = None,
    asset_cfg: str = "robot",
) -> torch.Tensor:
    """Terminate if robot falls below minimum height in specific terrain types.

    This function checks if the robot's root height is below a minimum threshold,
    but only terminates if the robot is currently on one of the specified terrain types.

    Args:
        env: The learning environment.
        minimum_height: Minimum height threshold for the root link (in m). Defaults to 0.5.
        terrain_types: List of terrain type names to check. If None or empty, checks all terrains.
                      Defaults to None.
        asset_cfg: Name of the robot asset. Defaults to "robot".

    Returns:
        A boolean tensor of shape (num_envs,) indicating which environments should terminate.
        True means the robot has fallen below the minimum height in the specified terrain types.
    """
    # Get robot asset
    asset: Articulation = env.scene[asset_cfg]
    
    # Get root height (z-coordinate of root link position)
    root_height = asset.data.root_link_pos_w[:, 2]  # (num_envs,)
    
    # Check if height is below minimum threshold
    is_below_minimum = root_height < minimum_height
    
    # If no terrain types specified, terminate based on height alone
    if not terrain_types or len(terrain_types) == 0:
        return is_below_minimum
    
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
    
    # Terminate if height is below minimum AND on desired terrain
    return is_below_minimum & is_desired_terrain

def _figure8_on_terrain_mask(
    env: ManagerBasedRLEnv,
    terrain_types: list[str] | tuple[str, ...],
) -> torch.Tensor:
    """当前 env 是否出生在指定子地形列（优先 ``terrain.terrain_types``，避免仅位置推断列号不准）。"""
    from NoetixE1.tasks.walkrun.noetix_e1.mdp.curriculums import get_terrain_type_names_to_col_map

    names_to_cols = get_terrain_type_names_to_col_map(env)
    desired_cols: list[int] = []
    for name in terrain_types:
        desired_cols.extend(names_to_cols.get(name, []))
    if not desired_cols:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    desired_t = torch.tensor(desired_cols, dtype=torch.long, device=env.device)
    terrain: TerrainImporter = env.scene.terrain
    on_by_spawn = torch.isin(terrain.terrain_types, desired_t)

    col_indices = getattr(env.unwrapped, "_current_terrain_col_indices", None)
    if col_indices is not None:
        on_by_pos = torch.isin(col_indices, desired_t)
        return on_by_spawn | on_by_pos
    return on_by_spawn


def _figure8_bridge_geometry_per_env(env: ManagerBasedRLEnv) -> tuple[float, torch.Tensor, float, float]:
    """从 ``figure8_bridge`` 子地形读取内缘半径、桥宽与中心挖空尺寸。"""
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
    bridge_width_m = bridge_width_for_terrain_levels(
        env.scene.terrain.terrain_levels, num_rows, widths
    )
    cutout_w = float(getattr(sub, "center_cutout_width_m", 0.0))
    cutout_l = float(getattr(sub, "center_cutout_length_m", 0.0))
    return inner_r, bridge_width_m, cutout_w, cutout_l


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


def _s_curve_region_masks(
    foot_xy: torch.Tensor,
    radius_m: float,
    path_width_m: torch.Tensor,
    center_offset_m: tuple[float, float],
    start_region_xy_range: tuple[float, float, float, float],
    end_region_xy_range: tuple[float, float, float, float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    center = torch.tensor(center_offset_m, dtype=foot_xy.dtype, device=foot_xy.device)
    rel = foot_xy - center.view(1, 1, 2)
    half_width = 0.5 * path_width_m.view(-1, 1)
    band_lo = (float(radius_m) - half_width).clamp(min=0.0)
    band_hi = float(radius_m) + half_width

    upper_center = torch.tensor((0.0, radius_m), dtype=foot_xy.dtype, device=foot_xy.device)
    lower_center = torch.tensor((0.0, -radius_m), dtype=foot_xy.dtype, device=foot_xy.device)
    upper_rel = rel - upper_center.view(1, 1, 2)
    lower_rel = rel - lower_center.view(1, 1, 2)
    upper_dist = torch.norm(upper_rel, dim=-1)
    lower_dist = torch.norm(lower_rel, dim=-1)
    on_upper = (rel[..., 0] >= 0.0) & (upper_dist >= band_lo) & (upper_dist <= band_hi)
    on_lower = (rel[..., 0] <= 0.0) & (lower_dist >= band_lo) & (lower_dist <= band_hi)
    on_bridge = on_upper | on_lower

    start_x_lo, start_x_hi, start_y_lo, start_y_hi = [float(v) for v in start_region_xy_range]
    end_x_lo, end_x_hi, end_y_lo, end_y_hi = [float(v) for v in end_region_xy_range]
    in_start = (
        (rel[..., 0] >= min(start_x_lo, start_x_hi))
        & (rel[..., 0] <= max(start_x_lo, start_x_hi))
        & (rel[..., 1] >= min(start_y_lo, start_y_hi))
        & (rel[..., 1] <= max(start_y_lo, start_y_hi))
    )
    in_end = (
        (rel[..., 0] >= min(end_x_lo, end_x_hi))
        & (rel[..., 0] <= max(end_x_lo, end_x_hi))
        & (rel[..., 1] >= min(end_y_lo, end_y_hi))
        & (rel[..., 1] <= max(end_y_lo, end_y_hi))
    )
    return on_bridge, in_start, in_end, on_bridge | in_start | in_end


def s_curve_reach_end_timeout(
    env: ManagerBasedRLEnv,
    terrain_types: list[str] | tuple[str, ...] = ("s_curve",),
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["l_leg_ankle_roll_link", "r_leg_ankle_roll_link"]
    ),
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
    contact_force_threshold: float = 0.5,
    start_region_xy_range: tuple[float, float, float, float] = (-0.9, 0.9, -3.6, -2.8),
    end_region_xy_range: tuple[float, float, float, float] = (-0.9, 0.9, 2.8, 3.6),
) -> torch.Tensor:
    """S 弯：脚接触终点区域时按 timeout 成功结束。"""
    on_terrain = _figure8_on_terrain_mask(env, terrain_types)
    if not on_terrain.any():
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    radius_m, path_width_m, center_offset_m = _s_curve_geometry_per_env(env)
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
    foot_body_ids = contact_sensor_cfg.body_ids
    foot_xy = asset.data.body_pos_w[:, foot_body_ids, :2] - env.scene.env_origins[:, :2].unsqueeze(1)
    _, _, in_end, _ = _s_curve_region_masks(
        foot_xy,
        radius_m,
        path_width_m,
        center_offset_m,
        start_region_xy_range=start_region_xy_range,
        end_region_xy_range=end_region_xy_range,
    )

    net_contact_forces = contact_sensor.data.net_forces_w_history
    feet_contact = (
        torch.max(torch.norm(net_contact_forces[:, :, foot_body_ids, :], dim=-1), dim=1)[0]
        > float(contact_force_threshold)
    )
    reached_end = feet_contact.all(dim=1) & in_end.all(dim=1)
    return on_terrain & reached_end


def s_curve_invalid_contact(
    env: ManagerBasedRLEnv,
    terrain_types: list[str] | tuple[str, ...] = ("s_curve",),
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["l_leg_ankle_roll_link", "r_leg_ankle_roll_link"]
    ),
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
    contact_force_threshold: float = 0.5,
    start_region_xy_range: tuple[float, float, float, float] = (-0.9, 0.9, -3.6, -2.8),
    end_region_xy_range: tuple[float, float, float, float] = (-0.9, 0.9, 2.8, 3.6),
) -> torch.Tensor:
    """S 弯：脚接触桥面、起点、终点以外区域则失败终止。"""
    on_terrain = _figure8_on_terrain_mask(env, terrain_types)
    if not on_terrain.any():
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    radius_m, path_width_m, center_offset_m = _s_curve_geometry_per_env(env)
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
    foot_body_ids = contact_sensor_cfg.body_ids
    foot_xy = asset.data.body_pos_w[:, foot_body_ids, :2] - env.scene.env_origins[:, :2].unsqueeze(1)
    _, _, _, valid = _s_curve_region_masks(
        foot_xy,
        radius_m,
        path_width_m,
        center_offset_m,
        start_region_xy_range=start_region_xy_range,
        end_region_xy_range=end_region_xy_range,
    )

    net_contact_forces = contact_sensor.data.net_forces_w_history
    feet_contact = (
        torch.max(torch.norm(net_contact_forces[:, :, foot_body_ids, :], dim=-1), dim=1)[0]
        > float(contact_force_threshold)
    )
    invalid_contact = (feet_contact & (~valid)).any(dim=1)
    return on_terrain & invalid_contact


def figure8_bridge_fall_off(
    env: ManagerBasedRLEnv,
    terrain_types: list[str] | tuple[str, ...],
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["l_leg_ankle_roll_link", "r_leg_ankle_roll_link"]
    ),
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
    contact_force_threshold: float = 0.5,
    root_drop_threshold_m: float = 0.06,
    annulus_outer_margin_m: float = 0.08,
    transition_y_extend_m: float = 0.06,
    transition_x_margin_m: float = 0.06,
) -> torch.Tensor:
    """8 字桥：离开有效区或 root 明显下沉则终止。

    有效区 = 双圆环带（外缘略扩）∪ 中心 cutout ∪ 上下过渡走廊（含内圈坑底）。
    须先在有效区内落地后，满足以下**任一**条件即终止：
    - 双脚 XY 均不在有效区（掉到环带外深坑）；
    - root 相对落地高度下沉超过 ``root_drop_threshold_m``，且当前**不在**上下过渡走廊
      （避免 bridge→坑底合法下坡被误杀）。

    注意：本仓库 ``step`` 在算终止前执行 ``episode_length_buf += 1``，因此新 episode
    第一次检查时 length 为 1（而非 0）。落地缓存必须在 ``length <= 1`` 时清零，
    否则会跨 episode 残留导致一出生就被 ``dropped`` / ``leave_valid`` 误杀。
    """
    device = env.device
    if not hasattr(env.unwrapped, "figure8_landed"):
        env.unwrapped.figure8_landed = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
        env.unwrapped.figure8_root_z_at_landing = torch.zeros(env.num_envs, dtype=torch.float, device=device)

    landed: torch.Tensor = env.unwrapped.figure8_landed
    root_z0: torch.Tensor = env.unwrapped.figure8_root_z_at_landing

    # Isaac Lab / AMP env: length += 1 happens before termination_manager.compute().
    # After reset length==0; first terminate of an episode therefore sees length==1.
    ep_start = env.episode_length_buf <= 1
    if ep_start.any():
        landed = torch.where(ep_start, torch.zeros_like(landed), landed)
        root_z0 = torch.where(ep_start, torch.zeros_like(root_z0), root_z0)
        env.unwrapped.figure8_landed = landed
        env.unwrapped.figure8_root_z_at_landing = root_z0

    on_terrain = _figure8_on_terrain_mask(env, terrain_types)
    if not on_terrain.any():
        return torch.zeros(env.num_envs, dtype=torch.bool, device=device)

    from NoetixE1.terrains.custom_terrain.figure8_geometry import (
        foot_xy_in_figure8_vertical_transition_torch,
        foot_xy_on_figure8_termination_valid_torch,
    )

    inner_radius_m, bridge_width_m, cutout_w, cutout_l = _figure8_bridge_geometry_per_env(env)

    asset: Articulation = env.scene[asset_cfg.name]
    origins_xy = env.scene.env_origins[:, :2].unsqueeze(1)
    foot_xy = asset.data.body_pos_w[:, asset_cfg.body_ids, :2] - origins_xy

    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    feet_contact = (
        torch.max(torch.norm(net_contact_forces[:, :, contact_sensor_cfg.body_ids, :], dim=-1), dim=1)[0]
        > float(contact_force_threshold)
    )
    any_contact = feet_contact.any(dim=1)

    term_kw = dict(
        inner_radius_m=inner_radius_m,
        bridge_width_m=bridge_width_m,
        cutout_width_m=cutout_w,
        cutout_length_m=cutout_l,
        annulus_outer_margin_m=float(annulus_outer_margin_m),
        transition_y_extend_m=float(transition_y_extend_m),
        transition_x_margin_m=float(transition_x_margin_m),
    )
    left_on = foot_xy_on_figure8_termination_valid_torch(foot_xy[:, 0], **term_kw)
    right_on = foot_xy_on_figure8_termination_valid_torch(foot_xy[:, 1], **term_kw)
    in_valid_xy = left_on | right_on

    left_transition = foot_xy_in_figure8_vertical_transition_torch(
        foot_xy[:, 0], cutout_w, cutout_l, float(transition_y_extend_m), float(transition_x_margin_m)
    )
    right_transition = foot_xy_in_figure8_vertical_transition_torch(
        foot_xy[:, 1], cutout_w, cutout_l, float(transition_y_extend_m), float(transition_x_margin_m)
    )
    on_transition = left_transition | right_transition

    just_landed = (~landed) & any_contact & in_valid_xy & on_terrain
    root_z_rel = asset.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    root_z0 = torch.where(just_landed, root_z_rel, root_z0)
    landed = landed | just_landed

    leave_valid = ~in_valid_xy
    # 坑底比桥面低 ~bridge_height；脚仍在上下过渡走廊时不算“跌落”
    dropped = ((root_z0 - root_z_rel) > float(root_drop_threshold_m)) & (~on_transition)
    terminate = landed & on_terrain & (leave_valid | dropped)

    env.unwrapped.figure8_landed = landed
    env.unwrapped.figure8_root_z_at_landing = root_z0
    return terminate


def figure8_bridge_riser_strike(
    env: ManagerBasedRLEnv,
    terrain_types: list[str] | tuple[str, ...],
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["l_leg_ankle_roll_link", "r_leg_ankle_roll_link"]
    ),
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
    contact_force_threshold: float = 0.5,
    danger_zone_margin_m: float = 0.2,
    strike_horiz_ratio: float = 0.5,
) -> torch.Tensor:
    """8 字桥：危险区内脚踝/足部撞击上地形截面（竖直棱边）则终止。

    危险区 = 中心挖空矩形各向外扩展 ``danger_zone_margin_m``（默认 0.2m）。
    截面撞击判定：该脚在危险区内有接触，且水平接触力显著大于竖直分量
    （``‖F_xy‖ > strike_horiz_ratio * |F_z|``，与 ``feet_impact`` 一致）。
    """
    on_terrain = _figure8_on_terrain_mask(env, terrain_types)
    if not on_terrain.any():
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    _, _, cutout_w, cutout_l = _figure8_bridge_geometry_per_env(env)
    from NoetixE1.terrains.custom_terrain.figure8_geometry import foot_xy_in_expanded_cutout_torch

    asset: Articulation = env.scene[asset_cfg.name]
    origins_xy = env.scene.env_origins[:, :2].unsqueeze(1)
    foot_xy = asset.data.body_pos_w[:, asset_cfg.body_ids, :2] - origins_xy

    in_danger = torch.stack(
        [
            foot_xy_in_expanded_cutout_torch(foot_xy[:, i], cutout_w, cutout_l, danger_zone_margin_m)
            for i in range(foot_xy.shape[1])
        ],
        dim=1,
    )

    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w[:, contact_sensor_cfg.body_ids, :]
    force_norm = torch.norm(contact_forces, dim=-1)
    is_contact = force_norm > float(contact_force_threshold)

    horizontal_force = torch.norm(contact_forces[..., :2], dim=-1)
    vertical_force = torch.abs(contact_forces[..., 2])
    is_strike = horizontal_force > float(strike_horiz_ratio) * vertical_force

    foot_strike = in_danger & is_contact & is_strike
    terminate = on_terrain & foot_strike.any(dim=1)
    return terminate


def interlaced_corridor_lateral_exit(
    env: ManagerBasedRLEnv,
    terrain_types: tuple[str, ...],
    local_x_range: tuple[float, float],
    max_abs_local_y: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate after the root leaves the lateral boundary of an interlaced corridor."""
    if max_abs_local_y <= 0.0:
        raise ValueError(f"max_abs_local_y must be positive, got {max_abs_local_y}.")
    x_lo, x_hi = local_x_range
    if x_hi <= x_lo:
        raise ValueError(f"local_x_range must be increasing, got {local_x_range}.")

    asset: Articulation = env.scene[asset_cfg.name]
    local_xy = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]

    from NoetixE1.tasks.walkrun.noetix_e1.mdp.curriculums import _check_terrain_types_from_positions

    col_indices = getattr(env.unwrapped, "_current_terrain_col_indices", None)
    if col_indices is None:
        on_terrain = _check_terrain_types_from_positions(env, asset.data.root_pos_w, terrain_types)
    else:
        on_terrain = _check_terrain_types_from_positions(env, None, terrain_types, col_indices=col_indices)

    in_corridor_x = (local_xy[:, 0] >= x_lo) & (local_xy[:, 0] <= x_hi)
    outside_lateral_boundary = torch.abs(local_xy[:, 1]) > float(max_abs_local_y)
    return on_terrain & in_corridor_x & outside_lateral_boundary


def _continuous_wave_on_terrain_mask(
    env: ManagerBasedRLEnv,
    terrain_types: tuple[str, ...] | list[str],
    asset: Articulation,
) -> torch.Tensor:
    from NoetixE1.tasks.walkrun.noetix_e1.mdp.curriculums import _check_terrain_types_from_positions

    col_indices = getattr(env.unwrapped, "_current_terrain_col_indices", None)
    if col_indices is None:
        return _check_terrain_types_from_positions(env, asset.data.root_pos_w, terrain_types)
    return _check_terrain_types_from_positions(env, None, terrain_types, col_indices=col_indices)


def continuous_wave_slopes_lateral_exit(
    env: ManagerBasedRLEnv,
    terrain_types: tuple[str, ...] | list[str] = ("continuous_wave_slopes",),
    max_abs_root_y_m: float = 1.00,
    max_abs_contact_y_m: float = 1.20,
    contact_force_threshold: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["l_leg_ankle_roll_link", "r_leg_ankle_roll_link"]
    ),
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg(
        "contact_sensor", body_names=".*_ankle_roll_link"
    ),
) -> torch.Tensor:
    """Fail on a root lateral exit or a support-foot contact beyond the terrain edge."""
    if max_abs_root_y_m <= 0.0 or max_abs_contact_y_m <= 0.0:
        raise ValueError("Continuous-wave lateral bounds must be positive.")

    asset: Articulation = env.scene[asset_cfg.name]
    on_terrain = _continuous_wave_on_terrain_mask(env, terrain_types, asset)
    local_root_xy = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    foot_xy = (
        asset.data.body_pos_w[:, asset_cfg.body_ids, :2]
        - env.scene.env_origins[:, :2].unsqueeze(1)
    )

    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    feet_contact = (
        torch.max(
            torch.norm(net_contact_forces[:, :, contact_sensor_cfg.body_ids, :], dim=-1), dim=1
        )[0]
        > float(contact_force_threshold)
    )
    support_contact_outside = (
        feet_contact & (torch.abs(foot_xy[..., 1]) > float(max_abs_contact_y_m))
    ).any(dim=1)
    root_outside = torch.abs(local_root_xy[:, 1]) > float(max_abs_root_y_m)
    return on_terrain & (root_outside | support_contact_outside)


def continuous_wave_slopes_forbidden_contact(
    env: ManagerBasedRLEnv,
    terrain_types: tuple[str, ...] | list[str] = ("continuous_wave_slopes",),
    threshold: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["l_leg_ankle_roll_link", "r_leg_ankle_roll_link"]
    ),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg(
        "contact_sensor",
        body_names=[r"^(?!l_leg_ankle_roll_link$)(?!r_leg_ankle_roll_link$).+$"],
    ),
) -> torch.Tensor:
    """Fail when a non-foot robot body contacts the terrain."""
    asset: Articulation = env.scene[asset_cfg.name]
    on_terrain = _continuous_wave_on_terrain_mask(env, terrain_types, asset)

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    illegal_contact = torch.max(torch.norm(contact_forces, dim=-1), dim=1)[0] > float(threshold)
    return on_terrain & illegal_contact.any(dim=1)


def terrain_out_of_bounds(
    env: ManagerBasedRLEnv,
    asset_cfg: str = "robot",
    distance_buffer: float = 3.0,
    print_reason: bool = False,
) -> torch.Tensor:
    """Terminate when the actor move too close to the edge of the terrain.

    If the actor moves too close to the edge of the terrain, the termination is activated. The distance
    to the edge of the terrain is calculated based on the size of the terrain and the distance buffer.
    """
    if env.scene.cfg.terrain.terrain_type == "plane":
        return torch.zeros(
            (env.num_envs,), device=env.device, dtype=torch.bool
        )  # we have infinite terrain because it is a plane
    elif env.scene.cfg.terrain.terrain_type == "generator":
        # obtain the size of the sub-terrains
        terrain_gen_cfg = env.scene.terrain.cfg.terrain_generator
        grid_width, grid_length = terrain_gen_cfg.size
        n_rows, n_cols = terrain_gen_cfg.num_rows, terrain_gen_cfg.num_cols
        border_width = terrain_gen_cfg.border_width
        # compute the size of the map
        map_width = n_rows * grid_width + 2 * border_width
        map_height = n_cols * grid_length + 2 * border_width

        # extract the used quantities (to enable type-hinting)
        asset: RigidObject = env.scene[asset_cfg]

        # check if the agent is out of bounds
        x_out_of_bounds = torch.abs(asset.data.root_pos_w[:, 0]) > 0.5 * map_width - distance_buffer
        y_out_of_bounds = torch.abs(asset.data.root_pos_w[:, 1]) > 0.5 * map_height - distance_buffer
        return_ = torch.logical_or(x_out_of_bounds, y_out_of_bounds)
        if print_reason and return_.any():
            print("The base is out of the terrain border:", return_.sum())
        return return_
    else:
        raise ValueError("Received unsupported terrain type, must be either 'plane' or 'generator'.")


class illegal_reset_contact(ManagerTermBase):
    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.threshold = cfg.params["threshold"]
        self.sensor_cfg = cfg.params["sensor_cfg"]
        self.print_reason = cfg.params.get("print_reason", False)
        self.episode_length_threshold = cfg.params.get("episode_length_threshold", 1)
        self.illegal_contact_counter = torch.zeros(env.num_envs, device=env.device, dtype=torch.int)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        threshold: float,
        sensor_cfg: SceneEntityCfg,
        print_reason: bool = False,
        episode_length_threshold: int = 1,
    ) -> torch.Tensor:
        """Timeout if the robot is reset with some undesired penetration with the environment.
        within the first episode_length_threshold steps.
        """
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        net_contact_forces = contact_sensor.data.net_forces_w_history
        contacts = torch.any(
            torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold, dim=1
        )
        self.illegal_contact_counter += contacts.int()
        return_ = torch.logical_and(
            self.illegal_contact_counter >= episode_length_threshold,
            env.episode_length_buf <= episode_length_threshold,
        )
        if return_.any() and print_reason:
            print(f"illegal_reset_contact: {return_.sum()} envs")
        return return_

    def reset(self, env_ids: Sequence[int] | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.illegal_contact_counter[env_ids] = 0
