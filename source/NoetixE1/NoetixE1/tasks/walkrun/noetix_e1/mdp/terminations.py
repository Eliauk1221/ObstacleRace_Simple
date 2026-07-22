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


def _s_curve_geometry_per_env(env: ManagerBasedRLEnv) -> tuple[float, torch.Tensor]:
    """读取 ``s_curve`` 当前课程等级的内缘半径和桥宽。"""
    from NoetixE1.terrains.custom_terrain.figure8_geometry import bridge_width_for_runtime_terrain_levels

    terrain_gen_cfg = env.scene.terrain.cfg.terrain_generator
    sub = terrain_gen_cfg.sub_terrains.get("s_curve")
    if sub is None:
        raise KeyError("s_curve sub_terrain not found in terrain_generator.sub_terrains")
    widths = tuple(getattr(sub, "bridge_width_by_level", ()) or ())
    if not widths:
        raise ValueError("s_curve bridge_width_by_level must not be empty")
    bridge_width_m = bridge_width_for_runtime_terrain_levels(
        env.scene.terrain.terrain_levels,
        int(getattr(terrain_gen_cfg, "num_rows", 1)),
        widths,
        getattr(terrain_gen_cfg, "difficulty_range", None),
    )
    return float(sub.inner_radius_m), bridge_width_m


def figure8_bridge_fall_off(
    env: ManagerBasedRLEnv,
    terrain_types: list[str] | tuple[str, ...],
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["l_leg_ankle_roll_link", "r_leg_ankle_roll_link"]
    ),
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
    contact_force_threshold: float = 0.5,
    root_drop_threshold_m: float = 0.06,
) -> torch.Tensor:
    """8 字桥：离开有效区或 root 明显下沉则终止。

    有效区 = 双圆环带 ∪ 中心矩形台地（坑底挖空区）。
    须先在有效区内落地后，满足以下**任一**条件即终止：
    - 双脚 XY 均不在有效区（走到坑底其它区域）；
    - root 相对落地高度下沉超过 ``root_drop_threshold_m``（从桥面跌落）。
    """
    on_terrain = _figure8_on_terrain_mask(env, terrain_types)
    if not on_terrain.any():
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    from NoetixE1.terrains.custom_terrain.figure8_geometry import foot_xy_on_figure8_support_torch

    inner_radius_m, bridge_width_m, cutout_w, cutout_l = _figure8_bridge_geometry_per_env(env)

    # Lazy-init per-env landing cache on the unwrapped env
    device = env.device
    if not hasattr(env.unwrapped, "figure8_landed"):
        env.unwrapped.figure8_landed = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
        env.unwrapped.figure8_root_z_at_landing = torch.zeros(env.num_envs, dtype=torch.float, device=device)

    landed: torch.Tensor = env.unwrapped.figure8_landed
    root_z0: torch.Tensor = env.unwrapped.figure8_root_z_at_landing

    # Clear cache at episode start to avoid cross-episode leakage
    ep0 = env.episode_length_buf == 0
    if ep0.any():
        landed = torch.where(ep0, torch.zeros_like(landed), landed)
        root_z0 = torch.where(ep0, torch.zeros_like(root_z0), root_z0)

    asset: Articulation = env.scene[asset_cfg.name]
    origins_xy = env.scene.env_origins[:, :2].unsqueeze(1)
    foot_xy = asset.data.body_pos_w[:, asset_cfg.body_ids, :2] - origins_xy

    # Determine if feet are contacting (to define "landed")
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    feet_contact = (
        torch.max(torch.norm(net_contact_forces[:, :, contact_sensor_cfg.body_ids, :], dim=-1), dim=1)[0]
        > float(contact_force_threshold)
    )  # (num_envs, 2)
    any_contact = feet_contact.any(dim=1)

    # 有效区：环带 + 中心台地
    left_on = foot_xy_on_figure8_support_torch(
        foot_xy[:, 0], inner_radius_m, bridge_width_m, cutout_w, cutout_l
    )
    right_on = foot_xy_on_figure8_support_torch(
        foot_xy[:, 1], inner_radius_m, bridge_width_m, cutout_w, cutout_l
    )
    in_valid_xy = left_on | right_on
    any_on = in_valid_xy

    # Cache root height at first landing on valid support
    just_landed = (~landed) & any_contact & any_on & on_terrain
    root_z_rel = asset.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    root_z0 = torch.where(just_landed, root_z_rel, root_z0)
    landed = landed | just_landed

    leave_valid = ~in_valid_xy
    dropped = (root_z0 - root_z_rel) > float(root_drop_threshold_m)
    terminate = landed & on_terrain & (leave_valid | dropped)

    # Write back caches
    env.unwrapped.figure8_landed = landed
    env.unwrapped.figure8_root_z_at_landing = root_z0
    return terminate


def s_curve_fall_off(
    env: ManagerBasedRLEnv,
    terrain_types: list[str] | tuple[str, ...],
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["l_leg_ankle_roll_link", "r_leg_ankle_roll_link"]
    ),
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
    contact_force_threshold: float = 0.5,
    root_drop_threshold_m: float = 0.06,
) -> torch.Tensor:
    """S 弯桥：首次在有效半环带落地后，离桥或明显下沉即终止。"""
    on_terrain = _figure8_on_terrain_mask(env, terrain_types)
    if not on_terrain.any():
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    from NoetixE1.terrains.custom_terrain.figure8_geometry import foot_xy_on_s_curve_torch

    inner_radius_m, bridge_width_m = _s_curve_geometry_per_env(env)
    device = env.device
    if not hasattr(env.unwrapped, "s_curve_landed"):
        env.unwrapped.s_curve_landed = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
        env.unwrapped.s_curve_root_z_at_landing = torch.zeros(env.num_envs, dtype=torch.float, device=device)

    landed: torch.Tensor = env.unwrapped.s_curve_landed
    root_z0: torch.Tensor = env.unwrapped.s_curve_root_z_at_landing
    episode_start = env.episode_length_buf == 0
    landed = torch.where(episode_start, torch.zeros_like(landed), landed)
    root_z0 = torch.where(episode_start, torch.zeros_like(root_z0), root_z0)

    asset: Articulation = env.scene[asset_cfg.name]
    foot_xy = asset.data.body_pos_w[:, asset_cfg.body_ids, :2] - env.scene.env_origins[:, None, :2]
    left_on = foot_xy_on_s_curve_torch(foot_xy[:, 0], inner_radius_m, bridge_width_m)
    right_on = foot_xy_on_s_curve_torch(foot_xy[:, 1], inner_radius_m, bridge_width_m)
    any_on = left_on | right_on

    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    feet_contact = (
        torch.max(torch.norm(net_contact_forces[:, :, contact_sensor_cfg.body_ids, :], dim=-1), dim=1)[0]
        > float(contact_force_threshold)
    )
    just_landed = (~landed) & feet_contact.any(dim=1) & any_on & on_terrain
    root_z_rel = asset.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    root_z0 = torch.where(just_landed, root_z_rel, root_z0)
    landed = landed | just_landed

    leave_bridge = ~(left_on | right_on)
    dropped = (root_z0 - root_z_rel) > float(root_drop_threshold_m)
    terminate = landed & on_terrain & (leave_bridge | dropped)
    env.unwrapped.s_curve_landed = landed
    env.unwrapped.s_curve_root_z_at_landing = root_z0
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
