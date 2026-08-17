# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
import numpy as np
import torch
from isaaclab.actuators import ImplicitActuator
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCamera

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Global state for continuous disturbance
_disturbance_state: dict[str, dict] = {}


def _resolve_terrain_rsi_allowed_trajectory_indices(
    motion_files: list[str],
    terrain_specific_excluded_motion_files: dict[str, list[str]],
) -> dict[str, tuple[int, ...]]:
    """Resolve per-terrain RSI trajectory indices after applying file exclusions."""
    all_indices = tuple(range(len(motion_files)))
    allowed_by_terrain = {}
    for terrain_type, excluded_paths in terrain_specific_excluded_motion_files.items():
        missing_paths = [path for path in excluded_paths if path not in motion_files]
        if missing_paths:
            raise ValueError(f"Unknown RSI-excluded motion files for terrain '{terrain_type}': {missing_paths}")

        excluded_indices = {motion_files.index(path) for path in excluded_paths}
        allowed_indices = tuple(index for index in all_indices if index not in excluded_indices)
        if not allowed_indices:
            raise ValueError(f"RSI motion exclusion removed every trajectory for terrain '{terrain_type}'.")
        allowed_by_terrain[terrain_type] = allowed_indices
    return allowed_by_terrain


def _terrain_env_mask(
    terrain_types_for_envs: torch.Tensor,
    col_indices: list[int],
    num_envs: int,
    device: torch.device,
) -> torch.Tensor:
    is_this_terrain = torch.zeros(num_envs, dtype=torch.bool, device=device)
    for col_idx in col_indices:
        is_this_terrain = is_this_terrain | (terrain_types_for_envs == col_idx)
    return is_this_terrain


def _sample_figure8_path_spawn_xy_yaw(
    num_samples: int,
    device: torch.device,
    inner_radius_m: float,
    bridge_width_m: float,
    path_loop_ranges: list[dict],
    position_noise: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """在双圆 8 字中心线指定弧段上采样出生点（env 局部坐标）及切向 yaw。"""
    from NoetixE1.terrains.custom_terrain.figure8_geometry import sample_path_spawn_xy_yaw

    return sample_path_spawn_xy_yaw(
        num_samples, device, inner_radius_m, bridge_width_m, path_loop_ranges, position_noise
    )


def _apply_figure8_path_spawn_to_rand_samples(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    rand_samples: torch.Tensor,
    terrain_types_for_envs: torch.Tensor,
    terrain_type_names_to_col_map: dict[str, list[int]],
    figure8_spawn_cfg: dict,
) -> None:
    """将 figure8 环境在 rand_samples 中的 x, y, yaw 设为桥面上的路径采样点。"""
    if "figure8_bridge" not in terrain_type_names_to_col_map:
        return
    if not figure8_spawn_cfg.get("path_loop_ranges"):
        return

    col_indices = terrain_type_names_to_col_map["figure8_bridge"]
    is_f8 = _terrain_env_mask(terrain_types_for_envs, col_indices, len(env_ids), env.device)
    if not is_f8.any():
        return

    num_match = int(is_f8.sum().item())
    x, y, yaw = _sample_figure8_path_spawn_xy_yaw(
        num_match,
        env.device,
        float(figure8_spawn_cfg.get("inner_radius_m", 2.5)),
        float(figure8_spawn_cfg.get("bridge_width_m", 0.5)),
        list(figure8_spawn_cfg["path_loop_ranges"]),
        float(figure8_spawn_cfg.get("position_noise", 0.15)),
    )
    rand_samples[is_f8, 0] = x
    rand_samples[is_f8, 1] = y
    rand_samples[is_f8, 5] = yaw


def _apply_figure8_center_cutout_spawn_to_rand_samples(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    rand_samples: torch.Tensor,
    terrain_types_for_envs: torch.Tensor,
    terrain_type_names_to_col_map: dict[str, list[int]],
    figure8_spawn_cfg: dict,
) -> None:
    """figure8 环境：在原点矩形挖空区采样出生点与朝向。"""
    if "figure8_bridge" not in terrain_type_names_to_col_map:
        return

    col_indices = terrain_type_names_to_col_map["figure8_bridge"]
    is_f8 = _terrain_env_mask(terrain_types_for_envs, col_indices, len(env_ids), env.device)
    if not is_f8.any():
        return

    from NoetixE1.terrains.custom_terrain.figure8_geometry import sample_center_cutout_spawn_xy_yaw

    num_match = int(is_f8.sum().item())
    x, y, yaw = sample_center_cutout_spawn_xy_yaw(
        num_match,
        env.device,
        float(figure8_spawn_cfg.get("center_cutout_width_m", 0.8)),
        float(figure8_spawn_cfg.get("center_cutout_length_m", 0.8)),
        tuple(figure8_spawn_cfg.get("base_yaw_angles", (1.57, 3.14))),
        float(figure8_spawn_cfg.get("yaw_jitter_deg", 75.0)),
        float(figure8_spawn_cfg.get("position_noise", 0.0)),
    )
    rand_samples[is_f8, 0] = x
    rand_samples[is_f8, 1] = y
    rand_samples[is_f8, 5] = yaw


def _apply_figure8_spawn_to_rand_samples(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    rand_samples: torch.Tensor,
    terrain_types_for_envs: torch.Tensor,
    terrain_type_names_to_col_map: dict[str, list[int]],
    figure8_spawn_cfg: dict,
) -> None:
    """按 ``spawn_mode`` 在 8 字地形上采样出生点。"""
    mode = figure8_spawn_cfg.get("spawn_mode", "path")
    if mode == "center_cutout":
        _apply_figure8_center_cutout_spawn_to_rand_samples(
            env, env_ids, rand_samples, terrain_types_for_envs, terrain_type_names_to_col_map, figure8_spawn_cfg
        )
    elif figure8_spawn_cfg.get("path_loop_ranges"):
        _apply_figure8_path_spawn_to_rand_samples(
            env, env_ids, rand_samples, terrain_types_for_envs, terrain_type_names_to_col_map, figure8_spawn_cfg
        )


def _figure8_spawn_cfg_active(figure8_spawn_cfg: dict | None) -> bool:
    if figure8_spawn_cfg is None:
        return False
    if figure8_spawn_cfg.get("spawn_mode") == "center_cutout":
        return True
    return bool(figure8_spawn_cfg.get("path_loop_ranges"))


def _write_figure8_root_pose_from_spawn(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    positions: torch.Tensor,
    orientations: torch.Tensor,
    rand_samples: torch.Tensor,
    is_f8: torch.Tensor,
    figure8_spawn_cfg: dict,
    default_root_z: float,
) -> None:
    """8 字 env：根位置落在采样点；桥面或中心坑底由 ``spawn_mode`` 决定 Z 偏移。"""
    if not is_f8.any():
        return
    origins = env.scene.env_origins[env_ids]
    positions[is_f8, 0] = origins[is_f8, 0] + rand_samples[is_f8, 0]
    positions[is_f8, 1] = origins[is_f8, 1] + rand_samples[is_f8, 1]
    spawn_on_bridge = figure8_spawn_cfg.get("spawn_mode", "path") != "center_cutout"
    if spawn_on_bridge:
        bridge_h = float(figure8_spawn_cfg.get("bridge_height", 0.1))
        positions[is_f8, 2] = origins[is_f8, 2] + float(default_root_z) + bridge_h
    else:
        # origin.z is aligned with the bridge deck; pit spawns need the deck-to-pit offset.
        bridge_h = float(figure8_spawn_cfg.get("bridge_height", 0.1))
        holes_depth = float(figure8_spawn_cfg.get("holes_depth", 0.0))
        positions[is_f8, 2] = origins[is_f8, 2] + float(default_root_z) - (bridge_h - holes_depth)
    n = int(is_f8.sum().item())
    device = env.device
    roll = torch.zeros(n, device=device)
    pitch = torch.zeros(n, device=device)
    orientations[is_f8] = math_utils.quat_from_euler_xyz(roll, pitch, rand_samples[is_f8, 5])


def _apply_terrain_specific_spawn_z_offset(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    positions: torch.Tensor,
    terrain_specific_spawn_z_offset: dict[str, float] | None,
    device: torch.device,
) -> None:
    """Add a constant z offset on reset for named sub-terrains (compensates flat-motion height on slopes)."""
    if not terrain_specific_spawn_z_offset:
        return
    from NoetixE1.tasks.walkrun.noetix_e1.mdp.curriculums import _adaptive_terrain_state

    env_id_str = str(id(env))
    if env_id_str not in _adaptive_terrain_state:
        return
    terrain_type_names_to_col_map = _adaptive_terrain_state[env_id_str]["terrain_type_names_to_col_map"]
    terrain_types_for_envs = env.scene.terrain.terrain_types[env_ids]
    for terrain_type, dz in terrain_specific_spawn_z_offset.items():
        if terrain_type not in terrain_type_names_to_col_map:
            continue
        col_indices = terrain_type_names_to_col_map[terrain_type]
        is_this_terrain = torch.zeros(len(env_ids), dtype=torch.bool, device=device)
        for col_idx in col_indices:
            is_this_terrain = is_this_terrain | (terrain_types_for_envs == col_idx)
        if is_this_terrain.any():
            positions[is_this_terrain, 2] += float(dz)


def reset_root_state_amp(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    root_pos_range: dict[str, tuple[float, float]],
    root_vel_range: dict[str, tuple[float, float]],
    joint_pos_range: tuple[float, float],
    joint_vel_range: tuple[float, float],
    reference_state_initialization: bool = False,
    prob_rsi: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    terrain_specific_yaw: dict[str, list[float] | tuple[float, float]] | None = None,
    terrain_specific_heading_target: dict[str, dict] | None = None,
    terrain_specific_pos_range: dict[str, dict[str, tuple[float, float]]] | None = None,
    default_motion_files: list[str] | tuple[str, ...] | None = None,
    terrain_specific_motion_files: dict[str, list[str]] | None = None,
    terrain_specific_rsi_excluded_motion_files: dict[str, list[str]] | None = None,
    terrain_specific_prob_rsi: dict[str, float] | None = None,
    terrain_specific_joint_vel_range: dict[str, tuple[float, float]] | None = None,
    terrain_specific_motion_level_range: tuple[int, int] | None = None,
    terrain_specific_reset_height_range: dict[str, tuple[float, float]] | None = None,
    terrain_reset_frames: dict[str, tuple[int, ...]] | None = None,
    terrain_specific_spawn_z_offset: dict[str, float] | None = None,
    terrain_specific_figure8_spawn: dict | None = None,
    terrain_specific_l_curve_bidirectional_spawn: dict[str, dict] | None = None,
):
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # cast env_ids to allow broadcasting
    if asset_cfg.joint_ids != slice(None):
        iter_env_ids = env_ids[:, None]
    else:
        iter_env_ids = env_ids

    # Unified random sampling for x, y, yaw (regardless of AMP motion usage)
    # poses - sample x, y, yaw uniformly
    range_list = [root_pos_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    def _sample_joint_velocity_noise(
        target_env_ids: torch.Tensor,
        shape: torch.Size | tuple[int, ...],
    ) -> torch.Tensor:
        noise = math_utils.sample_uniform(*joint_vel_range, shape, asset.device)
        if not terrain_specific_joint_vel_range:
            return noise

        from NoetixE1.tasks.walkrun.noetix_e1.mdp.curriculums import get_terrain_type_names_to_col_map

        names_to_cols = get_terrain_type_names_to_col_map(env)
        terrain_types = env.scene.terrain.terrain_types[target_env_ids]
        for terrain_name, velocity_range in terrain_specific_joint_vel_range.items():
            columns = names_to_cols.get(terrain_name, ())
            if not columns:
                continue
            terrain_mask = _terrain_env_mask(
                terrain_types, columns, len(target_env_ids), asset.device
            )
            if terrain_mask.any():
                noise[terrain_mask] = math_utils.sample_uniform(
                    *velocity_range,
                    noise[terrain_mask].shape,
                    asset.device,
                )
        return noise

    if terrain_specific_l_curve_bidirectional_spawn:
        from .l_railed_platform import L_CURVE_REVERSE_DIRECTION_ATTR

        reverse_direction = getattr(env.unwrapped, L_CURVE_REVERSE_DIRECTION_ATTR, None)
        if reverse_direction is None:
            reverse_direction = torch.zeros(env.num_envs, dtype=torch.bool, device=asset.device)
            setattr(env.unwrapped, L_CURVE_REVERSE_DIRECTION_ATTR, reverse_direction)
        reverse_direction[env_ids] = False

    # Handle terrain-specific configurations (yaw angles and position ranges)
    if (
        (terrain_specific_yaw is not None and len(terrain_specific_yaw) > 0)
        or (terrain_specific_heading_target is not None and len(terrain_specific_heading_target) > 0)
        or (terrain_specific_pos_range is not None and len(terrain_specific_pos_range) > 0)
    ):
        from NoetixE1.tasks.walkrun.noetix_e1.mdp.curriculums import _adaptive_terrain_state
        
        env_id_str = str(id(env))
        if env_id_str in _adaptive_terrain_state:
            terrain_type_names_to_col_map = _adaptive_terrain_state[env_id_str]["terrain_type_names_to_col_map"]
            
            # Get terrain types for each environment in env_ids
            terrain_types_for_envs = env.scene.terrain.terrain_types[env_ids]  # (len(env_ids),)
            
            # Handle terrain-specific yaw angles
            if terrain_specific_yaw is not None and len(terrain_specific_yaw) > 0:
                # Initialize yaw samples with default random values
                yaw_samples = rand_samples[:, 5].clone()
                
                # For each terrain type with specific yaw angles
                for terrain_type, fixed_yaw_angles in terrain_specific_yaw.items():
                    if terrain_type not in terrain_type_names_to_col_map:
                        continue
                    
                    # Get column indices for this terrain type
                    col_indices = terrain_type_names_to_col_map[terrain_type]
                    
                    # Create mask for environments on this terrain type
                    is_this_terrain = torch.zeros(len(env_ids), dtype=torch.bool, device=asset.device)
                    for col_idx in col_indices:
                        is_this_terrain = is_this_terrain | (terrain_types_for_envs == col_idx)
                    
                    if is_this_terrain.any():
                        num_matching_envs = is_this_terrain.sum().item()
                        if isinstance(fixed_yaw_angles, tuple) and len(fixed_yaw_angles) == 2:
                            yaw_lo, yaw_hi = float(fixed_yaw_angles[0]), float(fixed_yaw_angles[1])
                            yaw_samples[is_this_terrain] = math_utils.sample_uniform(
                                yaw_lo, yaw_hi, (num_matching_envs,), device=asset.device
                            )
                        else:
                            fixed_yaw_tensor = torch.tensor(fixed_yaw_angles, device=asset.device)
                            selected_indices = torch.randint(
                                0, len(fixed_yaw_angles), (num_matching_envs,), device=asset.device
                            )
                            yaw_samples[is_this_terrain] = fixed_yaw_tensor[selected_indices]
                
                # Replace yaw in rand_samples
                rand_samples[:, 5] = yaw_samples
            
            # Handle terrain-specific position ranges (x, y)
            if terrain_specific_pos_range is not None and len(terrain_specific_pos_range) > 0:
                # Initialize x, y samples with default random values
                x_samples = rand_samples[:, 0].clone()
                y_samples = rand_samples[:, 1].clone()
                
                # For each terrain type with specific position ranges
                for terrain_type, pos_range in terrain_specific_pos_range.items():
                    if terrain_type not in terrain_type_names_to_col_map:
                        continue
                    
                    # Get column indices for this terrain type
                    col_indices = terrain_type_names_to_col_map[terrain_type]
                    
                    # Create mask for environments on this terrain type
                    is_this_terrain = torch.zeros(len(env_ids), dtype=torch.bool, device=asset.device)
                    for col_idx in col_indices:
                        is_this_terrain = is_this_terrain | (terrain_types_for_envs == col_idx)
                    
                    # For environments on this terrain, sample from specific position ranges
                    if is_this_terrain.any():
                        num_matching_envs = is_this_terrain.sum().item()
                        
                        # Sample x from terrain-specific range if provided
                        if "x" in pos_range:
                            x_range = pos_range["x"]
                            x_samples[is_this_terrain] = math_utils.sample_uniform(
                                x_range[0], x_range[1], (num_matching_envs,), device=asset.device
                            )
                        
                        # Sample y from terrain-specific range if provided
                        if "y" in pos_range:
                            y_range = pos_range["y"]
                            y_samples[is_this_terrain] = math_utils.sample_uniform(
                                y_range[0], y_range[1], (num_matching_envs,), device=asset.device
                            )
                
                # Replace x, y in rand_samples
                rand_samples[:, 0] = x_samples
                rand_samples[:, 1] = y_samples

            # Apply heading targets after position sampling so the yaw points
            # from each selected spawn pose toward its configured target.
            if terrain_specific_heading_target:
                for terrain_type, target_cfg in terrain_specific_heading_target.items():
                    if terrain_type not in terrain_type_names_to_col_map:
                        continue
                    if not isinstance(target_cfg, dict):
                        raise ValueError(
                            "terrain-specific heading targets must be dictionaries, "
                            f"got {target_cfg!r} for {terrain_type!r}."
                        )
                    if "target_x" not in target_cfg or "target_y_range" not in target_cfg:
                        raise ValueError(
                            "terrain-specific heading targets must define target_x "
                            f"and target_y_range, got {target_cfg!r} for {terrain_type!r}."
                        )

                    target_y_range = target_cfg["target_y_range"]
                    if not isinstance(target_y_range, (tuple, list)) or len(target_y_range) != 2:
                        raise ValueError(
                            "terrain-specific target_y_range must contain two values, "
                            f"got {target_y_range!r} for {terrain_type!r}."
                        )
                    target_y_low, target_y_high = sorted(float(value) for value in target_y_range)

                    yaw_offset_range = target_cfg.get("yaw_offset_range", (0.0, 0.0))
                    if not isinstance(yaw_offset_range, (tuple, list)) or len(yaw_offset_range) != 2:
                        raise ValueError(
                            "terrain-specific yaw_offset_range must contain two values, "
                            f"got {yaw_offset_range!r} for {terrain_type!r}."
                        )
                    yaw_offset_low, yaw_offset_high = (float(value) for value in yaw_offset_range)

                    is_this_terrain = torch.zeros(len(env_ids), dtype=torch.bool, device=asset.device)
                    for col_idx in terrain_type_names_to_col_map[terrain_type]:
                        is_this_terrain |= terrain_types_for_envs == col_idx

                    outside_spawn_y_range = target_cfg.get("outside_spawn_y_range")
                    if outside_spawn_y_range is not None:
                        if (
                            not isinstance(outside_spawn_y_range, (tuple, list))
                            or len(outside_spawn_y_range) != 2
                        ):
                            raise ValueError(
                                "terrain-specific outside_spawn_y_range must contain two values, "
                                f"got {outside_spawn_y_range!r} for {terrain_type!r}."
                            )
                        central_y_low, central_y_high = sorted(
                            float(value) for value in outside_spawn_y_range
                        )
                        spawn_y = rand_samples[:, 1]
                        is_this_terrain &= (spawn_y < central_y_low) | (spawn_y > central_y_high)
                    if not is_this_terrain.any():
                        continue

                    spawn_x = rand_samples[is_this_terrain, 0]
                    spawn_y = rand_samples[is_this_terrain, 1]
                    target_x = torch.full_like(spawn_x, float(target_cfg["target_x"]))
                    target_y = spawn_y.clamp(target_y_low, target_y_high)
                    yaw_samples = torch.atan2(target_y - spawn_y, target_x - spawn_x)
                    yaw_samples += math_utils.sample_uniform(
                        yaw_offset_low,
                        yaw_offset_high,
                        yaw_samples.shape,
                        device=asset.device,
                    )
                    rand_samples[is_this_terrain, 5] = torch.atan2(
                        torch.sin(yaw_samples), torch.cos(yaw_samples)
                    )

    if terrain_specific_l_curve_bidirectional_spawn:
        from NoetixE1.tasks.walkrun.noetix_e1.mdp.curriculums import _adaptive_terrain_state
        from .l_railed_platform import L_CURVE_REVERSE_DIRECTION_ATTR, l_railed_platform_geometry

        state = _adaptive_terrain_state.get(str(id(env)))
        if state is not None:
            names_to_cols = state["terrain_type_names_to_col_map"]
            terrain_types = env.scene.terrain.terrain_types[env_ids]
            reverse_direction = getattr(env.unwrapped, L_CURVE_REVERSE_DIRECTION_ATTR)
            for terrain_type, spawn_cfg in terrain_specific_l_curve_bidirectional_spawn.items():
                columns = names_to_cols.get(terrain_type, ())
                if not columns:
                    continue
                on_terrain = torch.zeros(len(env_ids), dtype=torch.bool, device=asset.device)
                for column in columns:
                    on_terrain |= terrain_types == column
                reverse_probability = float(spawn_cfg.get("reverse_probability", 0.5))
                if not 0.0 <= reverse_probability <= 1.0:
                    raise ValueError("L-curve reverse_probability must be in [0, 1].")
                reverse_spawn = on_terrain & (
                    torch.rand(len(env_ids), device=asset.device) < reverse_probability
                )
                reverse_direction[env_ids[on_terrain]] = reverse_spawn[on_terrain]
                count = int(reverse_spawn.sum().item())
                if count == 0:
                    continue
                geometry = l_railed_platform_geometry(
                    env, spawn_cfg.get("sub_terrain_key", terrain_type)
                )
                longitudinal = spawn_cfg.get("longitudinal_offset_range", (-0.1, 0.3))
                lateral = spawn_cfg.get("lateral_offset_range", (-0.1, 0.1))
                yaw_jitter = spawn_cfg.get(
                    "yaw_jitter_range", (-0.2617993877991494, 0.2617993877991494)
                )
                longitudinal_sample = math_utils.sample_uniform(
                    *longitudinal, (count,), device=asset.device
                )
                lateral_sample = math_utils.sample_uniform(*lateral, (count,), device=asset.device)
                yaw_sample = math_utils.sample_uniform(*yaw_jitter, (count,), device=asset.device)
                rand_samples[reverse_spawn, 0] = geometry.corner_x_m + lateral_sample
                rand_samples[reverse_spawn, 1] = (
                    geometry.route_end_y_m + geometry.route_start_x_m - longitudinal_sample
                )
                reverse_yaw = -0.5 * torch.pi + yaw_sample
                rand_samples[reverse_spawn, 5] = torch.atan2(
                    torch.sin(reverse_yaw), torch.cos(reverse_yaw)
                )

    if _figure8_spawn_cfg_active(terrain_specific_figure8_spawn):
        from NoetixE1.tasks.walkrun.noetix_e1.mdp.curriculums import _adaptive_terrain_state

        env_id_str = str(id(env))
        if env_id_str in _adaptive_terrain_state:
            terrain_type_names_to_col_map = _adaptive_terrain_state[env_id_str]["terrain_type_names_to_col_map"]
            terrain_types_for_envs = env.scene.terrain.terrain_types[env_ids]
            _apply_figure8_spawn_to_rand_samples(
                env,
                env_ids,
                rand_samples,
                terrain_types_for_envs,
                terrain_type_names_to_col_map,
                terrain_specific_figure8_spawn,
            )

    rsi_draw = torch.rand(1, dtype=torch.float32, device=env.device)
    rsi_probabilities = torch.full(
        (len(env_ids),), float(prob_rsi), dtype=torch.float32, device=env.device
    )
    if terrain_specific_prob_rsi:
        from NoetixE1.tasks.walkrun.noetix_e1.mdp.curriculums import get_terrain_type_names_to_col_map

        terrain_names_to_cols = get_terrain_type_names_to_col_map(env)
        terrain_types_for_envs = env.scene.terrain.terrain_types[env_ids]
        for terrain_name, terrain_prob_rsi in terrain_specific_prob_rsi.items():
            terrain_prob_rsi = float(terrain_prob_rsi)
            if not 0.0 <= terrain_prob_rsi <= 1.0:
                raise ValueError(
                    f"terrain-specific prob_rsi for '{terrain_name}' must be in [0, 1], got {terrain_prob_rsi}"
                )
            columns = terrain_names_to_cols.get(terrain_name, ())
            if columns:
                terrain_mask = _terrain_env_mask(
                    terrain_types_for_envs, columns, len(env_ids), asset.device
                )
                rsi_probabilities[terrain_mask] = terrain_prob_rsi
    rsi_mask = rsi_draw < rsi_probabilities

    if env.unwrapped.motion_loader is not None and reference_state_initialization and rsi_mask.any():
        motion_loader = env.unwrapped.motion_loader

        def _sample_full_frames_at_reset_keyframes(
            num_frames: int,
            allowed_traj_idxs: tuple[int, ...] | list[int],
            terrain_name: str,
        ) -> torch.Tensor:
            keyframes = (terrain_reset_frames or {}).get(terrain_name, (0,)) or (0,)
            trajectory_pool = np.asarray(allowed_traj_idxs, dtype=np.int64)
            trajectory_indices = np.random.choice(trajectory_pool, size=num_frames, replace=True)
            frame_indices = np.random.choice(
                np.asarray(keyframes, dtype=np.int64), size=num_frames, replace=True
            )
            max_frame_indices = np.maximum(
                motion_loader.trajectory_num_frames[trajectory_indices].astype(np.int64) - 1,
                0,
            )
            frame_indices = np.minimum(frame_indices, max_frame_indices)
            frame_times = (
                frame_indices.astype(np.float32)
                * motion_loader.trajectory_frame_durations[trajectory_indices].astype(np.float32)
            )
            return motion_loader.get_full_frame_at_time_batch(trajectory_indices, frame_times)

        # get frames of amp motions (optionally from terrain-specific motion files)
        use_full_motion_state = torch.zeros(len(env_ids), dtype=torch.bool, device=env.device)
        terrain_specific_motion_files = terrain_specific_motion_files or {}
        terrain_specific_rsi_excluded_motion_files = terrain_specific_rsi_excluded_motion_files or {}
        motion_files_list = list(getattr(env.unwrapped.cfg, "motion_files", []) or [])
        default_motion_indices: tuple[int, ...] | None = None
        if default_motion_files:
            missing_default_files = [path for path in default_motion_files if path not in motion_files_list]
            if missing_default_files:
                raise ValueError(f"Unknown default RSI motion files: {missing_default_files}")
            default_motion_indices = tuple(motion_files_list.index(path) for path in default_motion_files)
        has_terrain_specific_rsi_sampling = bool(
            terrain_specific_motion_files or terrain_specific_rsi_excluded_motion_files
        )
        if has_terrain_specific_rsi_sampling and motion_files_list:
            from NoetixE1.tasks.walkrun.noetix_e1.mdp.curriculums import get_terrain_type_names_to_col_map

            terrain_to_indices = {}
            for terrain_type, paths in terrain_specific_motion_files.items():
                indices = []
                for p in paths:
                    try:
                        i = motion_files_list.index(p)
                        indices.append(i)
                    except ValueError:
                        pass
                if indices:
                    terrain_to_indices[terrain_type] = indices

            terrain_to_rsi_allowed_indices = _resolve_terrain_rsi_allowed_trajectory_indices(
                motion_files_list, terrain_specific_rsi_excluded_motion_files
            )

            # 仅在指定地形等级范围内启用“指定地形用指定 motion”；None 表示全部等级启用
            if terrain_specific_motion_level_range is not None:
                level_min, level_max = terrain_specific_motion_level_range
                terrain_levels_for_envs = env.scene.terrain.terrain_levels[env_ids]
                level_ok = (terrain_levels_for_envs >= level_min) & (terrain_levels_for_envs <= level_max)
            else:
                level_ok = torch.ones(len(env_ids), dtype=torch.bool, device=env.device)

            terrain_type_names_to_col_map = get_terrain_type_names_to_col_map(env)
            if terrain_type_names_to_col_map:
                terrain_types_for_envs = env.scene.terrain.terrain_types[env_ids]
                col_to_terrain_name = {}
                for tname, cols in terrain_type_names_to_col_map.items():
                    for c in cols:
                        col_to_terrain_name[c] = tname
                groups = {}
                for i in range(len(env_ids)):
                    col = terrain_types_for_envs[i].item()
                    tname = col_to_terrain_name.get(col, None)
                    if tname not in groups:
                        groups[tname] = []
                    groups[tname].append(i)
                frame_dim = env.motion_loader.preloaded_states.shape[2]
                frames = torch.empty(
                    (len(env_ids), frame_dim), device=env.device, dtype=env.motion_loader.preloaded_states.dtype
                )
                all_trajectory_indices = tuple(range(len(motion_files_list)))
                default_indices = default_motion_indices or all_trajectory_indices
                for tname, local_indices in groups.items():
                    terrain_allowed = terrain_to_rsi_allowed_indices.get(tname, all_trajectory_indices)
                    rsi_allowed = tuple(index for index in default_indices if index in terrain_allowed)
                    if not rsi_allowed:
                        raise ValueError(f"No default RSI motions remain available for terrain '{tname}'.")
                    excluded = set(all_trajectory_indices) - set(terrain_allowed)

                    full_state_allowed = terrain_to_indices.get(tname) if tname else None
                    if full_state_allowed is not None:
                        full_state_allowed = tuple(index for index in full_state_allowed if index not in excluded)
                        if not full_state_allowed:
                            raise ValueError(
                                f"RSI motion exclusion removed every full-state trajectory for terrain '{tname}'."
                            )

                    if full_state_allowed is not None and len(full_state_allowed) > 0:
                        # In-range samples use the full motion state; others keep the normal reset pose.
                        level_ok_sub = level_ok[local_indices]
                        in_range_where = torch.where(level_ok_sub)[0]
                        out_range_where = torch.where(~level_ok_sub)[0]
                        if in_range_where.numel() > 0:
                            batch_in = _sample_full_frames_at_reset_keyframes(
                                int(in_range_where.numel()), full_state_allowed, tname
                            )
                            for j, pos in enumerate(in_range_where):
                                li = local_indices[pos.item()]
                                frames[li] = batch_in[j]
                                use_full_motion_state[li] = True
                        if out_range_where.numel() > 0:
                            batch_out = env.motion_loader.get_full_frame_batch(
                                out_range_where.numel(), allowed_traj_idxs=rsi_allowed
                            )
                            for j, pos in enumerate(out_range_where):
                                li = local_indices[pos.item()]
                                frames[li] = batch_out[j]
                    else:
                        batch = env.motion_loader.get_full_frame_batch(
                            len(local_indices), allowed_traj_idxs=rsi_allowed
                        )
                        for j, li in enumerate(local_indices):
                            frames[li] = batch[j]
            else:
                if terrain_specific_rsi_excluded_motion_files:
                    raise RuntimeError("Cannot apply terrain-specific RSI motion exclusions without terrain mapping.")
                frames = env.motion_loader.get_full_frame_batch(
                    len(env_ids), allowed_traj_idxs=default_motion_indices
                )
        else:
            frames = env.motion_loader.get_full_frame_batch(
                len(env_ids), allowed_traj_idxs=default_motion_indices
            )

        use_full_motion_state &= rsi_mask

        # get root state of amp motions (raw: motion frame, no origin)
        raw_amp_positions = env.unwrapped.motion_loader_class.get_root_pos_batch(frames)
        quat_xyzw = env.unwrapped.motion_loader_class.get_root_rot_batch(frames)  # xyzw
        amp_orientations = torch.cat((quat_xyzw[:, -1].unsqueeze(1), quat_xyzw[:, :-1]), dim=1)  # xyzw -> wxyz

        # Positions: 地形专属 motion 时完全采用 motion 位置 + 地形原点偏置；否则在原点附近随机 x,y，z 用 motion
        positions = torch.empty((len(env_ids), 3), device=env.device, dtype=raw_amp_positions.dtype)
        origins = env.scene.env_origins[env_ids]
        positions[~use_full_motion_state, 0] = origins[~use_full_motion_state, 0] + rand_samples[~use_full_motion_state, 0]
        positions[~use_full_motion_state, 1] = origins[~use_full_motion_state, 1] + rand_samples[~use_full_motion_state, 1]
        positions[~use_full_motion_state, 2] = raw_amp_positions[~use_full_motion_state, 2] + origins[~use_full_motion_state, 2] + 0.05
        positions[use_full_motion_state, :2] = raw_amp_positions[use_full_motion_state, :2] + origins[use_full_motion_state, :2]
        positions[use_full_motion_state, 2] = raw_amp_positions[use_full_motion_state, 2] + 0.05

        # 地形专属 reset 高度：按 difficulty 在配置的 (height_min, height_max) 内插值，仅对 use_full_motion_state 且地形在配置内的 env 生效
        if (
            terrain_specific_reset_height_range
            and use_full_motion_state.any()
        ):
            from NoetixE1.tasks.walkrun.noetix_e1.mdp.curriculums import _adaptive_terrain_state

            env_id_str = str(id(env))
            if env_id_str in _adaptive_terrain_state:
                terrain_type_names_to_col_map = _adaptive_terrain_state[env_id_str]["terrain_type_names_to_col_map"]
                terrain_types_for_envs = env.scene.terrain.terrain_types[env_ids]
                terrain_levels_for_envs = env.scene.terrain.terrain_levels[env_ids].float()
                num_rows = getattr(
                    getattr(env.scene.terrain.cfg, "terrain_generator", None), "num_rows", 10
                )
                difficulty = terrain_levels_for_envs / max(1, num_rows - 1)
                col_to_terrain_name = {}
                for tname, cols in terrain_type_names_to_col_map.items():
                    for c in cols:
                        col_to_terrain_name[c] = tname
                for tname, (height_min, height_max) in terrain_specific_reset_height_range.items():
                    if tname not in terrain_type_names_to_col_map:
                        continue
                    col_indices = terrain_type_names_to_col_map[tname]
                    is_this_terrain = torch.zeros(len(env_ids), dtype=torch.bool, device=env.device)
                    for c in col_indices:
                        is_this_terrain = is_this_terrain | (terrain_types_for_envs == c)
                    mask = use_full_motion_state & is_this_terrain
                    if mask.any():
                        height_above = height_min + difficulty[mask] * (height_max - height_min)
                        positions[mask, 2] = positions[mask, 2] + height_above

        # Orientations: 地形专属 motion 时完全采用 motion 朝向；否则 roll/pitch 用 motion，yaw 用 rand_samples
        roll, pitch, _ = math_utils.euler_xyz_from_quat(amp_orientations)
        orientations = torch.empty((len(env_ids), 4), device=env.device, dtype=amp_orientations.dtype)
        orientations[~use_full_motion_state] = math_utils.quat_from_euler_xyz(
            roll[~use_full_motion_state], pitch[~use_full_motion_state], rand_samples[~use_full_motion_state, 5]
        )
        orientations[use_full_motion_state] = amp_orientations[use_full_motion_state]

        # base root velocities (body frame from motion) -> world
        base_lin_vel = env.unwrapped.motion_loader_class.get_linear_vel_batch(frames)
        base_ang_vel = env.unwrapped.motion_loader_class.get_angular_vel_batch(frames)
        lin_vel = math_utils.quat_apply(orientations, base_lin_vel)
        ang_vel = math_utils.quat_apply(orientations, base_ang_vel)
        velocities = torch.cat([lin_vel, ang_vel], dim=-1)

        # get joint state of amp motions
        joint_pos = env.unwrapped.motion_loader_class.get_joint_pose_batch(frames)[
            ..., env.unwrapped.motion_loader.joint_mapping["motion2sim"]
        ]
        joint_vel = env.unwrapped.motion_loader_class.get_joint_vel_batch(frames)[
            ..., env.unwrapped.motion_loader.joint_mapping["motion2sim"]
        ]

        # Terrain-specific RSI probabilities may leave part of this reset batch on the normal reset path.
        no_rsi_mask = ~rsi_mask
        if no_rsi_mask.any():
            no_rsi_env_ids = env_ids[no_rsi_mask]
            root_states = asset.data.default_root_state[no_rsi_env_ids].clone()
            positions[no_rsi_mask] = (
                root_states[:, 0:3]
                + env.scene.env_origins[no_rsi_env_ids]
                + rand_samples[no_rsi_mask, 0:3]
            )
            orientations_delta = math_utils.quat_from_euler_xyz(
                rand_samples[no_rsi_mask, 3],
                rand_samples[no_rsi_mask, 4],
                rand_samples[no_rsi_mask, 5],
            )
            orientations[no_rsi_mask] = math_utils.quat_mul(
                root_states[:, 3:7], orientations_delta
            )
            velocity_ranges = torch.tensor(
                [root_vel_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]],
                device=asset.device,
            )
            velocity_noise = math_utils.sample_uniform(
                velocity_ranges[:, 0],
                velocity_ranges[:, 1],
                (int(no_rsi_mask.sum().item()), 6),
                device=asset.device,
            )
            velocities[no_rsi_mask] = root_states[:, 7:13] + velocity_noise
            default_joint_pos = asset.data.default_joint_pos[no_rsi_env_ids][:, asset_cfg.joint_ids].clone()
            default_joint_vel = asset.data.default_joint_vel[no_rsi_env_ids][:, asset_cfg.joint_ids].clone()
            default_joint_pos += math_utils.sample_uniform(
                *joint_pos_range, default_joint_pos.shape, default_joint_pos.device
            )
            default_joint_vel += _sample_joint_velocity_noise(
                no_rsi_env_ids, default_joint_vel.shape
            )
            joint_pos[no_rsi_mask] = default_joint_pos
            joint_vel[no_rsi_mask] = default_joint_vel
    else:
        # get default root state
        root_states = asset.data.default_root_state[env_ids].clone()

        # Use unified random sampling for x, y, yaw (already processed above)
        # Build positions and orientations using rand_samples
        positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
        orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
        # base root velocities
        range_list = [root_vel_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=asset.device)
        rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

        # world root velocities
        velocities = root_states[:, 7:13] + rand_samples
        lin_vel = velocities[:, 0:3]
        ang_vel = velocities[:, 3:6]
        # base root velocities
        base_lin_vel = math_utils.quat_apply_inverse(orientations, lin_vel)
        base_ang_vel = math_utils.quat_apply_inverse(orientations, ang_vel)

        # get default joint state
        joint_pos = asset.data.default_joint_pos[iter_env_ids, asset_cfg.joint_ids].clone()
        joint_vel = asset.data.default_joint_vel[iter_env_ids, asset_cfg.joint_ids].clone()

        # bias these values randomly
        joint_pos += math_utils.sample_uniform(*joint_pos_range, joint_pos.shape, joint_pos.device)
        joint_vel += _sample_joint_velocity_noise(env_ids, joint_vel.shape)

    if _figure8_spawn_cfg_active(terrain_specific_figure8_spawn):
        from NoetixE1.tasks.walkrun.noetix_e1.mdp.curriculums import _adaptive_terrain_state

        env_id_str = str(id(env))
        if env_id_str in _adaptive_terrain_state:
            terrain_type_names_to_col_map = _adaptive_terrain_state[env_id_str]["terrain_type_names_to_col_map"]
            if "figure8_bridge" in terrain_type_names_to_col_map:
                terrain_types_for_envs = env.scene.terrain.terrain_types[env_ids]
                col_indices = terrain_type_names_to_col_map["figure8_bridge"]
                is_f8 = _terrain_env_mask(
                    terrain_types_for_envs, col_indices, len(env_ids), asset.device
                )
                if is_f8.any():
                    default_root_z = float(asset.data.default_root_state[0, 2].item())
                    _write_figure8_root_pose_from_spawn(
                        env,
                        env_ids,
                        positions,
                        orientations,
                        rand_samples,
                        is_f8,
                        terrain_specific_figure8_spawn,
                        default_root_z,
                    )

    _apply_terrain_specific_spawn_z_offset(
        env, env_ids, positions, terrain_specific_spawn_z_offset, asset.device
    )

    # clamp joint pos to limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[iter_env_ids, asset_cfg.joint_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    # clamp joint vel to limits
    joint_vel_limits = asset.data.soft_joint_vel_limits[iter_env_ids, asset_cfg.joint_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    # set into the physics simulation
    asset.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def randomize_joint_default_pos(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize the joint default positions which may be different from URDF due to calibration errors.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if pos_distribution_params is not None:
        pos = asset.data.default_joint_pos.to(asset.device).clone()
        pos = _randomize_prop_by_op(
            pos, pos_distribution_params, env_ids, joint_ids, operation=operation, distribution=distribution
        )[env_ids][:, joint_ids]

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        asset.data.default_joint_pos[env_ids, joint_ids] = pos
        # update the offset in action since it is not updated automatically
        env.action_manager.get_term("joint_pos")._offset[env_ids, joint_ids] = pos


def randomize_camera_intrinsics(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    sensor_cfg: SceneEntityCfg,
    intrinsic_range: dict[str, tuple[float, float]],
):
    """随机化相机内参矩阵，通过在初始内参的基础上添加随机值来实现域随机化。

    内参矩阵格式：
    [f_x, 0, c_x]
    [0, f_y, c_y]
    [0, 0, 1]

    其中：
    - f_x, f_y: 焦距（focal length）
    - c_x, c_y: 主点坐标（principal point）

    .. note::
        此函数在环境初始化时使用。它会在初始内参的基础上添加随机值，以提高模型的泛化性。

    Args:
        env: 环境实例
        env_ids: 要随机化的环境ID。如果为None，则随机化所有环境
        sensor_cfg: 传感器配置，指定要随机化的相机传感器
        intrinsic_range: 内参随机化范围字典，包含以下键：
            - "f_x": (min, max) - 焦距x的随机化范围（相对于初始值的比例或绝对值）
            - "f_y": (min, max) - 焦距y的随机化范围
            - "c_x": (min, max) - 主点x的随机化范围
            - "c_y": (min, max) - 主点y的随机化范围
    """
    # 提取传感器（启用类型提示）
    camera: RayCasterCamera = env.scene.sensors[sensor_cfg.name]

    # 解析环境ID
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=camera.device)
    else:
        env_ids = env_ids.to(camera.device)

    # 获取当前内参矩阵 (num_envs, 3, 3)
    intrinsic_matrices = camera._data.intrinsic_matrices.clone()

    # 保存初始内参值（如果还没有保存）
    if not hasattr(camera, "_initial_intrinsic_matrices"):
        camera._initial_intrinsic_matrices = intrinsic_matrices.clone()

    # 提取当前内参值
    f_x_current = intrinsic_matrices[:, 0, 0]
    f_y_current = intrinsic_matrices[:, 1, 1]
    c_x_current = intrinsic_matrices[:, 0, 2]
    c_y_current = intrinsic_matrices[:, 1, 2]

    # 为每个环境采样随机值
    num_envs_to_randomize = len(env_ids)

    # 采样焦距随机值
    f_x_range = intrinsic_range.get("f_x", (0.0, 0.0))
    f_y_range = intrinsic_range.get("f_y", (0.0, 0.0))
    f_x_delta = math_utils.sample_uniform(
        f_x_range[0], f_x_range[1], (num_envs_to_randomize,), device=camera.device
    )
    f_y_delta = math_utils.sample_uniform(
        f_y_range[0], f_y_range[1], (num_envs_to_randomize,), device=camera.device
    )

    # 采样主点随机值
    c_x_range = intrinsic_range.get("c_x", (0.0, 0.0))
    c_y_range = intrinsic_range.get("c_y", (0.0, 0.0))
    c_x_delta = math_utils.sample_uniform(
        c_x_range[0], c_x_range[1], (num_envs_to_randomize,), device=camera.device
    )
    c_y_delta = math_utils.sample_uniform(
        c_y_range[0], c_y_range[1], (num_envs_to_randomize,), device=camera.device
    )

    # 应用随机化（在初始值基础上添加）
    intrinsic_matrices[env_ids, 0, 0] = f_x_current[env_ids] + f_x_delta
    intrinsic_matrices[env_ids, 1, 1] = f_y_current[env_ids] + f_y_delta
    intrinsic_matrices[env_ids, 0, 2] = c_x_current[env_ids] + c_x_delta
    intrinsic_matrices[env_ids, 1, 2] = c_y_current[env_ids] + c_y_delta

    # 确保内参矩阵的有效性（焦距和主点应该为正）
    intrinsic_matrices[env_ids, 0, 0] = torch.clamp(intrinsic_matrices[env_ids, 0, 0], min=1.0)
    intrinsic_matrices[env_ids, 1, 1] = torch.clamp(intrinsic_matrices[env_ids, 1, 1], min=1.0)
    intrinsic_matrices[env_ids, 0, 2] = torch.clamp(intrinsic_matrices[env_ids, 0, 2], min=0.0)
    intrinsic_matrices[env_ids, 1, 2] = torch.clamp(intrinsic_matrices[env_ids, 1, 2], min=0.0)

    # 设置新的内参矩阵
    # 将 env_ids 转换为 list 以匹配 API 要求
    env_ids_list = env_ids.cpu().tolist() if isinstance(env_ids, torch.Tensor) else env_ids
    camera.set_intrinsic_matrices(intrinsic_matrices[env_ids], env_ids=env_ids_list)


def randomize_camera_extrinsics(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    sensor_cfg: SceneEntityCfg,
    position_range: dict[str, tuple[float, float]],
    orientation_range: dict[str, tuple[float, float]],
):
    """随机化相机外参（位置和姿态），通过在初始 offset 的基础上添加随机值来实现域随机化。

    外参包括：
    - 位置（position）：相对于父坐标系（机器人基座）的 x, y, z 偏移（米）
    - 姿态（orientation）：相对于父坐标系的欧拉角 roll, pitch, yaw（弧度）

    .. note::
        此函数在环境初始化时使用。它会在初始 offset 的基础上添加随机值，以提高模型的泛化性。
        通过修改 offset 而不是世界坐标，可以确保相机始终相对于机器人基座保持正确的相对位置。

    Args:
        env: 环境实例
        env_ids: 要随机化的环境ID。如果为None，则随机化所有环境
        sensor_cfg: 传感器配置，指定要随机化的相机传感器
        position_range: 位置随机化范围字典，包含以下键：
            - "x": (min, max) - x方向的随机化范围（米）
            - "y": (min, max) - y方向的随机化范围（米）
            - "z": (min, max) - z方向的随机化范围（米）
        orientation_range: 姿态随机化范围字典，包含以下键：
            - "roll": (min, max) - 绕x轴旋转的随机化范围（弧度）
            - "pitch": (min, max) - 绕y轴旋转的随机化范围（弧度）
            - "yaw": (min, max) - 绕z轴旋转的随机化范围（弧度）
    """
    # 提取传感器（启用类型提示）
    camera: RayCasterCamera = env.scene.sensors[sensor_cfg.name]

    # 解析环境ID
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=camera.device)
    else:
        env_ids = env_ids.to(camera.device)

    # 保存初始外参值（如果还没有保存）
    if not hasattr(camera, "_initial_offset_pos"):
        camera._initial_offset_pos = camera._offset_pos.clone()
        camera._initial_offset_quat = camera._offset_quat.clone()

    # 获取当前的 offset（相对于父坐标系）
    current_offset_pos = camera._offset_pos[env_ids].clone()
    current_offset_quat = camera._offset_quat[env_ids].clone()

    # 为每个环境采样位置随机值
    num_envs_to_randomize = len(env_ids)

    # 采样位置随机偏移（相对于父坐标系）
    pos_range_list = [position_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    pos_ranges = torch.tensor(pos_range_list, device=camera.device)
    pos_deltas = math_utils.sample_uniform(
        pos_ranges[:, 0], pos_ranges[:, 1], (num_envs_to_randomize, 3), device=camera.device
    )

    # 采样姿态随机值（欧拉角）
    ori_range_list = [orientation_range.get(key, (0.0, 0.0)) for key in ["roll", "pitch", "yaw"]]
    ori_ranges = torch.tensor(ori_range_list, device=camera.device)
    ori_deltas = math_utils.sample_uniform(
        ori_ranges[:, 0], ori_ranges[:, 1], (num_envs_to_randomize, 3), device=camera.device
    )

    # 将欧拉角增量转换为四元数
    ori_delta_quat = math_utils.quat_from_euler_xyz(ori_deltas[:, 0], ori_deltas[:, 1], ori_deltas[:, 2])

    # 应用位置随机化（在初始 offset 基础上添加）
    # offset 位置是相对于父坐标系的，直接相加即可
    camera._offset_pos[env_ids] = current_offset_pos + pos_deltas

    # 应用姿态随机化（在初始 offset 姿态基础上旋转）
    # offset 四元数是相对于父坐标系的，需要组合旋转
    camera._offset_quat[env_ids] = math_utils.quat_mul(current_offset_quat, ori_delta_quat)


def randomize_actuator_effort_limit(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    effort_limit_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"] = "scale",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """随机化执行器的 effort_limit_sim。

    此函数允许随机化执行器的 effort_limit_sim 属性。函数从给定的分布参数中采样随机值，
    并根据操作类型（add、scale 或 abs）应用到执行器属性中。

    .. tip::
        对于隐式执行器，此函数使用 CPU 张量来分配执行器的 effort_limit_sim 到仿真中。
        在这种情况下，建议仅在环境初始化时使用此函数。

    Args:
        env: 环境实例
        env_ids: 要随机化的环境ID。如果为None，则随机化所有环境
        asset_cfg: 资产配置，指定要随机化的执行器
        effort_limit_distribution_params: effort_limit 的分布参数 (min, max)
        operation: 操作类型，"add"（添加）、"scale"（缩放）或 "abs"（绝对值）
        distribution: 分布类型，"uniform"（均匀）、"log_uniform"（对数均匀）或 "gaussian"（高斯）
    """
    # 提取资产（启用类型提示）
    asset: Articulation = env.scene[asset_cfg.name]

    # 解析环境ID
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # 遍历所有执行器并随机化 effort_limit_sim
    for actuator in asset.actuators.values():
        if isinstance(asset_cfg.joint_ids, slice):
            # 我们取执行器的所有关节
            actuator_indices = slice(None)
            if isinstance(actuator.joint_indices, slice):
                pass  # 使用 slice(None) 作为 actuator_indices
            elif isinstance(actuator.joint_indices, torch.Tensor):
                pass  # 使用 slice(None) 作为 actuator_indices
            else:
                raise TypeError("执行器关节索引必须是 slice 或 torch.Tensor。")
        elif isinstance(actuator.joint_indices, slice):
            # 我们取资产配置中定义的关节
            actuator_indices = torch.tensor(asset_cfg.joint_ids, device=asset.device)
        else:
            # 我们取执行器关节和资产配置关节的交集
            actuator_joint_indices = actuator.joint_indices
            asset_joint_ids = torch.tensor(asset_cfg.joint_ids, device=asset.device)
            # 需要随机化的执行器中的关节索引
            actuator_indices = torch.nonzero(torch.isin(actuator_joint_indices, asset_joint_ids)).view(-1)
            if len(actuator_indices) == 0:
                continue

        # 随机化 effort_limit_sim
        effort_limit = actuator.effort_limit_sim[env_ids].clone()
        # 从默认值开始应用随机化
        # 这确保在多次调用函数时，随机化应用于默认值而不是之前随机化的值
        # 注意：effort_limit_sim 的形状是 (num_envs, num_joints_in_actuator)
        # 我们需要确保从默认值开始，但如果没有默认值，就使用当前值

        effort_limit = _randomize_prop_by_op(
            effort_limit,
            effort_limit_distribution_params,
            dim_0_ids=None,
            dim_1_ids=actuator_indices,
            operation=operation,
            distribution=distribution,
        )
        actuator.effort_limit_sim[env_ids] = effort_limit

        # 如果是隐式执行器，需要写入仿真
        if isinstance(actuator, ImplicitActuator):
            # 使用 asset 的方法写入 effort_limit 到仿真
            asset.write_joint_effort_limit_to_sim(effort_limit, joint_ids=actuator.joint_indices, env_ids=env_ids)


def apply_continuous_disturbance(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    push_force_range: tuple[float, float],
    push_torque_range: tuple[float, float],
    disturbance_probability: float,
    disturbance_interval: tuple[int, int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Apply continuous external force and torque disturbance to the robot.

    This function implements a continuous disturbance system that:
    1. Randomly starts disturbances with a given probability
    2. Maintains constant force and torque during the disturbance period
    3. Automatically stops after a specified number of steps

    Args:
        env: The learning environment.
        env_ids: The environment IDs to update. If None, updates all environments.
        push_force_range: Range of force magnitude (min, max).
        push_torque_range: Range of torque magnitude (min, max).
        disturbance_probability: Probability of starting a new disturbance per step.
        disturbance_interval: Range of steps to maintain the disturbance (min, max).
        asset_cfg: The asset configuration for the robot.
    """
    # Extract the used quantities
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    device = asset.device
    num_envs = env.scene.num_envs

    # Resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(num_envs, device=device)
    else:
        env_ids = env_ids.to(device)

    # Initialize disturbance state for this environment if not exists
    env_id_str = str(id(env))
    if env_id_str not in _disturbance_state:
        _disturbance_state[env_id_str] = {
            "disturbance_state": torch.zeros(num_envs, dtype=torch.bool, device=device),
            "remaining_steps": torch.zeros(num_envs, dtype=torch.int32, device=device),
            "disturbance_force": torch.zeros(num_envs, 3, device=device),
            "disturbance_torque": torch.zeros(num_envs, 3, device=device),
        }

    state = _disturbance_state[env_id_str]

    # Ensure state tensors match current number of environments
    if state["disturbance_state"].shape[0] != num_envs:
        state["disturbance_state"] = torch.zeros(num_envs, dtype=torch.bool, device=device)
        state["remaining_steps"] = torch.zeros(num_envs, dtype=torch.int32, device=device)
        state["disturbance_force"] = torch.zeros(num_envs, 3, device=device)
        state["disturbance_torque"] = torch.zeros(num_envs, 3, device=device)

    # Update disturbance state and remaining steps
    random_probabilities = torch.rand(num_envs, device=device)
    new_disturbance_mask = (random_probabilities < disturbance_probability) & (~state["disturbance_state"])
    state["remaining_steps"][new_disturbance_mask] = torch.randint(
        disturbance_interval[0],
        disturbance_interval[1] + 1,
        (new_disturbance_mask.sum(),),
        dtype=torch.int32,
        device=device,
    )
    state["remaining_steps"][state["disturbance_state"]] -= 1
    state["disturbance_state"][state["remaining_steps"] <= 0] = False
    state["disturbance_state"][new_disturbance_mask] = True

    # Generate new disturbance force and torque for newly started disturbances
    if new_disturbance_mask.any():
        num_new_disturbances = new_disturbance_mask.sum().item()
        # Sample force: each component independently sampled from range with 50% chance for negative
        new_disturbance_force = torch.where(
            torch.rand(num_new_disturbances, 3, device=device) < 0.5,
            math_utils.sample_uniform(
                -push_force_range[1], -push_force_range[0], (num_new_disturbances, 3), device=device
            ),
            math_utils.sample_uniform(
                push_force_range[0], push_force_range[1], (num_new_disturbances, 3), device=device
            ),
        )

        # Sample torque: each component independently sampled from range with 50% chance for negative
        new_disturbance_torque = torch.where(
            torch.rand(num_new_disturbances, 3, device=device) < 0.5,
            math_utils.sample_uniform(
                -push_torque_range[1], -push_torque_range[0], (num_new_disturbances, 3), device=device
            ),
            math_utils.sample_uniform(
                push_torque_range[0], push_torque_range[1], (num_new_disturbances, 3), device=device
            ),
        )

        # Assign new force and torque to the correct environments
        state["disturbance_force"][new_disturbance_mask] = new_disturbance_force
        state["disturbance_torque"][new_disturbance_mask] = new_disturbance_torque

    # Set force and torque to zero for environments not in disturbance state
    no_disturbance_mask = ~state["disturbance_state"]
    state["disturbance_force"][no_disturbance_mask] = 0.0
    state["disturbance_torque"][no_disturbance_mask] = 0.0

    # Apply force and torque to the specified body (default: root body, body_id=0)
    # Shape: (num_envs, 3) -> (len(env_ids), 1, 3) for set_external_force_and_torque
    forces = state["disturbance_force"][env_ids].unsqueeze(1)  # (len(env_ids), 1, 3)
    torques = state["disturbance_torque"][env_ids].unsqueeze(1)  # (len(env_ids), 1, 3)

    asset.set_external_force_and_torque(
        forces=forces, torques=torques, env_ids=env_ids, body_ids=asset_cfg.body_ids
    )


def register_virtual_obstacle_to_sensor(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    sensor_cfgs: list[SceneEntityCfg] | SceneEntityCfg,
):
    """Make each sensor accessible to the terrain virtual obstacle by providing `sensor.register_virtual_obstacles` with
    `terrain.virtual_obstacles` dict.

    """
    if isinstance(sensor_cfgs, SceneEntityCfg):
        sensor_cfgs = [sensor_cfgs]

    virtual_obstacles: dict = env.scene.terrain.virtual_obstacles

    for sensor_cfg in sensor_cfgs:
        sensor = env.scene[sensor_cfg.name]
        if not hasattr(sensor, "register_virtual_obstacles"):
            raise ValueError(f"Sensor {sensor_cfg.name} does not support virtual obstacles.")

        sensor.register_virtual_obstacles(virtual_obstacles)


def push_by_setting_velocity_terrain_specific(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    terrain_specific_velocity_range: dict[str, dict[str, tuple[float, float]]] | None = None,
):
    """Push the asset by setting the root velocity to a random value within the given ranges.

    This creates an effect similar to pushing the asset with a random impulse that changes the asset's velocity.
    It samples the root velocity from the given ranges and sets the velocity into the physics simulation.

    The function takes a dictionary of velocity ranges for each axis and rotation. The keys of the dictionary
    are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form ``(min, max)``.
    If the dictionary does not contain a key, the velocity is set to zero for that axis.

    Args:
        env: The environment instance.
        env_ids: The environment IDs to apply the push to.
        velocity_range: Default velocity ranges for all axes and rotations.
        asset_cfg: The asset configuration.
        terrain_specific_velocity_range: Optional dictionary mapping terrain type names to velocity range dictionaries.
            Each terrain-specific velocity range can override the default ranges for specific axes.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # velocities
    vel_w = asset.data.root_vel_w[env_ids]

    # Initialize velocity ranges with default values
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)

    # Handle terrain-specific velocity ranges
    if terrain_specific_velocity_range is not None and len(terrain_specific_velocity_range) > 0:
        from NoetixE1.tasks.walkrun.noetix_e1.mdp.curriculums import _adaptive_terrain_state

        env_id_str = str(id(env))
        if env_id_str in _adaptive_terrain_state:
            terrain_type_names_to_col_map = _adaptive_terrain_state[env_id_str]["terrain_type_names_to_col_map"]

            # Get terrain types for each environment in env_ids
            terrain_types_for_envs = env.scene.terrain.terrain_types[env_ids]  # (len(env_ids),)

            # Initialize velocity deltas for all environments (will be filled)
            vel_deltas = torch.zeros((len(env_ids), 6), device=asset.device)
            # Track which environments have been assigned terrain-specific velocities
            assigned_mask = torch.zeros(len(env_ids), dtype=torch.bool, device=asset.device)

            # For each terrain type with specific velocity ranges
            for terrain_type, terrain_velocity_range in terrain_specific_velocity_range.items():
                if terrain_type not in terrain_type_names_to_col_map:
                    continue

                # Get column indices for this terrain type
                col_indices = terrain_type_names_to_col_map[terrain_type]

                # Create mask for environments on this terrain type
                is_this_terrain = torch.zeros(len(env_ids), dtype=torch.bool, device=asset.device)
                for col_idx in col_indices:
                    is_this_terrain = is_this_terrain | (terrain_types_for_envs == col_idx)

                # Only process environments that haven't been assigned yet
                is_this_terrain = is_this_terrain & ~assigned_mask

                # For environments on this terrain, sample from terrain-specific velocity ranges
                if is_this_terrain.any():
                    num_matching_envs = is_this_terrain.sum().item()

                    # Build terrain-specific range list (use terrain-specific if provided, otherwise use default)
                    terrain_range_list = []
                    for idx, key in enumerate(["x", "y", "z", "roll", "pitch", "yaw"]):
                        if key in terrain_velocity_range:
                            terrain_range_list.append(terrain_velocity_range[key])
                        else:
                            terrain_range_list.append(velocity_range.get(key, (0.0, 0.0)))

                    terrain_ranges = torch.tensor(terrain_range_list, device=asset.device)

                    # Sample velocities for this terrain type
                    terrain_vel_deltas = math_utils.sample_uniform(
                        terrain_ranges[:, 0], terrain_ranges[:, 1], (num_matching_envs, 6), device=asset.device
                    )

                    # Store the deltas for this terrain type
                    vel_deltas[is_this_terrain] = terrain_vel_deltas
                    assigned_mask[is_this_terrain] = True

            # Apply default velocities for environments without terrain-specific config
            default_mask = ~assigned_mask
            if default_mask.any():
                default_vel_deltas = math_utils.sample_uniform(
                    ranges[:, 0], ranges[:, 1], (default_mask.sum(), 6), device=asset.device
                )
                vel_deltas[default_mask] = default_vel_deltas

            # Apply velocity deltas
            vel_w += vel_deltas
        else:
            # Fallback to default behavior if terrain state is not available
            vel_w += math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], vel_w.shape, device=asset.device)
    else:
        # Use default velocity ranges for all environments
        vel_w += math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], vel_w.shape, device=asset.device)

    # set the velocities into the physics simulation
    # print(vel_w)
    asset.write_root_velocity_to_sim(vel_w, env_ids=env_ids)
