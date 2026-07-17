# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Global state for adaptive terrain sampling (similar to mimic task's adaptive sampling)
_adaptive_terrain_state: dict[str, dict] = {}


def _get_terrain_type_from_position(
    env: ManagerBasedRLEnv,
    positions_w: torch.Tensor,
) -> torch.Tensor:
    """Get terrain type (column index) for each position based on world coordinates.

    This function computes the terrain grid column index for each position by:
    1. Converting world coordinates to terrain grid coordinates
    2. Computing the column index based on the grid position
    3. Clamping to valid range

    Args:
        env: The learning environment.
        positions_w: World positions of shape (num_positions, 2) or (num_positions, 3).
                     Only x and y coordinates are used.

    Returns:
        A tensor of shape (num_positions,) containing the column index (terrain type) for each position.
        Returns -1 for positions outside the terrain grid.
    """
    if env.scene.cfg.terrain.terrain_type != "generator":
        # For non-generator terrains, return -1 (invalid)
        return torch.full((positions_w.shape[0],), -1, dtype=torch.long, device=positions_w.device)

    terrain_gen_cfg = env.scene.terrain.cfg.terrain_generator
    if not hasattr(terrain_gen_cfg, "num_rows") or not hasattr(terrain_gen_cfg, "num_cols"):
        return torch.full((positions_w.shape[0],), -1, dtype=torch.long, device=positions_w.device)

    # Extract x, y coordinates
    pos_xy = positions_w[:, :2]  # (num_positions, 2)

    # Get terrain grid parameters
    grid_width, grid_length = terrain_gen_cfg.size
    n_rows, n_cols = terrain_gen_cfg.num_rows, terrain_gen_cfg.num_cols

    # Compute total map size
    map_width = n_rows * grid_width
    map_height = n_cols * grid_length

    # Convert world coordinates to grid coordinates
    # Terrain grid is centered at origin, so we shift by half the map size
    grid_x = (pos_xy[:, 0] + map_width / 2.0) / grid_width
    grid_y = (pos_xy[:, 1] + map_height / 2.0) / grid_length

    # Compute row and column indices
    row_indices = torch.floor(grid_x).long()
    col_indices = torch.floor(grid_y).long()

    # Clamp to valid range and check bounds
    row_indices = torch.clamp(row_indices, 0, n_rows - 1)
    col_indices = torch.clamp(col_indices, 0, n_cols - 1)

    # Check if positions are within terrain bounds (excluding border)
    in_bounds_x = (grid_x >= 0) & (grid_x < n_rows)
    in_bounds_y = (grid_y >= 0) & (grid_y < n_cols)
    in_bounds = in_bounds_x & in_bounds_y
    
    # Set invalid positions to -1
    col_indices = torch.where(in_bounds, col_indices, torch.full_like(col_indices, -1))
    
    return col_indices


def _check_terrain_types_from_positions(
    env: ManagerBasedRLEnv,
    positions_w: torch.Tensor,
    desired_terrain_types: tuple[str, ...] | list[str],
    col_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Check if positions are on desired terrain types (batch processing).
    
    Args:
        env: The learning environment.
        positions_w: World positions of shape (num_positions, 2) or (num_positions, 3).
                     Only used if col_indices is None.
        desired_terrain_types: Tuple or list of terrain type names to check.
        col_indices: Optional pre-computed terrain type (column index) tensor of shape (num_positions,).
                    If provided, positions_w is ignored and this is used instead.
                    If None, terrain types are computed from positions_w.
    
    Returns:
        A boolean tensor of shape (num_positions,) indicating which positions are on desired terrain types.
    """
    env_id_str = str(id(env))
    if env_id_str not in _adaptive_terrain_state:
        # If terrain state is not available, return all False
        num_positions = col_indices.shape[0] if col_indices is not None else positions_w.shape[0]
        return torch.zeros(num_positions, dtype=torch.bool, device=positions_w.device)
    
    terrain_type_names_to_col_map = _adaptive_terrain_state[env_id_str]["terrain_type_names_to_col_map"]
    
    # Get terrain type (column index) for each position
    # Use cached col_indices if provided, otherwise compute from positions
    if col_indices is None:
        col_indices = _get_terrain_type_from_position(env, positions_w)  # (num_positions,)
    
    # Build mask for desired terrain types (batch processing)
    is_desired_terrain = torch.zeros(col_indices.shape[0], dtype=torch.bool, device=col_indices.device)
    
    # Collect all column indices for desired terrain types
    desired_col_indices = []
    for terrain_type in desired_terrain_types:
        if terrain_type in terrain_type_names_to_col_map:
            desired_col_indices.extend(terrain_type_names_to_col_map[terrain_type])
    
    if desired_col_indices:
        # Convert to tensor for batch comparison
        desired_cols_tensor = torch.tensor(desired_col_indices, dtype=col_indices.dtype, device=col_indices.device)
        # Use broadcasting to check if any col_index matches desired_cols (batch processing)
        is_desired_terrain = (col_indices.unsqueeze(1) == desired_cols_tensor.unsqueeze(0)).any(dim=1)
    
    return is_desired_terrain


def _build_col_to_terrain_type_map(terrain_gen_cfg) -> dict[int, str]:
    """Build a mapping from column index to terrain type name.

    Args:
        terrain_gen_cfg: Terrain generator configuration.

    Returns:
        Dictionary mapping column indices to terrain type names.
    """
    import numpy as np

    terrain_type_names = list(terrain_gen_cfg.sub_terrains.keys())
    col_to_terrain_type_map = {}

    if terrain_gen_cfg.curriculum:
        proportions = np.array([sub_cfg.proportion for sub_cfg in terrain_gen_cfg.sub_terrains.values()])
        proportions /= np.sum(proportions)
        for col_idx in range(terrain_gen_cfg.num_cols):
            sub_index = np.min(np.where(col_idx / terrain_gen_cfg.num_cols + 0.001 < np.cumsum(proportions))[0])
            col_to_terrain_type_map[col_idx] = terrain_type_names[sub_index]
    else:
        num_terrain_types = len(terrain_type_names)
        for col_idx in range(terrain_gen_cfg.num_cols):
            terrain_type_idx = col_idx % num_terrain_types
            col_to_terrain_type_map[col_idx] = terrain_type_names[terrain_type_idx]

    return col_to_terrain_type_map


def get_terrain_type_names_to_col_map(env: ManagerBasedRLEnv) -> dict[str, list[int]]:
    """子地形名 → 列索引列表。优先用课程状态，否则按 terrain_generator 配置即时构建。"""
    env_id_str = str(id(env))
    if env_id_str in _adaptive_terrain_state:
        return _adaptive_terrain_state[env_id_str]["terrain_type_names_to_col_map"]

    terrain_gen = getattr(env.scene.terrain.cfg, "terrain_generator", None)
    if terrain_gen is None:
        return {}

    col_to_name = _build_col_to_terrain_type_map(terrain_gen)
    names_to_cols: dict[str, list[int]] = {}
    for col_idx, type_name in col_to_name.items():
        names_to_cols.setdefault(type_name, []).append(int(col_idx))
    return names_to_cols

def _initialize_adaptive_state(
    env: ManagerBasedRLEnv,
    terrain_gen_cfg,
    terrain_type_names: list[str],
    adaptive_uniform_ratio: float,
    adaptive_alpha: float,
    min_prob: float,
) -> dict:
    """Initialize adaptive terrain state for the environment.

    Args:
        env: The learning environment.
        terrain_gen_cfg: Terrain generator configuration.
        terrain_type_names: List of terrain type names.
        adaptive_uniform_ratio: Ratio of uniform sampling probability.
        adaptive_alpha: Alpha for the adaptive sampling.
        min_prob: Minimum probability for the adaptive sampling.
    Returns:
        Initialized state dictionary.
    """
    num_rows = terrain_gen_cfg.num_rows
    num_cols = terrain_gen_cfg.num_cols
    num_terrain_types = len(terrain_type_names)
    col_to_terrain_type_map = _build_col_to_terrain_type_map(terrain_gen_cfg)

    terrain_mean_level = {}
    terrain_max_level = {}
    terrain_env_nums = {}
    for terrain_type_name in terrain_type_names:
        terrain_mean_level[terrain_type_name] = 0
        terrain_max_level[terrain_type_name] = 0
        terrain_env_nums[terrain_type_name] = 0

    terrain_type_names_to_col_map = {}
    for col_idx, type_name in col_to_terrain_type_map.items():
        if type_name not in terrain_type_names_to_col_map:
            terrain_type_names_to_col_map[type_name] = []
        terrain_type_names_to_col_map[type_name].append(col_idx)

    state = {
        "num_rows": num_rows,
        "num_cols": num_cols,
        "num_terrain_types": num_terrain_types,
        "col_to_terrain_type_map": col_to_terrain_type_map,
        "terrain_type_names_to_col_map": terrain_type_names_to_col_map,
        "terrain_type_names": terrain_type_names,
        "adaptive_uniform_ratio": adaptive_uniform_ratio,
        "terrain_type_stats": {},
        "terrain_mean_level": terrain_mean_level,
        "terrain_max_level": terrain_max_level,
        "terrain_env_nums": terrain_env_nums,
        "type_sampling_probs": torch.ones(num_terrain_types, dtype=torch.float, device=env.device) / num_terrain_types,  # 初始化为均匀分布
        "adaptive_alpha": adaptive_alpha,  # 低通滤波的平滑系数
        "min_prob": min_prob,  # 最小概率，防止出现0概率
    }
    return state


def _update_terrain_state(
    state: dict,
    terrain,
) -> None:
    """Update terrain state using batch processing.

    Args:
        state: Adaptive terrain state dictionary.
        terrain_types: Current terrain types (column indices) for all environments.
        terrain_levels: Current terrain levels (row indices) for all environments.
    """
    terrain_levels_float = terrain.terrain_levels.float()
    terrain_types = terrain.terrain_types
    device = terrain.terrain_types.device

    # 构建列索引到地形类型名称的映射（反向查找，提高效率）
    col_to_type_name = state["col_to_terrain_type_map"]

    # 批量计算每个地形类型的统计信息
    for terrain_type_name in state["terrain_type_names"]:
        # 找到该地形类型对应的所有列索引
        terrain_cols = [
            col_idx
            for col_idx, type_name in col_to_type_name.items()
            if type_name == terrain_type_name
        ]
        if not terrain_cols:
            state["terrain_mean_level"][terrain_type_name] = 0.0
            state["terrain_max_level"][terrain_type_name] = 0.0
            state["terrain_env_nums"][terrain_type_name] = 0.0
            continue

        # 使用向量化操作找到匹配的环境（批量处理）
        terrain_cols_tensor = torch.tensor(terrain_cols, device=device, dtype=terrain_types.dtype)
        # 使用广播和any操作创建mask
        mask = (terrain_types.unsqueeze(1) == terrain_cols_tensor.unsqueeze(0)).any(dim=1)

        if mask.any():
            # 批量计算平均值、最大值和数量
            matching_levels = terrain_levels_float[mask]
            state["terrain_mean_level"][terrain_type_name] = matching_levels.mean().item()
            state["terrain_max_level"][terrain_type_name] = matching_levels.max().item()
            state["terrain_env_nums"][terrain_type_name] = float(mask.sum().item())
        else:
            state["terrain_mean_level"][terrain_type_name] = 0.0
            state["terrain_max_level"][terrain_type_name] = 0.0
            state["terrain_env_nums"][terrain_type_name] = 0.0


def _compute_type_sampling_probs(state: dict) -> torch.Tensor:
    """Compute adaptive sampling probabilities for terrain types.

    Args:
        state: Adaptive terrain state dictionary.

    Returns:
        Normalized sampling probabilities for each terrain type.
    """
    # 自适应采样的几个原则
    # 1. 先根据当前地形等级均值，求出比例
    # 2. 结合最低保护
    # 3. 低通形式更新采样概率，保证时序平滑
    # 4. 采样概率归一化
    # 5. 采样概率截断，防止出现0概率

    num_terrain_types = state["num_terrain_types"]
    terrain_type_names = state["terrain_type_names"]
    terrain_mean_level = state["terrain_mean_level"]
    adaptive_uniform_ratio = state["adaptive_uniform_ratio"]
    device = state["type_sampling_probs"].device

    # 1. 根据当前地形等级均值，计算目标采样比例
    # 等级越高，说明已经掌握得越好，应该降低采样概率（让机器人更多练习其他地形）
    target_probs = torch.zeros(num_terrain_types, dtype=torch.float, device=device)
    num_rows = state["num_rows"]
    max_level = float(num_rows - 1)  # 最大等级（0-based索引）

    # 计算每个地形类型与最大等级的差值（差值越大，说明还有更多提升空间）
    # 使用 max_level - mean_level + epsilon 来确保差值始终为正
    epsilon = 0.1  # 小的偏移量，避免除零和确保低等级也有采样机会
    total_remaining_space = 0.0
    remaining_spaces = []

    for terrain_type_name in terrain_type_names:
        mean_level = terrain_mean_level.get(terrain_type_name, 0.0)
        # 剩余提升空间 = 最大等级 - 当前平均等级 + epsilon
        remaining_space = max_level - mean_level + epsilon
        remaining_spaces.append(remaining_space)
        total_remaining_space += remaining_space

    # 如果总剩余空间为0（所有地形都达到最高等级），使用均匀分布
    if total_remaining_space < 1e-6:
        target_probs.fill_(1.0 / num_terrain_types)
    else:
        # 根据剩余提升空间计算目标概率（剩余空间越大，概率越大）
        for i, remaining_space in enumerate(remaining_spaces):
            target_probs[i] = remaining_space / total_remaining_space

    # 2. 结合最低保护（uniform ratio）
    # 确保每个地形类型都有最小采样概率
    min_prob_per_type = adaptive_uniform_ratio / float(num_terrain_types)
    target_probs = target_probs * (1.0 - adaptive_uniform_ratio) + min_prob_per_type

    # 3. 低通形式更新采样概率，保证时序平滑（EMA: Exponential Moving Average）
    alpha = state.get("adaptive_alpha", 0.1)
    current_probs = state["type_sampling_probs"]
    # EMA更新: new = alpha * target + (1 - alpha) * old
    new_probs = alpha * target_probs + (1.0 - alpha) * current_probs

    # 4. 采样概率归一化
    new_probs = new_probs / (new_probs.sum() + 1e-8)

    # 5. 采样概率截断，防止出现0概率
    min_prob = state.get("min_prob", 1e-6)
    new_probs = torch.clamp(new_probs, min=min_prob)
    # 重新归一化
    new_probs = new_probs / (new_probs.sum() + 1e-8)

    # 更新state中的采样概率
    state["type_sampling_probs"] = new_probs

    return new_probs


def _decide_terrain_batch(
    state: dict,
    terrain_types: torch.Tensor,
    terrain_levels: torch.Tensor,
    move_up: torch.Tensor,
    move_down: torch.Tensor,
    env: ManagerBasedRLEnv,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decide new terrain types and levels for a batch of environments using vectorized operations.

    Args:
        state: Adaptive terrain state dictionary.
        terrain_types: Current terrain type column indices for all environments.
        terrain_levels: Current terrain level row indices for all environments.
        move_up: Boolean tensor indicating which environments should move up.
        move_down: Boolean tensor indicating which environments should move down.
        env: The learning environment.

    Returns:
        Tuple of (new_terrain_types, new_terrain_levels) as tensors.
    """
    device = terrain_types.device
    max_level = state["num_rows"] - 1
    terrain_mean_level = state["terrain_mean_level"]
    type_sampling_probs = state["type_sampling_probs"]
    col_to_terrain_type_map = state["col_to_terrain_type_map"]

    # 初始化新地形类型和等级（默认保持当前）
    new_types = terrain_types.clone()
    new_levels = terrain_levels.clone()

    # Case 1: move_down - 下降一个等级
    move_down_mask = move_down.bool()
    if move_down_mask.any():
        new_levels[move_down_mask] = torch.clamp(new_levels[move_down_mask] - 1, min=0)

    # Case 2: move_up 且当前已在最高等级 - 需要重新分配地形
    move_up_at_max_mask = move_up.bool() & (terrain_levels >= max_level)
    if move_up_at_max_mask.any():
        num_to_resample = move_up_at_max_mask.sum().item()
        resample_indices = torch.where(move_up_at_max_mask)[0]

        # 批量采样地形类型索引
        sampled_type_indices = torch.multinomial(
            type_sampling_probs.unsqueeze(0).expand(num_to_resample, -1),
            num_samples=1,
        ).squeeze(-1)

        # 构建地形类型名称到列索引的映射（反向查找，取第一个匹配的列）
        type_name_to_col = {}
        for col_idx, type_name in col_to_terrain_type_map.items():
            if type_name not in type_name_to_col:
                type_name_to_col[type_name] = col_idx

        # 批量获取当前地形类型名称和索引
        current_cols = terrain_types[resample_indices]  # Shape: [num_to_resample]
        terrain_type_names = state["terrain_type_names"]
        num_terrain_types = len(terrain_type_names)

        # 构建列索引到地形类型索引的映射（预先构建，避免循环中查找）
        col_to_type_idx = {}
        for col_idx, type_name in col_to_terrain_type_map.items():
            if type_name in terrain_type_names:
                col_to_type_idx[col_idx] = terrain_type_names.index(type_name)

        # 批量查找当前地形类型索引（使用向量化操作）
        current_type_indices = torch.full((num_to_resample,), -1, dtype=torch.long, device=device)
        valid_mask = torch.zeros(num_to_resample, dtype=torch.bool, device=device)

        # 批量查找：对于每个列索引，查找对应的类型索引
        for i in range(num_to_resample):
            col_idx_item = current_cols[i].item()
            if col_idx_item in col_to_type_idx:
                current_type_indices[i] = col_to_type_idx[col_idx_item]
                valid_mask[i] = True

        # 批量检查是否满足重新分配条件（向量化）
        min_prob = state.get("min_prob", 0.02)
        resample_mask = torch.zeros(num_to_resample, dtype=torch.bool, device=device)

        # 批量计算每个地形类型的环境比例（预先计算）
        terrain_env_ratios = torch.zeros(num_terrain_types, dtype=torch.float, device=device)
        for type_idx, type_name in enumerate(terrain_type_names):
            terrain_env_ratios[type_idx] = state["terrain_env_nums"][type_name] / env.num_envs

        # 向量化检查条件
        valid_indices = current_type_indices[valid_mask]
        if len(valid_indices) > 0:
            valid_ratios = terrain_env_ratios[valid_indices]
            resample_mask[valid_mask] = valid_ratios > min_prob

        # 对于不满足条件的环境，随机选择等级
        not_resample_mask = valid_mask & ~resample_mask
        if not_resample_mask.any():
            num_random = not_resample_mask.sum().item()
            random_levels = torch.randint(0, max_level + 1, (num_random,), device=device)
            new_levels[resample_indices[not_resample_mask]] = random_levels

        # 对于满足条件的环境，批量处理地形分配
        if resample_mask.any():
            resample_sub_indices = torch.where(resample_mask)[0]
            resample_env_indices = resample_indices[resample_sub_indices]
            sampled_type_indices_sub = sampled_type_indices[resample_sub_indices]

            # 批量获取采样的地形类型名称和列索引（向量化）
            # 预先构建类型索引到列索引的映射
            type_idx_to_col = torch.zeros(num_terrain_types, dtype=terrain_types.dtype, device=device)
            for type_idx, type_name in enumerate(terrain_type_names):
                type_idx_to_col[type_idx] = type_name_to_col.get(type_name, -1)

            # 批量获取采样的列索引
            sampled_cols = type_idx_to_col[sampled_type_indices_sub]
            # 对于无效的列索引（-1），使用当前列索引
            invalid_col_mask = sampled_cols == -1
            if invalid_col_mask.any():
                sampled_cols[invalid_col_mask] = current_cols[resample_sub_indices][invalid_col_mask]

            # 批量判断是否切换到其他地形
            current_cols_sub = current_cols[resample_sub_indices]
            switch_terrain_mask = sampled_cols != current_cols_sub

            # 批量获取平均等级（预先构建）
            terrain_mean_levels_tensor = torch.zeros(num_terrain_types, dtype=torch.float, device=device)
            for type_idx, type_name in enumerate(terrain_type_names):
                terrain_mean_levels_tensor[type_idx] = terrain_mean_level.get(type_name, 0.0)

            # 批量获取采样的平均等级
            sampled_mean_levels = terrain_mean_levels_tensor[sampled_type_indices_sub]

            # 批量设置新的地形类型和等级
            # 切换到其他地形：使用平均等级
            switch_mask = switch_terrain_mask
            if switch_mask.any():
                mean_levels_rounded = torch.round(sampled_mean_levels[switch_mask]).long()
                mean_levels_clamped = torch.clamp(mean_levels_rounded, min=0, max=max_level)
                new_types[resample_env_indices[switch_mask]] = sampled_cols[switch_mask]
                new_levels[resample_env_indices[switch_mask]] = mean_levels_clamped

            # 保持当前地形：随机选择等级
            stay_mask = ~switch_terrain_mask
            if stay_mask.any():
                num_stay = stay_mask.sum().item()
                random_levels = torch.randint(0, max_level + 1, (num_stay,), device=device)
                new_levels[resample_env_indices[stay_mask]] = random_levels

    # Case 3: move_up 但未达到最高等级 - 上升一个等级
    move_up_not_max_mask = move_up.bool() & (terrain_levels < max_level)
    if move_up_not_max_mask.any():
        new_levels[move_up_not_max_mask] = torch.clamp(new_levels[move_up_not_max_mask] + 1, max=max_level)

    return new_types, new_levels


def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    
    # Get terrain configuration
    terrain_gen_cfg = terrain.cfg.terrain_generator
    
    # Initialize or update _adaptive_terrain_state if terrain generator supports it
    env_id_str = str(id(env))
    if hasattr(terrain_gen_cfg, "sub_terrains") and hasattr(terrain_gen_cfg, "num_cols"):
        terrain_type_names = list(terrain_gen_cfg.sub_terrains.keys())
        if len(terrain_type_names) > 0:
            # Initialize state if not exists
            if env_id_str not in _adaptive_terrain_state:
                state = _initialize_adaptive_state(
                    env,
                    terrain_gen_cfg,
                    terrain_type_names,
                    adaptive_uniform_ratio=0.1,  # Not used in vel curriculum, but required for initialization
                    adaptive_alpha=0.1,  # Not used in vel curriculum, but required for initialization
                    min_prob=0.02,  # Not used in vel curriculum, but required for initialization
                )
                _adaptive_terrain_state[env_id_str] = state
            
            # Update terrain state statistics
            state = _adaptive_terrain_state[env_id_str]
            _update_terrain_state(state, terrain)
    
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain_gen_cfg.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())


def _env_ids_on_terrain_types(
    env: ManagerBasedRLEnv,
    env_ids_tensor: torch.Tensor,
    terrain_types: tuple[str, ...] | list[str],
) -> torch.Tensor:
    """``env_ids`` 子集中哪些机器人当前落在指定地形列上，形状 ``(len(env_ids),)`` bool。"""
    asset: Articulation = env.scene["robot"]
    col_indices = getattr(env.unwrapped, "_current_terrain_col_indices", None)
    if col_indices is not None:
        col_indices = col_indices[env_ids_tensor]
    positions = asset.data.root_link_pos_w[env_ids_tensor]
    return _check_terrain_types_from_positions(env, positions, terrain_types, col_indices=col_indices)

def terrain_levels_vel_with_termination(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    interlaced_corridor_progress_start_x: float = -2.25,
    interlaced_corridor_required_progress_m: float = 5.0,
    interlaced_corridor_max_final_abs_y: float = 0.6,
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

  This function extends terrain_levels_vel with improved logic for handling cases without forward velocity commands.

    **figure8_bridge**
    - move_up：沿 8 字中心线累计弧长 ≥ 单圆半周（``pi * r_c``）且未掉桥
    - move_down：仅 ``figure8_bridge_fall_off``（脚 XY 离开 8 字桥面环带）
    - 弧长由 ``figure8_bridge_path_progress_reward`` 写入 ``env.figure8_episode_arc_m``

    **其它地形**
    - 有前进速度指令：原距离课程
    - 无前进速度指令：超时不变档；仅 ``fall_down_in_some_terrain`` 降档（不含姿态类终止）

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Args:
        env: The learning environment.
        env_ids: The environment indices to update.
        asset_cfg: The asset configuration for the robot.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")

    # Get terrain configuration
    terrain_gen_cfg = terrain.cfg.terrain_generator

    # Initialize or update _adaptive_terrain_state if terrain generator supports it
    env_id_str = str(id(env))
    if hasattr(terrain_gen_cfg, "sub_terrains") and hasattr(terrain_gen_cfg, "num_cols"):
        terrain_type_names = list(terrain_gen_cfg.sub_terrains.keys())
        if len(terrain_type_names) > 0:
            # Initialize state if not exists
            if env_id_str not in _adaptive_terrain_state:
                state = _initialize_adaptive_state(
                    env,
                    terrain_gen_cfg,
                    terrain_type_names,
                    adaptive_uniform_ratio=0.1,  # Not used in vel curriculum, but required for initialization
                    adaptive_alpha=0.1,  # Not used in vel curriculum, but required for initialization
                    min_prob=0.02,  # Not used in vel curriculum, but required for initialization
                )
                _adaptive_terrain_state[env_id_str] = state

            # Update terrain state statistics
            state = _adaptive_terrain_state[env_id_str]
            _update_terrain_state(state, terrain)

    # Convert env_ids to tensor for batch processing
    env_ids_tensor = torch.tensor(env_ids, device=env.device, dtype=torch.long)

    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids_tensor, :2] - env.scene.env_origins[env_ids_tensor, :2], dim=1)
    # Get command velocity magnitude (forward velocity)
    command_vel_magnitude = command[env_ids_tensor, 0]

    # Check if command has forward velocity (threshold: 0.2 m/s to account for numerical errors)
    has_forward_velocity = command_vel_magnitude > 0.2

    # Initialize move_up and move_down tensors
    move_up = torch.zeros(len(env_ids), dtype=torch.bool, device=env.device)
    move_down = torch.zeros(len(env_ids), dtype=torch.bool, device=env.device)

    is_figure8 = _env_ids_on_terrain_types(env, env_ids_tensor, ("figure8_bridge",))
    non_figure8 = ~is_figure8

    # 8 字桥：沿中心线走过半圈（pi*r_c）升档；掉桥降档
    if is_figure8.any():
        f8_local = torch.where(is_figure8)[0]
        f8_env_ids = env_ids_tensor[f8_local]
        fell_off = env.termination_manager.get_term("figure8_bridge_fall_off")[f8_env_ids]

        f8_sub = terrain_gen_cfg.sub_terrains.get("figure8_bridge")
        inner_r = float(getattr(f8_sub, "inner_radius_m", 2.5)) if f8_sub is not None else 2.5
        widths = tuple(getattr(f8_sub, "bridge_width_by_level", (0.8, 0.7, 0.6, 0.5, 0.45)))
        from NoetixE1.terrains.custom_terrain.figure8_geometry import half_loop_arc_length_m

        half_loop = half_loop_arc_length_m(
            terrain.terrain_levels[f8_env_ids],
            int(terrain_gen_cfg.num_rows),
            inner_r,
            widths,
        )
        episode_arc = getattr(env.unwrapped, "figure8_episode_arc_m", None)
        if episode_arc is None:
            arc_ok = torch.zeros(len(f8_local), dtype=torch.bool, device=env.device)
        else:
            arc_ok = episode_arc[f8_env_ids] >= half_loop

        move_up[f8_local] = arc_ok & ~fell_off
        move_down[f8_local] = fell_off

    # Interlaced corridor: success is forward progress from its fixed spawn point.
    # Euclidean distance would incorrectly count lateral drift as task progress.
    is_interlaced = _env_ids_on_terrain_types(env, env_ids_tensor, ("interlaced_corridor_wedges",))
    if is_interlaced.any():
        ic_local = torch.where(is_interlaced)[0]
        ic_env_ids = env_ids_tensor[ic_local]
        ic_has_forward = has_forward_velocity[ic_local]
        ic_local_x = asset.data.root_pos_w[ic_env_ids, 0] - env.scene.env_origins[ic_env_ids, 0]
        ic_local_y = asset.data.root_pos_w[ic_env_ids, 1] - env.scene.env_origins[ic_env_ids, 1]
        ic_progress = ic_local_x - float(interlaced_corridor_progress_start_x)
        ic_success = (
            ic_has_forward
            & (ic_progress > float(interlaced_corridor_required_progress_m))
            & (torch.abs(ic_local_y) <= float(interlaced_corridor_max_final_abs_y))
        )
        move_up[ic_local] = ic_success
        move_down[ic_local] = (
            ic_has_forward
            & ~ic_success
            & (ic_progress < 0.5 * float(interlaced_corridor_required_progress_m))
        )

    # Other non-figure8 terrains retain the distance / velocity curriculum.
    other_non_figure8 = non_figure8 & ~is_interlaced
    if other_non_figure8.any():
        nf_local = torch.where(other_non_figure8)[0]
        has_vel_mask = has_forward_velocity[nf_local]
        if has_vel_mask.any():
            sub = nf_local[torch.where(has_vel_mask)[0]]
            move_up[sub] = distance[sub] > terrain_gen_cfg.size[0] / 2
            move_down_temp = distance[sub] < command_vel_magnitude[sub] * env.max_episode_length_s * 0.5
            move_down_temp *= ~move_up[sub]
            move_down[sub] = move_down_temp

        no_vel_mask = ~has_forward_velocity[nf_local]
        if no_vel_mask.any():
            sub = nf_local[torch.where(no_vel_mask)[0]]
            move_up[sub] = distance[sub] > terrain_gen_cfg.size[0] / 2
            sub_env_ids = env_ids_tensor[sub]
            fell_down = env.termination_manager.get_term("fall_down_in_some_terrain")[sub_env_ids]
            move_down[sub] = fell_down
            move_down[sub] *= ~move_up[sub]

    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())


def terrain_levels_adaptive(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    adaptive_uniform_ratio: float = 0.1,
    adaptive_alpha: float = 0.1,
    min_prob: float = 0.02,
) -> torch.Tensor:
    """Adaptive terrain curriculum based on terrain type and level tracking.

    This function implements an adaptive terrain curriculum inspired by the mimic task's adaptive sampling:
    1. Tracks success/failure for each terrain TYPE (column) and LEVEL (row) separately
    2. When robot successfully passes the highest difficulty level in a terrain type,
       allows adaptive sampling to other terrain types
    3. Retains some probability on the mastered terrain type to prevent forgetting
    4. Uses adaptive sampling to focus on difficult terrain types/levels that need more practice

    Terrain structure:
    - Terrain TYPE (column): stairs, gap, stepping_stones, etc. (terrain_types)
    - Terrain LEVEL (row): difficulty level within each type (0-9) (terrain_levels)

    The key idea is:
    - Track failures per terrain type and level
    - When a terrain type is mastered (highest level passed), allow sampling other types
    - Use adaptive sampling based on failure counts to focus on difficult terrains
    - Retain some probability on mastered terrain to prevent forgetting

    Args:
        env: The learning environment.
        env_ids: The environment IDs to update.
        asset_cfg: The asset configuration for the robot.
        adaptive_uniform_ratio: Ratio of uniform sampling probability. Defaults to 0.1.
        adaptive_alpha: Alpha for the adaptive sampling. Defaults to 0.1.
        min_prob: Minimum probability for the adaptive sampling. Defaults to 0.02.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # Extract the used quantities
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")

    # Get terrain configuration
    terrain_gen_cfg = terrain.cfg.terrain_generator
    if not hasattr(terrain_gen_cfg, "num_rows") or not hasattr(terrain_gen_cfg, "num_cols"):
        return terrain_levels_vel(env, env_ids, asset_cfg)

    if not hasattr(terrain_gen_cfg, "sub_terrains"):
        return terrain_levels_vel(env, env_ids, asset_cfg)

    terrain_type_names = list(terrain_gen_cfg.sub_terrains.keys())
    if len(terrain_type_names) == 0:
        return terrain_levels_vel(env, env_ids, asset_cfg)

    # Initialize or get adaptive terrain state
    env_id_str = str(id(env))
    if env_id_str not in _adaptive_terrain_state:
        state = _initialize_adaptive_state(
            env,
            terrain_gen_cfg,
            terrain_type_names,
            adaptive_uniform_ratio,
            adaptive_alpha,
            min_prob,
        )
        _adaptive_terrain_state[env_id_str] = state

    state = _adaptive_terrain_state[env_id_str]

    # Compute distance walked for terrain level updates
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    move_up = distance > terrain_gen_cfg.size[0] / 2
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up

    # Get current terrain types and levels
    terrain_types = terrain.terrain_types[env_ids].long()
    terrain_levels = terrain.terrain_levels[env_ids].long()

    # update terrain state
    # maybe move to the end of the function
    _update_terrain_state(state, terrain)

    # Compute adaptive sampling probabilities for terrain types
    _compute_type_sampling_probs(state)

    # Decide new terrain types and levels for all environments using batch processing
    new_types, new_levels = _decide_terrain_batch(
        state,
        terrain_types,
        terrain_levels,
        move_up,
        move_down,
        env,
    )

    # Clamp to valid range
    new_levels = torch.clamp(new_levels, 0, state["num_rows"] - 1)
    new_types = torch.clamp(new_types, 0, state["num_cols"] - 1)

    # Update terrain types and levels
    terrain.terrain_types[env_ids] = new_types
    terrain.terrain_levels[env_ids] = new_levels

    return torch.mean(terrain.terrain_levels.float())
