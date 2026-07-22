# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

"""Functions to generate custom terrains with noise."""

from __future__ import annotations

import math

import numpy as np
from typing import TYPE_CHECKING

from isaaclab.terrains.height_field.utils import height_field_to_mesh

if TYPE_CHECKING:
    from . import custom_terrains_cfg


def _resolve_feature_extent(cfg) -> tuple[float, float]:
    """Return the effective feature width and length inside the terrain cell."""
    extent = getattr(cfg, "feature_extent_m", None)
    if extent is not None:
        extent = float(extent)
        return extent, extent
    return float(cfg.size[0]), float(cfg.size[1])


@height_field_to_mesh
def noisy_pyramid_sloped_terrain(
    difficulty: float, cfg: custom_terrains_cfg.HfNoisyPyramidSlopedTerrainCfg
) -> np.ndarray:
    """Generate a terrain with a truncated pyramid structure and added noise.

    This is similar to the standard pyramid sloped terrain, but with random noise added to simulate
    rough or uneven surfaces.

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        The height field of the terrain as a 2D numpy array with discretized heights.
        The shape of the array is (width, length), where width and length are the number of points
        along the x and y axis, respectively.
    """
    # resolve terrain configuration
    if cfg.inverted:
        slope = -cfg.slope_range[0] - difficulty * (cfg.slope_range[1] - cfg.slope_range[0])
    else:
        slope = cfg.slope_range[0] + difficulty * (cfg.slope_range[1] - cfg.slope_range[0])

    # switch parameters to discrete units
    # -- horizontal scale
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    feature_w, feature_h = _resolve_feature_extent(cfg)
    feature_half_w = feature_w / 2.0
    feature_half_h = feature_h / 2.0
    # -- height
    # We want the height to be 1/2 of the feature width since the terrain is a pyramid.
    height_max = int(slope * feature_w / 2 / cfg.vertical_scale)
    # -- center of the terrain
    center_x = int(width_pixels / 2)
    center_y = int(length_pixels / 2)

    # Build a centered feature region; the outer margin stays at ground height.
    h_scale = cfg.horizontal_scale
    ix = np.arange(width_pixels, dtype=np.float64)[:, None]
    iy = np.arange(length_pixels, dtype=np.float64)[None, :]
    px = (ix + 0.5) * h_scale - cfg.size[0] * 0.5
    py = (iy + 0.5) * h_scale - cfg.size[1] * 0.5
    norm_x = np.clip((feature_half_w - np.abs(px)) / max(feature_half_w, 1e-9), 0.0, 1.0)
    norm_y = np.clip((feature_half_h - np.abs(py)) / max(feature_half_h, 1e-9), 0.0, 1.0)
    hf_raw = height_max * norm_x * norm_y

    # create a flat platform at the center of the terrain
    platform_width = int(cfg.platform_width / cfg.horizontal_scale / 2)
    # get the height of the platform at the corner of the platform
    x_pf = width_pixels // 2 - platform_width
    y_pf = length_pixels // 2 - platform_width
    z_pf = hf_raw[x_pf, y_pf]
    hf_raw = np.clip(hf_raw, min(0, z_pf), max(0, z_pf))

    # add noise to the height field
    noise_min = int(cfg.noise_range[0] / cfg.vertical_scale)
    noise_max = int(cfg.noise_range[1] / cfg.vertical_scale)
    noise_step = int(cfg.noise_step / cfg.vertical_scale)
    # create range of noise heights possible
    noise_range = np.arange(noise_min, noise_max + noise_step, noise_step)
    # sample noise randomly
    noise = np.random.choice(noise_range, size=(width_pixels, length_pixels))
    # add noise to the height field
    hf_raw = hf_raw + noise

    # add random obstacles if configured
    if cfg.num_obstacles > 0:
        if isinstance(cfg.obstacle_height, tuple):
            obstacle_height_min = cfg.obstacle_height[0]
            obstacle_height_max = cfg.obstacle_height[1]
        else:
            obstacle_height_min = cfg.obstacle_height
            obstacle_height_max = cfg.obstacle_height
        
        obstacle_height_min_pixels = int(obstacle_height_min / cfg.vertical_scale)
        obstacle_height_max_pixels = int(obstacle_height_max / cfg.vertical_scale)
        
        # Determine obstacle size range
        if isinstance(cfg.obstacle_size, tuple):
            obstacle_size_min = cfg.obstacle_size[0]
            obstacle_size_max = cfg.obstacle_size[1]
        else:
            obstacle_size_min = cfg.obstacle_size
            obstacle_size_max = cfg.obstacle_size
        
        # Convert to pixels
        obstacle_size_min_pixels = int(obstacle_size_min / cfg.horizontal_scale)
        obstacle_size_max_pixels = int(obstacle_size_max / cfg.horizontal_scale)
        
        # Calculate platform bounds to avoid placing obstacles on the platform
        platform_half_width_pixels = int(cfg.platform_width / cfg.horizontal_scale / 2)
        platform_min_x = max(0, center_x - platform_half_width_pixels)
        platform_max_x = min(width_pixels, center_x + platform_half_width_pixels)
        platform_min_y = max(0, center_y - platform_half_width_pixels)
        platform_max_y = min(length_pixels, center_y + platform_half_width_pixels)
        
        # Add obstacles
        for _ in range(cfg.num_obstacles):
            # Random obstacle size
            if obstacle_size_min_pixels == obstacle_size_max_pixels:
                obstacle_size_pixels = obstacle_size_min_pixels
            else:
                obstacle_size_pixels = np.random.randint(obstacle_size_min_pixels, obstacle_size_max_pixels + 1)
            
            # Random position (avoid platform center)
            max_attempts = 100
            for attempt in range(max_attempts):
                # Random position in the terrain
                obstacle_center_x = np.random.randint(obstacle_size_pixels, width_pixels - obstacle_size_pixels)
                obstacle_center_y = np.random.randint(obstacle_size_pixels, length_pixels - obstacle_size_pixels)
                
                # Check if obstacle overlaps with platform
                obstacle_min_x = obstacle_center_x - obstacle_size_pixels // 2
                obstacle_max_x = obstacle_center_x + obstacle_size_pixels // 2
                obstacle_min_y = obstacle_center_y - obstacle_size_pixels // 2
                obstacle_max_y = obstacle_center_y + obstacle_size_pixels // 2
                
                # Skip if overlaps with platform
                if (obstacle_max_x < platform_min_x or obstacle_min_x > platform_max_x or
                    obstacle_max_y < platform_min_y or obstacle_min_y > platform_max_y):
                    # Place obstacle: add height to the terrain at obstacle location
                    obstacle_min_x = max(0, obstacle_min_x)
                    obstacle_max_x = min(width_pixels, obstacle_max_x)
                    obstacle_min_y = max(0, obstacle_min_y)
                    obstacle_max_y = min(length_pixels, obstacle_max_y)
                    
                    # Add obstacle height on top of existing terrain
                    hf_raw[obstacle_min_x:obstacle_max_x, obstacle_min_y:obstacle_max_y] += np.random.randint(obstacle_height_min_pixels, obstacle_height_max_pixels)
                    break

    # round off the heights to the nearest vertical step
    return np.rint(hf_raw).astype(np.int16)


@height_field_to_mesh
def noisy_pyramid_stairs_terrain(
    difficulty: float, cfg: custom_terrains_cfg.HfNoisyPyramidStairsTerrainCfg
) -> np.ndarray:
    """Generate a terrain with a pyramid stair pattern and added noise.

    This creates a height field version of pyramid stairs with random noise added to simulate
    rough or uneven step surfaces.

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        The height field of the terrain as a 2D numpy array with discretized heights.
        The shape of the array is (width, length), where width and length are the number of points
        along the x and y axis, respectively.
    """
    # resolve terrain configuration
    step_height = cfg.step_height_range[0] + difficulty * (cfg.step_height_range[1] - cfg.step_height_range[0])
    if cfg.inverted:
        step_height *= -1
    # switch parameters to discrete units
    # -- terrain
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # -- stairs: use pixel-based width if available to avoid precision loss
    if cfg.step_width_pixels_range is not None:
        # Use integer pixel range (越难越窄: width decreases with difficulty)
        # step_width_pixels_range[0] is width at difficulty=0 (easier, wider)
        # step_width_pixels_range[1] is width at difficulty=1 (harder, narrower)
        step_width = int(
            cfg.step_width_pixels_range[0] 
            + difficulty * (cfg.step_width_pixels_range[1] - cfg.step_width_pixels_range[0])
        )
        # Note: difficulty values are not uniformly distributed (0.1, 0.2, 0.3...)
        # When curriculum=True, difficulty is dynamically adjusted based on robot performance
        # in each environment, resulting in values like 0.077, 0.143, 0.285, etc.
    elif cfg.step_width is not None:
        # Fallback to float-based width (for backward compatibility)
        step_width = int(cfg.step_width / cfg.horizontal_scale)
    else:
        raise ValueError("Either step_width_pixels_range or step_width must be provided")
    step_height = int(step_height / cfg.vertical_scale)
    # -- platform: use pixel-based width if available to avoid precision loss
    if cfg.platform_width_pixels is not None:
        platform_width = cfg.platform_width_pixels
    else:
        platform_width = int(cfg.platform_width / cfg.horizontal_scale)

    feature_w, feature_h = _resolve_feature_extent(cfg)
    feature_w_pixels = int(feature_w / cfg.horizontal_scale)
    feature_h_pixels = int(feature_h / cfg.horizontal_scale)
    x0 = max(0, (width_pixels - feature_w_pixels) // 2)
    y0 = max(0, (length_pixels - feature_h_pixels) // 2)

    # create a terrain with a flat platform at the center
    hf_raw = np.zeros((width_pixels, length_pixels))
    # Add steps inside the centered feature region; the outer margin remains flat.
    current_step_height = 0
    start_x, start_y = x0, y0
    stop_x, stop_y = x0 + feature_w_pixels, y0 + feature_h_pixels
    while (stop_x - start_x) > platform_width and (stop_y - start_y) > platform_width:
        # increment position
        # -- x
        start_x += step_width
        stop_x -= step_width
        # -- y
        start_y += step_width
        stop_y -= step_width
        # increment height
        current_step_height += step_height
        # add the step
        hf_raw[start_x:stop_x, start_y:stop_y] = current_step_height

    # add noise to the height field
    noise_min = int(cfg.noise_range[0] / cfg.vertical_scale)
    noise_max = int(cfg.noise_range[1] / cfg.vertical_scale)
    noise_step = int(cfg.noise_step / cfg.vertical_scale)
    # create range of noise heights possible
    noise_range = np.arange(noise_min, noise_max + noise_step, noise_step)
    # sample noise randomly
    noise = np.random.choice(noise_range, size=(width_pixels, length_pixels))
    # add noise to the height field
    hf_raw = hf_raw + noise

    # round off the heights to the nearest vertical step
    return np.rint(hf_raw).astype(np.int16)


@height_field_to_mesh
def multi_gap_terrain(
    difficulty: float, cfg: custom_terrains_cfg.HfMultiGapTerrainCfg
) -> np.ndarray:
    """Generate a terrain with nested "回字形" (ring-shaped) gaps of configurable depth.

    This creates a height field terrain with nested ring-shaped gap structures (一环套一环).
    The structure from outside to inside:
    - Outer border (solid ground)
    - Gap ring 1 (hollow, with configurable depth)
    - Ground ring 1 (solid ground)
    - Gap ring 2 (hollow, with configurable depth)
    - Ground ring 2 (solid ground)
    - ...
    - Inner platform (solid ground at center)

    Unlike the original gap terrain which has infinite depth, these gaps have a configurable depth.

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        The height field of the terrain as a 2D numpy array with discretized heights.
        The shape of the array is (width, length), where width and length are the number of points
        along the x and y axis, respectively.
    """
    # switch parameters to discrete units
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)

    # create a flat terrain at ground level (height = 0)
    hf_raw = np.zeros((width_pixels, length_pixels), dtype=np.float32)

    # resolve gap configuration based on difficulty
    gap_depth = cfg.gap_depth_range[0] + difficulty * (cfg.gap_depth_range[1] - cfg.gap_depth_range[0])
    step_height = cfg.step_height_range[0] + difficulty * (cfg.step_height_range[1] - cfg.step_height_range[0])
    platform_width = cfg.platform_width
    
    # Get gap width: use pixel-based width if available to avoid precision loss
    if cfg.gap_width_pixels_range is not None:
        # Use integer pixel range (越难越宽: width increases with difficulty)
        # gap_width_pixels_range[0] is width at difficulty=0 (easier, narrower)
        # gap_width_pixels_range[1] is width at difficulty=1 (harder, wider)
        gap_width_pixels = int(
            cfg.gap_width_pixels_range[0] 
            + difficulty * (cfg.gap_width_pixels_range[1] - cfg.gap_width_pixels_range[0])
        )
        # Convert to physical units for ground ring width calculation
        gap_width = gap_width_pixels * cfg.horizontal_scale
    elif cfg.gap_width_range is not None:
        # Fallback to float-based width (for backward compatibility)
        gap_width = cfg.gap_width_range[0] + difficulty * (cfg.gap_width_range[1] - cfg.gap_width_range[0])
        gap_width_pixels = int(gap_width / cfg.horizontal_scale)
    else:
        raise ValueError("Either gap_width_pixels_range or gap_width_range must be provided")
    
    # Ground ring width is automatically calculated as: 1.0m - gap_width
    # This ensures gap_width + ground_ring_width = 1.0m
    ground_ring_width = 1.0 - gap_width

    # convert gap parameters to discrete units
    gap_depth_pixels = int(gap_depth / cfg.vertical_scale)
    step_height_pixels = int(step_height / cfg.vertical_scale)
    platform_width_pixels = int(platform_width / cfg.horizontal_scale)
    ground_ring_width_pixels = int(ground_ring_width / cfg.horizontal_scale)
    
    # Calculate the maximum number of gap rings that can fit
    # Available space from center to edge: terrain_size / 2
    # Used by platform: platform_width / 2
    # Reserved for outer ground: outer_ground_width
    # Each gap ring needs: gap_width + ground_ring_width (on each side, so * 2 total)
    feature_w, feature_h = _resolve_feature_extent(cfg)
    terrain_half_size = min(feature_w, feature_h) / 2
    available_space = terrain_half_size - (platform_width / 2)
    ring_pair_width = gap_width + ground_ring_width  # One gap + one ground ring
    
    # Calculate maximum number of gap rings that can fit
    if ring_pair_width > 0:
        max_num_gaps = int(available_space / ring_pair_width)
        # Ensure at least 0 gaps (can't be negative)
        num_gaps = max(0, max_num_gaps)
    else:
        num_gaps = 0

    # Gap structure is always centered at terrain center
    gap_center_x = cfg.size[0] / 2
    gap_center_y = cfg.size[1] / 2

    # convert to pixel coordinates
    gap_center_x_pixels = int(gap_center_x / cfg.horizontal_scale)
    gap_center_y_pixels = int(gap_center_y / cfg.horizontal_scale)

    # Build nested structure from inside to outside
    # Structure from inside to outside: 
    # platform (height 0) -> ground_ring_1 (height +step_height) -> gap_ring_1 (depth gap_depth) 
    # -> ground_ring_2 (height -step_height) -> gap_ring_2 (depth gap_depth) 
    # -> ground_ring_3 (height +2*step_height) -> gap_ring_3 (depth gap_depth)
    # -> ground_ring_4 (height -2*step_height) -> gap_ring_4 (depth gap_depth) -> ... -> outer_ground
    # Height alternates between positive and negative, creating a wave-like pattern
    
    # Start with the innermost platform at height 0
    current_inner_size = platform_width_pixels
    current_inner_half = current_inner_size // 2
    
    # Set platform to height 0
    platform_start_x = max(0, gap_center_x_pixels - current_inner_half)
    platform_end_x = min(width_pixels, gap_center_x_pixels + current_inner_half)
    platform_start_y = max(0, gap_center_y_pixels - current_inner_half)
    platform_end_y = min(length_pixels, gap_center_y_pixels + current_inner_half)
    hf_raw[platform_start_x:platform_end_x, platform_start_y:platform_end_y] = 0
    
    # Create nested rings: from inside to outside
    for gap_idx in range(num_gaps):
        # Calculate the height for this ground ring (alternating increase and decrease)
        # Pattern: +step_height, -step_height, +2*step_height, -2*step_height, +3*step_height, -3*step_height, ...
        # For gap_idx=0: height = +step_height
        # For gap_idx=1: height = -step_height
        # For gap_idx=2: height = +2*step_height
        # For gap_idx=3: height = -2*step_height
        # etc.
        if gap_idx % 2 == 0:
            # Even indices: positive height (increasing)
            ground_ring_height = ((gap_idx // 2) + 1) * step_height_pixels
        else:
            # Odd indices: negative height (decreasing)
            ground_ring_height = -((gap_idx // 2) + 1) * step_height_pixels
        
        # First, add a ground ring (solid ground between gaps) with increasing height
        prev_inner_size = current_inner_size
        prev_inner_half = prev_inner_size // 2
        current_inner_size += 2 * ground_ring_width_pixels
        current_inner_half = current_inner_size // 2
        
        # Calculate bounds for ground ring
        ground_outer_start_x = max(0, gap_center_x_pixels - current_inner_half)
        ground_outer_end_x = min(width_pixels, gap_center_x_pixels + current_inner_half)
        ground_outer_start_y = max(0, gap_center_y_pixels - current_inner_half)
        ground_outer_end_y = min(length_pixels, gap_center_y_pixels + current_inner_half)
        
        # Set ground ring to the calculated height
        # Use mask to set only the ring area, ensuring clean boundaries
        if ground_outer_start_x < ground_outer_end_x and ground_outer_start_y < ground_outer_end_y:
            x_coords = np.arange(ground_outer_start_x, ground_outer_end_x)
            y_coords = np.arange(ground_outer_start_y, ground_outer_end_y)
            if len(x_coords) > 0 and len(y_coords) > 0:
                x_grid, y_grid = np.meshgrid(x_coords, y_coords, indexing='ij')
                
                # Calculate Manhattan distance (L-infinity norm) from center for each pixel
                dx = np.abs(x_grid - gap_center_x_pixels)
                dy = np.abs(y_grid - gap_center_y_pixels)
                max_dist = np.maximum(dx, dy)
                
                # Create mask for the ring area (between prev_inner and current_inner)
                # Use >= for inner boundary to include boundary pixels, <= for outer boundary
                # This ensures proper boundary handling without gaps or overlaps
                ring_mask = (max_dist >= prev_inner_half) & (max_dist < current_inner_half)
                
                # Set ground ring to the calculated height (only the ring area)
                hf_raw[ground_outer_start_x:ground_outer_end_x, ground_outer_start_y:ground_outer_end_y][ring_mask] = ground_ring_height
        
        # Now add the gap ring
        prev_inner_size = current_inner_size
        prev_inner_half = prev_inner_size // 2
        current_inner_size += 2 * gap_width_pixels
        current_inner_half = current_inner_size // 2
        
        # Calculate bounds for gap ring
        gap_outer_start_x = max(0, gap_center_x_pixels - current_inner_half)
        gap_outer_end_x = min(width_pixels, gap_center_x_pixels + current_inner_half)
        gap_outer_start_y = max(0, gap_center_y_pixels - current_inner_half)
        gap_outer_end_y = min(length_pixels, gap_center_y_pixels + current_inner_half)

        # Set gap ring to gap depth (negative, relative to the ground ring height)
        # The gap depth is measured from the ground ring height downward
        gap_bottom_height = ground_ring_height - gap_depth_pixels
        
        # Set gap ring area using mask to ensure clean boundaries
        if gap_outer_start_x < gap_outer_end_x and gap_outer_start_y < gap_outer_end_y:
            x_coords = np.arange(gap_outer_start_x, gap_outer_end_x)
            y_coords = np.arange(gap_outer_start_y, gap_outer_end_y)
            if len(x_coords) > 0 and len(y_coords) > 0:
                x_grid, y_grid = np.meshgrid(x_coords, y_coords, indexing='ij')
                
                # Calculate Manhattan distance (L-infinity norm) from center for each pixel
                dx = np.abs(x_grid - gap_center_x_pixels)
                dy = np.abs(y_grid - gap_center_y_pixels)
                max_dist = np.maximum(dx, dy)
                
                # Create mask for the gap ring area (between prev_inner and current_inner)
                # Use >= for inner boundary to include boundary pixels, < for outer boundary
                # This ensures proper boundary handling without gaps or overlaps
                gap_ring_mask = (max_dist >= prev_inner_half) & (max_dist < current_inner_half)
                
                # Set the gap ring area to gap depth (only the ring area)
                hf_raw[gap_outer_start_x:gap_outer_end_x, gap_outer_start_y:gap_outer_end_y][gap_ring_mask] = gap_bottom_height

    # round off the heights to the nearest vertical step
    return np.rint(hf_raw).astype(np.int16)


@height_field_to_mesh
def single_gap_terrain(
    difficulty: float, cfg: custom_terrains_cfg.HfSingleGapTerrainCfg
) -> np.ndarray:
    """单环回字形 gap：中心平台 → 一圈地面环 → 一圈 gap → 外围地面。

    与 multi_gap_terrain 不同的是，这里只有一圈 gap，
    并且中心平台与相邻的内侧地面环高度一致（无台阶）。"""
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    hf_raw = np.zeros((width_pixels, length_pixels), dtype=np.float32)

    gap_depth = cfg.gap_depth_range[0] + difficulty * (cfg.gap_depth_range[1] - cfg.gap_depth_range[0])
    step_height = cfg.step_height_range[0] + difficulty * (cfg.step_height_range[1] - cfg.step_height_range[0])
    platform_width = cfg.platform_width

    if cfg.gap_width_pixels_range is not None:
        gap_width_pixels = int(
            cfg.gap_width_pixels_range[0]
            + difficulty * (cfg.gap_width_pixels_range[1] - cfg.gap_width_pixels_range[0])
        )
        gap_width = gap_width_pixels * cfg.horizontal_scale
    elif getattr(cfg, "gap_width_range", None) is not None:
        gap_width = cfg.gap_width_range[0] + difficulty * (cfg.gap_width_range[1] - cfg.gap_width_range[0])
        gap_width_pixels = int(gap_width / cfg.horizontal_scale)
    else:
        raise ValueError("Either gap_width_pixels_range or gap_width_range must be provided")

    ground_ring_width = 1.0 - gap_width
    gap_depth_pixels = int(gap_depth / cfg.vertical_scale)
    step_height_pixels = int(step_height / cfg.vertical_scale)
    platform_width_pixels = int(platform_width / cfg.horizontal_scale)
    ground_ring_width_pixels = int(ground_ring_width / cfg.horizontal_scale)

    gap_center_x = cfg.size[0] / 2
    gap_center_y = cfg.size[1] / 2
    gap_center_x_pixels = int(gap_center_x / cfg.horizontal_scale)
    gap_center_y_pixels = int(gap_center_y / cfg.horizontal_scale)

    # 中心平台：高度与相邻地面环一致（ground_ring_height），保证平台与内圈地面平齐
    current_inner_size = platform_width_pixels
    current_inner_half = current_inner_size // 2
    platform_start_x = max(0, gap_center_x_pixels - current_inner_half)
    platform_end_x = min(width_pixels, gap_center_x_pixels + current_inner_half)
    platform_start_y = max(0, gap_center_y_pixels - current_inner_half)
    platform_end_y = min(length_pixels, gap_center_y_pixels + current_inner_half)
    ground_ring_height = step_height_pixels
    hf_raw[platform_start_x:platform_end_x, platform_start_y:platform_end_y] = ground_ring_height

    # 仅一圈：ground ring -> gap ring
    prev_inner_size = current_inner_size
    prev_inner_half = prev_inner_size // 2
    current_inner_size += 2 * ground_ring_width_pixels
    current_inner_half = current_inner_size // 2

    ground_outer_start_x = max(0, gap_center_x_pixels - current_inner_half)
    ground_outer_end_x = min(width_pixels, gap_center_x_pixels + current_inner_half)
    ground_outer_start_y = max(0, gap_center_y_pixels - current_inner_half)
    ground_outer_end_y = min(length_pixels, gap_center_y_pixels + current_inner_half)
    if ground_outer_start_x < ground_outer_end_x and ground_outer_start_y < ground_outer_end_y:
        x_coords = np.arange(ground_outer_start_x, ground_outer_end_x)
        y_coords = np.arange(ground_outer_start_y, ground_outer_end_y)
        if len(x_coords) > 0 and len(y_coords) > 0:
            x_grid, y_grid = np.meshgrid(x_coords, y_coords, indexing="ij")
            dx = np.abs(x_grid - gap_center_x_pixels)
            dy = np.abs(y_grid - gap_center_y_pixels)
            max_dist = np.maximum(dx, dy)
            ring_mask = (max_dist >= prev_inner_half) & (max_dist < current_inner_half)
            hf_raw[ground_outer_start_x:ground_outer_end_x, ground_outer_start_y:ground_outer_end_y][
                ring_mask
            ] = ground_ring_height

    prev_inner_size = current_inner_size
    prev_inner_half = prev_inner_size // 2
    current_inner_size += 2 * gap_width_pixels
    current_inner_half = current_inner_size // 2
    gap_outer_start_x = max(0, gap_center_x_pixels - current_inner_half)
    gap_outer_end_x = min(width_pixels, gap_center_x_pixels + current_inner_half)
    gap_outer_start_y = max(0, gap_center_y_pixels - current_inner_half)
    gap_outer_end_y = min(length_pixels, gap_center_y_pixels + current_inner_half)
    gap_bottom_height = ground_ring_height - gap_depth_pixels
    if gap_outer_start_x < gap_outer_end_x and gap_outer_start_y < gap_outer_end_y:
        x_coords = np.arange(gap_outer_start_x, gap_outer_end_x)
        y_coords = np.arange(gap_outer_start_y, gap_outer_end_y)
        if len(x_coords) > 0 and len(y_coords) > 0:
            x_grid, y_grid = np.meshgrid(x_coords, y_coords, indexing="ij")
            dx = np.abs(x_grid - gap_center_x_pixels)
            dy = np.abs(y_grid - gap_center_y_pixels)
            max_dist = np.maximum(dx, dy)
            gap_ring_mask = (max_dist >= prev_inner_half) & (max_dist < current_inner_half)
            hf_raw[gap_outer_start_x:gap_outer_end_x, gap_outer_start_y:gap_outer_end_y][
                gap_ring_mask
            ] = gap_bottom_height

    return np.rint(hf_raw).astype(np.int16)


@height_field_to_mesh
def stepping_stones_terrain(difficulty: float, cfg: custom_terrains_cfg.HfSteppingStonesTerrainCfg) -> np.ndarray:
    """Generate a terrain with a stepping stones pattern.

    The terrain is a stepping stones pattern which trims to a flat platform at the center of the terrain.

    .. image:: ../../_static/terrains/height_field/stepping_stones_terrain.jpg
       :width: 40%
       :align: center

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        The height field of the terrain as a 2D numpy array with discretized heights.
        The shape of the array is (width, length), where width and length are the number of points
        along the x and y axis, respectively.
    """
    # resolve terrain configuration
    stone_width = cfg.stone_width_range[1] - difficulty * (cfg.stone_width_range[1] - cfg.stone_width_range[0])
    stone_distance = cfg.stone_distance_range[0] + difficulty * (
        cfg.stone_distance_range[1] - cfg.stone_distance_range[0]
    )
    holes_depth = cfg.holes_depth[0] + difficulty * (cfg.holes_depth[1] - cfg.holes_depth[0])

    # switch parameters to discrete units
    # -- terrain
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # -- stones
    stone_distance = int(stone_distance / cfg.horizontal_scale)
    stone_width = int(stone_width / cfg.horizontal_scale)
    stone_height_max = int(cfg.stone_height_max / cfg.vertical_scale)
    # -- holes
    holes_depth = int(holes_depth / cfg.vertical_scale)
    # -- platform
    platform_width = int(cfg.platform_width / cfg.horizontal_scale)
    # create range of heights
    stone_height_range = np.arange(-stone_height_max - 1, stone_height_max, step=1)

    # create a terrain with a flat platform at the center
    hf_raw = np.full((width_pixels, length_pixels), holes_depth)
    # add the stones
    start_x, start_y = 0, 0
    # -- if the terrain is longer than it is wide then fill the terrain column by column
    if length_pixels >= width_pixels:
        while start_y < length_pixels:
            # ensure that stone stops along y-axis
            stop_y = min(length_pixels, start_y + stone_width)
            # randomly sample x-position
            start_x = np.random.randint(0, stone_width)
            stop_x = max(0, start_x - stone_distance)
            # fill first stone
            hf_raw[0:stop_x, start_y:stop_y] = np.random.choice(stone_height_range)
            # fill row with stones
            while start_x < width_pixels:
                stop_x = min(width_pixels, start_x + stone_width)
                hf_raw[start_x:stop_x, start_y:stop_y] = np.random.choice(stone_height_range)
                start_x += stone_width + stone_distance
            # update y-position
            start_y += stone_width + stone_distance
    elif width_pixels > length_pixels:
        while start_x < width_pixels:
            # ensure that stone stops along x-axis
            stop_x = min(width_pixels, start_x + stone_width)
            # randomly sample y-position
            start_y = np.random.randint(0, stone_width)
            stop_y = max(0, start_y - stone_distance)
            # fill first stone
            hf_raw[start_x:stop_x, 0:stop_y] = np.random.choice(stone_height_range)
            # fill column with stones
            while start_y < length_pixels:
                stop_y = min(length_pixels, start_y + stone_width)
                hf_raw[start_x:stop_x, start_y:stop_y] = np.random.choice(stone_height_range)
                start_y += stone_width + stone_distance
            # update x-position
            start_x += stone_width + stone_distance
    # add the platform in the center
    x1 = (width_pixels - platform_width) // 2
    x2 = (width_pixels + platform_width) // 2
    y1 = (length_pixels - platform_width) // 2
    y2 = (length_pixels + platform_width) // 2
    hf_raw[x1:x2, y1:y2] = 0
    # round off the heights to the nearest vertical step
    return np.rint(hf_raw).astype(np.int16)


@height_field_to_mesh
def cross_stepping_stones_terrain(difficulty: float, cfg: custom_terrains_cfg.HfCrossSteppingStonesTerrainCfg) -> np.ndarray:
    """Generate a terrain with a cross-shaped stepping stones pattern.

    The terrain has stepping stones arranged in a cross shape (十字形), with two rows/columns
    of stones on each branch. The rest of the terrain is filled with holes.

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        The height field of the terrain as a 2D numpy array with discretized heights.
        The shape of the array is (width, length), where width and length are the number of points
        along the x and y axis, respectively.
    """
    # resolve terrain configuration
    stone_width = cfg.stone_width_range[1] - difficulty * (cfg.stone_width_range[1] - cfg.stone_width_range[0])
    stone_distance = cfg.stone_distance_range[0] + difficulty * (
        cfg.stone_distance_range[1] - cfg.stone_distance_range[0]
    )
    holes_depth = cfg.holes_depth[0] + difficulty * (cfg.holes_depth[1] - cfg.holes_depth[0])

    # switch parameters to discrete units
    # -- terrain
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # -- stones
    stone_distance = int(stone_distance / cfg.horizontal_scale)
    stone_width = int(stone_width / cfg.horizontal_scale)
    stone_height_max = int(cfg.stone_height_max / cfg.vertical_scale)
    # -- holes
    holes_depth = int(holes_depth / cfg.vertical_scale)
    # -- platform
    platform_width = int(cfg.platform_width / cfg.horizontal_scale)
    # -- border
    border_width = int(cfg.border_width / cfg.horizontal_scale)
    # create range of heights
    stone_height_range = np.arange(-stone_height_max - 1, stone_height_max, step=1)

    # create a terrain filled with holes
    hf_raw = np.full((width_pixels, length_pixels), holes_depth)

    # find center of terrain
    center_x = width_pixels // 2
    center_y = length_pixels // 2

    # Get number of stone rows/columns
    num_rows = cfg.num_stone_rows

    # fill horizontal branch (left-right) with num_rows rows of stepping stones
    # Distribute rows symmetrically around center with proper spacing to avoid overlap
    for row_idx in range(num_rows):
        # Calculate y position for this row
        # Use stone_width + stone_distance as spacing to ensure no overlap
        row_spacing = stone_width + stone_distance
        if num_rows == 1:
            row_y = center_y - stone_width // 2
        else:
            # Calculate offset from center: distribute evenly with proper spacing
            # For even number of rows: distribute around center
            # For odd number of rows: one row at center, others distributed around
            offset_pixels = (row_idx - (num_rows - 1) / 2) * row_spacing
            row_y = center_y - stone_width // 2 + int(offset_pixels)
        
        row_y = max(0, min(length_pixels - stone_width, row_y))
        
        # Fill this row with stepping stones
        start_x = 0
        while start_x < width_pixels:
            stop_x = min(width_pixels, start_x + stone_width)
            if row_y + stone_width <= length_pixels and row_y >= 0:
                hf_raw[start_x:stop_x, row_y:row_y + stone_width] = np.random.choice(stone_height_range)
            start_x += stone_width + stone_distance

    # fill vertical branch (up-down) with num_rows columns of stepping stones
    # Distribute columns symmetrically around center with proper spacing to avoid overlap
    for col_idx in range(num_rows):
        # Calculate x position for this column
        # Use stone_width + stone_distance as spacing to ensure no overlap
        col_spacing = stone_width + stone_distance
        if num_rows == 1:
            col_x = center_x - stone_width // 2
        else:
            # Calculate offset from center: distribute evenly with proper spacing
            offset_pixels = (col_idx - (num_rows - 1) / 2) * col_spacing
            col_x = center_x - stone_width // 2 + int(offset_pixels)
        
        col_x = max(0, min(width_pixels - stone_width, col_x))
        
        # Fill this column with stepping stones
        start_y = 0
        while start_y < length_pixels:
            stop_y = min(length_pixels, start_y + stone_width)
            if col_x + stone_width <= width_pixels and col_x >= 0:
                hf_raw[col_x:col_x + stone_width, start_y:stop_y] = np.random.choice(stone_height_range)
            start_y += stone_width + stone_distance

    # add the platform in the center
    x1 = (width_pixels - platform_width) // 2
    x2 = (width_pixels + platform_width) // 2
    y1 = (length_pixels - platform_width) // 2
    y2 = (length_pixels + platform_width) // 2
    hf_raw[x1:x2, y1:y2] = 0

    # add safe border platform around the edges
    if border_width > 0:
        # Top border
        hf_raw[:, :border_width] = 0
        # Bottom border
        hf_raw[:, length_pixels - border_width:] = 0
        # Left border
        hf_raw[:border_width, :] = 0
        # Right border
        hf_raw[width_pixels - border_width:, :] = 0

    # round off the heights to the nearest vertical step
    return np.rint(hf_raw).astype(np.int16)


@height_field_to_mesh
def frame_platform_terrain(
    difficulty: float, cfg: custom_terrains_cfg.HfFramePlatformTerrainCfg
) -> np.ndarray:
    """回字形高台：中心为平地，外围一圈高台（框）。高度、边框宽度、内沿距中心距离均可设课程。

    类似 Isaac Lab 的 rails_terrain 的 height-field 版。内孔为平地(0)，框上为指定高度。
    """
    height = cfg.height_range[0] + difficulty * (cfg.height_range[1] - cfg.height_range[0])
    width = cfg.width_range[0] + difficulty * (cfg.width_range[1] - cfg.width_range[0])
    distance_from_center = cfg.distance_range[0] + difficulty * (
        cfg.distance_range[1] - cfg.distance_range[0]
    )

    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    center_x = width_pixels // 2
    center_y = length_pixels // 2

    inner_half_px = distance_from_center / cfg.horizontal_scale
    width_px = width / cfg.horizontal_scale
    outer_half_px = inner_half_px + width_px
    height_px = int(round(height / cfg.vertical_scale))

    ii = np.arange(width_pixels, dtype=np.float64)
    jj = np.arange(length_pixels, dtype=np.float64)
    dx = np.abs(ii - center_x)
    dy = np.abs(jj - center_y)
    dx_2d = np.broadcast_to(dx.reshape(-1, 1), (width_pixels, length_pixels))
    dy_2d = np.broadcast_to(dy.reshape(1, -1), (width_pixels, length_pixels))

    inside_inner = (dx_2d <= inner_half_px) & (dy_2d <= inner_half_px)
    outside_outer = (dx_2d >= outer_half_px) | (dy_2d >= outer_half_px)
    on_frame = ~inside_inner & ~outside_outer

    hf_raw = np.zeros((width_pixels, length_pixels), dtype=np.float64)
    hf_raw[on_frame] = height_px

    return np.rint(hf_raw).astype(np.int16)


@height_field_to_mesh
def double_frame_platform_terrain(
    difficulty: float, cfg: custom_terrains_cfg.HfDoubleFramePlatformTerrainCfg
) -> np.ndarray:
    """双回字形高台：中心平地 → 内圈高台 → 平地环 → 外圈高台 → 外围平地。两个高台的高、宽、距离可分别配置。"""
    # 内圈
    height_1 = cfg.height_range_1[0] + difficulty * (cfg.height_range_1[1] - cfg.height_range_1[0])
    width_1 = cfg.width_range_1[0] + difficulty * (cfg.width_range_1[1] - cfg.width_range_1[0])
    distance_1 = cfg.distance_range_1[0] + difficulty * (
        cfg.distance_range_1[1] - cfg.distance_range_1[0]
    )
    # 外圈
    height_2 = cfg.height_range_2[0] + difficulty * (cfg.height_range_2[1] - cfg.height_range_2[0])
    width_2 = cfg.width_range_2[0] + difficulty * (cfg.width_range_2[1] - cfg.width_range_2[0])
    distance_2 = cfg.distance_range_2[0] + difficulty * (
        cfg.distance_range_2[1] - cfg.distance_range_2[0]
    )

    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    center_x = width_pixels // 2
    center_y = length_pixels // 2

    scale_h = cfg.horizontal_scale
    scale_v = cfg.vertical_scale

    inner1_half_px = distance_1 / scale_h
    outer1_half_px = inner1_half_px + width_1 / scale_h
    inner2_half_px = distance_2 / scale_h
    outer2_half_px = inner2_half_px + width_2 / scale_h
    height1_px = int(round(height_1 / scale_v))
    height2_px = int(round(height_2 / scale_v))

    ii = np.arange(width_pixels, dtype=np.float64)
    jj = np.arange(length_pixels, dtype=np.float64)
    dx = np.abs(ii - center_x)
    dy = np.abs(jj - center_y)
    dx_2d = np.broadcast_to(dx.reshape(-1, 1), (width_pixels, length_pixels))
    dy_2d = np.broadcast_to(dy.reshape(1, -1), (width_pixels, length_pixels))

    inside_inner_1 = (dx_2d <= inner1_half_px) & (dy_2d <= inner1_half_px)
    outside_outer_1 = (dx_2d >= outer1_half_px) | (dy_2d >= outer1_half_px)
    on_frame_1 = ~inside_inner_1 & ~outside_outer_1

    inside_inner_2 = (dx_2d <= inner2_half_px) & (dy_2d <= inner2_half_px)
    outside_outer_2 = (dx_2d >= outer2_half_px) | (dy_2d >= outer2_half_px)
    on_frame_2 = ~inside_inner_2 & ~outside_outer_2

    hf_raw = np.zeros((width_pixels, length_pixels), dtype=np.float64)
    hf_raw[on_frame_2] = height2_px
    hf_raw[on_frame_1] = height1_px

    return np.rint(hf_raw).astype(np.int16)


@height_field_to_mesh
def hurdle_terrain(
    difficulty: float, cfg: custom_terrains_cfg.HfHurdleTerrainCfg
) -> np.ndarray:
    """Generate a terrain with nested "回字形" (ring-shaped) hurdles.

    This creates a height field terrain with nested ring-shaped hurdle structures (一环套一环).
    The structure from outside to inside:
    - Outer border (solid ground at height 0)
    - Ground ring 1 (solid ground at height 0)
    - Hurdle ring 1 (raised barrier, height increases with difficulty)
    - Ground ring 2 (solid ground at height 0)
    - Hurdle ring 2 (raised barrier, height increases with difficulty)
    - ...
    - Inner platform (solid ground at height 0)

    Unlike gaps which are depressions, hurdles are raised barriers that the robot must step over.

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        The height field of the terrain as a 2D numpy array with discretized heights.
        The shape of the array is (width, length), where width and length are the number of points
        along the x and y axis, respectively.
    """
    # switch parameters to discrete units
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)

    # create a flat terrain at ground level (height = 0)
    hf_raw = np.zeros((width_pixels, length_pixels), dtype=np.float32)

    # resolve hurdle configuration based on difficulty
    # Hurdle height increases with difficulty
    hurdle_height = cfg.hurdle_height_range[0] + difficulty * (cfg.hurdle_height_range[1] - cfg.hurdle_height_range[0])
    platform_width = cfg.platform_width
    
    # Fixed number of hurdles: 3
    num_hurdles = 3
    
    # Get hurdle widths (can be single value or tuple of 3 values)
    if isinstance(cfg.hurdle_width_pixels, tuple):
        if len(cfg.hurdle_width_pixels) != 3:
            raise ValueError("hurdle_width_pixels tuple must have exactly 3 elements")
        hurdle_width_pixels_list = list(cfg.hurdle_width_pixels)
    else:
        # Single value: all hurdles have the same width
        hurdle_width_pixels_list = [cfg.hurdle_width_pixels] * 3
    
    # convert to discrete units
    hurdle_height_pixels = int(hurdle_height / cfg.vertical_scale)
    platform_width_pixels = int(platform_width / cfg.horizontal_scale)
    
    # Hurdle structure is always centered at terrain center
    hurdle_center_x = cfg.size[0] / 2
    hurdle_center_y = cfg.size[1] / 2

    # convert to pixel coordinates
    hurdle_center_x_pixels = int(hurdle_center_x / cfg.horizontal_scale)
    hurdle_center_y_pixels = int(hurdle_center_y / cfg.horizontal_scale)
    
    # Calculate available space for uniform distribution
    terrain_half_size_pixels = min(width_pixels, length_pixels) // 2
    platform_half_size_pixels = platform_width_pixels // 2
    available_space_pixels = terrain_half_size_pixels - platform_half_size_pixels
    
    # Calculate total width of all hurdles
    total_hurdle_width_pixels = sum(hurdle_width_pixels_list)
    
    # Calculate uniform spacing between hurdles (4 gaps: platform->hurdle1, hurdle1->hurdle2, hurdle2->hurdle3, hurdle3->edge)
    # We have: platform + gap1 + hurdle1 + gap2 + hurdle2 + gap3 + hurdle3 + gap4 = terrain_edge
    # gap1 = gap2 = gap3 = gap4 for uniform distribution
    remaining_space = available_space_pixels - total_hurdle_width_pixels
    uniform_gap_pixels = remaining_space // 4  # 4 gaps for uniform distribution
    
    # Build nested structure from inside to outside
    # Structure from inside to outside: 
    # platform (height 0) -> gap1 -> hurdle_ring_1 (height hurdle_height) 
    # -> gap2 -> hurdle_ring_2 (height hurdle_height) 
    # -> gap3 -> hurdle_ring_3 (height hurdle_height) -> gap4 -> outer_ground
    
    # Start with the innermost platform at height 0
    current_inner_size = platform_width_pixels
    current_inner_half = current_inner_size // 2
    
    # Set platform to height 0
    platform_start_x = max(0, hurdle_center_x_pixels - current_inner_half)
    platform_end_x = min(width_pixels, hurdle_center_x_pixels + current_inner_half)
    platform_start_y = max(0, hurdle_center_y_pixels - current_inner_half)
    platform_end_y = min(length_pixels, hurdle_center_y_pixels + current_inner_half)
    hf_raw[platform_start_x:platform_end_x, platform_start_y:platform_end_y] = 0
    
    # Create 3 hurdles uniformly distributed: from inside to outside
    for hurdle_idx in range(num_hurdles):
        # Add gap (ground ring) before this hurdle
        prev_inner_size = current_inner_size
        prev_inner_half = prev_inner_size // 2
        current_inner_size += 2 * uniform_gap_pixels
        current_inner_half = current_inner_size // 2
        
        # Calculate bounds for ground ring (gap)
        ground_outer_start_x = max(0, hurdle_center_x_pixels - current_inner_half)
        ground_outer_end_x = min(width_pixels, hurdle_center_x_pixels + current_inner_half)
        ground_outer_start_y = max(0, hurdle_center_y_pixels - current_inner_half)
        ground_outer_end_y = min(length_pixels, hurdle_center_y_pixels + current_inner_half)
        
        # Set ground ring to height 0 (only the ring area)
        if ground_outer_start_x < ground_outer_end_x and ground_outer_start_y < ground_outer_end_y:
            x_coords = np.arange(ground_outer_start_x, ground_outer_end_x)
            y_coords = np.arange(ground_outer_start_y, ground_outer_end_y)
            if len(x_coords) > 0 and len(y_coords) > 0:
                x_grid, y_grid = np.meshgrid(x_coords, y_coords, indexing='ij')
                
                # Calculate Manhattan distance (L-infinity norm) from center for each pixel
                dx = np.abs(x_grid - hurdle_center_x_pixels)
                dy = np.abs(y_grid - hurdle_center_y_pixels)
                max_dist = np.maximum(dx, dy)
                
                # Create mask for the ring area (between prev_inner and current_inner)
                ring_mask = (max_dist >= prev_inner_half) & (max_dist < current_inner_half)
                
                # Set ground ring to height 0 (only the ring area)
                hf_raw[ground_outer_start_x:ground_outer_end_x, ground_outer_start_y:ground_outer_end_y][ring_mask] = 0
        
        # Now add the hurdle ring (raised barrier) with its specific width
        hurdle_width_pixels = hurdle_width_pixels_list[hurdle_idx]
        prev_inner_size = current_inner_size
        prev_inner_half = prev_inner_size // 2
        current_inner_size += 2 * hurdle_width_pixels
        current_inner_half = current_inner_size // 2
        
        # Calculate bounds for hurdle ring
        hurdle_outer_start_x = max(0, hurdle_center_x_pixels - current_inner_half)
        hurdle_outer_end_x = min(width_pixels, hurdle_center_x_pixels + current_inner_half)
        hurdle_outer_start_y = max(0, hurdle_center_y_pixels - current_inner_half)
        hurdle_outer_end_y = min(length_pixels, hurdle_center_y_pixels + current_inner_half)
        
        # Set hurdle ring to hurdle height (raised barrier)
        if hurdle_outer_start_x < hurdle_outer_end_x and hurdle_outer_start_y < hurdle_outer_end_y:
            x_coords = np.arange(hurdle_outer_start_x, hurdle_outer_end_x)
            y_coords = np.arange(hurdle_outer_start_y, hurdle_outer_end_y)
            if len(x_coords) > 0 and len(y_coords) > 0:
                x_grid, y_grid = np.meshgrid(x_coords, y_coords, indexing='ij')
                
                # Calculate Manhattan distance (L-infinity norm) from center for each pixel
                dx = np.abs(x_grid - hurdle_center_x_pixels)
                dy = np.abs(y_grid - hurdle_center_y_pixels)
                max_dist = np.maximum(dx, dy)
                
                # Create mask for the hurdle ring area (between prev_inner and current_inner)
                hurdle_ring_mask = (max_dist >= prev_inner_half) & (max_dist < current_inner_half)
                
                # Set the hurdle ring area to hurdle height (only the ring area)
                hf_raw[hurdle_outer_start_x:hurdle_outer_end_x, hurdle_outer_start_y:hurdle_outer_end_y][hurdle_ring_mask] = hurdle_height_pixels

    # round off the heights to the nearest vertical step
    return np.rint(hf_raw).astype(np.int16)


def _smoothstep01(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

@height_field_to_mesh
def interlaced_corridor_wedges_terrain(
    difficulty: float, cfg: custom_terrains_cfg.HfInterlacedCorridorWedgesTerrainCfg
) -> np.ndarray:
    """Cross-slopes along +x: every segment uses y=±W (default ±1.5), width 3 m, slope direction alternates."""
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    h_scale = cfg.horizontal_scale
    v_scale = cfg.vertical_scale

    sx, sy = float(cfg.size[0]), float(cfg.size[1])
    W = float(cfg.corridor_half_width_m)
    L = float(cfg.segment_length_m)
    n_seg = int(cfg.num_segments)
    if W <= 1e-6 or L <= 1e-6:
        raise ValueError("corridor_half_width_m and segment_length_m must be positive.")
    if n_seg < 1:
        raise ValueError(f"num_segments must be >= 1, got {n_seg}")

    blend = max(float(cfg.edge_blend_m), 1e-4)
    blend_x = float(cfg.longitudinal_blend_m) if cfg.longitudinal_blend_m is not None else blend
    blend_x = max(blend_x, 1e-4)
    fade_y = max(float(cfg.corridor_edge_fade_m), 1e-4)

    total_len = n_seg * L
    full_width = 2.0 * W
    y_lo = -W
    y_hi = W

    d_cur = float(min(1.0, max(0.0, difficulty)))
    lo, hi = float(cfg.slope_angle_range[0]), float(cfg.slope_angle_range[1])
    angle_deg = lo + d_cur * (hi - lo)
    slope_rad = math.radians(angle_deg)
    h_peak = math.tan(slope_rad) * full_width

    x0 = float(cfg.pattern_origin_x_m)
    ix = np.arange(width_pixels, dtype=np.float64)[:, None]
    iy = np.arange(length_pixels, dtype=np.float64)[None, :]
    px_m = (ix + 0.5) * h_scale - sx * 0.5
    py_m = (iy + 0.5) * h_scale - sy * 0.5

    rel_x = px_m - x0
    in_pattern = (rel_x >= 0.0) & (rel_x < total_len)
    kx = np.floor(rel_x / L)
    valid_seg = in_pattern & (kx >= 0.0) & (kx < float(n_seg))
    slope_up = np.mod(kx, 2) == 0

    foot_x = _smoothstep01(rel_x / (2.0 * blend_x)) * _smoothstep01((total_len - rel_x) / (2.0 * blend_x))
    foot_x = np.where(valid_seg, foot_x, 0.0)

    ramp_up = np.clip((py_m - y_lo) / full_width, 0.0, 1.0)
    ramp_down = np.clip((y_hi - py_m) / full_width, 0.0, 1.0)
    edge_y = _smoothstep01((py_m - y_lo + blend) / (2.0 * blend)) * _smoothstep01((y_hi + blend - py_m) / (2.0 * blend))

    h_up = h_peak * ramp_up * edge_y * foot_x
    h_down = h_peak * ramp_down * edge_y * foot_x
    h_m = np.where(slope_up, h_up, h_down)
    h_m = np.where(valid_seg, h_m, 0.0)

    path_mask = _smoothstep01((W + fade_y - np.abs(py_m)) / (2.0 * fade_y))
    h_m *= path_mask

    hf_raw = np.rint(h_m / v_scale).astype(np.float64)
    return hf_raw.astype(np.int16)

def _point_to_closed_polyline_dist(
    px: np.ndarray, py: np.ndarray, pts: np.ndarray
) -> np.ndarray:
    """点到闭合折线的最短距离。

    ``px`` 为 ``(W, 1)``、``py`` 为 ``(1, L)``（与高度场网格一致），返回 ``(W, L)`` 距离（米）。
    """
    px_b = np.asarray(px, dtype=np.float64).reshape(-1, 1)
    py_b = np.asarray(py, dtype=np.float64).reshape(1, -1)
    width_pixels, length_pixels = px_b.shape[0], py_b.shape[1]

    a = pts
    b = np.roll(pts, -1, axis=0)
    ab = b - a
    ab2 = np.maximum(np.sum(ab * ab, axis=1), 1e-12)

    px_flat = np.broadcast_to(px_b, (width_pixels, length_pixels)).reshape(-1)
    py_flat = np.broadcast_to(py_b, (width_pixels, length_pixels)).reshape(-1)
    flat = np.stack([px_flat, py_flat], axis=1)
    ap = flat[:, None, :] - a[None, :, :]
    t = np.sum(ap * ab[None, :, :], axis=2) / ab2[None, :]
    t = np.clip(t, 0.0, 1.0)
    closest = a[None, :, :] + t[:, :, None] * ab[None, :, :]
    diff = flat[:, None, :] - closest
    dist = np.sqrt(np.min(np.sum(diff * diff, axis=2), axis=1))
    return dist.reshape(width_pixels, length_pixels)


def _figure8_annulus_footprint(
    dist: np.ndarray, inner_r: float, outer_r: float, fade: float
) -> np.ndarray:
    """圆环桥面软边：``inner_r <= dist <= outer_r`` 为 1。"""
    outside_inner = _smoothstep01((dist - inner_r + fade) / fade)
    inside_outer = _smoothstep01((outer_r + fade - dist) / fade)
    return outside_inner * inside_outer


def _apply_figure8_center_cutout(
    footprint: np.ndarray,
    px_m: np.ndarray,
    py_m: np.ndarray,
    cutout_width_m: float,
    cutout_length_m: float,
) -> np.ndarray:
    """原点挖空：强制为坑底（footprint=0），与 ``holes_depth`` 一致。"""
    from .figure8_geometry import center_cutout_mask_numpy

    cutout = center_cutout_mask_numpy(px_m, py_m, cutout_width_m, cutout_length_m)
    if not np.any(cutout):
        return footprint
    return np.where(cutout, 0.0, footprint)


@height_field_to_mesh
def figure8_bridge_terrain(
    difficulty: float, cfg: custom_terrains_cfg.HfFigure8BridgeTerrainCfg
) -> np.ndarray:
    """平面闭合双 S（8 字）窄桥：坑底 + 可变宽桥面，中心交叉区加宽，无地面高台。

    中心线：``X = R sin t``, ``Y = R sin t cos t``，``t ∈ [0, 2π)``，原点在 8 字几何中心。
    """
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    h_scale = cfg.horizontal_scale
    v_scale = cfg.vertical_scale

    widths = getattr(cfg, "bridge_width_by_level", None)
    if widths is not None and len(widths) > 0:
        level_max = len(widths) - 1
        level_idx = int(np.clip(np.round(float(difficulty) * level_max), 0, level_max))
        bridge_width = float(widths[level_idx])
    else:
        bridge_width = cfg.bridge_width_range[0] + difficulty * (
            cfg.bridge_width_range[1] - cfg.bridge_width_range[0]
        )

    holes_depth_m = cfg.holes_depth[0] + difficulty * (cfg.holes_depth[1] - cfg.holes_depth[0])
    h_top = float(cfg.bridge_height)
    fade = max(float(cfg.radial_edge_fade_m), 1e-4)

    sx, sy = float(cfg.size[0]), float(cfg.size[1])
    ix = np.arange(width_pixels, dtype=np.float64)[:, None]
    iy = np.arange(length_pixels, dtype=np.float64)[None, :]
    px_m = (ix + 0.5) * h_scale - sx * 0.5
    py_m = (iy + 0.5) * h_scale - sy * 0.5

    from .figure8_geometry import centerline_radius_m, outer_radius_m

    inner_r = float(cfg.inner_radius_m)
    outer_r = outer_radius_m(inner_r, bridge_width)
    r_c = centerline_radius_m(inner_r, bridge_width)
    d_left = np.hypot(px_m + r_c, py_m)
    d_right = np.hypot(px_m - r_c, py_m)
    footprint = np.maximum(
        _figure8_annulus_footprint(d_left, inner_r, outer_r, fade),
        _figure8_annulus_footprint(d_right, inner_r, outer_r, fade),
    )
    footprint = _apply_figure8_center_cutout(
        footprint,
        px_m,
        py_m,
        float(getattr(cfg, "center_cutout_width_m", 0.0)),
        float(getattr(cfg, "center_cutout_length_m", 0.0)),
    )

    h_m = np.full((width_pixels, length_pixels), holes_depth_m, dtype=np.float64)
    h_m = np.maximum(h_m, h_top * footprint)

    hf_raw = np.rint(h_m / v_scale).astype(np.float64)
    return hf_raw.astype(np.int16)
