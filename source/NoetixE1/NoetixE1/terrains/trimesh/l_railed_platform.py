# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

"""Raised 90-degree L-shaped platform with an outer hollow guardrail."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import trimesh

if TYPE_CHECKING:
    from .l_railed_platform_cfg import MeshLRailedPlatformTerrainCfg


L_RAILED_PLATFORM_METADATA_KEY = "l_railed_platform"


def _box(extents, center, yaw: float = 0.0) -> trimesh.Trimesh:
    transform = trimesh.transformations.translation_matrix(center)
    if yaw != 0.0:
        transform = transform @ trimesh.transformations.rotation_matrix(yaw, (0.0, 0.0, 1.0))
    return trimesh.creation.box(extents=extents, transform=transform)


def _validate_cfg(cfg: MeshLRailedPlatformTerrainCfg) -> None:
    size_x, size_y = map(float, cfg.size)
    start_x, start_y = map(float, cfg.first_leg_start_center_m)
    num_steps = int(cfg.num_steps)
    num_width_levels = int(cfg.num_width_levels)
    width_easy, width_hard = map(float, cfg.width_range_m)
    dimensions = (
        size_x, size_y, start_x, start_y, float(cfg.step_height_m),
        float(cfg.tread_depth_m), float(cfg.leg_length_m), width_easy, width_hard,
        float(cfg.guardrail_height_m), float(cfg.guardrail_depth_m),
        float(cfg.guardrail_rail_height_m), float(cfg.hollow_opening_width_m),
        float(cfg.hollow_opening_height_m), float(cfg.hollow_post_width_m),
        float(cfg.spawn_distance_before_stairs_m), float(cfg.ground_thickness_m),
    )
    if not all(np.isfinite(value) for value in dimensions):
        raise ValueError("L-railed-platform configuration values must all be finite.")
    if size_x <= 0.0 or size_y <= 0.0:
        raise ValueError(f"size must be positive, got {cfg.size}.")
    if num_steps < 1 or float(cfg.num_steps) != num_steps:
        raise ValueError(f"num_steps must be a positive integer, got {cfg.num_steps}.")
    if num_width_levels < 2 or float(cfg.num_width_levels) != num_width_levels:
        raise ValueError(f"num_width_levels must be an integer of at least two, got {cfg.num_width_levels}.")
    if cfg.step_height_m <= 0.0 or cfg.tread_depth_m <= 0.0 or cfg.leg_length_m <= 0.0:
        raise ValueError("step height, tread depth, and L-leg length must be positive.")
    if width_easy <= 0.0 or width_hard <= 0.0 or width_hard > width_easy:
        raise ValueError(f"width_range_m must progress from wide to narrow, got {cfg.width_range_m}.")
    if cfg.guardrail_height_m <= 0.0 or cfg.guardrail_depth_m <= 0.0:
        raise ValueError("guardrail dimensions must be positive.")
    assembled_height = 2.0 * cfg.guardrail_rail_height_m + cfg.hollow_opening_height_m
    if not np.isclose(assembled_height, cfg.guardrail_height_m, atol=1.0e-9):
        raise ValueError("guardrail_height_m must equal two rail heights plus the opening height.")
    stairs_run = num_steps * float(cfg.tread_depth_m)
    corner_x = start_x + float(cfg.leg_length_m)
    exit_y = start_y + float(cfg.leg_length_m)
    guardrail_reach = width_easy / 2.0 + float(cfg.guardrail_depth_m)
    bounds = (
        start_x - stairs_run - float(cfg.spawn_distance_before_stairs_m),
        start_y - guardrail_reach,
        corner_x + guardrail_reach,
        exit_y + stairs_run,
    )
    if bounds[0] < 0.0 or bounds[1] < 0.0 or bounds[2] > size_x or bounds[3] > size_y:
        raise ValueError(f"L platform does not fit inside terrain cell: bounds={bounds}, size={cfg.size}.")


def _append_walkable_box(meshes, x_min, x_max, y_min, y_max, surface_height) -> None:
    if surface_height > 0.0:
        meshes.append(_box((x_max - x_min, y_max - y_min, surface_height),
                           ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0, surface_height / 2.0)))


def _append_hollow_guardrail_segment(meshes, start_xy, end_xy, surface_height, cfg) -> None:
    p0 = np.asarray(start_xy, dtype=np.float64)
    p1 = np.asarray(end_xy, dtype=np.float64)
    delta = p1 - p0
    length = float(np.linalg.norm(delta))
    tangent = delta / length
    yaw = float(np.arctan2(delta[1], delta[0]))
    center_xy = (p0 + p1) / 2.0
    depth = float(cfg.guardrail_depth_m)
    total_height = float(cfg.guardrail_height_m)
    rail_height = float(cfg.guardrail_rail_height_m)
    opening_height = float(cfg.hollow_opening_height_m)
    meshes.append(_box((length, depth, rail_height),
                       (center_xy[0], center_xy[1], surface_height + rail_height / 2.0), yaw))
    meshes.append(_box((length, depth, rail_height),
                       (center_xy[0], center_xy[1], surface_height + total_height - rail_height / 2.0), yaw))
    opening_width = float(cfg.hollow_opening_width_m)
    post_width = float(cfg.hollow_post_width_m)
    post_centers = list(np.arange(post_width / 2.0, length - post_width / 2.0 + 1.0e-9,
                                  post_width + opening_width))
    end_post_center = length - post_width / 2.0
    if not post_centers or end_post_center - post_centers[-1] > post_width + 1.0e-9:
        post_centers.append(end_post_center)
    for distance in post_centers:
        post_xy = p0 + tangent * distance
        meshes.append(_box((post_width, depth, opening_height),
                           (float(post_xy[0]), float(post_xy[1]),
                            surface_height + rail_height + opening_height / 2.0), yaw))


def l_railed_platform_terrain(difficulty: float, cfg: MeshLRailedPlatformTerrainCfg):
    """Generate a raised left-turn L course with stairs at both ends."""
    _validate_cfg(cfg)
    difficulty = float(np.clip(difficulty, 0.0, 1.0))
    width_easy, width_hard = map(float, cfg.width_range_m)
    level = min(int(np.floor(difficulty * cfg.num_width_levels)), cfg.num_width_levels - 1)
    passage_width = width_easy + level / (cfg.num_width_levels - 1) * (width_hard - width_easy)
    size_x, size_y = map(float, cfg.size)
    start_x, start_y = map(float, cfg.first_leg_start_center_m)
    num_steps = int(cfg.num_steps)
    step_height = float(cfg.step_height_m)
    tread_depth = float(cfg.tread_depth_m)
    leg_length = float(cfg.leg_length_m)
    platform_height = num_steps * step_height
    stairs_run = num_steps * tread_depth
    half_width = passage_width / 2.0
    corner_x = start_x + leg_length
    exit_y = start_y + leg_length
    entry_stairs_x_min = start_x - stairs_run
    exit_stairs_y_max = exit_y + stairs_run
    meshes = [_box((size_x, size_y, cfg.ground_thickness_m),
                   (size_x / 2.0, size_y / 2.0, -cfg.ground_thickness_m / 2.0))]
    for index in range(num_steps):
        x_min = entry_stairs_x_min + index * tread_depth
        _append_walkable_box(meshes, x_min, x_min + tread_depth, start_y - half_width,
                             start_y + half_width, (index + 1) * step_height)
    _append_walkable_box(meshes, start_x, corner_x - half_width, start_y - half_width,
                         start_y + half_width, platform_height)
    _append_walkable_box(meshes, corner_x - half_width, corner_x + half_width,
                         start_y - half_width, start_y + half_width, platform_height)
    _append_walkable_box(meshes, corner_x - half_width, corner_x + half_width,
                         start_y + half_width, exit_y, platform_height)
    for index in range(num_steps):
        y_min = exit_y + index * tread_depth
        _append_walkable_box(meshes, corner_x - half_width, corner_x + half_width, y_min,
                             y_min + tread_depth, (num_steps - index - 1) * step_height)
    rail_offset = (passage_width + cfg.guardrail_depth_m) / 2.0
    rail_corner = (corner_x + rail_offset, start_y - rail_offset)
    _append_hollow_guardrail_segment(meshes, (start_x, start_y - rail_offset), rail_corner,
                                     platform_height, cfg)
    _append_hollow_guardrail_segment(meshes, rail_corner, (corner_x + rail_offset, exit_y),
                                     platform_height, cfg)
    meshes.append(_box((cfg.hollow_post_width_m, cfg.hollow_post_width_m, cfg.hollow_opening_height_m),
                       (rail_corner[0], rail_corner[1], platform_height + cfg.guardrail_rail_height_m
                        + cfg.hollow_opening_height_m / 2.0)))
    combined = trimesh.util.concatenate(meshes)
    combined.metadata[L_RAILED_PLATFORM_METADATA_KEY] = {
        "difficulty": difficulty,
        "width_level_index": level,
        "passage_width_m": passage_width,
        "platform_height_m": platform_height,
        "leg_lengths_m": np.asarray([leg_length, leg_length], dtype=np.float32),
        "path_centerline_xy_m": np.asarray(
            [[start_x, start_y], [corner_x, start_y], [corner_x, exit_y]], dtype=np.float32),
        "stair_endpoints_xy_m": np.asarray(
            [[entry_stairs_x_min, start_y], [corner_x, exit_stairs_y_max]], dtype=np.float32),
    }
    origin = np.asarray(
        [entry_stairs_x_min - cfg.spawn_distance_before_stairs_m, start_y, 0.0], dtype=np.float64)
    return [combined], origin
