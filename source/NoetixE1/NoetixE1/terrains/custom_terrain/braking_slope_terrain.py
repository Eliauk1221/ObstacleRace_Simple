"""Trimesh generator for the ordered braking-slope curriculum."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import trimesh

if TYPE_CHECKING:
    from .custom_terrains_cfg import MeshBrakingSlopeTerrainCfg


class BrakingSlopeGeometry(NamedTuple):
    """Level-specific local geometry in meters."""

    step_height: float
    tread_depth: float
    stairs_height: float
    stairs_start_x: float
    slope_top_x: float
    slope_foot_x: float
    slope_surface_length: float
    slope_horizontal_length: float


def level_index_for_difficulty(difficulty: float, num_levels: int) -> int:
    """Map a normalized curriculum difficulty to an ordered level index."""
    if num_levels <= 0:
        raise ValueError(f"num_levels must be positive, got {num_levels}.")
    normalized = float(np.clip(difficulty, 0.0, 1.0))
    return min(int(math.floor(normalized * num_levels)), num_levels - 1)


def braking_slope_geometry(
    difficulty: float,
    step_height_by_level: tuple[float, ...],
    tread_depth_by_level: tuple[float, ...],
    num_steps: int,
    slope_angle_deg: float,
    slope_foot_x_m: float,
) -> BrakingSlopeGeometry:
    """Compute the exact stair and slope junctions for one curriculum level."""
    if len(step_height_by_level) != len(tread_depth_by_level) or not step_height_by_level:
        raise ValueError("step-height and tread-depth schedules must have the same non-zero length.")
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}.")
    if not 0.0 < slope_angle_deg < 90.0:
        raise ValueError(f"slope_angle_deg must lie in (0, 90), got {slope_angle_deg}.")

    level = level_index_for_difficulty(difficulty, len(step_height_by_level))
    step_height = float(step_height_by_level[level])
    tread_depth = float(tread_depth_by_level[level])
    if step_height <= 0.0 or tread_depth <= 0.0:
        raise ValueError("step heights and tread depths must be positive.")

    angle = math.radians(float(slope_angle_deg))
    stairs_height = num_steps * step_height
    slope_surface_length = stairs_height / math.sin(angle)
    slope_horizontal_length = slope_surface_length * math.cos(angle)
    slope_top_x = float(slope_foot_x_m) - slope_horizontal_length
    stairs_start_x = slope_top_x - num_steps * tread_depth
    return BrakingSlopeGeometry(
        step_height=step_height,
        tread_depth=tread_depth,
        stairs_height=stairs_height,
        stairs_start_x=stairs_start_x,
        slope_top_x=slope_top_x,
        slope_foot_x=float(slope_foot_x_m),
        slope_surface_length=slope_surface_length,
        slope_horizontal_length=slope_horizontal_length,
    )


def _down_slope_prism(
    top_x: float,
    foot_x: float,
    y_min: float,
    y_max: float,
    height: float,
) -> trimesh.Trimesh:
    """Create a closed triangular prism descending from ``top_x`` to ``foot_x``."""
    vertices = np.array(
        [
            [top_x, y_min, 0.0],
            [top_x, y_min, height],
            [foot_x, y_min, 0.0],
            [top_x, y_max, 0.0],
            [top_x, y_max, height],
            [foot_x, y_max, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [3, 5, 4],
            [0, 3, 4],
            [0, 4, 1],
            [1, 4, 5],
            [1, 5, 2],
            [0, 2, 5],
            [0, 5, 3],
        ],
        dtype=np.int64,
    )
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if mesh.volume < 0.0:
        mesh.invert()
    return mesh


def braking_slope_terrain(
    difficulty: float, cfg: MeshBrakingSlopeTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate four rising treads and a down slope on a fully collidable flat tile."""
    size_x, size_y = map(float, cfg.size)
    walkable_width = float(cfg.walkable_width_m)
    finish_flat_length = float(cfg.finish_flat_length_m)
    success_region_length = float(cfg.success_region_length_m)
    ground_thickness = float(cfg.ground_thickness_m)
    geometry = braking_slope_geometry(
        difficulty,
        tuple(cfg.step_height_by_level),
        tuple(cfg.tread_depth_by_level),
        int(cfg.num_steps),
        float(cfg.slope_angle_deg),
        float(cfg.slope_foot_x_m),
    )

    if walkable_width <= 0.0 or walkable_width > size_y:
        raise ValueError(f"walkable_width_m must lie in (0, {size_y}], got {walkable_width}.")
    if finish_flat_length <= 0.0 or ground_thickness <= 0.0:
        raise ValueError("finish-flat length and ground thickness must be positive.")
    if not 0.0 < success_region_length <= finish_flat_length:
        raise ValueError("success-region length must lie within the finish flat.")
    if geometry.stairs_start_x < -0.5 * size_x:
        raise ValueError("The stairs extend beyond the negative x boundary of the tile.")
    if geometry.slope_foot_x + finish_flat_length > 0.5 * size_x:
        raise ValueError("The finish flat extends beyond the positive x boundary of the tile.")

    center_x = 0.5 * size_x
    center_y = 0.5 * size_y
    y_min = center_y - 0.5 * walkable_width
    y_max = center_y + 0.5 * walkable_width

    def to_tile_x(local_x: float) -> float:
        return center_x + local_x

    meshes = [
        trimesh.creation.box(
            extents=(size_x, size_y, ground_thickness),
            transform=trimesh.transformations.translation_matrix(
                (center_x, center_y, -0.5 * ground_thickness)
            ),
        )
    ]
    for step_index in range(int(cfg.num_steps)):
        x_min = geometry.stairs_start_x + step_index * geometry.tread_depth
        x_max = (
            geometry.slope_top_x
            if step_index == int(cfg.num_steps) - 1
            else x_min + geometry.tread_depth
        )
        height = (step_index + 1) * geometry.step_height
        x_min_tile = to_tile_x(x_min)
        x_max_tile = to_tile_x(x_max)
        step_mesh = trimesh.creation.box(
            extents=(x_max - x_min, walkable_width, height),
            transform=trimesh.transformations.translation_matrix(
                (0.5 * (x_min_tile + x_max_tile), center_y, 0.5 * height)
            ),
        )
        x_center_tile = 0.5 * (x_min_tile + x_max_tile)
        step_mesh.vertices[step_mesh.vertices[:, 0] < x_center_tile, 0] = x_min_tile
        step_mesh.vertices[step_mesh.vertices[:, 0] >= x_center_tile, 0] = x_max_tile
        step_mesh.vertices[step_mesh.vertices[:, 1] < center_y, 1] = y_min
        step_mesh.vertices[step_mesh.vertices[:, 1] >= center_y, 1] = y_max
        step_mesh.vertices[step_mesh.vertices[:, 2] < 0.5 * height, 2] = 0.0
        step_mesh.vertices[step_mesh.vertices[:, 2] >= 0.5 * height, 2] = height
        meshes.append(step_mesh)
    meshes.append(
        _down_slope_prism(
            to_tile_x(geometry.slope_top_x),
            to_tile_x(geometry.slope_foot_x),
            y_min,
            y_max,
            geometry.stairs_height,
        )
    )

    origin = np.array([center_x, center_y, 0.0], dtype=np.float64)
    return meshes, origin
