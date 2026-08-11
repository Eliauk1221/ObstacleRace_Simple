"""Trimesh generator for the straight symmetric V-slope obstacle."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import trimesh

if TYPE_CHECKING:
    from .custom_terrains_cfg import MeshSymmetricSlopeTerrainCfg


def gap_width_for_difficulty(difficulty: float, gap_width_by_level: tuple[float, ...]) -> float:
    """Select the ordered gap width nearest to the requested curriculum difficulty."""
    if not gap_width_by_level:
        raise ValueError("gap_width_by_level must not be empty.")
    normalized = float(np.clip(difficulty, 0.0, 1.0))
    level = min(int(math.floor(normalized * len(gap_width_by_level))), len(gap_width_by_level) - 1)
    return float(gap_width_by_level[level])


def _triangular_prism(
    x_min: float,
    x_max: float,
    y_outer: float,
    y_inner: float,
    outer_height: float,
) -> trimesh.Trimesh:
    """Create one closed wedge whose top rises from the inner edge to the outer edge."""
    vertices = np.array(
        [
            [x_min, y_outer, 0.0],
            [x_min, y_outer, outer_height],
            [x_min, y_inner, 0.0],
            [x_max, y_outer, 0.0],
            [x_max, y_outer, outer_height],
            [x_max, y_inner, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 2, 1],
            [3, 4, 5],
            [0, 1, 4],
            [0, 4, 3],
            [1, 2, 5],
            [1, 5, 4],
            [0, 3, 5],
            [0, 5, 2],
        ],
        dtype=np.int64,
    )
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if mesh.volume < 0.0:
        mesh.invert()
    return mesh


def symmetric_slope_terrain(
    difficulty: float, cfg: MeshSymmetricSlopeTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a 3 m straight V-slope with a flat, collidable center gap."""
    size_x, size_y = map(float, cfg.size)
    obstacle_length = float(cfg.obstacle_length_m)
    obstacle_width = float(cfg.obstacle_width_m)
    finish_flat_length = float(cfg.finish_flat_length_m)
    slope_angle_deg = float(cfg.slope_angle_deg)
    ground_thickness = float(cfg.ground_thickness_m)
    gap_width = gap_width_for_difficulty(difficulty, tuple(cfg.gap_width_by_level))

    if obstacle_length <= 0.0 or obstacle_width <= 0.0:
        raise ValueError("obstacle_length_m and obstacle_width_m must be positive.")
    if 0.5 * (size_x - obstacle_length) < finish_flat_length:
        raise ValueError("The centered obstacle and finish flat must fit inside the terrain x extent.")
    if obstacle_width > size_y:
        raise ValueError("obstacle_width_m must fit inside the terrain y extent.")
    if not 0.0 < gap_width < obstacle_width:
        raise ValueError(f"gap width must lie in (0, {obstacle_width}), got {gap_width}.")
    if not 0.0 < slope_angle_deg < 90.0:
        raise ValueError(f"slope_angle_deg must lie in (0, 90), got {slope_angle_deg}.")
    if finish_flat_length <= 0.0 or ground_thickness <= 0.0:
        raise ValueError("finish_flat_length_m and ground_thickness_m must be positive.")

    center_x = 0.5 * size_x
    center_y = 0.5 * size_y
    x_min = center_x - 0.5 * obstacle_length
    x_max = center_x + 0.5 * obstacle_length
    y_min = center_y - 0.5 * obstacle_width
    y_max = center_y + 0.5 * obstacle_width
    gap_y_min = center_y - 0.5 * gap_width
    gap_y_max = center_y + 0.5 * gap_width
    side_run = 0.5 * (obstacle_width - gap_width)
    outer_height = side_run * math.tan(math.radians(slope_angle_deg))

    ground = trimesh.creation.box(
        extents=(size_x, size_y, ground_thickness),
        transform=trimesh.transformations.translation_matrix(
            (center_x, center_y, -0.5 * ground_thickness)
        ),
    )
    left_wedge = _triangular_prism(x_min, x_max, y_min, gap_y_min, outer_height)
    right_wedge = _triangular_prism(x_min, x_max, y_max, gap_y_max, outer_height)

    origin = np.array([center_x, center_y, 0.0], dtype=np.float64)
    return [ground, left_wedge, right_wedge], origin
