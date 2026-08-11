"""Exact triangle-mesh generator for rotated pyramid stairs."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import trimesh

if TYPE_CHECKING:
    from .custom_terrains_cfg import MeshSpiralPyramidStairsTerrainCfg


class SpiralStairGeometry(NamedTuple):
    """Difficulty-scaled height and fixed horizontal stair geometry."""

    step_height: float
    half_extents: tuple[float, ...]
    angles_rad: tuple[float, ...]
    inner_tread_depth: float
    outer_tread_depth: float


def resolve_spiral_stair_geometry(
    difficulty: float,
    cfg: MeshSpiralPyramidStairsTerrainCfg,
) -> SpiralStairGeometry:
    """Resolve exact nested-square boundaries from the center platform outwards."""
    size_x, size_y = map(float, cfg.size)
    num_steps = int(cfg.num_steps)
    platform_width = float(cfg.platform_width)
    center_tread_depth = float(cfg.center_tread_depth)
    usable_tread_width = float(cfg.usable_tread_width)
    border_width = float(cfg.border_width)
    ground_thickness = float(cfg.ground_thickness)
    rotation_deg = float(cfg.rotation_per_step_deg)

    if size_x <= 0.0 or size_y <= 0.0:
        raise ValueError(f"Spiral stair size must be positive, got {cfg.size}.")
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}.")
    if platform_width <= 0.0 or platform_width >= min(size_x, size_y):
        raise ValueError("platform_width must be positive and smaller than the terrain size.")
    if center_tread_depth <= 0.0 or usable_tread_width <= 0.0:
        raise ValueError("center_tread_depth and usable_tread_width must be positive.")
    if usable_tread_width > platform_width:
        raise ValueError("usable_tread_width must not exceed platform_width.")
    if border_width < 0.0:
        raise ValueError(f"border_width must be non-negative, got {border_width}.")
    if ground_thickness <= 0.0:
        raise ValueError(f"ground_thickness must be positive, got {ground_thickness}.")
    if not 0.0 < rotation_deg < 90.0:
        raise ValueError("rotation_per_step_deg must lie in (0, 90).")

    height_min, height_max = map(float, cfg.step_height_range)
    if height_min <= 0.0 or height_max < height_min:
        raise ValueError(f"Invalid step_height_range: {cfg.step_height_range}.")
    normalized_difficulty = float(np.clip(difficulty, 0.0, 1.0))
    step_height = height_min + normalized_difficulty * (height_max - height_min)

    rotation_rad = math.radians(rotation_deg)
    cos_rotation = math.cos(rotation_rad)
    sin_rotation = math.sin(rotation_rad)
    half_usable_width = 0.5 * usable_tread_width
    inner_tread_depth = center_tread_depth - half_usable_width * math.tan(rotation_rad)
    outer_tread_depth = center_tread_depth + half_usable_width * math.tan(rotation_rad)
    if inner_tread_depth <= 0.0:
        raise ValueError(
            "rotation_per_step_deg and usable_tread_width leave no positive inner tread depth."
        )

    # For adjacent boundaries a (outer) and b (inner), the center-line depth is
    # a - b / cos(rotation). Reverse this relation so the center platform stays exact.
    half_extents_inner_to_outer = [0.5 * platform_width]
    for _ in range(num_steps):
        inner_half_extent = half_extents_inner_to_outer[-1]
        outer_half_extent = inner_half_extent / cos_rotation + center_tread_depth
        half_extents_inner_to_outer.append(outer_half_extent)
    half_extents = tuple(reversed(half_extents_inner_to_outer))

    max_outer_half_extent = 0.5 * min(size_x, size_y) - border_width
    if half_extents[0] > max_outer_half_extent + 1.0e-9:
        raise ValueError(
            "Fixed spiral stair geometry does not fit inside the terrain and requested border: "
            f"outer half extent {half_extents[0]:.3f} m > {max_outer_half_extent:.3f} m."
        )

    for outer_half_extent, inner_half_extent in zip(half_extents, half_extents[1:]):
        min_clearance = outer_half_extent - inner_half_extent * (cos_rotation + sin_rotation)
        if min_clearance <= 0.0:
            raise ValueError(
                "Adjacent rotated stair boundaries overlap; increase center_tread_depth or reduce rotation."
            )

    direction = -1.0 if cfg.inverted else 1.0
    angles_rad = tuple(direction * level * rotation_rad for level in range(num_steps + 1))
    return SpiralStairGeometry(
        step_height=step_height,
        half_extents=half_extents,
        angles_rad=angles_rad,
        inner_tread_depth=inner_tread_depth,
        outer_tread_depth=outer_tread_depth,
    )


def _square_corners(
    center_xy: tuple[float, float], half_extent: float, angle: float, z: float
) -> np.ndarray:
    """Return counterclockwise corners for one exact rotated square."""
    local_xy = np.array(
        [
            [-half_extent, -half_extent],
            [half_extent, -half_extent],
            [half_extent, half_extent],
            [-half_extent, half_extent],
        ],
        dtype=np.float64,
    )
    cos_angle, sin_angle = math.cos(angle), math.sin(angle)
    rotation = np.array(
        [[cos_angle, -sin_angle], [sin_angle, cos_angle]], dtype=np.float64
    )
    corners_xy = local_xy @ rotation.T
    corners_xy += np.asarray(center_xy, dtype=np.float64)
    return np.column_stack((corners_xy, np.full(4, z, dtype=np.float64)))


def _horizontal_ring(
    outer_corners: np.ndarray, inner_corners: np.ndarray, z: float
) -> trimesh.Trimesh:
    """Triangulate the exact planar ring between two nested quadrilaterals."""
    vertices = np.vstack((outer_corners.copy(), inner_corners.copy()))
    vertices[:, 2] = z
    faces: list[list[int]] = []
    for index in range(4):
        next_index = (index + 1) % 4
        faces.append([index, next_index, 4 + next_index])
        faces.append([index, 4 + next_index, 4 + index])
    return trimesh.Trimesh(
        vertices=vertices, faces=np.asarray(faces, dtype=np.int64), process=False
    )


def _square_surface(corners: np.ndarray, z: float, upward: bool) -> trimesh.Trimesh:
    """Create one square face with a requested normal direction."""
    vertices = corners.copy()
    vertices[:, 2] = z
    faces = (
        np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        if upward
        else np.array([[0, 2, 1], [0, 3, 2]], dtype=np.int64)
    )
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _vertical_square_wall(
    corners: np.ndarray,
    z_low: float,
    z_high: float,
    normal_outward: bool,
) -> trimesh.Trimesh:
    """Create four exact vertical faces along a square boundary."""
    if z_high <= z_low:
        raise ValueError(f"Vertical wall requires z_high > z_low, got {z_high} <= {z_low}.")
    low = corners.copy()
    high = corners.copy()
    low[:, 2] = z_low
    high[:, 2] = z_high
    vertices = np.vstack((low, high))
    faces: list[list[int]] = []
    for index in range(4):
        next_index = (index + 1) % 4
        quad = [index, next_index, 4 + next_index, 4 + index]
        if normal_outward:
            faces.append([quad[0], quad[1], quad[2]])
            faces.append([quad[0], quad[2], quad[3]])
        else:
            faces.append([quad[0], quad[2], quad[1]])
            faces.append([quad[0], quad[3], quad[2]])
    return trimesh.Trimesh(
        vertices=vertices, faces=np.asarray(faces, dtype=np.int64), process=False
    )


def _railing_segment(
    start_xy: tuple[float, float],
    stop_xy: tuple[float, float],
    bottom_z: float,
    top_z: float,
    thickness: float,
) -> trimesh.Trimesh:
    """Create one vertical segment in a spiral fan railing."""
    delta_x = stop_xy[0] - start_xy[0]
    delta_y = stop_xy[1] - start_xy[1]
    length = math.hypot(delta_x, delta_y) + min(0.02, 0.5 * thickness)
    height = top_z - bottom_z
    if length <= 0.0 or height <= 0.0:
        raise ValueError("Spiral railing segments must have positive length and height.")
    transform = trimesh.transformations.rotation_matrix(
        math.atan2(delta_y, delta_x), (0.0, 0.0, 1.0)
    )
    transform[:3, 3] = (
        0.5 * (start_xy[0] + stop_xy[0]),
        0.5 * (start_xy[1] + stop_xy[1]),
        0.5 * (bottom_z + top_z),
    )
    return trimesh.creation.box(extents=(length, thickness, height), transform=transform)


def _spiral_stair_surface_mesh(
    cfg: MeshSpiralPyramidStairsTerrainCfg,
    geometry: SpiralStairGeometry,
) -> tuple[trimesh.Trimesh, list[np.ndarray]]:
    """Build one closed stair solid from exact tread and riser faces."""
    size_x, size_y = map(float, cfg.size)
    center_xy = (0.5 * size_x, 0.5 * size_y)
    height_direction = -1.0 if cfg.inverted else 1.0
    num_steps = int(cfg.num_steps)
    step_height = geometry.step_height

    boundaries = [
        _square_corners(center_xy, half_extent, angle, 0.0)
        for half_extent, angle in zip(geometry.half_extents, geometry.angles_rad)
    ]
    tile_corners = np.array(
        [[0.0, 0.0, 0.0], [size_x, 0.0, 0.0], [size_x, size_y, 0.0], [0.0, size_y, 0.0]],
        dtype=np.float64,
    )

    pieces: list[trimesh.Trimesh] = []
    pieces.append(_horizontal_ring(tile_corners, boundaries[1], 0.0))
    for level in range(1, num_steps):
        level_z = height_direction * level * step_height
        pieces.append(_horizontal_ring(boundaries[level], boundaries[level + 1], level_z))

    platform_z = height_direction * num_steps * step_height
    pieces.append(_square_surface(boundaries[-1], platform_z, upward=True))

    for level in range(1, num_steps + 1):
        outer_z = height_direction * (level - 1) * step_height
        inner_z = height_direction * level * step_height
        pieces.append(
            _vertical_square_wall(
                boundaries[level],
                min(outer_z, inner_z),
                max(outer_z, inner_z),
                normal_outward=not cfg.inverted,
            )
        )

    bottom_z = min(0.0, platform_z) - float(cfg.ground_thickness)
    pieces.append(_square_surface(tile_corners, bottom_z, upward=False))
    pieces.append(_vertical_square_wall(tile_corners, bottom_z, 0.0, normal_outward=True))

    mesh = trimesh.util.concatenate(pieces)
    mesh.merge_vertices()
    mesh.remove_unreferenced_vertices()
    return mesh, boundaries


def spiral_pyramid_stairs_terrain(
    difficulty: float, cfg: MeshSpiralPyramidStairsTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate fixed-run rotated stairs with exact, non-rasterized edges."""
    geometry = resolve_spiral_stair_geometry(difficulty, cfg)
    terrain_mesh, boundaries = _spiral_stair_surface_mesh(cfg, geometry)
    meshes = [terrain_mesh]

    tile_probability = float(cfg.railing_tile_probability)
    side_probability = float(cfg.railing_side_probability)
    if not 0.0 <= tile_probability <= 1.0 or not 0.0 <= side_probability <= 1.0:
        raise ValueError("railing probabilities must lie in [0, 1].")
    if cfg.railing_thickness <= 0.0 or cfg.railing_top_clearance <= 0.0:
        raise ValueError("railing thickness and top clearance must be positive.")

    if tile_probability > 0.0 and np.random.uniform() < tile_probability:
        selected_corners = np.random.uniform(size=4) < side_probability
        if cfg.ensure_one_railing and not selected_corners.any():
            selected_corners[int(np.random.randint(0, 4))] = True

        height_direction = -1.0 if cfg.inverted else 1.0
        highest_z = max(0.0, height_direction * int(cfg.num_steps) * geometry.step_height)
        top_z = highest_z + float(cfg.railing_top_clearance)
        for level in range(int(cfg.num_steps)):
            outer_z = height_direction * level * geometry.step_height
            inner_z = height_direction * (level + 1) * geometry.step_height
            bottom_z = min(outer_z, inner_z)
            for corner_index, selected in enumerate(selected_corners):
                if not selected:
                    continue
                start_xy = tuple(boundaries[level][corner_index, :2])
                stop_xy = tuple(boundaries[level + 1][corner_index, :2])
                meshes.append(
                    _railing_segment(
                        start_xy,
                        stop_xy,
                        bottom_z,
                        top_z,
                        float(cfg.railing_thickness),
                    )
                )

    center_x, center_y = 0.5 * float(cfg.size[0]), 0.5 * float(cfg.size[1])
    platform_z = (-1.0 if cfg.inverted else 1.0) * int(cfg.num_steps) * geometry.step_height
    origin = np.array([center_x, center_y, platform_z], dtype=np.float64)
    return meshes, origin
