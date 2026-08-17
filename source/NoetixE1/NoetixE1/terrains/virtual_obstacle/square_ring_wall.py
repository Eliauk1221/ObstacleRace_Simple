# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import trimesh
from isaaclab.markers import VisualizationMarkers

from .virtual_obstacle_base import VirtualObstacleBase

if TYPE_CHECKING:
    from .square_ring_wall_cfg import SquareRingWallOuterBandVirtualObstacleCfg


def _square_cell_flat_index(row, col, num_rows: int, num_cols: int, curriculum: bool):
    """Map a square-ring terrain cell to its generation-order index."""
    if curriculum:
        return col * num_rows + row
    return row * num_cols + col


class SquareRingWallOuterBandVirtualObstacle(VirtualObstacleBase):
    """Analytical inner- or outer-side bands around each square-ring wall hurdle."""

    def __init__(self, cfg: SquareRingWallOuterBandVirtualObstacleCfg):
        self.cfg = cfg
        self._terrain_generator = None
        self.device = torch.device("cpu")
        self._active_cells: torch.Tensor | None = None
        self._cell_centers_xy: torch.Tensor | None = None
        self._ring_inner_halves: torch.Tensor | None = None
        self._ring_outer_halves: torch.Tensor | None = None
        self._ring_z_tops: torch.Tensor | None = None
        self._num_rows = 0
        self._num_cols = 0
        self._curriculum = False
        self._sub_size = (0.0, 0.0)

    def bind_terrain_generator(self, terrain_generator) -> None:
        self._terrain_generator = terrain_generator

    def generate(self, mesh: trimesh.Trimesh, device: torch.device | str = "cpu") -> None:
        del mesh
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self._collect_cell_params_from_generator()

    def _collect_cell_params_from_generator(self) -> None:
        tg = self._terrain_generator
        if tg is None:
            raise RuntimeError("Square-ring outer bands require a bound terrain generator.")
        keys = getattr(tg, "_subterrain_keys", None)
        cfgs = getattr(tg, "_subterrain_specific_cfgs", None)
        if keys is None or cfgs is None:
            raise RuntimeError("Square-ring outer bands require per-cell terrain metadata.")

        num_rows, num_cols = int(tg.cfg.num_rows), int(tg.cfg.num_cols)
        num_cells = num_rows * num_cols
        if len(keys) != num_cells or len(cfgs) != num_cells:
            raise RuntimeError(
                "Square-ring terrain metadata is incomplete: "
                f"keys={len(keys)}, configs={len(cfgs)}, expected={num_cells}."
            )

        curriculum = bool(tg.cfg.curriculum)
        size_x, size_y = float(tg.cfg.size[0]), float(tg.cfg.size[1])
        active = np.zeros(num_cells, dtype=np.bool_)
        centers_xy = np.zeros((num_cells, 2), dtype=np.float32)
        inner_halves = np.zeros((num_cells, 3), dtype=np.float32)
        outer_halves = np.zeros((num_cells, 3), dtype=np.float32)
        z_tops = np.zeros((num_cells, 3), dtype=np.float32)

        matched = 0
        for index, (key, sub_cfg) in enumerate(zip(keys, cfgs)):
            if key != self.cfg.subterrain_key:
                continue
            ring_outer_halves = _hurdle_ring_outer_halves_m(sub_cfg)
            ring_widths = _hurdle_ring_widths_m(sub_cfg)
            ring_heights = _hurdle_ring_heights_m(sub_cfg)
            if ring_outer_halves is None or ring_widths is None or ring_heights is None:
                raise RuntimeError("square_ring_wall must use HfHurdleTerrainCfg-compatible geometry.")
            if curriculum:
                row, col = index % num_rows, index // num_rows
            else:
                row, col = index // num_cols, index % num_cols
            centers_xy[index] = (
                (row + 0.5) * size_x - num_rows * size_x * 0.5,
                (col + 0.5) * size_y - num_cols * size_y * 0.5,
            )
            inner_halves[index] = ring_outer_halves - ring_widths
            outer_halves[index] = ring_outer_halves
            z_tops[index] = ring_heights + float(self.cfg.z_top_margin)
            active[index] = True
            matched += 1

        self._num_rows, self._num_cols = num_rows, num_cols
        self._curriculum = curriculum
        self._sub_size = (size_x, size_y)
        self._active_cells = torch.as_tensor(active, dtype=torch.bool, device=self.device)
        self._cell_centers_xy = torch.as_tensor(centers_xy, dtype=torch.float32, device=self.device)
        self._ring_inner_halves = torch.as_tensor(inner_halves, dtype=torch.float32, device=self.device)
        self._ring_outer_halves = torch.as_tensor(outer_halves, dtype=torch.float32, device=self.device)
        self._ring_z_tops = torch.as_tensor(z_tops, dtype=torch.float32, device=self.device)
        print(
            f"[INFO] SquareRingWallOuterBandVirtualObstacle: {matched} cells, "
            f"side={self.cfg.band_side}, band_width={self.cfg.band_width:.3f} m."
        )

    def disable_visualizer(self) -> None:
        if hasattr(self, "_band_visualizer"):
            self._band_visualizer.set_visibility(False)

    def visualize(self) -> None:
        if (
            self._active_cells is None
            or self._cell_centers_xy is None
            or self._ring_inner_halves is None
            or self._ring_outer_halves is None
            or self._ring_z_tops is None
        ):
            raise RuntimeError("Square-ring side bands have not been generated.")

        active_indices = torch.nonzero(self._active_cells, as_tuple=False).flatten()
        if active_indices.numel() == 0:
            self.disable_visualizer()
            return

        translations: list[torch.Tensor] = []
        scales: list[torch.Tensor] = []
        band_width = float(self.cfg.band_width)
        z_bottom = float(self.cfg.z_bottom)
        for cell_index in active_indices.tolist():
            center_xy = self._cell_centers_xy[cell_index]
            for ring_index in range(3):
                if self.cfg.band_side == "inner":
                    shell_outer = float(self._ring_inner_halves[cell_index, ring_index])
                    shell_inner = shell_outer - band_width
                elif self.cfg.band_side == "outer":
                    shell_inner = float(self._ring_outer_halves[cell_index, ring_index])
                    shell_outer = shell_inner + band_width
                else:
                    raise ValueError(f"Unsupported square-ring band_side: {self.cfg.band_side!r}.")
                if shell_inner < 0.0:
                    raise ValueError("Square-ring visualization band extends past the ring center.")

                z_top = float(self._ring_z_tops[cell_index, ring_index])
                z_height = z_top - z_bottom
                if z_height <= 0.0:
                    continue
                z_center = z_bottom + 0.5 * z_height
                side_center = shell_inner + 0.5 * band_width
                cx, cy = float(center_xy[0]), float(center_xy[1])

                translations.extend(
                    (
                        torch.tensor((cx, cy + side_center, z_center), device=self.device),
                        torch.tensor((cx, cy - side_center, z_center), device=self.device),
                        torch.tensor((cx + side_center, cy, z_center), device=self.device),
                        torch.tensor((cx - side_center, cy, z_center), device=self.device),
                    )
                )
                scales.extend(
                    (
                        torch.tensor((2.0 * shell_outer, band_width, z_height), device=self.device),
                        torch.tensor((2.0 * shell_outer, band_width, z_height), device=self.device),
                        torch.tensor((band_width, 2.0 * shell_inner, z_height), device=self.device),
                        torch.tensor((band_width, 2.0 * shell_inner, z_height), device=self.device),
                    )
                )

        if not translations:
            self.disable_visualizer()
            return
        if not hasattr(self, "_band_visualizer"):
            self._band_visualizer = VisualizationMarkers(self.cfg.visualizer)
        self._band_visualizer.set_visibility(True)
        self._band_visualizer.visualize(
            translations=torch.stack(translations),
            scales=torch.stack(scales),
        )

    def get_points_penetration_offset(self, points: torch.Tensor) -> torch.Tensor:
        if (
            self._active_cells is None
            or self._ring_inner_halves is None
            or self._ring_outer_halves is None
            or self._ring_z_tops is None
        ):
            raise RuntimeError("Square-ring side bands have not been generated.")
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"Expected points with shape (N, 3), got {tuple(points.shape)}.")

        size_x, size_y = self._sub_size
        total_x, total_y = size_x * self._num_rows, size_y * self._num_cols
        row = torch.floor((points[:, 0] + total_x * 0.5) / size_x).long()
        col = torch.floor((points[:, 1] + total_y * 0.5) / size_y).long()
        valid = (row >= 0) & (row < self._num_rows) & (col >= 0) & (col < self._num_cols)
        index = _square_cell_flat_index(
            row.clamp(0, self._num_rows - 1),
            col.clamp(0, self._num_cols - 1),
            self._num_rows,
            self._num_cols,
            self._curriculum,
        ).long()
        active = valid & self._active_cells[index]
        if not torch.any(active):
            return torch.zeros_like(points)

        centers = self._cell_centers_xy[index]
        local_abs_x = torch.abs(points[:, 0] - centers[:, 0])
        local_abs_y = torch.abs(points[:, 1] - centers[:, 1])
        chebyshev_radius = torch.maximum(local_abs_x, local_abs_y)

        band_width = float(self.cfg.band_width)
        if self.cfg.band_side == "inner":
            boundary_halves = self._ring_inner_halves[index]
            xy_distance = boundary_halves - chebyshev_radius[:, None]
        elif self.cfg.band_side == "outer":
            boundary_halves = self._ring_outer_halves[index]
            xy_distance = chebyshev_radius[:, None] - boundary_halves
        else:
            raise ValueError(f"Unsupported square-ring band_side: {self.cfg.band_side!r}.")
        in_xy = (xy_distance >= 0.0) & (xy_distance < band_width)
        xy_depth = torch.minimum(xy_distance, band_width - xy_distance)

        z_bottom = float(self.cfg.z_bottom)
        z_tops = self._ring_z_tops[index]
        point_z = points[:, 2, None]
        in_z = (point_z >= z_bottom) & (point_z <= z_tops)
        z_depth = torch.minimum(point_z - z_bottom, z_tops - point_z)
        depth_by_ring = torch.minimum(xy_depth, z_depth)
        depth_by_ring = torch.where(
            in_xy & in_z & (depth_by_ring > 0.0),
            depth_by_ring,
            torch.zeros_like(depth_by_ring),
        )
        depth = depth_by_ring.max(dim=1).values

        offset = torch.zeros_like(points)
        inside = active & (depth > 0.0)
        offset[inside, 2] = depth[inside]
        return offset


def _hurdle_ring_outer_halves_m(cfg) -> np.ndarray | None:
    """Return the three square-ring outer half-widths in meters."""
    if not hasattr(cfg, "hurdle_width_pixels") or not hasattr(cfg, "platform_width"):
        return None
    if isinstance(cfg.hurdle_width_pixels, tuple):
        if len(cfg.hurdle_width_pixels) != 3:
            raise ValueError("hurdle_width_pixels must have exactly 3 elements")
        widths_pixels = [int(width) for width in cfg.hurdle_width_pixels]
    else:
        widths_pixels = [int(cfg.hurdle_width_pixels)] * 3
    if any(width < 1 for width in widths_pixels):
        raise ValueError("hurdle_width_pixels values must be positive")

    horizontal_scale = float(cfg.horizontal_scale)
    width_pixels = int(float(cfg.size[0]) / horizontal_scale)
    length_pixels = int(float(cfg.size[1]) / horizontal_scale)
    platform_width_pixels = int(float(cfg.platform_width) / horizontal_scale)
    available_space_pixels = min(width_pixels, length_pixels) // 2 - platform_width_pixels // 2
    remaining_space = available_space_pixels - sum(widths_pixels)
    uniform_gap_pixels = remaining_space // 4
    if uniform_gap_pixels < 0:
        raise ValueError("Square-ring widths and platform do not fit inside the terrain cell.")

    current_inner_size = platform_width_pixels
    outer_halves = []
    for width_pixels_i in widths_pixels:
        current_inner_size += 2 * uniform_gap_pixels + 2 * width_pixels_i
        outer_halves.append((current_inner_size // 2) * horizontal_scale)
    return np.asarray(outer_halves, dtype=np.float32)


def _hurdle_ring_widths_m(cfg) -> np.ndarray | None:
    """Return the physical width of each ring in meters."""
    if not hasattr(cfg, "hurdle_width_pixels") or not hasattr(cfg, "horizontal_scale"):
        return None
    if isinstance(cfg.hurdle_width_pixels, tuple):
        if len(cfg.hurdle_width_pixels) != 3:
            raise ValueError("hurdle_width_pixels must have exactly 3 elements")
        widths_pixels = [int(width) for width in cfg.hurdle_width_pixels]
    else:
        widths_pixels = [int(cfg.hurdle_width_pixels)] * 3
    if any(width < 1 for width in widths_pixels):
        raise ValueError("hurdle_width_pixels values must be positive")
    return np.asarray(widths_pixels, dtype=np.float32) * float(cfg.horizontal_scale)


def _hurdle_ring_heights_m(cfg) -> np.ndarray | None:
    """Return fixed or curriculum-resolved height for each ring."""
    if not hasattr(cfg, "hurdle_height_range"):
        return None
    fixed_heights = getattr(cfg, "hurdle_heights_m", None)
    if fixed_heights is not None:
        if len(fixed_heights) != 3:
            raise ValueError("hurdle_heights_m must have exactly 3 elements")
        heights = np.asarray(fixed_heights, dtype=np.float32)
    else:
        difficulty = float(getattr(cfg, "difficulty", 0.0))
        height = float(cfg.hurdle_height_range[0]) + difficulty * (
            float(cfg.hurdle_height_range[1]) - float(cfg.hurdle_height_range[0])
        )
        heights = np.full(3, height, dtype=np.float32)
    if np.any(heights < 0.0):
        raise ValueError("Square-ring heights must be non-negative.")
    return heights
