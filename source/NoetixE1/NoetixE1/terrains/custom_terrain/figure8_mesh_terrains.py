# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

"""Trimesh 参数化 8 字桥：光滑圆环桥面，参数与 ``HfFigure8BridgeTerrainCfg`` 对齐。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import trimesh

if TYPE_CHECKING:
    from . import custom_terrains_cfg


def _resolve_bridge_width(difficulty: float, cfg: custom_terrains_cfg.MeshFigure8BridgeTerrainCfg) -> float:
    widths = getattr(cfg, "bridge_width_by_level", None)
    if widths is not None and len(widths) > 0:
        level_max = len(widths) - 1
        level_idx = int(np.clip(np.round(float(difficulty) * level_max), 0, level_max))
        return float(widths[level_idx])
    return float(cfg.bridge_width_range[0] + difficulty * (cfg.bridge_width_range[1] - cfg.bridge_width_range[0]))


def _resolve_holes_depth(difficulty: float, cfg: custom_terrains_cfg.MeshFigure8BridgeTerrainCfg) -> float:
    return float(cfg.holes_depth[0] + difficulty * (cfg.holes_depth[1] - cfg.holes_depth[0]))


def _remove_faces_in_axis_cutout(
    mesh: trimesh.Trimesh,
    cutout_half_w: float,
    cutout_half_l: float,
) -> trimesh.Trimesh:
    """去掉中心矩形挖空区内的三角面（env 局部系，原点在 8 字中心）。"""
    if cutout_half_w <= 0.0 or cutout_half_l <= 0.0 or mesh.faces.shape[0] == 0:
        return mesh
    centers = mesh.triangles_center
    in_cutout = (np.abs(centers[:, 0]) <= cutout_half_w) & (np.abs(centers[:, 1]) <= cutout_half_l)
    if not np.any(in_cutout):
        return mesh
    keep = np.where(~in_cutout)[0]
    if keep.size == 0:
        return trimesh.Trimesh()
    return mesh.submesh([keep], append=True, repair=False)


def _build_annulus_prism(
    center_x: float,
    center_y: float,
    inner_r: float,
    outer_r: float,
    z_bottom: float,
    z_top: float,
    num_sections: int,
    cutout_half_w: float = 0.0,
    cutout_half_l: float = 0.0,
) -> trimesh.Trimesh:
    """用 ``trimesh.creation.annulus`` 生成闭合圆环棱柱，再裁掉中心挖空区。"""
    height = float(z_top) - float(z_bottom)
    if height <= 0.0 or outer_r <= inner_r:
        return trimesh.Trimesh()

    mesh = trimesh.creation.annulus(
        float(inner_r),
        float(outer_r),
        height,
        sections=max(16, int(num_sections)),
    )
    mesh.apply_translation((float(center_x), float(center_y), float(z_bottom)))
    return _remove_faces_in_axis_cutout(mesh, cutout_half_w, cutout_half_l)


def _build_pit_ground(
    size_x: float,
    size_y: float,
    holes_depth: float,
    ground_thickness: float = 0.1,
) -> trimesh.Trimesh:
    """坑底整格地面（顶面在 ``holes_depth``）。"""
    return trimesh.creation.box(
        extents=(size_x, size_y, ground_thickness),
        transform=trimesh.transformations.translation_matrix(
            (0.5 * size_x, 0.5 * size_y, holes_depth - 0.5 * ground_thickness)
        ),
    )


def build_figure8_bridge_mesh(
    size: tuple[float, float],
    inner_radius_m: float,
    bridge_width_m: float,
    bridge_height: float,
    holes_depth_m: float,
    center_cutout_width_m: float = 0.0,
    center_cutout_length_m: float = 0.0,
    samples_per_loop: int = 128,
    ground_thickness: float = 0.1,
) -> trimesh.Trimesh:
    """在 terrain mesh 坐标系（原点为格子左下角）中构建 8 字桥 mesh。"""
    from .figure8_geometry import centerline_radius_m

    inner_r = float(inner_radius_m)
    r_c = centerline_radius_m(inner_r, bridge_width_m)
    cutout_hw = 0.5 * float(center_cutout_width_m)
    cutout_hl = 0.5 * float(center_cutout_length_m)
    n_sec = max(16, int(samples_per_loop))

    sx, sy = float(size[0]), float(size[1])
    offset = np.array([0.5 * sx, 0.5 * sy, 0.0], dtype=np.float64)

    left = _build_annulus_prism(
        -r_c, 0.0, inner_r, inner_r + bridge_width_m, holes_depth_m, bridge_height, n_sec, cutout_hw, cutout_hl
    )
    right = _build_annulus_prism(
        r_c, 0.0, inner_r, inner_r + bridge_width_m, holes_depth_m, bridge_height, n_sec, cutout_hw, cutout_hl
    )

    parts: list[trimesh.Trimesh] = [_build_pit_ground(sx, sy, holes_depth_m, ground_thickness)]
    for part in (left, right):
        if part.vertices.shape[0] > 0:
            part.apply_translation(offset)
            parts.append(part)

    mesh = trimesh.util.concatenate(parts)
    mesh.merge_vertices()
    mesh.remove_unreferenced_vertices()
    return mesh


def figure8_bridge_mesh_terrain(
    difficulty: float, cfg: custom_terrains_cfg.MeshFigure8BridgeTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Trimesh 版闭合 8 字桥：双圆环 + 坑底 + 可选中心挖空。"""
    bridge_width = _resolve_bridge_width(difficulty, cfg)
    holes_depth = _resolve_holes_depth(difficulty, cfg)
    bridge_height = float(cfg.bridge_height)

    mesh = build_figure8_bridge_mesh(
        size=cfg.size,
        inner_radius_m=float(cfg.inner_radius_m),
        bridge_width_m=bridge_width,
        bridge_height=bridge_height,
        holes_depth_m=holes_depth,
        center_cutout_width_m=float(getattr(cfg, "center_cutout_width_m", 0.0)),
        center_cutout_length_m=float(getattr(cfg, "center_cutout_length_m", 0.0)),
        samples_per_loop=int(getattr(cfg, "samples_per_loop", 128)),
    )

    # 与 HF ``height_field_to_mesh`` 一致：中心附近最高点为桥面高度，供 env_origin / 出生高度对齐。
    origin = np.array([cfg.size[0] / 2.0, cfg.size[1] / 2.0, bridge_height], dtype=np.float64)
    return [mesh], origin
