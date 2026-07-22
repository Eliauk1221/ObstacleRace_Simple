# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

"""双圆相切 8 字桥几何：内缘半径 + 固定桥宽，中心线在两圆上。"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import torch

Figure8Loop = Literal["left", "right"]


def _torch_device(device: torch.device | str) -> torch.device:
    return device if isinstance(device, torch.device) else torch.device(device)


def _circle_angles(num: int, device: torch.device | str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """``[0, 2π)`` 上均匀 ``num`` 个角（不含重复端点，与 ``np.linspace(..., endpoint=False)`` 一致）。"""
    n = int(num)
    dev = _torch_device(device)
    if n <= 0:
        return torch.empty(0, device=dev, dtype=dtype)
    return torch.arange(n, device=dev, dtype=dtype) * (2.0 * math.pi / n)


def centerline_radius_m(inner_radius_m: float, bridge_width_m: float) -> float:
    """桥面中心线相对圆心的半径（米）。"""
    return float(inner_radius_m) + 0.5 * float(bridge_width_m)


def outer_radius_m(inner_radius_m: float, bridge_width_m: float) -> float:
    """桥面外缘半径（米）。"""
    return float(inner_radius_m) + float(bridge_width_m)


def bridge_width_from_difficulty(
    difficulty: float,
    bridge_width_by_level: tuple[float, ...] | list[float],
) -> float:
    """将 ``difficulty∈[0,1]`` 映射到离散桥宽档位。"""
    widths = tuple(float(w) for w in bridge_width_by_level)
    if len(widths) == 0:
        raise ValueError("bridge_width_by_level must not be empty.")
    level_max = len(widths) - 1
    level_idx = int(max(0, min(level_max, round(float(difficulty) * level_max))))
    return widths[level_idx]


def circle_centers_x(centerline_radius_m: float) -> tuple[float, float]:
    """左右圆心 x 坐标（局部系原点在 8 字中心）。"""
    r = float(centerline_radius_m)
    return (-r, r)


def centerline_points_numpy(
    inner_radius_m: float,
    bridge_width_m: float,
    samples_per_loop: int,
) -> np.ndarray:
    """闭合折线用采样：左圆 + 右圆各一圈，形状 ``(2*N, 2)``。"""
    r_c = centerline_radius_m(inner_radius_m, bridge_width_m)
    n = int(samples_per_loop)
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    left = np.stack([-r_c + r_c * np.cos(theta), r_c * np.sin(theta)], axis=1)
    right = np.stack([r_c + r_c * np.cos(theta), r_c * np.sin(theta)], axis=1)
    return np.concatenate([left, right], axis=0).astype(np.float64)


def centerline_points_torch(
    inner_radius_m: float,
    bridge_width_m: float,
    samples_per_loop: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Torch 版中心线点 ``(2*N, 2)``。"""
    r_c = centerline_radius_m(inner_radius_m, bridge_width_m)
    theta = _circle_angles(samples_per_loop, device)
    left = torch.stack([-r_c + r_c * torch.cos(theta), r_c * torch.sin(theta)], dim=1)
    right = torch.stack([r_c + r_c * torch.cos(theta), r_c * torch.sin(theta)], dim=1)
    return torch.cat([left, right], dim=0)


def loop_centerline_xy_yaw(
    loop: Figure8Loop,
    theta: torch.Tensor,
    inner_radius_m: float,
    bridge_width_m: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """单圆上的中心线点与切向 yaw（env 局部坐标）。"""
    r_c = centerline_radius_m(inner_radius_m, bridge_width_m)
    if loop == "left":
        x = -r_c + r_c * torch.cos(theta)
        y = r_c * torch.sin(theta)
        dx = -r_c * torch.sin(theta)
        dy = r_c * torch.cos(theta)
    else:
        x = r_c + r_c * torch.cos(theta)
        y = r_c * torch.sin(theta)
        dx = -r_c * torch.sin(theta)
        dy = r_c * torch.cos(theta)
    yaw = torch.atan2(dy, dx)
    return x, y, yaw


def sample_path_spawn_xy_yaw(
    num_samples: int,
    device: torch.device,
    inner_radius_m: float,
    bridge_width_m: float,
    path_loop_ranges: list[dict],
    position_noise: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """在指定圆环的弧段上采样出生点。

    ``path_loop_ranges`` 元素示例::
        {"loop": "right", "theta": (0.55 * math.pi, 0.95 * math.pi)}
    """
    if num_samples <= 0:
        empty = torch.empty(0, device=device)
        return empty, empty, empty

    n_ranges = len(path_loop_ranges)
    range_idx = torch.randint(0, n_ranges, (num_samples,), device=device)

    theta = torch.empty(num_samples, device=device, dtype=torch.float32)
    x = torch.empty(num_samples, device=device, dtype=torch.float32)
    y = torch.empty(num_samples, device=device, dtype=torch.float32)
    yaw = torch.empty(num_samples, device=device, dtype=torch.float32)

    for i, spec in enumerate(path_loop_ranges):
        mask = range_idx == i
        count = int(mask.sum().item())
        if count == 0:
            continue
        loop: Figure8Loop = spec["loop"]
        th0, th1 = spec["theta"]
        th = torch.empty(count, device=device, dtype=torch.float32).uniform_(float(th0), float(th1))
        xi, yi, yawi = loop_centerline_xy_yaw(loop, th, inner_radius_m, bridge_width_m)
        theta[mask] = th
        x[mask] = xi
        y[mask] = yi
        yaw[mask] = yawi

    if position_noise > 0.0:
        x = x + torch.empty_like(x).uniform_(-position_noise, position_noise)
        y = y + torch.empty_like(y).uniform_(-position_noise, position_noise)

    return x, y, yaw


def bridge_width_for_terrain_levels(
    terrain_levels: torch.Tensor,
    num_rows: int,
    bridge_width_by_level: tuple[float, ...] | list[float],
) -> torch.Tensor:
    """按课程行号映射离散桥宽，形状与 ``terrain_levels`` 相同。"""
    return bridge_width_from_terrain_difficulty(
        terrain_difficulty_from_levels(terrain_levels, num_rows),
        bridge_width_by_level,
    )


def terrain_difficulty_from_levels(
    terrain_levels: torch.Tensor, num_rows: int
) -> torch.Tensor:
    """与 ``figure8_bridge_terrain`` 一致的 difficulty：``row / (num_rows - 1)``，单行时为 0。"""
    if int(num_rows) <= 1:
        return torch.zeros_like(terrain_levels, dtype=torch.float32)
    denom = float(int(num_rows) - 1)
    return terrain_levels.float().clamp(0.0, denom) / denom


def bridge_width_from_terrain_difficulty(
    difficulty: torch.Tensor,
    bridge_width_by_level: tuple[float, ...] | list[float],
) -> torch.Tensor:
    """与 ``figure8_bridge_terrain`` 一致：``round(difficulty * level_max)`` 取离散桥宽。"""
    widths = torch.tensor(list(bridge_width_by_level), device=difficulty.device, dtype=torch.float32)
    if widths.numel() == 0:
        raise ValueError("bridge_width_by_level must not be empty.")
    level_max = widths.numel() - 1
    idx = torch.round(difficulty * float(level_max)).long().clamp(0, level_max)
    return widths[idx]


def centerline_radius_batch(
    inner_radius_m: float, bridge_width_m: torch.Tensor
) -> torch.Tensor:
    """中心线半径 ``r_c = inner + bridge_width/2``（逐 env）。"""
    return float(inner_radius_m) + 0.5 * bridge_width_m


def path_param_delta_to_arc_m(delta_t: torch.Tensor, centerline_radius_m: torch.Tensor) -> torch.Tensor:
    """路径参数正向增量（弧度）→ 沿中心线弧长（米）。

    双圆 8 字折线中，单圆占路径参数约 ``pi`` 弧度、弧长 ``2*pi*r_c``，故 ``ds ≈ max(0, dt) * 2*r_c``。
    """
    return torch.clamp(delta_t, min=0.0) * (2.0 * centerline_radius_m)


def half_loop_arc_length_m(
    terrain_levels: torch.Tensor,
    num_rows: int,
    inner_radius_m: float,
    bridge_width_by_level: tuple[float, ...] | list[float],
) -> torch.Tensor:
    """单圆半周弧长 ``pi * r_c``（课程 move_up 阈值）。"""
    bw = bridge_width_for_terrain_levels(terrain_levels, num_rows, bridge_width_by_level)
    r_c = centerline_radius_batch(inner_radius_m, bw)
    return math.pi * r_c


def center_cutout_mask_numpy(
    px_m: np.ndarray,
    py_m: np.ndarray,
    cutout_width_m: float,
    cutout_length_m: float,
) -> np.ndarray:
    """原点轴对齐矩形挖空掩码（True=坑底，无桥面）。宽度沿 x、长度沿 y。"""
    if float(cutout_width_m) <= 0.0 or float(cutout_length_m) <= 0.0:
        return np.zeros_like(px_m, dtype=bool)
    half_w = 0.5 * float(cutout_width_m)
    half_l = 0.5 * float(cutout_length_m)
    return (np.abs(px_m) <= half_w) & (np.abs(py_m) <= half_l)


def foot_xy_on_center_cutout_torch(
    foot_xy: torch.Tensor,
    cutout_width_m: float,
    cutout_length_m: float,
) -> torch.Tensor:
    """脚底 XY 是否落在原点矩形挖空区（坑底/中心空地），形状 ``(N,)`` bool。"""
    n = foot_xy.shape[0]
    device = foot_xy.device
    cw = float(cutout_width_m)
    cl = float(cutout_length_m)
    if cw <= 0.0 or cl <= 0.0:
        return torch.zeros(n, dtype=torch.bool, device=device)
    half_w = 0.5 * cw
    half_l = 0.5 * cl
    return (foot_xy[:, 0].abs() <= half_w) & (foot_xy[:, 1].abs() <= half_l)


def foot_xy_in_expanded_cutout_torch(
    foot_xy: torch.Tensor,
    cutout_width_m: float,
    cutout_length_m: float,
    margin_m: float = 0.2,
) -> torch.Tensor:
    """脚底 XY 是否落在中心挖空区外扩 ``margin_m`` 的矩形危险区内，形状 ``(N,)`` bool。"""
    n = foot_xy.shape[0]
    device = foot_xy.device
    cw = float(cutout_width_m)
    cl = float(cutout_length_m)
    margin = float(margin_m)
    if cw <= 0.0 or cl <= 0.0:
        return torch.zeros(n, dtype=torch.bool, device=device)
    half_w = 0.5 * cw + margin
    half_l = 0.5 * cl + margin
    return (foot_xy[:, 0].abs() <= half_w) & (foot_xy[:, 1].abs() <= half_l)


def foot_xy_on_figure8_inner_pit_torch(
    foot_xy: torch.Tensor,
    inner_radius_m: float,
    bridge_width_m: torch.Tensor,
) -> torch.Tensor:
    """圆环**内圈**坑底（``d < inner_r``）：有地面 mesh，但不在桥面环带上。

    从中心 cutout 上桥必经此区；终止判定需计入，否则 ``leave_valid`` 误杀。
    """
    n = foot_xy.shape[0]
    device, dtype = foot_xy.device, foot_xy.dtype
    inner_r = torch.full((n,), float(inner_radius_m), device=device, dtype=dtype)
    bw = (
        torch.full((n,), float(bridge_width_m), device=device, dtype=dtype)
        if isinstance(bridge_width_m, (float, int))
        else bridge_width_m.to(device=device, dtype=dtype).reshape(-1)
    )
    r_c = inner_r + 0.5 * bw
    d_left = torch.hypot(foot_xy[:, 0] + r_c, foot_xy[:, 1])
    d_right = torch.hypot(foot_xy[:, 0] - r_c, foot_xy[:, 1])
    return (d_left < inner_r) | (d_right < inner_r)


def foot_xy_on_figure8_support_torch(
    foot_xy: torch.Tensor,
    inner_radius_m: float,
    bridge_width_m: torch.Tensor,
    cutout_width_m: float = 0.0,
    cutout_length_m: float = 0.0,
    annulus_outer_margin_m: float = 0.0,
    annulus_inner_margin_m: float = 0.0,
    include_inner_pit: bool = False,
) -> torch.Tensor:
    """环带桥面、中心挖空，或（终止用）圆环内圈坑底，均视为有效支撑面。"""
    on_bridge = foot_xy_on_bridge_annulus_torch(
        foot_xy,
        inner_radius_m,
        bridge_width_m,
        outer_margin_m=annulus_outer_margin_m,
        inner_margin_m=annulus_inner_margin_m,
    )
    on_cutout = foot_xy_on_center_cutout_torch(foot_xy, cutout_width_m, cutout_length_m)
    on_pit = (
        foot_xy_on_figure8_inner_pit_torch(foot_xy, inner_radius_m, bridge_width_m)
        if include_inner_pit
        else torch.zeros_like(on_bridge)
    )
    return on_bridge | on_cutout | on_pit


def foot_xy_on_figure8_termination_valid_torch(
    foot_xy: torch.Tensor,
    inner_radius_m: float,
    bridge_width_m: torch.Tensor,
    cutout_width_m: float = 0.0,
    cutout_length_m: float = 0.0,
    annulus_outer_margin_m: float = 0.08,
    transition_y_extend_m: float = 0.12,
    transition_x_margin_m: float = 0.06,
) -> torch.Tensor:
    """终止用有效支撑：环带（外缘略扩）∪ cutout ∪ 中心上下过渡区（含内圈坑底）。

    上下过渡区 = cutout 沿 ±y 外扩 ``transition_y_extend_m``、沿 ±x 外扩 ``transition_x_margin_m``，
    仅在该走廊内将内圈坑底计入有效区，避免全局 inner_pit 导致掉桥不终止。
    """
    on_bridge = foot_xy_on_bridge_annulus_torch(
        foot_xy,
        inner_radius_m,
        bridge_width_m,
        outer_margin_m=float(annulus_outer_margin_m),
    )
    on_cutout = foot_xy_on_center_cutout_torch(foot_xy, cutout_width_m, cutout_length_m)

    cw = float(cutout_width_m)
    cl = float(cutout_length_m)
    if cw > 0.0 and cl > 0.0:
        half_w = 0.5 * cw + float(transition_x_margin_m)
        half_l = 0.5 * cl + float(transition_y_extend_m)
        in_transition = (foot_xy[:, 0].abs() <= half_w) & (foot_xy[:, 1].abs() <= half_l)
        on_pit = foot_xy_on_figure8_inner_pit_torch(foot_xy, inner_radius_m, bridge_width_m) & in_transition
    else:
        on_pit = torch.zeros(foot_xy.shape[0], dtype=torch.bool, device=foot_xy.device)

    return on_bridge | on_cutout | on_pit


def foot_xy_in_figure8_vertical_transition_torch(
    foot_xy: torch.Tensor,
    cutout_width_m: float,
    cutout_length_m: float,
    transition_y_extend_m: float = 0.12,
    transition_x_margin_m: float = 0.06,
) -> torch.Tensor:
    """中心 cutout 上下延伸过渡走廊（含 cutout 本体），用于跌落判定的坑底豁免。"""
    n = foot_xy.shape[0]
    device = foot_xy.device
    cw = float(cutout_width_m)
    cl = float(cutout_length_m)
    if cw <= 0.0 or cl <= 0.0:
        return torch.zeros(n, dtype=torch.bool, device=device)
    half_w = 0.5 * cw + float(transition_x_margin_m)
    half_l = 0.5 * cl + float(transition_y_extend_m)
    return (foot_xy[:, 0].abs() <= half_w) & (foot_xy[:, 1].abs() <= half_l)


def sample_center_cutout_spawn_xy_yaw(
    num_samples: int,
    device: torch.device | str,
    cutout_width_m: float,
    cutout_length_m: float,
    base_yaw_angles: tuple[float, ...] | list[float],
    yaw_jitter_deg: float = 75.0,
    position_noise: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """在原点矩形挖空区内均匀采样出生点；朝向在 ``base_yaw_angles`` 中随机选一，并加 ±``yaw_jitter_deg``。"""
    if num_samples <= 0:
        empty = torch.empty(0, device=device)
        return empty, empty, empty

    dev = _torch_device(device)
    half_w = 0.5 * float(cutout_width_m)
    half_l = 0.5 * float(cutout_length_m)
    x = torch.empty(num_samples, device=dev, dtype=torch.float32).uniform_(-half_w, half_w)
    y = torch.empty(num_samples, device=dev, dtype=torch.float32).uniform_(-half_l, half_l)
    if position_noise > 0.0:
        x = x + torch.empty_like(x).uniform_(-position_noise, position_noise)
        y = y + torch.empty_like(y).uniform_(-position_noise, position_noise)
        x = x.clamp(-half_w, half_w)
        y = y.clamp(-half_l, half_l)

    bases = torch.tensor(list(base_yaw_angles), device=dev, dtype=torch.float32)
    if bases.numel() == 0:
        raise ValueError("base_yaw_angles must not be empty.")
    pick = torch.randint(0, bases.numel(), (num_samples,), device=dev)
    jitter = math.radians(float(yaw_jitter_deg))
    yaw = bases[pick] + torch.empty(num_samples, device=dev, dtype=torch.float32).uniform_(-jitter, jitter)
    return x, y, yaw


def foot_xy_on_bridge_annulus_torch(
    foot_xy: torch.Tensor,
    inner_radius_m: float,
    bridge_width_m: torch.Tensor,
    outer_margin_m: float = 0.0,
    inner_margin_m: float = 0.0,
) -> torch.Tensor:
    """脚底 XY（env 局部系）是否落在双圆 8 字桥面环形足迹内，形状 ``(N,)`` bool。"""
    n = foot_xy.shape[0]
    device, dtype = foot_xy.device, foot_xy.dtype
    inner_r = torch.full((n,), float(inner_radius_m), device=device, dtype=dtype)
    bw = (
        torch.full((n,), float(bridge_width_m), device=device, dtype=dtype)
        if isinstance(bridge_width_m, (float, int))
        else bridge_width_m.to(device=device, dtype=dtype).reshape(-1)
    )
    r_c = inner_r + 0.5 * bw
    outer_r = inner_r + bw
    inner_lo = inner_r - float(inner_margin_m)
    outer_hi = outer_r + float(outer_margin_m)
    d_left = torch.hypot(foot_xy[:, 0] + r_c, foot_xy[:, 1])
    d_right = torch.hypot(foot_xy[:, 0] - r_c, foot_xy[:, 1])
    on_left = (d_left >= inner_lo) & (d_left <= outer_hi)
    on_right = (d_right >= inner_lo) & (d_right <= outer_hi)
    return on_left | on_right
