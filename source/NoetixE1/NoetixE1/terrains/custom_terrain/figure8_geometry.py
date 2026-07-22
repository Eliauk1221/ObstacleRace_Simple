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


def s_curve_centerline_points_torch(
    inner_radius_m: float,
    bridge_width_m: float,
    samples_per_arc: int,
    device: torch.device | str,
) -> torch.Tensor:
    """半 8 字 S 桥的开口中心线，方向为左端 -> 原点 -> 右端。

    左侧采用上半圆，右侧采用下半圆；两段在原点有同向切线。返回的折线
    不闭合，供 S 弯的路径奖励和进度课程使用。
    """
    n = max(2, int(samples_per_arc))
    dev = _torch_device(device)
    r_c = centerline_radius_m(inner_radius_m, bridge_width_m)

    left_theta = torch.linspace(math.pi, 0.0, n, device=dev)
    right_theta = torch.linspace(math.pi, 2.0 * math.pi, n, device=dev)
    left = torch.stack(
        [-r_c + r_c * torch.cos(left_theta), r_c * torch.sin(left_theta)], dim=1
    )
    right = torch.stack(
        [r_c + r_c * torch.cos(right_theta), r_c * torch.sin(right_theta)], dim=1
    )
    return torch.cat([left[:-1], right], dim=0)


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


def sample_s_curve_path_spawn_xy_yaw(
    num_samples: int,
    device: torch.device | str,
    inner_radius_m: float,
    bridge_width_m: torch.Tensor | float,
    path_fraction_range: tuple[float, float] = (0.02, 0.15),
    position_noise: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """在 S 桥从左到右的前段路径上采样出生点与正向切线 yaw。

    归一化路径参数 0 对应左端，0.5 对应两半环的原点连接，1 对应右端。
    默认只在前 15% 采样，使单个 episode 有足够剩余弧长完成课程。
    """
    if num_samples <= 0:
        empty = torch.empty(0, device=_torch_device(device))
        return empty, empty, empty

    dev = _torch_device(device)
    lo, hi = (float(path_fraction_range[0]), float(path_fraction_range[1]))
    lo, hi = max(0.0, lo), min(1.0, hi)
    if hi < lo:
        raise ValueError("path_fraction_range must satisfy min <= max.")
    u = torch.empty(num_samples, device=dev, dtype=torch.float32).uniform_(lo, hi)
    if isinstance(bridge_width_m, (float, int)):
        r_c = torch.full((num_samples,), centerline_radius_m(inner_radius_m, bridge_width_m), device=dev)
    else:
        bridge_width_m = bridge_width_m.to(device=dev, dtype=torch.float32).reshape(-1)
        if bridge_width_m.numel() != num_samples:
            raise ValueError("bridge_width_m must have one value per sampled S-curve spawn.")
        r_c = float(inner_radius_m) + 0.5 * bridge_width_m
    on_left = u <= 0.5

    left_theta = math.pi - 2.0 * math.pi * u
    right_theta = math.pi + 2.0 * math.pi * (u - 0.5)
    theta = torch.where(on_left, left_theta, right_theta)
    center_x = torch.where(on_left, -r_c, r_c)
    x = center_x + r_c * torch.cos(theta)
    y = r_c * torch.sin(theta)

    # d/ds: left theta decreases; right theta increases.
    dx = torch.where(on_left, torch.sin(theta), -torch.sin(theta))
    dy = torch.where(on_left, -torch.cos(theta), torch.cos(theta))
    yaw = torch.atan2(dy, dx)

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


def bridge_width_for_runtime_terrain_levels(
    terrain_levels: torch.Tensor,
    num_rows: int,
    bridge_width_by_level: tuple[float, ...] | list[float],
    difficulty_range: tuple[float, float] | list[float] | None = None,
) -> torch.Tensor:
    """按运行时地形设置取桥宽，兼容固定难度的单行展示场景。

    普通课程由 ``terrain_levels`` 决定桥宽。单行且
    ``difficulty_range=(d, d)`` 时，生成器实际按 ``d`` 生成地形，不能把
    它误当作第 0 档。
    """
    if int(num_rows) <= 1 and difficulty_range is not None and len(difficulty_range) == 2:
        low, high = (float(difficulty_range[0]), float(difficulty_range[1]))
        if math.isclose(low, high):
            difficulty = torch.full_like(terrain_levels, low, dtype=torch.float32)
            return bridge_width_from_terrain_difficulty(difficulty, bridge_width_by_level)
    return bridge_width_for_terrain_levels(terrain_levels, num_rows, bridge_width_by_level)


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


def foot_xy_on_figure8_support_torch(
    foot_xy: torch.Tensor,
    inner_radius_m: float,
    bridge_width_m: torch.Tensor,
    cutout_width_m: float = 0.0,
    cutout_length_m: float = 0.0,
) -> torch.Tensor:
    """环带桥面或中心矩形挖空区均视为有效支撑面。"""
    on_bridge = foot_xy_on_bridge_annulus_torch(foot_xy, inner_radius_m, bridge_width_m)
    on_cutout = foot_xy_on_center_cutout_torch(foot_xy, cutout_width_m, cutout_length_m)
    return on_bridge | on_cutout


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
    d_left = torch.hypot(foot_xy[:, 0] + r_c, foot_xy[:, 1])
    d_right = torch.hypot(foot_xy[:, 0] - r_c, foot_xy[:, 1])
    on_left = (d_left >= inner_r) & (d_left <= outer_r)
    on_right = (d_right >= inner_r) & (d_right <= outer_r)
    return on_left | on_right


def foot_xy_on_s_curve_torch(
    foot_xy: torch.Tensor,
    inner_radius_m: float,
    bridge_width_m: torch.Tensor | float,
) -> torch.Tensor:
    """脚底 XY 是否在 S 桥有效半环带上。

    与 ``s_curve_terrain`` 完全一致：左圆只接受 ``y >= 0``，
    右圆只接受 ``y <= 0``。不能复用闭合 8 字的双圆环带掩码，否则会把
    S 桥坑底中的另外两个半圆误当成可支撑桥面。
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
    outer_r = inner_r + bw
    d_left = torch.hypot(foot_xy[:, 0] + r_c, foot_xy[:, 1])
    d_right = torch.hypot(foot_xy[:, 0] - r_c, foot_xy[:, 1])
    on_left = (d_left >= inner_r) & (d_left <= outer_r) & (foot_xy[:, 1] >= 0.0)
    on_right = (d_right >= inner_r) & (d_right <= outer_r) & (foot_xy[:, 1] <= 0.0)
    return on_left | on_right
