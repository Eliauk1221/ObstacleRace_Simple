# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

"""Shared geometry and path helpers for the raised L-shaped bridge."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import torch


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv



def _passage_width_for_terrain_levels(env, sub_terrain) -> torch.Tensor:
    """Reproduce the mesh generator's discrete width curriculum."""
    width_easy, width_hard = map(float, sub_terrain.width_range_m)
    num_levels = int(sub_terrain.num_width_levels)
    terrain_cfg = env.scene.terrain.cfg.terrain_generator
    num_rows = int(getattr(terrain_cfg, "num_rows", 1))
    if num_rows > 1:
        indices = torch.floor(
            env.scene.terrain.terrain_levels.float().clamp(0, num_rows - 1)
            / float(num_rows) * num_levels
        ).long()
    else:
        difficulty = float(getattr(terrain_cfg, "difficulty_range", (0.0, 0.0))[0])
        index = min(int(max(0.0, min(1.0, difficulty)) * num_levels), num_levels - 1)
        indices = torch.full((env.num_envs,), index, dtype=torch.long, device=env.device)
    ratio = indices.clamp(0, num_levels - 1).float() / float(num_levels - 1)
    return width_easy + ratio * (width_hard - width_easy)

class LRailedPlatformGeometry(NamedTuple):
    """Per-environment width and env-local bounds of the L bridge."""

    passage_width_m: torch.Tensor
    approach_start_x_m: float
    route_start_x_m: float
    platform_start_x_m: float
    corner_x_m: float
    platform_end_y_m: float
    route_end_y_m: float


class LCurvePathState(NamedTuple):
    """Projection of batched positions onto the rounded L reference path."""

    nearest_point: torch.Tensor
    tangent: torch.Tensor
    preview_point: torch.Tensor
    preview_tangent: torch.Tensor
    lateral_distance: torch.Tensor
    progress_m: torch.Tensor
    total_length_m: torch.Tensor
    arc_start_progress_m: torch.Tensor
    arc_end_progress_m: torch.Tensor
    arc_mid_progress_m: torch.Tensor


L_CURVE_REVERSE_DIRECTION_ATTR = "_l_curve_reverse_direction"


def l_curve_reverse_direction_mask(
    env: ManagerBasedRLEnv,
    device: torch.device,
) -> torch.Tensor:
    """Return the per-environment reverse-traversal state, if configured."""
    reverse_direction = getattr(
        env.unwrapped,
        L_CURVE_REVERSE_DIRECTION_ATTR,
        None,
    )
    if reverse_direction is None:
        return torch.zeros(
            env.num_envs,
            dtype=torch.bool,
            device=device,
        )
    reverse_direction = torch.as_tensor(
        reverse_direction,
        dtype=torch.bool,
        device=device,
    )
    if reverse_direction.shape != (env.num_envs,):
        raise ValueError(
            f"{L_CURVE_REVERSE_DIRECTION_ATTR} must have shape "
            f"({env.num_envs},), got {reverse_direction.shape}."
        )
    return reverse_direction


def l_curve_travel_direction_sign(
    env: ManagerBasedRLEnv,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return +1 for entry-to-exit travel and -1 for reverse travel."""
    reverse_direction = l_curve_reverse_direction_mask(env, device)
    return torch.where(
        reverse_direction,
        torch.full((env.num_envs,), -1.0, device=device, dtype=dtype),
        torch.ones(env.num_envs, device=device, dtype=dtype),
    )


def l_railed_platform_geometry(
    env: ManagerBasedRLEnv,
    sub_terrain_key: str = "l_curve",
) -> LRailedPlatformGeometry:
    """Resolve the active bridge width and route bounds from terrain config."""
    terrain_gen_cfg = env.scene.terrain.cfg.terrain_generator
    sub_terrain = terrain_gen_cfg.sub_terrains.get(sub_terrain_key)
    if sub_terrain is None:
        l_candidates = [
            cfg
            for cfg in terrain_gen_cfg.sub_terrains.values()
            if hasattr(cfg, "leg_length_m")
            and hasattr(cfg, "width_range_m")
            and hasattr(cfg, "first_leg_start_center_m")
        ]
        if len(l_candidates) != 1:
            raise KeyError(
                f"{sub_terrain_key!r} not found and no unique L-curve terrain is available."
            )
        sub_terrain = l_candidates[0]

    num_steps = int(sub_terrain.num_steps)
    stairs_run_m = num_steps * float(sub_terrain.tread_depth_m)
    spawn_distance_m = float(sub_terrain.spawn_distance_before_stairs_m)
    leg_length_m = float(sub_terrain.leg_length_m)

    route_start_x_m = spawn_distance_m
    platform_start_x_m = route_start_x_m + stairs_run_m
    corner_x_m = platform_start_x_m + leg_length_m

    return LRailedPlatformGeometry(
        passage_width_m=_passage_width_for_terrain_levels(env, sub_terrain),
        approach_start_x_m=0.0,
        route_start_x_m=route_start_x_m,
        platform_start_x_m=platform_start_x_m,
        corner_x_m=corner_x_m,
        platform_end_y_m=leg_length_m,
        route_end_y_m=leg_length_m + stairs_run_m,
    )


def _smoothstep01(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp(0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def project_to_rounded_l_path(
    local_xy: torch.Tensor,
    passage_width_m: torch.Tensor,
    corner_x_m: float,
    route_end_y_m: float,
    *,
    approach_start_x_m: float = 0.0,
    turn_radius_ratio: float = 0.4,
    lookahead_distance_m: float | torch.Tensor = 0.0,
    reverse_direction: torch.Tensor | None = None,
) -> LCurvePathState:
    """Project XY positions onto a line--quarter-circle--line L path.

    The robot approaches along local ``+x`` and exits along local ``+y``.
    The rounded turn radius is a fraction of the active curriculum width, so
    the reference path always remains inside the square corner platform.
    """
    if local_xy.ndim != 2 or local_xy.shape[1] != 2:
        raise ValueError(f"local_xy must have shape (N, 2), got {local_xy.shape}.")
    if passage_width_m.shape != local_xy.shape[:1]:
        raise ValueError(
            "passage_width_m must match the environment batch, got "
            f"{passage_width_m.shape} and {local_xy.shape[:1]}."
        )
    if not 0.0 < float(turn_radius_ratio) <= 0.5:
        raise ValueError(
            "turn_radius_ratio must be in (0, 0.5], got "
            f"{turn_radius_ratio}."
        )

    dtype = local_xy.dtype
    device = local_xy.device
    radius = passage_width_m.to(dtype=dtype).clamp_min(1.0e-6) * float(
        turn_radius_ratio
    )
    corner_x = torch.full_like(radius, float(corner_x_m))
    route_end_y = torch.full_like(radius, float(route_end_y_m))
    path_start_x = torch.full_like(radius, float(approach_start_x_m))
    arc_start_x = corner_x - radius

    # Candidate 1: horizontal approach and first bridge leg.
    horizontal_x = torch.maximum(
        torch.minimum(local_xy[:, 0], arc_start_x),
        path_start_x,
    )
    horizontal_point = torch.stack(
        (horizontal_x, torch.zeros_like(horizontal_x)), dim=1
    )
    horizontal_tangent = torch.stack(
        (torch.ones_like(horizontal_x), torch.zeros_like(horizontal_x)), dim=1
    )

    # Candidate 2: a left-turning quarter circle from -pi/2 to zero.
    arc_center = torch.stack((arc_start_x, radius), dim=1)
    delta_to_center = local_xy - arc_center
    angle = torch.atan2(delta_to_center[:, 1], delta_to_center[:, 0]).clamp(
        min=-0.5 * torch.pi,
        max=0.0,
    )
    arc_point = arc_center + radius.unsqueeze(1) * torch.stack(
        (torch.cos(angle), torch.sin(angle)), dim=1
    )
    arc_tangent = torch.stack((-torch.sin(angle), torch.cos(angle)), dim=1)

    # Candidate 3: second bridge leg and descending stairs.
    vertical_y = local_xy[:, 1].clamp_min(0.0)
    vertical_y = torch.maximum(vertical_y, radius)
    vertical_y = torch.minimum(vertical_y, route_end_y)
    vertical_point = torch.stack((corner_x, vertical_y), dim=1)
    vertical_tangent = torch.stack(
        (torch.zeros_like(vertical_y), torch.ones_like(vertical_y)), dim=1
    )

    candidate_points = torch.stack(
        (horizontal_point, arc_point, vertical_point), dim=1
    )
    candidate_tangents = torch.stack(
        (horizontal_tangent, arc_tangent, vertical_tangent), dim=1
    )
    candidate_distances = torch.linalg.vector_norm(
        local_xy.unsqueeze(1) - candidate_points,
        dim=-1,
    )
    nearest_segment = candidate_distances.argmin(dim=1)
    gather_xy = nearest_segment.view(-1, 1, 1).expand(-1, 1, 2)
    nearest_point = candidate_points.gather(1, gather_xy).squeeze(1)
    tangent = candidate_tangents.gather(1, gather_xy).squeeze(1)
    lateral_distance = candidate_distances.gather(
        1, nearest_segment.unsqueeze(1)
    ).squeeze(1)

    horizontal_length = arc_start_x - path_start_x
    arc_length = 0.5 * torch.pi * radius
    horizontal_progress = horizontal_x - path_start_x
    arc_progress = horizontal_length + radius * (angle + 0.5 * torch.pi)
    vertical_progress = horizontal_length + arc_length + (vertical_y - radius)
    candidate_progress = torch.stack(
        (horizontal_progress, arc_progress, vertical_progress), dim=1
    )
    progress = candidate_progress.gather(
        1, nearest_segment.unsqueeze(1)
    ).squeeze(1)
    total_length = horizontal_length + arc_length + (route_end_y - radius)

    if isinstance(lookahead_distance_m, torch.Tensor):
        lookahead = lookahead_distance_m.to(device=device, dtype=dtype)
        if lookahead.shape != progress.shape:
            raise ValueError(
                "lookahead_distance_m tensor must match the environment batch, "
                f"got {lookahead.shape} and {progress.shape}."
            )
    else:
        lookahead = torch.full_like(progress, float(lookahead_distance_m))
    if reverse_direction is None:
        reverse_direction = torch.zeros_like(progress, dtype=torch.bool)
    else:
        reverse_direction = reverse_direction.to(
            device=device,
            dtype=torch.bool,
        )
        if reverse_direction.shape != progress.shape:
            raise ValueError(
                "reverse_direction must match the environment batch, got "
                f"{reverse_direction.shape} and {progress.shape}."
            )
    travel_direction = torch.where(
        reverse_direction,
        torch.full_like(progress, -1.0),
        torch.ones_like(progress),
    )
    preview_progress = (
        progress + travel_direction * lookahead.clamp_min(0.0)
    ).clamp(
        min=0.0,
    )
    preview_progress = torch.minimum(preview_progress, total_length)
    preview_on_horizontal = preview_progress < horizontal_length
    preview_on_arc = (~preview_on_horizontal) & (
        preview_progress < horizontal_length + arc_length
    )
    preview_angle = (
        (preview_progress - horizontal_length) / radius - 0.5 * torch.pi
    ).clamp(min=-0.5 * torch.pi, max=0.0)
    preview_arc_tangent = torch.stack(
        (-torch.sin(preview_angle), torch.cos(preview_angle)), dim=1
    )
    preview_horizontal_point = torch.stack(
        (
            path_start_x + preview_progress,
            torch.zeros_like(preview_progress),
        ),
        dim=1,
    )
    preview_arc_point = arc_center + radius.unsqueeze(1) * torch.stack(
        (torch.cos(preview_angle), torch.sin(preview_angle)), dim=1
    )
    preview_vertical_y = (
        radius
        + preview_progress
        - horizontal_length
        - arc_length
    )
    preview_vertical_point = torch.stack(
        (corner_x, preview_vertical_y), dim=1
    )
    preview_point = torch.where(
        preview_on_horizontal.unsqueeze(1),
        preview_horizontal_point,
        preview_vertical_point,
    )
    preview_point = torch.where(
        preview_on_arc.unsqueeze(1), preview_arc_point, preview_point
    )
    preview_tangent = torch.zeros_like(local_xy)
    preview_tangent[:, 0] = preview_on_horizontal.to(dtype)
    preview_tangent[:, 1] = (~preview_on_horizontal & ~preview_on_arc).to(dtype)
    preview_tangent = torch.where(
        preview_on_arc.unsqueeze(1), preview_arc_tangent, preview_tangent
    )
    tangent = tangent * travel_direction.unsqueeze(1)
    preview_tangent = preview_tangent * travel_direction.unsqueeze(1)

    forward_arc_start_progress = horizontal_length
    forward_arc_end_progress = horizontal_length + arc_length
    forward_arc_mid_progress = horizontal_length + 0.5 * arc_length
    travel_progress = torch.where(
        reverse_direction,
        total_length - progress,
        progress,
    )
    arc_start_progress = torch.where(
        reverse_direction,
        total_length - forward_arc_end_progress,
        forward_arc_start_progress,
    )
    arc_end_progress = torch.where(
        reverse_direction,
        total_length - forward_arc_start_progress,
        forward_arc_end_progress,
    )
    arc_mid_progress = torch.where(
        reverse_direction,
        total_length - forward_arc_mid_progress,
        forward_arc_mid_progress,
    )
    return LCurvePathState(
        nearest_point=nearest_point,
        tangent=tangent,
        preview_point=preview_point,
        preview_tangent=preview_tangent,
        lateral_distance=lateral_distance,
        progress_m=travel_progress,
        total_length_m=total_length,
        arc_start_progress_m=arc_start_progress,
        arc_end_progress_m=arc_end_progress,
        arc_mid_progress_m=arc_mid_progress,
    )


def l_curve_path_state(
    env: ManagerBasedRLEnv,
    local_xy: torch.Tensor,
    *,
    sub_terrain_key: str = "l_curve",
    turn_radius_ratio: float = 0.4,
    lookahead_distance_m: float | torch.Tensor = 0.0,
    bidirectional: bool = False,
) -> tuple[LCurvePathState, LRailedPlatformGeometry]:
    """Resolve L-bridge geometry and project a batch of local positions."""
    geometry = l_railed_platform_geometry(env, sub_terrain_key)
    state = project_to_rounded_l_path(
        local_xy,
        geometry.passage_width_m,
        geometry.corner_x_m,
        geometry.route_end_y_m,
        approach_start_x_m=geometry.approach_start_x_m,
        turn_radius_ratio=turn_radius_ratio,
        lookahead_distance_m=lookahead_distance_m,
        reverse_direction=(
            l_curve_reverse_direction_mask(env, local_xy.device)
            if bidirectional
            else None
        ),
    )
    return state, geometry


def l_curve_corner_blend(
    state: LCurvePathState,
    approach_distance_m: float = 0.5,
    recovery_distance_m: float = 0.3,
) -> torch.Tensor:
    """Return a smooth zero-to-one-to-zero blend around the rounded turn."""
    if approach_distance_m <= 0.0 or recovery_distance_m <= 0.0:
        raise ValueError("Corner approach and recovery distances must be positive.")
    entry = _smoothstep01(
        (state.progress_m - (state.arc_start_progress_m - approach_distance_m))
        / float(approach_distance_m)
    )
    exit_blend = 1.0 - _smoothstep01(
        (state.progress_m - state.arc_end_progress_m)
        / float(recovery_distance_m)
    )
    return entry * exit_blend
