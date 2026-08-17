# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

"""Configuration for the raised 90-degree L-shaped platform."""

from __future__ import annotations

from isaaclab.terrains.terrain_generator_cfg import SubTerrainBaseCfg
from isaaclab.utils import configclass

from .l_railed_platform import l_railed_platform_terrain


@configclass
class MeshLRailedPlatformTerrainCfg(SubTerrainBaseCfg):
    """Two stairs up, two perpendicular legs, and two stairs down."""

    function = l_railed_platform_terrain

    size: tuple[float, float] = (8.0, 8.0)
    first_leg_start_center_m: tuple[float, float] = (1.5, 1.5)
    num_steps: int = 2
    step_height_m: float = 0.15
    tread_depth_m: float = 0.30
    leg_length_m: float = 4.0
    width_range_m: tuple[float, float] = (1.5, 0.7)
    num_width_levels: int = 10
    guardrail_height_m: float = 1.20
    guardrail_depth_m: float = 0.10
    guardrail_rail_height_m: float = 0.10
    hollow_opening_width_m: float = 0.20
    hollow_opening_height_m: float = 1.00
    hollow_post_width_m: float = 0.20
    spawn_distance_before_stairs_m: float = 0.60
    ground_thickness_m: float = 0.05
