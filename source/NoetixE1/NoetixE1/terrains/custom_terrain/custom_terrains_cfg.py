# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

"""Configuration classes for custom terrains with noise."""

from dataclasses import MISSING

from isaaclab.utils import configclass

from . import custom_terrains
from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg


@configclass
class HfNoisyPyramidSlopedTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a pyramid sloped height field terrain with noise."""

    function = custom_terrains.noisy_pyramid_sloped_terrain

    slope_range: tuple[float, float] = MISSING
    """The slope of the terrain (in radians)."""

    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""

    inverted: bool = False
    """Whether the pyramid is inverted. Defaults to False.

    If True, the terrain is inverted such that the platform is at the bottom and the slopes are upwards.
    """

    noise_range: tuple[float, float] = (0.0, 0.02)
    """The minimum and maximum height noise (in m) to add to the terrain. Defaults to (0.0, 0.02)."""

    noise_step: float = 0.005
    """The minimum height (in m) change between two noise points. Defaults to 0.005."""

    num_obstacles: int = 0
    """The number of random obstacles to add to the terrain. Defaults to 0 (no obstacles).
    
    Note:
        Obstacles are randomly placed and do not vary with difficulty.
    """

    obstacle_size: float | tuple[float, float] = 0.3
    """The size (width/length) of obstacles in meters. Defaults to 0.3.
    
    If a single float is provided, all obstacles have the same size.
    If a tuple (min, max) is provided, obstacle sizes are randomly sampled from this range.
    """

    obstacle_height: float | tuple[float, float] = 0.1
    """The height of obstacles in meters. Defaults to 0.1.
    
    Note:
        Obstacle height is fixed and does not vary with difficulty.
        The height is added on top of the terrain surface at the obstacle location.
    """


@configclass
class HfNoisyInvertedPyramidSlopedTerrainCfg(HfNoisyPyramidSlopedTerrainCfg):
    """Configuration for an inverted pyramid sloped height field terrain with noise.

    Note:
        This is a subclass of :class:`HfNoisyPyramidSlopedTerrainCfg` with :obj:`inverted` set to True.
    """

    inverted: bool = True


@configclass
class HfNoisyPyramidStairsTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a pyramid stair height field terrain with noise."""

    function = custom_terrains.noisy_pyramid_stairs_terrain

    border_width: float = 0.0
    """The width of the border around the terrain (in m). Defaults to 0.0."""

    step_height_range: tuple[float, float] = MISSING
    """The minimum and maximum height of the steps (in m)."""

    step_width: float | None = None
    """The width of the steps (in m). Defaults to None.
    
    If None, step_width_pixels_range will be used instead.
    This parameter is deprecated in favor of step_width_pixels_range to avoid precision loss.
    """

    step_width_pixels_range: tuple[int, int] | None = None
    """The minimum and maximum width of the steps in pixels (horizontal_scale multiples). Defaults to None.
    
    If provided, this takes precedence over step_width to avoid precision loss from float-to-int conversion.
    The first value is the width at difficulty=0 (easier, wider), 
    the second value is the width at difficulty=1 (harder, narrower).
    Step width decreases with difficulty (越难越窄).
    
    Example:
        If horizontal_scale=0.1m, step_width_pixels_range=(10, 5) means:
        - difficulty=0: step_width = 10 * 0.1 = 1.0m
        - difficulty=1: step_width = 5 * 0.1 = 0.5m
    """

    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""

    platform_width_pixels: int | None = None
    """The width of the square platform in pixels (horizontal_scale multiples). Defaults to None.
    
    If provided, this takes precedence over platform_width to avoid precision loss.
    """

    inverted: bool = False
    """Whether the stairs are inverted. Defaults to False."""

    noise_range: tuple[float, float] = (0.0, 0.02)
    """The minimum and maximum height noise (in m) to add to the terrain. Defaults to (0.0, 0.02)."""

    noise_step: float = 0.005
    """The minimum height (in m) change between two noise points. Defaults to 0.005."""


@configclass
class HfNoisyInvertedPyramidStairsTerrainCfg(HfNoisyPyramidStairsTerrainCfg):
    """Configuration for an inverted pyramid stair height field terrain with noise.

    Note:
        This is a subclass of :class:`HfNoisyPyramidStairsTerrainCfg` with :obj:`inverted` set to True.
    """

    inverted: bool = True


@configclass
class HfMultiGapTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a height field terrain with nested "回字形" (ring-shaped) gaps of configurable depth.

    Creates nested ring-shaped gap structures (一环套一环). The structure from outside to inside:
    - Outer border (solid ground)
    - Gap ring 1 (hollow, with configurable depth)
    - Ground ring 1 (solid ground)
    - Gap ring 2 (hollow, with configurable depth)
    - Ground ring 2 (solid ground)
    - ...
    - Inner platform (solid ground at center)
    
    The number of gap rings is automatically calculated based on:
    - Terrain size (from HfTerrainBaseCfg.size)
    - Platform width
    - Outer ground width
    - Gap width (varies with difficulty)
    - Ground ring width (varies with difficulty)
    """

    function = custom_terrains.multi_gap_terrain

    platform_width: float = 1.0
    """The width of the square platform at the center (in m). Defaults to 1.0."""

    gap_width_range: tuple[float, float] | None = None
    """The minimum and maximum width of each gap ring (in m). Defaults to None.
    
    Note:
        This parameter is deprecated in favor of gap_width_pixels_range to avoid precision loss.
        If None, gap_width_pixels_range will be used instead.
    """

    gap_width_pixels_range: tuple[int, int] | None = (2, 8)
    """The minimum and maximum width of each gap ring in pixels (horizontal_scale multiples). Defaults to (2, 8).
    
    If provided, this takes precedence over gap_width_range to avoid precision loss from float-to-int conversion.
    The first value is the width at difficulty=0 (easier, wider), 
    the second value is the width at difficulty=1 (harder, narrower).
    Gap width increases with difficulty (越难越宽).
    
    Note:
        The ground ring width between gaps is automatically calculated as: 1.0m - gap_width
        This ensures that gap_width + ground_ring_width = 1.0m (in physical units).
    
    Example:
        If horizontal_scale=0.1m, gap_width_pixels_range=(2, 8) means:
        - difficulty=0: gap_width = 2 * 0.1 = 0.2m, ground_ring_width = 1.0 - 0.2 = 0.8m
        - difficulty=1: gap_width = 8 * 0.1 = 0.8m, ground_ring_width = 1.0 - 0.8 = 0.2m
    """

    gap_depth_range: tuple[float, float] = (0.1, 0.5)
    """The minimum and maximum depth of each gap (in m). Defaults to (0.1, 0.5).
    
    Note:
        Gap depth is measured from the ground level (0.0) downward.
        Unlike the original gap terrain which has infinite depth, these gaps have a finite depth.
    """

    step_height_range: tuple[float, float] = (0.0, 0.2)
    """The minimum and maximum height difference between adjacent ground rings (in m). Defaults to (0.0, 0.2).
    
    Note:
        This creates a stepped effect where each ground ring is higher than the previous one.
        The height difference increases with difficulty.
        The innermost platform is at height 0, and each subsequent ground ring is higher.
    """


@configclass
class HfSingleGapTerrainCfg(HfTerrainBaseCfg):
    """单环回字形 gap：中心平台 → 一圈地面环 → 一圈 gap（坑）→ 外围地面。仅一个 gap，参数同 HfMultiGapTerrainCfg。"""

    function = custom_terrains.single_gap_terrain

    platform_width: float = 1.0
    """中心方形平台宽度（米）。"""

    gap_width_range: tuple[float, float] | None = None
    """每圈 gap 宽度范围（米）。若为 None 则使用 gap_width_pixels_range。"""

    gap_width_pixels_range: tuple[int, int] | None = (2, 8)
    """每圈 gap 宽度（像素）。difficulty=0 较窄，difficulty=1 较宽。地面环宽 = 1.0m - gap_width。"""

    gap_depth_range: tuple[float, float] = (0.1, 0.5)
    """Gap 深度（米），从地面向下。"""

    step_height_range: tuple[float, float] = (0.0, 0.2)
    """中心平台与 gap 内侧地面环的高度差（米），随 difficulty 插值。"""


@configclass
class HfSteppingStonesTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a stepping stones height field terrain."""

    function = custom_terrains.stepping_stones_terrain

    stone_height_max: float = MISSING
    """The maximum height of the stones (in m)."""

    stone_width_range: tuple[float, float] = MISSING
    """The minimum and maximum width of the stones (in m)."""

    stone_distance_range: tuple[float, float] = MISSING
    """The minimum and maximum distance between stones (in m)."""

    holes_depth: tuple[float, float] = [-0.3, -0.5]
    """The minimum and maximum depth of the holes (negative obstacles). Defaults to (-0.3, -0.5)."""

    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""


@configclass
class HfCrossSteppingStonesTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a cross-shaped stepping stones height field terrain."""

    function = custom_terrains.cross_stepping_stones_terrain

    stone_height_max: float = MISSING
    """The maximum height of the stones (in m)."""

    stone_width_range: tuple[float, float] = MISSING
    """The minimum and maximum width of the stones (in m)."""

    stone_distance_range: tuple[float, float] = MISSING
    """The minimum and maximum distance between stones (in m)."""

    holes_depth: tuple[float, float] = [-0.3, -0.5]
    """The minimum and maximum depth of the holes (negative obstacles). Defaults to (-0.3, -0.5)."""

    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""

    border_width: float = 0.5
    """The width of the safe border platform around the terrain edges (in m). Defaults to 0.5."""

    num_stone_rows: int = 2
    """The number of rows/columns of stepping stones on each branch of the cross.
    
    For horizontal branch (left-right): number of rows of stones.
    For vertical branch (up-down): number of columns of stones.
    Defaults to 2.
    """


@configclass
class HfHurdleTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a height field terrain with nested "回字形" (ring-shaped) hurdles.

    Creates exactly 3 nested ring-shaped hurdle structures (一环套一环), uniformly distributed.
    The structure from outside to inside:
    - Outer border (solid ground at height 0)
    - Ground ring 1 (solid ground at height 0)
    - Hurdle ring 1 (raised barrier, height increases with difficulty)
    - Ground ring 2 (solid ground at height 0)
    - Hurdle ring 2 (raised barrier, height increases with difficulty)
    - Ground ring 3 (solid ground at height 0)
    - Hurdle ring 3 (raised barrier, height increases with difficulty)
    - Inner platform (solid ground at height 0)

    Unlike gaps which are depressions, hurdles are raised barriers that the robot must step over.
    The hurdle height increases with difficulty, while the hurdle widths are configurable.
    The 3 hurdles are uniformly distributed from the platform to the terrain edge.
    """

    function = custom_terrains.hurdle_terrain

    platform_width: float = 1.0
    """The width of the square platform at the center (in m). Defaults to 1.0."""

    hurdle_height_range: tuple[float, float] = (0.05, 0.3)
    """The minimum and maximum height of hurdles (in m). Defaults to (0.05, 0.3).
    
    Hurdle height increases with difficulty. The first value is the height at difficulty=0,
    the second value is the height at difficulty=1.
    """

    hurdle_width_pixels: int | tuple[int, int, int] = 3
    """The width of each hurdle ring in pixels (horizontal_scale multiples). Defaults to 3.
    
    If a single int is provided, all 3 hurdles have the same width.
    If a tuple (width1, width2, width3) is provided, each hurdle can have a different width.
    Hurdle widths are fixed and do not vary with difficulty.
    
    Example:
        If horizontal_scale=0.1m:
        - hurdle_width_pixels=3 means all hurdles have width = 3 * 0.1 = 0.3m
        - hurdle_width_pixels=(2, 3, 4) means hurdle1=0.2m, hurdle2=0.3m, hurdle3=0.4m
    """


@configclass
class HfFramePlatformTerrainCfg(HfTerrainBaseCfg):
    """回字形高台：中心为平地，外围一圈高台（框）。类似 rails_terrain 的 height-field 版。

    - height_range: (min, max) 高台高度（米），随 difficulty 线性插值。
    - width_range: (min, max) 高台边框厚度（米），随 difficulty 线性插值。
    - distance_range: (min, max) 内沿距中心的距离（米，即内孔半宽），随 difficulty 线性插值。
    """

    function = custom_terrains.frame_platform_terrain
    height_range: tuple[float, float] = (0.1, 0.4)
    width_range: tuple[float, float] = (0.2, 0.6)
    distance_range: tuple[float, float] = (0.3, 0.8)


@configclass
class HfDoubleFramePlatformTerrainCfg(HfTerrainBaseCfg):
    """双回字形高台：中心平地 → 内圈高台 → 平地环 → 外圈高台 → 外围平地。两个高台的高、宽、距离可分别配置。

    - 高台1（内圈）：height_range_1, width_range_1, distance_range_1（内沿距中心）。
    - 高台2（外圈）：height_range_2, width_range_2, distance_range_2（内沿距中心）。
    需满足 distance_2 >= distance_1 + width_1 以便中间有平地环。
    """

    function = custom_terrains.double_frame_platform_terrain
    # 内圈回字
    height_range_1: tuple[float, float] = (0.1, 0.35)
    width_range_1: tuple[float, float] = (0.15, 0.5)
    distance_range_1: tuple[float, float] = (0.25, 0.6)
    # 外圈回字
    height_range_2: tuple[float, float] = (0.15, 0.4)
    width_range_2: tuple[float, float] = (0.2, 0.6)
    distance_range_2: tuple[float, float] = (0.5, 1.0)
