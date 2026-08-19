# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

"""Configuration classes for custom terrains with noise."""

from dataclasses import MISSING

from isaaclab.terrains.terrain_generator_cfg import SubTerrainBaseCfg
from isaaclab.utils import configclass

from . import custom_terrains
from . import braking_slope_terrain
from . import figure8_mesh_terrains
from . import spiral_pyramid_stairs_terrain
from . import symmetric_slope_terrain
from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg


@configclass
class MeshFlatPillarsTerrainCfg(SubTerrainBaseCfg):
    """Flat mesh terrain with sparse, non-overlapping vertical pillars."""

    function = custom_terrains.flat_pillars_terrain

    pillar_width_range: tuple[float, float] = (0.1, 0.2)
    """Pillar extent along the local x-axis, in meters."""

    pillar_thickness_range: tuple[float, float] = (0.1, 0.2)
    """Pillar extent along the local y-axis, in meters."""

    pillar_height_range: tuple[float, float] = (1.0, 1.5)
    """Pillar height above the flat ground, in meters."""

    num_pillars_range: tuple[int, int] = (4, 8)
    """Inclusive range for the number of pillars generated in each tile."""

    platform_width: float = 2.0
    """Width of the square pillar-free spawn area at the tile center."""

    boundary_margin: float = 0.5
    """Minimum distance between every pillar and the tile boundary."""

    min_pillar_spacing: float = 0.3
    """Minimum horizontal clearance between pillars and the center platform."""

    max_sampling_attempts: int = 500
    """Maximum placement attempts for each pillar."""


@configclass
class MeshSymmetricSlopeTerrainCfg(SubTerrainBaseCfg):
    """Straight V-slope with two 20-degree sides and a collidable center gap."""

    function = symmetric_slope_terrain.symmetric_slope_terrain

    obstacle_length_m: float = 3.0
    """Obstacle extent along the local positive x traversal direction."""

    obstacle_width_m: float = 3.5
    """Fixed outer width of both wedges and the center gap."""

    slope_angle_deg: float = 20.0
    """Angle of both transverse slope surfaces."""

    gap_width_by_level: tuple[float, ...] = (
        0.12,
        0.115,
        0.11,
        0.105,
        0.10,
        0.10,
        0.095,
        0.09,
        0.085,
        0.08,
    )
    """Ordered center-gap widths from the easiest to the hardest terrain level."""

    finish_flat_length_m: float = 0.2
    """Flat ground immediately after the wedges used as the success contact region."""

    ground_thickness_m: float = 0.1
    """Thickness of the collidable tile ground, including the center gap floor."""


@configclass
class MeshBrakingSlopeTerrainCfg(SubTerrainBaseCfg):
    """Four ordered stair treads connected directly to a 30-degree down slope."""

    function = braking_slope_terrain.braking_slope_terrain

    walkable_width_m: float = 2.0
    num_steps: int = 4
    slope_angle_deg: float = 30.0
    slope_foot_x_m: float = 1.1
    finish_flat_length_m: float = 0.4
    success_region_length_m: float = 0.2
    ground_thickness_m: float = 0.1
    step_height_by_level: tuple[float, ...] = (
        0.14, 0.1425, 0.145, 0.1475, 0.15,
        0.15, 0.1525, 0.155, 0.1575, 0.16,
    )
    tread_depth_by_level: tuple[float, ...] = (
        0.32, 0.315, 0.31, 0.305, 0.30,
        0.30, 0.295, 0.29, 0.285, 0.28,
    )


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
class HfThinObstaclesTerrainCfg(HfTerrainBaseCfg):
    """Flat height field with randomly placed, non-overlapping thin positive obstacles."""

    function = custom_terrains.thin_obstacles_terrain

    obstacle_width_range: tuple[float, float] = (0.1, 0.15)
    """The sampled short horizontal dimension of each obstacle, in meters."""

    obstacle_length: float = 1.5
    """The fixed long horizontal dimension of each obstacle, in meters."""

    obstacle_height_range: tuple[float, float] = (0.08, 0.18)
    """The sampled positive obstacle height of each obstacle, in meters."""

    num_obstacles: int = 30
    """The exact number of obstacles to place."""

    platform_width: float = 1.0
    """The square center platform kept clear for spawning, in meters."""

    boundary_margin: float = 0.5
    """Minimum distance from each terrain tile boundary, in meters."""

    max_sampling_attempts: int = 200
    """Maximum placement attempts per obstacle before generation fails."""


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

    railing_tile_probability: float = 0.0
    """Probability that a generated stair tile contains diagonal side railings."""

    railing_side_probability: float = 0.5
    """Independent selection probability for each of the four diagonal stair side seams."""

    ensure_one_railing: bool = True
    """Whether a selected railing tile must contain at least one railing side."""

    railing_thickness: float = 0.1
    """Horizontal thickness of each continuous solid railing, in meters."""

    railing_top_clearance_range: tuple[float, float] = (0.2, 0.5)
    """Railing top elevation above the highest stair tread, in meters."""


@configclass
class HfNoisyInvertedPyramidStairsTerrainCfg(HfNoisyPyramidStairsTerrainCfg):
    """Configuration for an inverted pyramid stair height field terrain with noise.

    Note:
        This is a subclass of :class:`HfNoisyPyramidStairsTerrainCfg` with :obj:`inverted` set to True.
    """

    inverted: bool = True


@configclass
class MeshSpiralPyramidStairsTerrainCfg(SubTerrainBaseCfg):
    """Exact rotated pyramid stairs with fixed horizontal geometry."""

    function = spiral_pyramid_stairs_terrain.spiral_pyramid_stairs_terrain

    step_height_range: tuple[float, float] = MISSING
    """Minimum and maximum step heights controlled by terrain difficulty."""

    num_steps: int = 7
    """Fixed number of risers from the outer tread to the center platform."""

    center_tread_depth: float = 0.33
    """Tread depth at the center of each stair side, in meters."""

    usable_tread_width: float = 2.0
    """Width used to validate the inner and outer tread depths, in meters."""

    platform_width: float = 2.0
    """Exact side length of the square center platform, in meters."""

    border_width: float = 0.5
    """Minimum flat clearance between the stairs and the tile boundary."""

    rotation_per_step_deg: float = 4.5
    """Adjacent-level rotation: higher is counterclockwise from lower, in degrees."""

    inverted: bool = False
    """Whether the center platform is below the outer terrain surface."""

    ground_thickness: float = 0.1
    """Thickness below the lowest stair surface used to close the collision mesh."""

    railing_tile_probability: float = 0.0
    """Probability that a generated stair tile contains fan-shaped corner railings."""

    railing_side_probability: float = 0.5
    """Independent selection probability for each of the four corner railings."""

    ensure_one_railing: bool = True
    """Whether a selected railing tile must contain at least one railing side."""

    railing_thickness: float = 0.1
    """Horizontal thickness of each continuous solid railing, in meters."""

    railing_top_clearance: float = 0.3
    """Railing top elevation above the terrain's highest stair tread, in meters."""


@configclass
class MeshSpiralInvertedPyramidStairsTerrainCfg(MeshSpiralPyramidStairsTerrainCfg):
    """Exact rotated pyramid stairs whose center platform is the lowest level."""

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

    Creates nested ring-shaped hurdle structures (一环套一环).
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
    By default, the hurdles are uniformly distributed from the platform to the terrain edge.
    """

    function = custom_terrains.hurdle_terrain

    platform_width: float = 1.0
    """The width of the square platform at the center (in m). Defaults to 1.0."""

    num_hurdles: int = 3
    """Number of concentric hurdle rings. Defaults to 3."""

    first_hurdle_distance: float | None = None
    """Optional distance from the terrain center to the first hurdle's inner edge (in m).

    When set, the first gap ends at this distance. Unless ``inter_hurdle_gap_range`` is set,
    remaining hurdle rings are distributed uniformly between the first hurdle and the terrain edge.
    """

    inter_hurdle_gap_range: tuple[float, float] | None = None
    """Optional clear gap between adjacent hurdle rings at difficulty 0 and 1 (in m).

    When set, the gap is linearly interpolated with terrain difficulty. The first hurdle
    remains in its normal position, and subsequent hurdles move to realize this gap.
    """

    hurdle_height_range: tuple[float, float] = (0.05, 0.3)
    """The minimum and maximum height of hurdles (in m). Defaults to (0.05, 0.3).
    
    Hurdle height increases with difficulty. The first value is the height at difficulty=0,
    the second value is the height at difficulty=1.
    """

    hurdle_heights_m: tuple[float, ...] | None = None
    """Optional fixed heights ordered from the innermost ring to the outermost ring.

    When set, these values override ``hurdle_height_range`` and must contain one height per hurdle.
    Configurations that leave this field as ``None`` retain the existing shared curriculum-height behavior.
    """

    hurdle_width_pixels: int | tuple[int, ...] = 3
    """The width of each hurdle ring in pixels (horizontal_scale multiples). Defaults to 3.
    
    If a single int is provided, all hurdles have the same width.
    If a tuple is provided, its length must equal ``num_hurdles``.
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


@configclass
class HfInterlacedCorridorWedgesTerrainCfg(HfTerrainBaseCfg):
    """沿 ``+x`` 的连续交叉斜坡：所有段横向边界统一为 ``y=±corridor_half_width_m``（默认 ±1.5，宽 3 m），
    奇偶段坡向交替（之字）。

    默认 4 段 × 1.5 m → 总长 6 m，坡度 15°。
    """

    function = custom_terrains.interlaced_corridor_wedges_terrain

    corridor_half_width_m: float = 1.5
    """走廊半宽（米）：每段 ``y_lo=-corridor_half_width_m``、``y_hi=+corridor_half_width_m``。"""

    segment_length_m: float = 1.5
    """沿 ``+x`` 每段三棱柱长度（米）。"""

    num_segments: int = 4
    """交叉坡段数；总长度 ``num_segments * segment_length_m``（默认 6 m）。"""

    slope_angle_range: tuple[float, float] = (15.0, 15.0)
    """坡面仰角（度）；``(15, 15)`` 为固定 15°，也可设课程区间。"""

    edge_blend_m: float = 0.02
    """横向棱边软过渡（米）；越小越利。"""

    longitudinal_blend_m: float | None = None
    """图案整体首尾沿 ``x`` 的 fade 宽度（米）；``None`` 表示与 ``edge_blend_m`` 相同。"""

    pattern_origin_x_m: float = -3.0
    """图案起点（米）：第 0 段从 ``pattern_origin_x_m`` 起，覆盖 ``num_segments * segment_length_m``。"""

    corridor_edge_fade_m: float = 0.06
    """在 ``|y| > corridor_half_width_m`` 外按此宽度将高度收到 0。"""


@configclass
class HfContinuousWaveSlopesTerrainCfg(HfTerrainBaseCfg):
    """沿 ``+x`` 连续布置两组“上坡-平台-下坡”的全宽斜坡。"""

    function = custom_terrains.continuous_wave_slopes_terrain

    slope_surface_length_m: float = 1.5
    """每个坡面的实际坡面长度（米），不是沿 x 轴的投影长度。"""

    slope_angle_deg: float = 15.0
    """坡面与水平面的夹角（度）。"""

    platform_length_m: float = 0.8
    """每个峰顶连接段沿 x 轴的长度（米）。"""

    half_width_m: float = 1.2
    """可行走地形半宽（米），对应总宽 2.4 m。"""

    edge_transition_m: float = 0.05
    """侧边高度收敛宽度（米）；默认等于一个高度场像素。"""


@configclass
class HfFigure8BridgeTerrainCfg(HfTerrainBaseCfg):
    """平面闭合 8 字桥：两圆于原点相切，圆环桥面 + 坑底。

    默认内缘半径 2.5 m、桥宽 0.5 m → 中心线半径 2.75 m，外缘 3.0 m；
    圆心 ``(±2.75, 0)``，建议场地 ``size >= (12, 12)`` m。
    """

    function = custom_terrains.figure8_bridge_terrain

    size: tuple[float, float] = (12.0, 12.0)
    """单格场地边长（米）；双圆外廓约 11.5 m，需大于默认 8×8。"""

    inner_radius_m: float = 2.5
    """桥面**内缘**相对圆心的半径（米）；中心线在其外缘再外扩 ``bridge_width/2``。"""

    bridge_height: float = 0.1
    """桥面顶面高度（米）。"""

    bridge_width_range: tuple[float, float] = (0.8, 0.45)
    """兼容旧逻辑：未设置 ``bridge_width_by_level`` 时在 ``[0]`` 与 ``[1]`` 间线性插值。"""

    bridge_width_by_level: tuple[float, ...] = (0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4)
    """离散等级桥宽（米）：与 num_rows=10 对齐，level 0→0.85 … level 9→0.4。"""

    holes_depth: tuple[float, float] = (-0.3, -0.5)
    """坑底深度（米，负值）；随 difficulty 插值。"""

    radial_edge_fade_m: float = 0.03
    """桥面内外缘软过渡宽度（米）。"""

    samples_per_loop: int = 256
    """每条圆环中心线弧长采样数（供 MDP 路径几何缓存）。"""

    center_cutout_width_m: float = 0.0
    """原点矩形挖空宽度（米，沿 x）；0 表示关闭。挖空区高度为坑底 ``holes_depth``。"""

    center_cutout_length_m: float = 0.0
    """原点矩形挖空长度（米，沿 y）；0 表示关闭。"""


@configclass
class MeshFigure8BridgeTerrainCfg(SubTerrainBaseCfg):
    """Trimesh 参数化闭合 8 字桥：几何与 :class:`HfFigure8BridgeTerrainCfg` 一致，桥面为光滑圆环 mesh。

    与高度场版相比不使用 ``radial_edge_fade_m`` / ``slope_threshold`` 离散化，内外缘为解析圆弧。
    """

    function = figure8_mesh_terrains.figure8_bridge_mesh_terrain

    size: tuple[float, float] = (12.0, 12.0)
    """单格场地边长（米）。"""

    inner_radius_m: float = 2.5
    """桥面内缘相对圆心的半径（米）。"""

    bridge_height: float = 0.1
    """桥面顶面高度（米）。"""

    bridge_width_range: tuple[float, float] = (0.8, 0.45)
    """未设置 ``bridge_width_by_level`` 时在区间内线性插值。"""

    bridge_width_by_level: tuple[float, ...] = (0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4)
    """离散等级桥宽（米）。"""

    holes_depth: tuple[float, float] = (-0.3, -0.5)
    """坑底深度（米，负值）；随 difficulty 插值。"""

    radial_edge_fade_m: float = 0.03
    """保留字段以与 HF 配置对齐；Trimesh 版不使用软边 fade。"""

    samples_per_loop: int = 128
    """每圈圆弧离散段数（越大越光滑）。"""

    center_cutout_width_m: float = 0.0
    """原点矩形挖空宽度（米，沿 x）；0 表示关闭。"""

    center_cutout_length_m: float = 0.0
    """原点矩形挖空长度（米，沿 y）；0 表示关闭。"""


@configclass
class MeshSCurveTerrainCfg(SubTerrainBaseCfg):
    """Trimesh 参数化 S 型高台：由两个相切的标准半圆组成。

    默认 S 中心在子地形原点，两个半圆半径 1.6 m；路径宽度随地形等级逐渐变窄。
    """

    function = figure8_mesh_terrains.s_curve_mesh_terrain

    size: tuple[float, float] = (8.0, 8.0)
    """单格场地边长（米）。"""

    radius_m: float = 1.6
    """两个标准半圆的中心线半径（米）。"""

    height_m: float = 0.1
    """兼容旧配置：未设置 ``height_range_m`` 时使用的固定高台高度（米）。"""

    height_range_m: tuple[float, float] | None = (0.05, 0.2)
    """S 型高台相对于平地的顶面高度范围（米），按 difficulty 从低到高线性插值。"""

    path_width_range: tuple[float, float] = (0.85, 0.4)
    """未设置 ``path_width_by_level`` 时在区间内线性插值；通常从宽到窄。"""

    path_width_by_level: tuple[float, ...] = (0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4)
    """离散等级路径宽度（米）：与 num_rows=10 对齐，level 越高越窄。"""

    center_offset_m: tuple[float, float] = (0.0, 0.0)
    """S 型中心点相对子地形原点的偏移（米，x/y）。"""

    samples_per_semicircle: int = 64
    """每个半圆的离散段数（越大越光滑）。"""
