from __future__ import annotations

from dataclasses import MISSING
from typing import List

from isaaclab.terrains.terrain_generator_cfg import SubTerrainBaseCfg
from isaaclab.utils import configclass

from ..height_field import PerlinPlaneTerrainCfg
from . import mesh_terrains


class WallTerrainCfgMixin:
    wall_prob: List[float] = [0.0, 0.0, 0.0, 0.0]  # Probability of generating walls on [left, right, front, back] sides
    wall_height: float = 5.0  # Height of the walls
    wall_thickness: float = 0.05  # Thickness of the walls


@configclass
class MotionMatchedTerrainCfg(SubTerrainBaseCfg):
    """Configuration for motion-matched terrain generation.

    ## Terrain Mesh Requirements
    - All terrain meshes must have the a border at the bottom.
    - The terrain origin (0, 0, 0) is at the surface of the terrain center, which means that the point should
        be above the terrain at (0, 0, t) given any t > 0 and below the terrain at (0, 0, t) given any t < 0.
    - The USER should ensure that the non-flat part of the terrain is within the size of the terrain.
    """

    function = mesh_terrains.motion_matched_terrain

    path: str = MISSING
    """Directory containing both terrains and the motions, so that these can be matched together.
    """

    metadata_yaml: str = MISSING
    """Path to metadata file (YAML or JSON) for motion matching.
    File extension .yaml/.yml uses YAML, .json uses JSON.

    You may use the `scripts/motion_matched_metadata_generator.py` to generate the metadata file if you arrange your
    dataset in the structure as described there.

    ## Typical metadata.yaml structure

    ```yaml
    motion_files:
      - motion_file: kneelClimbStep1-x-0.1-ziwen-retargeted.npz
        terrain_id: 0
    terrains:
      - terrain_file: box.stl          # relative to path (dataset dir)
        terrain_id: 0
        origin_offset: [0.0, 0.0, 0.0]   # optional: [x,y,z] mesh position relative to env.scene.env_origins
      - terrain_file: another.stl
        terrain_id: 1
    ```

    **origin_offset**: Optional 3-element list [x, y, z]. Defines the mesh position relative to the env origin
    (scene.env_origins). Only the imported mesh is translated; the cell origin and ground plane are unchanged.
    Omit this key or use [0,0,0] for default centering.
    """


@configclass
class ExternalMeshFromMetadataTerrainCfg(SubTerrainBaseCfg):
    """可选加载外部 mesh 的子地形：从 metadata.yaml 按 terrain_id 选择 mesh，可叠加在基础地形上，位置/角度可来自 metadata，偏移随难度线性插值。

    - base_terrain_cfg：可选基础地形（如 noisy_slope_up 等），先生成该地形再在其上叠加 mesh；None 则仅用平地。
    - terrain_id 为 None 时，用 difficulty 在 terrains 列表中选取。
    - position / angle：mesh 基准位置 (x,y,z) 与欧拉角 (roll,pitch,yaw 弧度)，可被 metadata 中同名键覆盖。
    - position_offset_range_* / angle_offset_range_*：每轴 (min, max)，随难度线性插值，offset = min + (max - min) * difficulty。
    path 与 metadata_yaml 可在运行时由 flat_env_cfg 等设置。
    """

    function = mesh_terrains.external_mesh_from_metadata_terrain
    path: str = MISSING
    """数据集目录，与 metadata.yaml 同根。运行时可由 env_cfg 覆盖。"""
    metadata_yaml: str = ""
    """metadata 文件路径，空则使用 os.path.join(path, 'metadata.yaml')。"""
    terrain_id: int | None = None
    """metadata.terrains 中要使用的条目 terrain_id；None 表示按 difficulty 选索引。"""
    base_terrain_cfg: SubTerrainBaseCfg | None = None
    """基础地形配置（如 HfNoisyPyramidSlopedTerrainCfg 等），先生成该地形再在其上叠加外部 mesh；None 表示仅用平地。"""
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """mesh 在格子内的基准位置 (x,y,z)，可被 metadata 中 "position": [x,y,z] 覆盖。"""
    angle: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """mesh 的基准欧拉角 (roll, pitch, yaw) 弧度，可被 metadata 中 "angle": [roll,pitch,yaw] 覆盖。"""
    # 各轴偏移范围 (min, max)：随难度线性插值，左为难度 0，右为难度 1
    position_offset_range_x: tuple[float, float] = (0.0, 0.0)
    position_offset_range_y: tuple[float, float] = (0.0, 0.0)
    position_offset_range_z: tuple[float, float] = (0.0, 0.0)
    angle_offset_range_roll: tuple[float, float] = (0.0, 0.0)
    angle_offset_range_pitch: tuple[float, float] = (0.0, 0.0)
    angle_offset_range_yaw: tuple[float, float] = (0.0, 0.0)


@configclass
class PerlinMeshFloatingBoxTerrainCfg(SubTerrainBaseCfg, WallTerrainCfgMixin):
    """Configuration for a floating box mesh terrain."""

    function = mesh_terrains.floating_box_terrain
    floating_height: tuple[float, float] | float = MISSING
    """The height of the box above the ground. Could be a fixed value or a range (min, max)."""
    box_length: tuple[float, float] | float = MISSING
    """The length of the box along the y-axis. Could be a fixed value or a range (min, max)."""
    box_width: float | None = None
    """The width of the box along the x-axis. If None, it will be equal to the width of the terrain."""
    box_height: tuple[float, float] | float = MISSING
    """The height of the box along the z-axis."""
    perlin_cfg: PerlinPlaneTerrainCfg | None = None

    # values used for perlin noise generation
    horizontal_scale: float = 0.1
    vertical_scale: float = 0.005
    slope_threshold: float | None = None
    no_perlin_at_obstacle: bool = True
    """If True, no perlin noise will be generated exactly below the box."""


@configclass
class PerlinMeshRandomMultiBoxTerrainCfg(SubTerrainBaseCfg, WallTerrainCfgMixin):
    """Configuration for a sub terrain with multiple random boxes with perlin noise."""

    function = mesh_terrains.random_multi_box_terrain
    box_height_mean: tuple[float, float] | float = MISSING
    box_height_range: float = MISSING
    box_length_mean: tuple[float, float] | float = MISSING
    box_length_range: float = MISSING
    box_width_mean: tuple[float, float] | float = MISSING
    box_width_range: float = MISSING
    platform_width: float = MISSING

    generation_ratio: float = MISSING

    perlin_cfg: PerlinPlaneTerrainCfg | None = None
    horizontal_scale: float = 0.1
    vertical_scale: float = 0.005
    slope_threshold: float | None = None
    no_perlin_at_obstacle: bool = False
    box_perlin_cfg: PerlinPlaneTerrainCfg | None = None
    """Used only when perlin_cfg is not None"""
