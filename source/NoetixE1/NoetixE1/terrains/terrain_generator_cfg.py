# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

"""
Configuration classes defining the different terrains available. Each configuration class must
inherit from ``isaaclab.terrains.terrains_cfg.TerrainConfig`` and define the following attributes:

- ``name``: Name of the terrain. This is used for the prim name in the USD stage.
- ``function``: Function to generate the terrain. This function must take as input the terrain difficulty
  and the configuration parameters and return a `tuple with the `trimesh`` mesh object and terrain origin.
"""

from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.terrains.height_field.hf_terrains_cfg import (
    HfRandomUniformTerrainCfg,
    HfDiscreteObstaclesTerrainCfg,
    HfPyramidSlopedTerrainCfg,
    HfInvertedPyramidSlopedTerrainCfg,
    HfWaveTerrainCfg,
    HfDiscreteObstaclesTerrainCfg,
)
from isaaclab.terrains.trimesh.mesh_terrains_cfg import (
    MeshPyramidStairsTerrainCfg,
    MeshInvertedPyramidStairsTerrainCfg,
    MeshStarTerrainCfg,
    MeshRailsTerrainCfg,
    MeshRandomGridTerrainCfg,
    MeshPitTerrainCfg,
)
from .custom_terrain import (
    HfNoisyInvertedPyramidSlopedTerrainCfg,
    HfNoisyInvertedPyramidStairsTerrainCfg,
    HfNoisyPyramidSlopedTerrainCfg,
    HfNoisyPyramidStairsTerrainCfg,
    MeshSpiralInvertedPyramidStairsTerrainCfg,
    MeshSpiralPyramidStairsTerrainCfg,
    HfThinObstaclesTerrainCfg,
    MeshFlatPillarsTerrainCfg,
    MeshBrakingSlopeTerrainCfg,
    HfMultiGapTerrainCfg,
    HfSteppingStonesTerrainCfg,
    HfCrossSteppingStonesTerrainCfg,
    HfHurdleTerrainCfg,
    HfFramePlatformTerrainCfg,
    HfDoubleFramePlatformTerrainCfg,
    HfSingleGapTerrainCfg,
    HfInterlacedCorridorWedgesTerrainCfg,
    HfContinuousWaveSlopesTerrainCfg,
    MeshFigure8BridgeTerrainCfg,
    MeshSCurveTerrainCfg,
    MeshSymmetricSlopeTerrainCfg,
)
from .trimesh import ExternalMeshFromMetadataTerrainCfg, MeshLRailedPlatformTerrainCfg

GRAVEL_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=10.0,
    num_rows=10,
    num_cols=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "random_rough": HfRandomUniformTerrainCfg(proportion=0.3, noise_range=(0.01, 0.1), noise_step=0.02),
        "discrete_obstacles": HfDiscreteObstaclesTerrainCfg(
            proportion=0.1, obstacle_width_range=(0.2, 1.0), obstacle_height_range=(0.05, 0.1), num_obstacles=10
        ),
        "slope_up": HfPyramidSlopedTerrainCfg(proportion=0.15, slope_range=(0.1, 0.5)),
        "slope_down": HfInvertedPyramidSlopedTerrainCfg(proportion=0.15, slope_range=(0.1, 0.5)),
        "stairs_up": MeshPyramidStairsTerrainCfg(
            proportion=0.15, step_height_range=(0.05, 0.2), step_width=0.3
        ),
        "stairs_down": MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.15, step_height_range=(0.05, 0.2), step_width=0.3
        ),
    },
)

DEPTH_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=10,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=0.9,
    use_cache=False,
    sub_terrains={
        "noisy_slope_up": HfNoisyPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.1, 0.4), noise_range=(0.01, 0.03), noise_step=0.01,
            num_obstacles=0, obstacle_size=(0.2, 0.4), obstacle_height=(0.5, 1.5), platform_width=2.0),
        "noisy_slope_down": HfNoisyInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.15, 0.4), noise_range=(0.01, 0.03), noise_step=0.01,
            num_obstacles=0, obstacle_size=(0.2, 0.4), obstacle_height=(0.5, 1.5), platform_width=2.0),
        "stairs_up": HfNoisyInvertedPyramidStairsTerrainCfg(
            proportion=0.2, step_height_range=(0.05, 0.25), step_width_pixels_range=(10, 5), platform_width=2.0, border_width=0.5,
            noise_range=(0.00, 0.00), noise_step=0.02),
        "stairs_down": HfNoisyPyramidStairsTerrainCfg(
            proportion=0.2, step_height_range=(0.05, 0.25), step_width_pixels_range=(10, 5), platform_width=2.0, border_width=0.5,
            noise_range=(0.00, 0.00), noise_step=0.02),
        # "noisy_stairs_up": HfNoisyInvertedPyramidStairsTerrainCfg(
        #     proportion=0.1, step_height_range=(0.05, 0.25), step_width_pixels_range=(10, 5), platform_width=2.0, border_width=0.5,
        #     noise_range=(0.01, 0.02), noise_step=0.02
        # ),
        # "noisy_stairs_down": HfNoisyPyramidStairsTerrainCfg(
        #     proportion=0.1, step_height_range=(0.05, 0.25), step_width_pixels_range=(10, 5), platform_width=2.0, border_width=0.5,
        #     noise_range=(0.01, 0.02), noise_step=0.02
        # ),
        "gap": HfSingleGapTerrainCfg(
            proportion=0.3, gap_width_pixels_range=(3, 10), gap_depth_range=(0.2, 1.0),
            step_height_range=(0.03, 0.1), platform_width=2.0),
        # "cross_stepping_stones1": HfCrossSteppingStonesTerrainCfg(
        #     proportion=0.5,
        #     stone_height_max=0.05,
        #     stone_width_range=(0.25, 0.5),
        #     stone_distance_range=(0.1, 0.35),
        #     holes_depth=(-0.3, -0.5),
        #     num_stone_rows=1,
        #     platform_width=1.5,
        # ),
        # "cross_stepping_stones2": HfCrossSteppingStonesTerrainCfg(
        #     proportion=0.5,
        #     stone_height_max=0.05,
        #     stone_width_range=(0.25, 0.5),
        #     stone_distance_range=(0.1, 0.35),
        #     holes_depth=(-0.3, -0.5),
        #     num_stone_rows=2,
        #     platform_width=1.5,
        # ),
        # "stepping_stones": HfSteppingStonesTerrainCfg(
        #     proportion=0.1,
        #     stone_height_max=0.05,
        #     stone_width_range=(0.25, 0.5),
        #     stone_distance_range=(0.1, 0.35),
        #     holes_depth=(-0.3, -0.5),
        #     border_width=0.5,
        #     platform_width=2.0,
        # ),
        # "beam": MeshStarTerrainCfg(
        #     proportion=0.1,
        #     num_bars=4,
        #     bar_width_range=(0.251, 0.5),
        #     bar_height_range=(0.3, 1.0),
        #     platform_width=2.0,
        # ),
        "discrete_obstacles": HfDiscreteObstaclesTerrainCfg(
            proportion=0.1, obstacle_width_range=(0.3, 1.0), obstacle_height_range=(0.05, 0.30), num_obstacles=30
        ),
        # "hurdle": HfHurdleTerrainCfg(
        #     proportion=0.1, platform_width=1.5, hurdle_height_range=(0.05, 0.35), hurdle_width_pixels=(1, 2, 3),
        # ),
    },
)


DEPTH_TERRAINS_CFG2 = TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=10,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=0.9,
    use_cache=False,
    sub_terrains={
        "rough_flat": HfNoisyPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.0), noise_range=(0.01, 0.05), noise_step=0.01,
            num_obstacles=0, obstacle_size=(0.2, 0.4), obstacle_height=(0.5, 1.5), platform_width=2.0),
        "wave": HfWaveTerrainCfg(proportion=0.1, num_waves=10.0, amplitude_range=(0.02, 0.05)),
        "noisy_slope_up": HfNoisyPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.1, 0.4), noise_range=(0.0, 0.0), noise_step=0.01,
            num_obstacles=0, obstacle_size=(0.2, 0.4), obstacle_height=(0.5, 1.5), platform_width=2.0),
        "noisy_slope_down": HfNoisyInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.15, 0.4), noise_range=(0.0, 0.0), noise_step=0.01,
            num_obstacles=0, obstacle_size=(0.2, 0.4), obstacle_height=(0.5, 1.5), platform_width=2.0),
        "stairs_up": HfNoisyInvertedPyramidStairsTerrainCfg(
            proportion=0.2, step_height_range=(0.05, 0.25), step_width_pixels_range=(10, 5), platform_width=2.0, border_width=0.5,
            noise_range=(0.00, 0.00), noise_step=0.02),
        "stairs_down": HfNoisyPyramidStairsTerrainCfg(
            proportion=0.2, step_height_range=(0.05, 0.25), step_width_pixels_range=(10, 5), platform_width=2.0, border_width=0.5,
            noise_range=(0.00, 0.00), noise_step=0.02),
        "gap": HfMultiGapTerrainCfg(
            proportion=0.1, gap_width_pixels_range=(3, 10), gap_depth_range=(0.2, 1.0),
            step_height_range=(0.03, 0.1), platform_width=2.0),
        "discrete_obstacles": HfDiscreteObstaclesTerrainCfg(
            proportion=0.1, obstacle_width_range=(0.3, 1.0), obstacle_height_range=(0.05, 0.30), num_obstacles=30
        ),
    },
)

DEPTH_TERRAINS_CFG3 = TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=15,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=0.9,
    use_cache=False,
    sub_terrains={
        "rough_flat": HfNoisyPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.0), noise_range=(0.01, 0.05), noise_step=0.01,
            num_obstacles=0, obstacle_size=(0.2, 0.4), obstacle_height=(0.5, 1.5), platform_width=2.0),
        "wave": HfWaveTerrainCfg(proportion=0.1, num_waves=10.0, amplitude_range=(0.02, 0.05)),
        "noisy_slope_up": HfNoisyPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.1, 0.4), noise_range=(0.0, 0.0), noise_step=0.01,
            num_obstacles=0, obstacle_size=(0.2, 0.4), obstacle_height=(0.5, 1.5), platform_width=2.0),
        "noisy_slope_down": HfNoisyInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.15, 0.4), noise_range=(0.0, 0.0), noise_step=0.01,
            num_obstacles=0, obstacle_size=(0.2, 0.4), obstacle_height=(0.5, 1.5), platform_width=2.0),
        "stairs_up": HfNoisyInvertedPyramidStairsTerrainCfg(
            proportion=0.2, step_height_range=(0.05, 0.25), step_width_pixels_range=(10, 5), platform_width=2.0, border_width=0.5,
            noise_range=(0.00, 0.00), noise_step=0.02),
        "stairs_down": HfNoisyPyramidStairsTerrainCfg(
            proportion=0.2, step_height_range=(0.05, 0.25), step_width_pixels_range=(10, 5), platform_width=2.0, border_width=0.5,
            noise_range=(0.00, 0.00), noise_step=0.02),
        "gap": HfMultiGapTerrainCfg(
            proportion=0.2, gap_width_pixels_range=(3, 10), gap_depth_range=(0.2, 1.0),
            step_height_range=(0.03, 0.1), platform_width=2.0),
        "discrete_obstacles": HfDiscreteObstaclesTerrainCfg(
            proportion=0.1, obstacle_width_range=(0.3, 1.0), obstacle_height_range=(0.05, 0.30), num_obstacles=30
        ),
        "cross_stepping_stones1": HfCrossSteppingStonesTerrainCfg(
            proportion=0.1,
            stone_height_max=0.05,
            stone_width_range=(0.25, 0.5),
            stone_distance_range=(0.1, 0.35),
            holes_depth=(-0.3, -0.5),
            num_stone_rows=1,
            platform_width=1.5,
        ),
        "cross_stepping_stones2": HfCrossSteppingStonesTerrainCfg(
            proportion=0.1,
            stone_height_max=0.05,
            stone_width_range=(0.25, 0.5),
            stone_distance_range=(0.1, 0.35),
            holes_depth=(-0.3, -0.5),
            num_stone_rows=2,
            platform_width=1.5,
        ),
        "stepping_stones": HfSteppingStonesTerrainCfg(
            proportion=0.2,
            stone_height_max=0.05,
            stone_width_range=(0.25, 0.5),
            stone_distance_range=(0.1, 0.35),
            holes_depth=(-0.3, -0.5),
            border_width=0.5,
            platform_width=2.0,
        ),
    },
)


DEPTH_TERRAINS_CFG4 = TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=15,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=0.9,
    use_cache=False,
    sub_terrains={
        "rough_flat": HfNoisyPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.0), noise_range=(0.01, 0.05), noise_step=0.01,
            num_obstacles=0, obstacle_size=(0.2, 0.4), obstacle_height=(0.5, 1.5), platform_width=2.0),
        "wave": HfWaveTerrainCfg(proportion=0.1, num_waves=10.0, amplitude_range=(0.02, 0.05)),
        "noisy_slope_up": HfNoisyPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.1, 0.4), noise_range=(0.0, 0.0), noise_step=0.01,
            num_obstacles=0, obstacle_size=(0.2, 0.4), obstacle_height=(0.5, 1.5), platform_width=2.0),
        "noisy_slope_down": HfNoisyInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.15, 0.4), noise_range=(0.0, 0.0), noise_step=0.01,
            num_obstacles=0, obstacle_size=(0.2, 0.4), obstacle_height=(0.5, 1.5), platform_width=2.0),
        "stairs_up": HfNoisyInvertedPyramidStairsTerrainCfg(
            proportion=0.2, step_height_range=(0.05, 0.25), step_width_pixels_range=(10, 5), platform_width=2.0, border_width=0.5,
            noise_range=(0.00, 0.00), noise_step=0.02),
        "stairs_down": HfNoisyPyramidStairsTerrainCfg(
            proportion=0.2, step_height_range=(0.05, 0.25), step_width_pixels_range=(10, 5), platform_width=2.0, border_width=0.5,
            noise_range=(0.00, 0.00), noise_step=0.02),
        "beam": MeshStarTerrainCfg(
            proportion=0.4,
            num_bars=2,
            bar_width_range=(0.1, 0.3),
            bar_height_range=(0.3, 1.0),
            platform_width=2.0,
        ),
        "pit": MeshRailsTerrainCfg(
            proportion=0.4,
            rail_thickness_range=(0.5, 1.0),
            rail_height_range=(0.45, 0.2),
            platform_width=2.0,
        ),
    },
)


DEPTH_TERRAINS_CFG5 = TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=15,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=0.9,
    use_cache=False,
    sub_terrains={
        "rough_flat": HfNoisyPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.0), noise_range=(0.01, 0.05), noise_step=0.01,
            num_obstacles=0, obstacle_size=(0.2, 0.4), obstacle_height=(0.5, 1.5), platform_width=2.0),
        "wave": HfWaveTerrainCfg(proportion=0.1, num_waves=10.0, amplitude_range=(0.02, 0.05)),
        "noisy_slope_up": HfNoisyPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.1, 0.4), noise_range=(0.0, 0.0), noise_step=0.01,
            num_obstacles=0, obstacle_size=(0.2, 0.4), obstacle_height=(0.5, 1.5), platform_width=2.0),
        "noisy_slope_down": HfNoisyInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.15, 0.4), noise_range=(0.0, 0.0), noise_step=0.01,
            num_obstacles=0, obstacle_size=(0.2, 0.4), obstacle_height=(0.5, 1.5), platform_width=2.0),
        "stairs_up": HfNoisyInvertedPyramidStairsTerrainCfg(
            proportion=0.2, step_height_range=(0.05, 0.25), step_width_pixels_range=(10, 5), platform_width=2.0, border_width=0.5,
            noise_range=(0.00, 0.00), noise_step=0.02),
        "stairs_down": HfNoisyPyramidStairsTerrainCfg(
            proportion=0.2, step_height_range=(0.05, 0.25), step_width_pixels_range=(10, 5), platform_width=2.0, border_width=0.5,
            noise_range=(0.00, 0.00), noise_step=0.02),
        "gap": HfMultiGapTerrainCfg(
            proportion=0.2, gap_width_pixels_range=(3, 10), gap_depth_range=(0.2, 1.0),
            step_height_range=(0.03, 0.1), platform_width=2.0),
        "cross_stepping_stones1": HfCrossSteppingStonesTerrainCfg(
            proportion=0.1,
            stone_height_max=0.05,
            stone_width_range=(0.25, 0.5),
            stone_distance_range=(0.1, 0.35),
            holes_depth=(-0.3, -0.5),
            num_stone_rows=1,
            platform_width=1.5,
        ),
        "stepping_stones": HfSteppingStonesTerrainCfg(
            proportion=0.2,
            stone_height_max=0.05,
            stone_width_range=(0.25, 0.5),
            stone_distance_range=(0.1, 0.35),
            holes_depth=(-0.3, -0.5),
            border_width=0.5,
            platform_width=2.0,
        ),
        "beam": MeshStarTerrainCfg(
            proportion=0.1,
            num_bars=2,
            bar_width_range=(0.15, 0.3),
            bar_height_range=(0.3, 1.0),
            platform_width=2.0,
        ),
        "pit": MeshRailsTerrainCfg(
            proportion=0.1,
            rail_thickness_range=(0.5, 1.0),
            rail_height_range=(0.1, 0.45),
            platform_width=2.0,
        ),
    },
)

HARD_DEPTH_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=10,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=0.9,
    use_cache=False,
    sub_terrains={
        "noisy_slope_up": HfNoisyPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.1, 0.4), noise_range=(0.01, 0.03), noise_step=0.01,
            num_obstacles=3, obstacle_size=(0.2, 0.4), obstacle_height=(0.5, 1.5), platform_width=2.0),
        "noisy_slope_down": HfNoisyInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.15, 0.4), noise_range=(0.01, 0.03), noise_step=0.01,
            num_obstacles=3, obstacle_size=(0.2, 0.4), obstacle_height=(0.5, 1.5), platform_width=2.0),
        "noisy_stairs_up": HfNoisyInvertedPyramidStairsTerrainCfg(
            proportion=0.1, step_height_range=(0.05, 0.25), step_width_pixels_range=(10, 5), platform_width=2.0, border_width=0.5,
            noise_range=(0.01, 0.02), noise_step=0.02
        ),
        "noisy_stairs_down": HfNoisyPyramidStairsTerrainCfg(
            proportion=0.1, step_height_range=(0.05, 0.25), step_width_pixels_range=(10, 5), platform_width=2.0, border_width=0.5,
            noise_range=(0.01, 0.02), noise_step=0.02
        ),
        "gap": HfMultiGapTerrainCfg(
            proportion=0.1, gap_width_pixels_range=(3, 10), gap_depth_range=(0.2, 1.0),
            step_height_range=(0.03, 0.1), platform_width=2.0),
        "cross_stepping_stones2": HfCrossSteppingStonesTerrainCfg(
            proportion=0.1,
            stone_height_max=0.025,
            stone_width_range=(0.3, 0.5),
            stone_distance_range=(0.1, 0.3),
            holes_depth=(-0.3, -0.5),
            num_stone_rows=2,
            platform_width=1.5,
        ),
        "stepping_stones": HfSteppingStonesTerrainCfg(
            proportion=0.1,
            stone_height_max=0.025,
            stone_width_range=(0.3, 0.5),
            stone_distance_range=(0.1, 0.3),
            holes_depth=(-0.3, -0.5),
            border_width=0.5,
            platform_width=2.0,
        ),
        "beam": MeshStarTerrainCfg(
            proportion=0.1,
            num_bars=4,
            bar_width_range=(0.251, 0.5),
            bar_height_range=(0.3, 1.0),
            platform_width=2.0,
        ),
        "discrete_obstacles": HfDiscreteObstaclesTerrainCfg(
            proportion=0.1, obstacle_width_range=(0.3, 1.0), obstacle_height_range=(0.05, 0.30), num_obstacles=30
        ),
        "hurdle": HfHurdleTerrainCfg(
            proportion=0.1, platform_width=1.5, hurdle_height_range=(0.05, 0.35), hurdle_width_pixels=(1, 2, 3),
        ),
    },
)

STAIR_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=2,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=0.9,
    use_cache=False,
    sub_terrains={
        "stairs_up": HfNoisyInvertedPyramidStairsTerrainCfg(
            proportion=0.2, step_height_range=(0.0, 0.25), step_width_pixels_range=(10, 5), platform_width=2.0, border_width=0.5,
            noise_range=(0.00, 0.00), noise_step=0.02),
        "stairs_down": HfNoisyPyramidStairsTerrainCfg(
            proportion=0.2, step_height_range=(0.0, 0.25), step_width_pixels_range=(10, 5), platform_width=2.0, border_width=0.5,
            noise_range=(0.00, 0.00), noise_step=0.02),
    },
)

PARKOUR_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(10.0, 10.0),
    border_width=20.0,
    num_rows=10,
    num_cols=10,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=0.9,
    use_cache=False,
    sub_terrains={
        "plane": HfNoisyPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.0), noise_range=(0.0, 0.0), noise_step=0.01,
            num_obstacles=0, obstacle_size=(0.2, 0.4), obstacle_height=(0.5, 1.5), platform_width=2.0),
        "rough_flat": HfNoisyPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.0), noise_range=(0.01, 0.05), noise_step=0.01,
            num_obstacles=0, obstacle_size=(0.2, 0.4), obstacle_height=(0.5, 1.5), platform_width=2.0),
        "wave": HfWaveTerrainCfg(proportion=0.1, num_waves=10.0, amplitude_range=(0.02, 0.05)),
        "noisy_slope_up": HfNoisyPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.1, 0.4), noise_range=(0.0, 0.0), noise_step=0.01,
            num_obstacles=0, obstacle_size=(0.2, 0.4), obstacle_height=(0.5, 1.5), platform_width=2.0),
        "noisy_slope_down": HfNoisyInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.15, 0.4), noise_range=(0.0, 0.0), noise_step=0.01,
            num_obstacles=0, obstacle_size=(0.2, 0.4), obstacle_height=(0.5, 1.5), platform_width=2.0),
        "frame_platform": HfDoubleFramePlatformTerrainCfg(
            proportion=0.5,
            height_range_1=(0.3, 0.5), width_range_1=(0.5, 0.5), distance_range_1=(1.25, 1.25),
            height_range_2=(0.3, 0.5), width_range_2=(1.0, 1.0), distance_range_2=(3.0, 3.0),
        ),
    },
)

PARKOUR_TEST_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(10.0, 10.0),
    border_width=20.0,
    num_rows=1,
    num_cols=2,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=0.9,
    use_cache=False,
    sub_terrains={
        "flat": HfNoisyPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.0), noise_range=(0.0, 0.0), noise_step=0.01,
            num_obstacles=0, obstacle_size=(0.2, 0.4), obstacle_height=(0.5, 1.5), platform_width=2.0),
        "frame_platform": HfDoubleFramePlatformTerrainCfg(
            proportion=0.1,
            height_range_1=(0.45, 0.45), width_range_1=(0.5, 0.5), distance_range_1=(1.25, 1.25),
            height_range_2=(0.45, 0.45), width_range_2=(1.0, 1.0), distance_range_2=(3.0, 3.0),
        ),
    },
)


MIMIC_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=False,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "wave": HfWaveTerrainCfg(proportion=0.5, num_waves=10.0, amplitude_range=(0.02, 0.05)),
        "random_rough": HfRandomUniformTerrainCfg(
            proportion=0.5, noise_range=(0.01, 0.05), noise_step=0.015
        ),
    },
)


ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs_28": MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.0, 0.23),
            step_width=0.28,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_30": MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.0, 0.23),
            step_width=0.30,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_32": MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.0, 0.23),
            step_width=0.32,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_34": MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.0, 0.23),
            step_width=0.34,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "boxes": MeshRandomGridTerrainCfg(
            proportion=0.15, grid_width=0.45, grid_height_range=(0.0, 0.15), platform_width=2.0
        ),
        "random_rough": HfRandomUniformTerrainCfg(
            proportion=0.15, noise_range=(-0.02, 0.04), noise_step=0.02, border_width=0.25
        ),
        "wave": HfWaveTerrainCfg(proportion=0.15, amplitude_range=(0.0, 0.2), num_waves=5.0),
        "high_platform": MeshPitTerrainCfg(
            proportion=0.15, pit_depth_range=(0.0, 0.3), platform_width=2.0, double_pit=True
        ),
        # "star": terrain_gen.MeshStarTerrainCfg(
        #     proportion=0.15, num_bars=6, bar_width_range=(0.05, 0.05), bar_height_range=(0.0, 0.25), platform_width=1.0
        # ),
        # "gap": terrain_gen.MeshGapTerrainCfg(
        #     proportion=0.15, gap_width_range=(0.1, 0.4), platform_width=2.0
        # )
    },
)

from isaaclab.terrains import TerrainGeneratorCfg as TerrainGeneratorCfgBase
from isaaclab.utils import configclass

from .terrain_generator import FiledTerrainGenerator


@configclass
class FiledTerrainGeneratorCfg(TerrainGeneratorCfgBase):
    class_type: type = FiledTerrainGenerator


from isaaclab.terrains import FlatPatchSamplingCfg

_DEPTH6_FLAT_PATCH_TARGET = {
    "target": FlatPatchSamplingCfg(
        num_patches=50,
        patch_radius=[0.05, 0.10, 0.15, 0.20],
        max_height_diff=0.05,
    ),
}

# DEPTH_TERRAINS_CFG6：walkrun 主用；含交叉斜坡的 approach 与 smooth entry/exit 变体
DEPTH_TERRAINS_CFG6 = FiledTerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=24,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=0.9,
    seed=42,
    use_cache=False,
    sub_terrains={
        "plane": HfNoisyPyramidSlopedTerrainCfg(
            proportion=0.1,
            slope_range=(0.0, 0.0),
            noise_range=(0.0, 0.0),
            noise_step=0.01,
            num_obstacles=0,
            obstacle_size=(0.2, 0.4),
            obstacle_height=(0.5, 1.5),
            platform_width=2.0,
            flat_patch_sampling=dict(_DEPTH6_FLAT_PATCH_TARGET),
        ),
        # "pillar_flat": MeshFlatPillarsTerrainCfg(
        #     proportion=0.1,
        #     pillar_width_range=(0.1, 0.2),
        #     pillar_thickness_range=(0.1, 0.2),
        #     pillar_height_range=(1.0, 1.5),
        #     num_pillars_range=(4, 8),
        #     platform_width=2.0,
        #     boundary_margin=0.5,
        #     min_pillar_spacing=0.3,
        #     flat_patch_sampling=dict(_DEPTH6_FLAT_PATCH_TARGET),
        # ),
        "rough_flat": HfNoisyPyramidSlopedTerrainCfg(
            proportion=0.1,
            slope_range=(0.0, 0.0),
            noise_range=(0.01, 0.05),
            noise_step=0.01,
            num_obstacles=0,
            obstacle_size=(0.2, 0.4),
            obstacle_height=(0.5, 1.5),
            platform_width=2.0,
            flat_patch_sampling=dict(_DEPTH6_FLAT_PATCH_TARGET),
        ),
        "noisy_slope_up": HfNoisyPyramidSlopedTerrainCfg(
            proportion=0.1,
            slope_range=(0.1, 0.4),
            noise_range=(0.01, 0.03),
            noise_step=0.01,
            num_obstacles=0,
            obstacle_size=(0.2, 0.4),
            obstacle_height=(0.5, 1.5),
            platform_width=2.0,
            flat_patch_sampling=dict(_DEPTH6_FLAT_PATCH_TARGET),
        ),
        "noisy_slope_down": HfNoisyInvertedPyramidSlopedTerrainCfg(
            proportion=0.1,
            slope_range=(0.15, 0.4),
            noise_range=(0.01, 0.03),
            noise_step=0.01,
            num_obstacles=0,
            obstacle_size=(0.2, 0.4),
            obstacle_height=(0.5, 1.5),
            platform_width=2.0,
            flat_patch_sampling=dict(_DEPTH6_FLAT_PATCH_TARGET),
        ),
        "stairs_up": HfNoisyInvertedPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.05, 0.25),
            step_width_pixels_range=(10, 5),
            platform_width=2.0,
            border_width=0.5,
            noise_range=(0.00, 0.00),
            noise_step=0.02,
            railing_tile_probability=0.5,
            railing_side_probability=0.5,
            ensure_one_railing=True,
            railing_thickness=0.1,
            railing_top_clearance_range=(0.2, 0.5),
            flat_patch_sampling=dict(_DEPTH6_FLAT_PATCH_TARGET),
        ),
        "spiral_stairs_up": MeshSpiralInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.25),
            num_steps=7,
            center_tread_depth=0.33,
            usable_tread_width=2.0,
            platform_width=2.0,
            border_width=0.5,
            rotation_per_step_deg=4.5,
            ground_thickness=0.1,
            railing_tile_probability=0.5,
            railing_side_probability=0.5,
            ensure_one_railing=True,
            railing_thickness=0.1,
            railing_top_clearance=0.3,
            flat_patch_sampling=dict(_DEPTH6_FLAT_PATCH_TARGET),
        ),
        "stairs_down": HfNoisyPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.05, 0.25),
            step_width_pixels_range=(10, 5),
            platform_width=2.0,
            border_width=0.5,
            noise_range=(0.00, 0.00),
            noise_step=0.02,
            railing_tile_probability=0.5,
            railing_side_probability=0.5,
            ensure_one_railing=True,
            railing_thickness=0.1,
            railing_top_clearance_range=(0.2, 0.5),
            flat_patch_sampling=dict(_DEPTH6_FLAT_PATCH_TARGET),
        ),
        "spiral_stairs_down": MeshSpiralPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.25),
            num_steps=7,
            center_tread_depth=0.33,
            usable_tread_width=2.0,
            platform_width=2.0,
            border_width=0.5,
            rotation_per_step_deg=4.5,
            ground_thickness=0.1,
            railing_tile_probability=0.5,
            railing_side_probability=0.5,
            ensure_one_railing=True,
            railing_thickness=0.1,
            railing_top_clearance=0.3,
            flat_patch_sampling=dict(_DEPTH6_FLAT_PATCH_TARGET),
        ),
        # "thin_obstacles": HfThinObstaclesTerrainCfg(
        #     proportion=0.1,
        #     obstacle_width_range=(0.1, 0.15),
        #     obstacle_length=1.5,
        #     obstacle_height_range=(0.08, 0.18),
        #     num_obstacles=30,
        #     platform_width=1.0,
        #     boundary_margin=0.5,
        #     flat_patch_sampling=dict(_DEPTH6_FLAT_PATCH_TARGET),
        # ),
        "square_ring_wall": HfHurdleTerrainCfg(
            proportion=0.2,
            platform_width=1.5,
            hurdle_height_range=(0.05, 0.20),
            hurdle_width_pixels=(2, 2, 3),
            flat_patch_sampling=dict(_DEPTH6_FLAT_PATCH_TARGET),
        ),
        "hurdle": HfHurdleTerrainCfg(
            proportion=0.4,
            platform_width=1.5,
            num_hurdles=1,
            first_hurdle_distance=2.0,
            hurdle_height_range=(0.25, 0.35),
            hurdle_width_pixels=2,
            flat_patch_sampling=dict(_DEPTH6_FLAT_PATCH_TARGET),
        ),
        # 保留地形名用于速度曲线对比；任务侧沿用 2026-07-24 坡前出生的稳定穿越设置。
        "interlaced_corridor_wedges_smooth_entry_exit": HfInterlacedCorridorWedgesTerrainCfg(
            proportion=0.3,
            corridor_half_width_m=1.5,
            segment_length_m=1.5,
            num_segments=4,
            slope_angle_range=(15.0, 15.0),
            edge_blend_m=0.02,
            corridor_edge_fade_m=0.06,
            pattern_origin_x_m=-3.0,
            flat_patch_sampling=dict(_DEPTH6_FLAT_PATCH_TARGET),
        ),
        # "continuous_wave_slopes": HfContinuousWaveSlopesTerrainCfg(
        #     proportion=0.1,
        #     slope_surface_length_m=1.5,
        #     slope_angle_deg=15.0,
        #     platform_length_m=0.8,
        #     half_width_m=1.2,
        #     edge_transition_m=0.05,
        #     flat_patch_sampling=dict(_DEPTH6_FLAT_PATCH_TARGET),
        # ),
        "symmetric_slope": MeshSymmetricSlopeTerrainCfg(
            proportion=0.1,
            obstacle_length_m=3.0,
            obstacle_width_m=3.5,
            slope_angle_deg=20.0,
            gap_width_by_level=(0.12, 0.115, 0.11, 0.105, 0.10, 0.10, 0.095, 0.09, 0.085, 0.08),
            finish_flat_length_m=0.2,
            ground_thickness_m=0.1,
            flat_patch_sampling=dict(_DEPTH6_FLAT_PATCH_TARGET),
        ),
        "braking_slope": MeshBrakingSlopeTerrainCfg(
            proportion=0.1,
            walkable_width_m=2.0,
            num_steps=4,
            slope_angle_deg=30.0,
            slope_foot_x_m=1.1,
            finish_flat_length_m=0.4,
            success_region_length_m=0.2,
            ground_thickness_m=0.1,
            step_height_by_level=(
                0.14, 0.1425, 0.145, 0.1475, 0.15,
                0.15, 0.1525, 0.155, 0.1575, 0.16,
            ),
            tread_depth_by_level=(
                0.32, 0.315, 0.31, 0.305, 0.30,
                0.30, 0.295, 0.29, 0.285, 0.28,
            ),
            flat_patch_sampling=dict(_DEPTH6_FLAT_PATCH_TARGET),
        ),
        "s_curve": MeshSCurveTerrainCfg(
            proportion=0.1,
            size=(8.0, 8.0),
            radius_m=1.6,
            height_range_m=(0.08, 0.12),
            path_width_by_level=(1.0, 0.9, 0.81, 0.73, 0.66, 0.6, 0.55, 0.5, 0.45, 0.4),
            center_offset_m=(0.0, 0.0),
            samples_per_semicircle=64,
        ),
        "l_curve": MeshLRailedPlatformTerrainCfg(
            proportion=0.1,
            size=(8.0, 8.0),
            first_leg_start_center_m=(1.5, 1.5),
            num_steps=2,
            step_height_m=0.15,
            tread_depth_m=0.30,
            leg_length_m=4.0,
            width_range_m=(1.5, 0.7),
            num_width_levels=10,
            guardrail_height_m=1.20,
            guardrail_depth_m=0.10,
            guardrail_rail_height_m=0.10,
            hollow_opening_width_m=0.20,
            hollow_opening_height_m=1.00,
            hollow_post_width_m=0.20,
            spawn_distance_before_stairs_m=0.60,
            ground_thickness_m=0.05,
        ),
        "thin_l_railed_platform": MeshLRailedPlatformTerrainCfg(
            proportion=0.1,
            size=(8.0, 8.0),
            first_leg_start_center_m=(1.5, 1.5),
            num_steps=2,
            step_height_m=0.15,
            tread_depth_m=0.30,
            leg_length_m=4.0,
            width_range_m=(1.5, 0.7),
            num_width_levels=10,
            guardrail_height_m=1.20,
            guardrail_depth_m=0.10,
            guardrail_rail_height_m=0.05,
            hollow_opening_width_m=0.35,
            hollow_opening_height_m=1.10,
            hollow_post_width_m=0.05,
            spawn_distance_before_stairs_m=0.60,
            ground_thickness_m=0.05,
        ),
    },
)


# 用于 mimic depth：多种子地形（楼梯、梅花桩、gap 等）按占比生成，并可选用 metadata 中的外部 mesh（terrain_id + 难度微调位置/角度）
MIMIC_DEPTH_WITH_EXTERNAL_CFG = FiledTerrainGeneratorCfg(
    curriculum=False,
    size=(10.0, 10.0),
    border_width=0.0,
    num_rows=10,
    num_cols=3,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.9,
    use_cache=False,
    sub_terrains={
        "flat_box": ExternalMeshFromMetadataTerrainCfg(
            proportion=0.1,
            path="",
            metadata_yaml="",
            terrain_id=0,
            base_terrain_cfg=HfNoisyPyramidSlopedTerrainCfg(
                proportion=0.1, slope_range=(0.0, 0.0), noise_range=(0.0, 0.0), noise_step=0.01,
                num_obstacles=0, obstacle_size=(0.2, 0.4), obstacle_height=(0.5, 1.5), platform_width=0.0
            ),
            position=(0.0, 0.0, 0.0),
            angle=(0.0, 0.0, 0.0),
            position_offset_range_x=(0.0, 0.0),
            position_offset_range_y=(0.0, 0.0),
            position_offset_range_z=(-0.1, 0.1),
            angle_offset_range_roll=(-0.05, 0.05),
            angle_offset_range_pitch=(-0.05, 0.05),
            angle_offset_range_yaw=(-0.1, 0.1),
        ),
        "rough_box": ExternalMeshFromMetadataTerrainCfg(
            proportion=0.1,
            path="",
            metadata_yaml="",
            terrain_id=0,
            base_terrain_cfg=HfNoisyPyramidSlopedTerrainCfg(
                proportion=0.1, slope_range=(0.0, 0.0), noise_range=(0.01, 0.05), noise_step=0.01,
                num_obstacles=0, obstacle_size=(0.2, 0.4), obstacle_height=(0.5, 1.5), platform_width=0.0
            ),
            position=(0.0, 0.0, 0.0),
            angle=(0.0, 0.0, 0.0),
            position_offset_range_x=(0.0, 0.0),
            position_offset_range_y=(0.0, 0.0),
            position_offset_range_z=(-0.1, 0.1),
            angle_offset_range_roll=(-0.05, 0.05),
            angle_offset_range_pitch=(-0.05, 0.05),
            angle_offset_range_yaw=(-0.1, 0.1),
        ),
        "wave_box": ExternalMeshFromMetadataTerrainCfg(
            proportion=0.1,
            path="",
            metadata_yaml="",
            terrain_id=0,
            base_terrain_cfg=HfWaveTerrainCfg(
                proportion=0.1, num_waves=10.0, amplitude_range=(0.02, 0.05)
            ),
            position=(0.0, 0.0, 0.0),
            angle=(0.0, 0.0, 0.0),
            position_offset_range_x=(0.0, 0.0),
            position_offset_range_y=(0.0, 0.0),
            position_offset_range_z=(-0.1, 0.1),
            angle_offset_range_roll=(-0.05, 0.05),
            angle_offset_range_pitch=(-0.05, 0.05),
            angle_offset_range_yaw=(-0.1, 0.1),
        ),
    },
)

MIMIC_DEPTH_TERRAIN_REFINE_CFG = FiledTerrainGeneratorCfg(
    curriculum=False,
    size=(10.0, 10.0),
    border_width=0.0,
    num_rows=1,
    num_cols=1,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.9,
    use_cache=False,
    sub_terrains={
        "flat_box": ExternalMeshFromMetadataTerrainCfg(
            proportion=0.1,
            path="",
            metadata_yaml="",
            terrain_id=0,
            base_terrain_cfg=HfNoisyPyramidSlopedTerrainCfg(
                proportion=0.1, slope_range=(0.0, 0.0), noise_range=(0.0, 0.0), noise_step=0.01,
                num_obstacles=0, obstacle_size=(0.2, 0.4), obstacle_height=(0.5, 1.5), platform_width=0.0
            ),
            position=(0.0, 0.0, 0.0),
            angle=(0.0, 0.0, 0.0),
            position_offset_range_x=(0.0, 0.0),
            position_offset_range_y=(0.0, 0.0),
            position_offset_range_z=(0.0, 0.0),
            angle_offset_range_roll=(0.0, 0.0),
            angle_offset_range_pitch=(0.0, 0.0),
            angle_offset_range_yaw=(0.0, 0.),
        ),
    },
)
