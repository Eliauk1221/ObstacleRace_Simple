# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

from __future__ import annotations

from typing import Literal

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import configclass

from .square_ring_wall import SquareRingWallOuterBandVirtualObstacle
from .virtual_obstacle_base import VirtualObstacleCfg


@configclass
class SquareRingWallOuterBandVirtualObstacleCfg(VirtualObstacleCfg):
    """Inner or outer virtual collision band for each ring in ``square_ring_wall`` terrain."""

    class_type: type = SquareRingWallOuterBandVirtualObstacle
    visualizer: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/squareRingWallOuterBands",
        markers={
            "band": sim_utils.CuboidCfg(
                size=(1.0, 1.0, 1.0),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.9, 0.1, 0.1), opacity=0.2
                ),
            ),
        },
    )
    subterrain_key: str = "square_ring_wall"
    band_side: Literal["inner", "outer"] = "outer"
    band_width: float = 0.02
    z_bottom: float = 0.0
    z_top_margin: float = 0.0
