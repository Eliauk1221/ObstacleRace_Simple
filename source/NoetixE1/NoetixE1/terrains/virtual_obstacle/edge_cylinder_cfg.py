from __future__ import annotations

import math
from dataclasses import MISSING
from typing import TYPE_CHECKING, Literal

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.sensors import patterns
from isaaclab.utils import configclass

from .edge_cylinder import GreedyconcatEdgeCylinder, PluckerEdgeCylinder, RansacEdgeCylinder, RayEdgeCylinder
from .virtual_obstacle_base import VirtualObstacleCfg


@configclass
class EdgeCylinderCfg(VirtualObstacleCfg):
    """The class to use for the edge cylinder detector."""

    class_type: type = MISSING
    """The class to use for the edge detector."""
    angle_threshold: float = 70.0
    """The angle threshold to consider an edge as sharp."""

    cylinder_radius: float = 0.2
    """Default radius of edge cylinders (non-classified edges)."""

    inner_edge_cylinder_radius: float | None = None
    """If set with ``outer_edge_cylinder_radius``, figure8 inner annulus edges use this (smaller) radius."""

    outer_edge_cylinder_radius: float | None = None
    """If set with ``inner_edge_cylinder_radius``, figure8 outer annulus edges use this (larger) radius."""

    figure8_annulus_inner_radius_m: float = 2.5
    """figure8 bridge deck inner radius (m), used to classify inner/outer edges."""

    figure8_bridge_widths: tuple[float, ...] = (0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4)
    """Candidate bridge widths for classifying figure8 annulus edges."""

    figure8_cell_size_m: float = 12.0
    """figure8 sub-terrain cell size (m), used to fold world XY into a local tile frame."""

    figure8_edge_radial_tol_m: float = 0.10
    """Radial match tolerance (m) when classifying an edge as inner/outer annulus."""

    figure8_edge_z_range_m: tuple[float, float] = (0.02, 0.20)
    """Only edges whose midpoint z is in this band are classified as figure8 deck edges."""

    num_grid_cells: int = 64**3
    """The number of grid cells to use for spatial partitioning of the edge cylinders.
    Usually the power of 2, e.g., 64^3 = 262144.
    """

    visualizer: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/edgeMarkers",
        markers={
            "cylinder": sim_utils.CylinderCfg(
                radius=1,
                height=1,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.9), opacity=0.2),
            )
        },
    )


@configclass
class PluckerEdgeCylinderCfg(EdgeCylinderCfg):
    """Configuration for the plucker edge cylinder generator."""

    class_type: type = PluckerEdgeCylinder
    """The class to use for the sharp edge detector."""


@configclass
class RansacEdgeCylinderCfg(EdgeCylinderCfg):
    """The class to use for the ransac edge cylinder generator."""

    class_type: type = RansacEdgeCylinder

    max_iter: int = 500
    """The maximum number of iterations."""

    point_distance_threshold: float = 0.04
    """The distance threshold to consider a point as an inlier."""

    min_points: int = 5
    """The minimum number of points required to fit."""

    cluster_eps: float = 0.08
    """The maximum distance between points in a cluster."""


@configclass
class GreedyconcatEdgeCylinderCfg(EdgeCylinderCfg):
    """The class to use for the greedy-concat edge cylinder generator."""

    class_type: type = GreedyconcatEdgeCylinder

    adjacent_angle_threshold: float = 30.0
    """The angle threshold to consider two edges as adjacent."""

    point_distance_threshold: float = 0.06
    """The distance threshold to consider a point as an inlier."""

    min_points: int = 5
    """The minimum number of points in one line."""


@configclass
class RayEdgeCylinderCfg(VirtualObstacleCfg):
    """The class to use for the ray-based edge cylinder generator."""

    class_type: type = RayEdgeCylinder

    cylinder_radius: float = 0.2
    """The radius of the edge cylinder, which is used to treat the edge cylinders as a virtual obstacle."""
    num_grid_cells: int = 64**3
    """The number of grid cells to use for spatial partitioning of the edge cylinders.
    Usually the power of 2, e.g., 64^3 = 262144.
    """
    max_iter: int = 500
    """The maximum number of iterations."""

    point_distance_threshold: float = 0.005
    """The distance threshold to consider a point as an inlier."""

    min_points: int = 15
    """The minimum number of points required to fit."""

    cluster_eps: float = 0.08
    """The maximum distance between points in a cluster."""

    ray_pattern: patterns.GridPatternCfg = patterns.GridPatternCfg(
        resolution=0.01,
        size=[6, 6],
        direction=(0.0, 0.0, -1.0),
    )
    """The pattern to use for ray sampling."""

    ray_offset_pos: list[float] = [0.0, 0.0, 1.0]
    """The offset position of the rays."""

    ray_rotate_axes: list[list[float]] = [
        [1.0, 1.0, 0.0],
        [-1.0, 1.0, 0.0],
        [1.0, -1.0, 0.0],
        [-1.0, -1.0, 0.0],
    ]

    ray_rotate_angle: list[float] = [math.pi * 0.25, math.pi * 0.25, math.pi * 0.25, math.pi * 0.25]
    """The axes and angles to rotate the rays."""

    max_ray_depth: float = 8.0
    """The maximum depth of the rays to sample."""

    depth_canny_thresholds: list[float] = [250, 300]
    """The thresholds for the Canny edge detector to detect edges in the depth image."""

    normal_canny_thresholds: list[float] = [80, 250]
    """The thresholds for the Canny edge detector to detect edges in the normal image."""

    cutoff_z_height: float = 0.1
    """The height threshold to filter out rays that are too close to the ground."""

    visualizer: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/edgeMarkers",
        markers={
            "cylinder": sim_utils.CylinderCfg(
                radius=1,
                height=1,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.9), opacity=0.2),
            )
        },
    )

    points_visualizer: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/edgePoints",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=0.01,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.5, 0.5)),
            ),
        },
    )
