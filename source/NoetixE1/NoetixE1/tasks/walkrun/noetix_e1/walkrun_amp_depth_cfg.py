# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from NoetixE1.sensors import MultiMeshRayCasterCameraCfg
from isaaclab.sensors.ray_caster.patterns import PinholeCameraPatternCfg
from NoetixE1.sensors.volume_points import Grid3dPointsGeneratorCfg, VolumePointsCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from NoetixE1.assets import ASSET_DIR
from NoetixE1.terrains import (
    DEPTH_TERRAINS_CFG2,
    DEPTH_TERRAINS_CFG3,
    DEPTH_TERRAINS_CFG4,
    DEPTH_TERRAINS_CFG5,
    DEPTH_TERRAINS_CFG6,
    GreedyconcatEdgeCylinderCfg,
    SquareRingWallOuterBandVirtualObstacleCfg,
    TerrainImporterCfg,
)

from . import mdp

##
# Pre-defined configs
##

from NoetixE1.assets.robots.e1.e1 import E1_23DOF_CFG, E1_23DOF_ACTION_SCALE  # isort:skip

E1_RUNNING_MOTION_FILE = f"{ASSET_DIR}/datasets/walkrun/e1/e1_running.txt"
HURDLE_MOTION_FILE = (
    f"{ASSET_DIR}/datasets/parkour/e1/sport_meet_23_old_urdf/"
    "sport_meet_hurdle_amp_2m_continuous_110_175_old_urdf.json"
)
DEFAULT_RSI_MOTION_FILES = (
    f"{ASSET_DIR}/datasets/walkrun/e1/e1_standing.txt",
    f"{ASSET_DIR}/datasets/walkrun/e1/e1_sidewalk.txt",
    f"{ASSET_DIR}/datasets/walkrun/e1/e1_walk_backward.txt",
    f"{ASSET_DIR}/datasets/walkrun/e1/e1_turning.txt",
    f"{ASSET_DIR}/datasets/walkrun/e1/e1_walk_forward.txt",
    E1_RUNNING_MOTION_FILE,
)
L_CURVE_TERRAIN_KEYS = ("l_curve", "thin_l_railed_platform")
_S_CURVE_ENTRY_CENTER_XY = (0.0, -3.2)
_S_CURVE_ENTRY_X_RANGE = (0.25, 1.25)
_S_CURVE_ENTRY_Y_RANGE = (-3.60, -2.80)
_S_CURVE_ENTRY_CENTER_Y_RANGE = (-3.45, -2.95)
L_CURVE_RSI_EXCLUDED_MOTION_FILES = (
    f"{ASSET_DIR}/datasets/walkrun/e1/e1_sidewalk.txt",
    f"{ASSET_DIR}/datasets/walkrun/e1/e1_walk_backward.txt",
    f"{ASSET_DIR}/datasets/walkrun/e1/e1_turning.txt",
    E1_RUNNING_MOTION_FILE,
)


##
# Scene definition
##


@configclass
class E1SceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",  # plane generator
        terrain_generator=DEPTH_TERRAINS_CFG6,
        max_init_terrain_level=1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
        virtual_obstacles={
            "edges": GreedyconcatEdgeCylinderCfg(
                cylinder_radius=0.1,
                terrain_specific_cylinder_radius={"hurdle": 0.05},
                # Closed figure8 uses asymmetric virtual edge radii.
                inner_edge_cylinder_radius=0.07,
                outer_edge_cylinder_radius=0.12,
                figure8_annulus_inner_radius_m=2.5,
                figure8_bridge_widths=(0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4),
                figure8_cell_size_m=12.0,
                min_points=2,
            ),
            "square_ring_wall_outer_bands": SquareRingWallOuterBandVirtualObstacleCfg(
                subterrain_key="square_ring_wall",
                band_side="outer",
                band_width=0.02,
                visualizer=VisualizationMarkersCfg(
                    prim_path="/Visuals/squareRingWallOuterBands",
                    markers={
                        "band": sim_utils.CuboidCfg(
                            size=(1.0, 1.0, 1.0),
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(0.0, 0.85, 0.95), opacity=0.25
                            ),
                        ),
                    },
                ),
            ),
            "square_ring_wall_inner_bands": SquareRingWallOuterBandVirtualObstacleCfg(
                subterrain_key="square_ring_wall",
                band_side="inner",
                band_width=0.02,
                visualizer=VisualizationMarkersCfg(
                    prim_path="/Visuals/squareRingWallInnerBands",
                    markers={
                        "band": sim_utils.CuboidCfg(
                            size=(1.0, 1.0, 1.0),
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(0.95, 0.1, 0.1), opacity=0.25
                            ),
                        ),
                    },
                ),
            ),
        },
    )

    # robot
    robot: ArticulationCfg = E1_23DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True, force_threshold=10.0, debug_vis=True
    )
    # height_scanner = None
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        update_period=1 / 50,
        offset=RayCasterCfg.OffsetCfg(pos=(0, 0, 0.5)),
        mesh_prim_paths=["/World/ground"],
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(
            resolution=0.05, size=[1.1, 0.7]
        ),  # data = ((1.1/0.05)+1)*((0.7/0.05)+1) = 345
        debug_vis=False,
    )
    left_foot_height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/l_leg_ankle_roll_link",
        update_period=1 / 50,
        offset=RayCasterCfg.OffsetCfg(pos=(0.025, 0, -0.05)),
        mesh_prim_paths=["/World/ground"],
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(
            resolution=0.05, size=[0.20, 0.05]
        ),  # 2*5=10
        debug_vis=True,
    )
    right_foot_height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/r_leg_ankle_roll_link",
        update_period=1 / 50,
        offset=RayCasterCfg.OffsetCfg(pos=(0.025, 0, -0.05)),
        mesh_prim_paths=["/World/ground"],
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(
            resolution=0.05, size=[0.20, 0.05]
        ),  # 2*5=10
        debug_vis=True,
    )
    leg_volume_points = VolumePointsCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_ankle_roll_link",
        points_generator=Grid3dPointsGeneratorCfg(
            x_min=-0.06,
            x_max=0.14,
            x_num=10,
            y_min=-0.05,
            y_max=0.05,
            y_num=5,
            z_min=-0.05,
            z_max=0.0,
            z_num=2,
        ),
        debug_vis=False,
    )
    hurdle_ankle_volume_points = VolumePointsCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_ankle_roll_link",
        points_generator=Grid3dPointsGeneratorCfg(
            x_min=-0.08,
            x_max=0.16,
            x_num=10,
            y_min=-0.05,
            y_max=0.05,
            y_num=5,
            z_min=-0.08,
            z_max=0.05,
            z_num=3,
        ),
        debug_vis=False,
    )
    raycaster_camera = MultiMeshRayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/waist_yaw_link",
        mesh_prim_paths=[
            "/World/ground",
            MultiMeshRayCasterCameraCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/Robot/l_leg.*/visuals", is_shared=True),
            MultiMeshRayCasterCameraCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/Robot/r_leg.*/visuals", is_shared=True),
            MultiMeshRayCasterCameraCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/Robot/l_arm_elbow.*/visuals", is_shared=True),
            MultiMeshRayCasterCameraCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/Robot/r_arm_elbow.*/visuals", is_shared=True),
        ],
        # pitch 30deg
        # offset=RayCasterCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.5), rot=(0.3536, -0.6124, 0.6124, -0.3536), convention="ros"),
        # pitch 45deg
        # offset=MultiMeshRayCasterCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.5), rot=(0.2705981, -0.6532815, 0.6532815, -0.2705981), convention="ros"),
        # # pitch 55deg
        # offset=MultiMeshRayCasterCameraCfg.OffsetCfg(pos=(0.15, 0.0, 0.4), rot=(0.2126311, -0.6743797, 0.6743797, -0.2126311), convention="ros"),
        # real 55deg
        # offset=MultiMeshRayCasterCameraCfg.OffsetCfg(pos=(0.13499, -0.01796, 0.37879), rot=(0.2126311, -0.6743797, 0.6743797, -0.2126311), convention="ros"),
        # 60 deg
        offset=MultiMeshRayCasterCameraCfg.OffsetCfg(pos=(0.099726, 0.0175, 0.50523), rot=(0.1830127, -0.6830127, 0.6830127, -0.1830127), convention="ros"),

        data_types=["distance_to_image_plane"],
        pattern_cfg=PinholeCameraPatternCfg.from_intrinsic_matrix(
            # 内参矩阵参数: [f_x, 0, c_x, 0, f_y, c_y, 0, 0, 1]
            intrinsic_matrix=[
                241.418 * 0.1, 0.0, (242.5 - 120) * 0.1,
                0.0, 241.418 * 0.1, (130.0 - 110) * 0.1,
                0.0, 0.0, 1.0     # 齐次坐标
            ],
            width=24,
            height=16,
            focal_length=24.0  # 可选，默认24.0cm
        ),
        # drift_range=(-0.05, 0.05),  # 机器人有中等定位误差
        # ray_cast_drift_range={
        #     "x": (-0.01, 0.01),  # 传感器有轻微测量噪声
        #     "y": (-0.01, 0.01),
        #     "z": (-0.005, 0.005)
        # },
        update_period=1 / 60,
        depth_clipping_behavior="zero",
        max_distance=4.0,
        debug_vis=True,
        # 配置多个颜色的marker用于注意力权重可视化
        visualizer_cfg=VisualizationMarkersCfg(
            prim_path="/Visuals/RayCasterCamera",
            markers={
                **{f"attn_{i}": sim_utils.SphereCfg(
                    radius=0.02,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(i / 9.0, 0.0, 1.0 - i / 9.0)  # RGB: 从蓝色(低注意力)到红色(高注意力)
                    ),
                ) for i in range(10)}
            },
        ),
    )

##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.TurnSequenceVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 10.0),
        rel_standing_envs=0.1,
        high_speed_reward_gate_floor=0.3,
        debug_vis=True,
        ranges=mdp.UniformVelocityWithZeroCommandCfg.Ranges(
            lin_vel_x=(-0.6, 1.5), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.0, 1.0), zero_prob=[0.4, 0.8, 0.4]
        ),
        terrain_specific_ranges={
            "interlaced_corridor_wedges_smooth_entry_exit": mdp.UniformVelocityWithZeroCommandCfg.Ranges(
                lin_vel_x=(0.3, 1.5),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(0.0, 0.0),
                zero_prob=[0.5, 0.9, 0.5],
            ),
            "continuous_wave_slopes": mdp.UniformVelocityWithZeroCommandCfg.Ranges(
                lin_vel_x=(-0.6, 1.5),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(0.0, 0.0),
                zero_prob=[0.1, 0.9, 0.5],
            ),
            "symmetric_slope": mdp.UniformVelocityWithZeroCommandCfg.Ranges(
                lin_vel_x=(-0.6, 1.5),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(0.0, 0.0),
                zero_prob=[0.1, 0.9, 0.5],
            ),
            "braking_slope": mdp.UniformVelocityWithZeroCommandCfg.Ranges(
                lin_vel_x=(-0.6, 1.5),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(0.0, 0.0),
                zero_prob=[0.1, 0.9, 0.5],
            ),
            "stairs_up": mdp.UniformVelocityWithZeroCommandCfg.Ranges(
                lin_vel_x=(-0.6, 1.5),
                lin_vel_y=(-0.5, 0.5),
                ang_vel_z=(-1.0, 1.0),
                zero_prob=[0.0, 0.9, 0.9],
            ),
            "spiral_stairs_up": mdp.UniformVelocityWithZeroCommandCfg.Ranges(
                lin_vel_x=(-0.6, 1.5),
                lin_vel_y=(-0.5, 0.5),
                ang_vel_z=(-1.0, 1.0),
                zero_prob=[0.0, 0.9, 0.9],
            ),
            "stairs_down": mdp.UniformVelocityWithZeroCommandCfg.Ranges(
                lin_vel_x=(-0.6, 1.5),
                lin_vel_y=(-0.5, 0.5),
                ang_vel_z=(-1.0, 1.0),
                zero_prob=[0.0, 0.9, 0.9],
            ),
            "spiral_stairs_down": mdp.UniformVelocityWithZeroCommandCfg.Ranges(
                lin_vel_x=(-0.6, 1.5),
                lin_vel_y=(-0.5, 0.5),
                ang_vel_z=(-1.0, 1.0),
                zero_prob=[0.0, 0.9, 0.9],
            ),
            "square_ring_wall": mdp.UniformVelocityWithZeroCommandCfg.Ranges(
                lin_vel_x=(-0.6, 1.5),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(-0.5, 0.5),
                zero_prob=[0.5, 0.9, 0.5],
            ),
            "hurdle": mdp.UniformVelocityWithZeroCommandCfg.Ranges(
                lin_vel_x=(1.0, 2.0),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(0.0, 0.0),
                zero_prob=[0.5, 0.9, 0.5],
            ),
            "s_curve": mdp.UniformVelocityWithZeroCommandCfg.Ranges(
                lin_vel_x=(0.8, 1.5),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(0, 0),
                zero_prob=[0.1, 0.9, 0.5],
            ),
            "l_curve": mdp.UniformVelocityWithZeroCommandCfg.Ranges(
                lin_vel_x=(0.6, 2.0),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(0.0, 0.0),
                zero_prob=[0.0, 1.0, 1.0],
            ),
            "thin_l_railed_platform": mdp.UniformVelocityWithZeroCommandCfg.Ranges(
                lin_vel_x=(0.6, 2.0),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(0.0, 0.0),
                zero_prob=[0.0, 1.0, 1.0],
            ),
        },
        terrain_specific_high_speed_ranges={
            "plane": mdp.UniformVelocityWithZeroCommandCfg.HighSpeedRanges(
                lin_vel_x=(1.5, 3.0),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(-1.0, 1.0),
                zero_prob=[0.0, 1.0, 0.5],
                probability=0.3,
            ),
            "rough_flat": mdp.UniformVelocityWithZeroCommandCfg.HighSpeedRanges(
                lin_vel_x=(1.5, 3.0),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(-1.0, 1.0),
                zero_prob=[0.0, 1.0, 0.5],
                probability=0.3,
            ),
            "noisy_slope_up": mdp.UniformVelocityWithZeroCommandCfg.HighSpeedRanges(
                lin_vel_x=(1.5, 3.0),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(-1.0, 1.0),
                zero_prob=[0.0, 1.0, 0.5],
                probability=0.3,
            ),
            "noisy_slope_down": mdp.UniformVelocityWithZeroCommandCfg.HighSpeedRanges(
                lin_vel_x=(1.5, 3.0),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(-1.0, 1.0),
                zero_prob=[0.0, 1.0, 0.5],
                probability=0.3,
            ),
            "interlaced_corridor_wedges_smooth_entry_exit": mdp.UniformVelocityWithZeroCommandCfg.HighSpeedRanges(
                lin_vel_x=(1.5, 2.0),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(0.0, 0.0),
                zero_prob=[0.0, 1.0, 1.0],
                probability=0.3,
            ),
            "continuous_wave_slopes": mdp.UniformVelocityWithZeroCommandCfg.HighSpeedRanges(
                lin_vel_x=(1.5, 3.0),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(0.0, 0.0),
                zero_prob=[0.0, 1.0, 1.0],
                probability=0.3,
            ),
            "symmetric_slope": mdp.UniformVelocityWithZeroCommandCfg.HighSpeedRanges(
                lin_vel_x=(1.5, 3.0),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(0.0, 0.0),
                zero_prob=[0.0, 1.0, 1.0],
                probability=0.3,
            ),
        },
        # TURN ABLATION: set probability to 0.0 to restore the original command distribution.
        turn_terrain_types=("interlaced_corridor_wedges",),
        turn_sequence_probability=0.1,
        turn_duration_range=(1.0, 5.0),
        yaw_speed_range=(0.2, 1.0),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], use_default_offset=True)
    joint_pos.scale = E1_23DOF_ACTION_SCALE


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        """如果要用HIM估计状态，需要保持policy和privileged观测的一致性，privileged多出来的观测项可以被HIM选择索引估计"""
        # observation terms (order preserved)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2), clip=(-18, 18))
        euler_angles = ObsTerm(func=mdp.euler_angles, noise=Unoise(n_min=-0.025, n_max=0.025), clip=(-18, 18))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.05, n_max=0.05), clip=(-18, 18))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5), clip=(-360, 360), scale=0.05)
        actions = ObsTerm(func=mdp.last_action, clip=(-18.0, 18.0))

        def __post_init__(self) -> None:
            self.concatenate_terms = True
            self.enable_corruption = True
            self.history_length = 5
            self.flatten_history_dim = False

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, clip=(-18, 18))
        euler_angles = ObsTerm(func=mdp.euler_angles, clip=(-18, 18))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, clip=(-18, 18))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, clip=(-360, 360), scale=0.05)
        actions = ObsTerm(func=mdp.last_action, clip=(-18.0, 18.0))
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-18, 18))
        payload = ObsTerm(func=mdp.payload, params={"asset_cfg": SceneEntityCfg("robot", body_names=["waist_yaw_link"])}, scale=0.2)
        # static_friction, dynamic_friction, restitution, N*3
        left_friction = ObsTerm(func=mdp.friction, params={"asset_cfg": SceneEntityCfg("robot", joint_names="l_leg_ankle_roll_joint")})
        # static_friction, dynamic_friction, restitution, N*3
        right_friction = ObsTerm(func=mdp.friction, params={"asset_cfg": SceneEntityCfg("robot", joint_names="r_leg_ankle_roll_joint")})
        # stiffness, N*23
        kp_params = ObsTerm(func=mdp.kp_params, scale=0.005)
        # damping, N*23
        kd_params = ObsTerm(func=mdp.kd_params, scale=0.05)
        # effort_limit, N*23
        effort_limit = ObsTerm(func=mdp.effort_limit, scale=0.01)
        # depth intrinsic param delta, N*4 (f_x_delta, f_y_delta, c_x_delta, c_y_delta)
        camera_intrinsic_delta = ObsTerm(
            func=mdp.camera_intrinsic_delta, params={"sensor_cfg": SceneEntityCfg("raycaster_camera")}, scale=2
        )
        # depth extrinsic param delta, N*6 (x_delta, y_delta, z_delta, roll_delta, pitch_delta, yaw_delta)
        camera_extrinsic_delta = ObsTerm(
            func=mdp.camera_extrinsic_delta, params={"sensor_cfg": SceneEntityCfg("raycaster_camera")}, scale=40
        )
        feet_contact = ObsTerm(
            func=mdp.current_feet_contact,
            params={
                "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
            },
        )
        feet_stumble = ObsTerm(
            func=mdp.current_feet_stumble,
            params={
                "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link")
            },
        )
        left_foot_height_scan = ObsTerm(
            func=mdp.height_scan,
            params={
                "sensor_cfg": SceneEntityCfg("left_foot_height_scanner"),
                "offset": 0.06,
            },
            clip=(-1.0, 1.0),
        )
        right_foot_height_scan = ObsTerm(
            func=mdp.height_scan,
            params={
                "sensor_cfg": SceneEntityCfg("right_foot_height_scanner"),
                "offset": 0.06,
            },
            clip=(-1.0, 1.0),
        )
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={
                "sensor_cfg": SceneEntityCfg("height_scanner"),
            },
            clip=(-1.0, 1.0),
        )

        def __post_init__(self) -> None:
            self.concatenate_terms = True
            self.enable_corruption = False

    @configclass
    class AmpObsCfg(ObsGroup):
        """Observations for discriminator group."""

        # observation terms (order preserved)
        joint_pos = ObsTerm(func=mdp.amp_joint_pos)
        key_pos = ObsTerm(func=mdp.amp_key_pos)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_vel = ObsTerm(func=mdp.amp_joint_vel)
        feet_contact = ObsTerm(
            func=mdp.current_feet_contact,
            params={
                "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
            },
        )

        def __post_init__(self) -> None:
            self.concatenate_terms = True
            self.enable_corruption = False
            self.flatten_history_dim = False

    @configclass
    class DepthCfg(ObsGroup):
        """Observations for depth group."""

        # observation terms (order preserved)
        # depths = ObsTerm(
        #     func=mdp.raycaster_depth,
        #     params={
        #         "distance_noise": 0.0,  # 随距离增加的噪声系数
        #         "gaussian_std": 0.00,  # 高斯噪声的标准差系数（相对于深度值）
        #         "salt_pepper_prob": 0.0,  # 椒盐噪声的概率
        #         "stripe_prob": 0.00,  # 条纹失效的概率（随机整行或整列为0）
        #         "edge_drag_prob": 0.0,  # 边缘拖影的概率
        #         "min_range": 0.3,  # 深度最小值
        #         "max_range": 2.0,  # 将从相机配置的 max_distance 中自动读取，无需手动指定
        #         "high_reflection_prob": 0.0,  # 高反噪声的概率
        #         "high_reflection_num_patches": (1, 2),  # 每个样本高反块数量范围
        #         "high_reflection_rect_ratio": (0.05, 0.1),  # 高反块占比范围（相对于图像尺寸的比例）
        #         "high_reflection_coefficient_range": (1.1, 1.3),  # 高反系数范围，每个高反块内的系数相同
        #         "high_reflection_persist_steps": 10,  # 高反块持续步数
        #         "mask_prob": 0.0,  # 随机矩形mask的概率
        #         "mask_num_masks": (1, 3),  # 每个样本mask数量范围
        #         "mask_rect_ratio": (0.1, 0.2),  # mask占比范围（相对于图像尺寸的比例）
        #         "mask_persist_steps": 20,  # mask持续步数
        #         "full_image_mask_prob": 0.00,  # 整张图片mask的概率
        #         "full_image_mask_persist_steps": 10,  # 整张图片mask持续步数
        #         "full_image_random_prob": 0.0,  # 整张图片随机噪声的概率
        #     },
        # )

        depths = ObsTerm(
            func=mdp.raycaster_depth,
            params={
                "distance_noise": 0.01,  # 随距离增加的噪声系数
                "gaussian_std": 0.005,  # 高斯噪声的标准差系数（相对于深度值）
                "salt_pepper_prob": 0.01,  # 椒盐噪声的概率
                "stripe_prob": 0.005,  # 条纹失效的概率（随机整行或整列为0）
                "edge_drag_prob": 0.01,  # 边缘拖影的概率
                "min_range": 0.3,  # 深度最小值
                "max_range": 2.0,  # 将从相机配置的 max_distance 中自动读取，无需手动指定
                "high_reflection_prob": 0.01,  # 高反噪声的概率
                "high_reflection_num_patches": (1, 2),  # 每个样本高反块数量范围
                "high_reflection_rect_ratio": (0.05, 0.1),  # 高反块占比范围（相对于图像尺寸的比例）
                "high_reflection_coefficient_range": (1.1, 1.3),  # 高反系数范围，每个高反块内的系数相同
                "high_reflection_persist_steps": 10,  # 高反块持续步数
                "mask_prob": 0.03,  # 随机矩形mask的概率
                "mask_num_masks": (1, 3),  # 每个样本mask数量范围
                "mask_rect_ratio": (0.1, 0.2),  # mask占比范围（相对于图像尺寸的比例）
                "mask_persist_steps": 20,  # mask持续步数
                "full_image_mask_prob": 0.005,  # 整张图片mask的概率
                "full_image_mask_persist_steps": 10,  # 整张图片mask持续步数
                "full_image_random_prob": 0.001,  # 整张图片随机噪声的概率
            },
        )

        def __post_init__(self) -> None:
            self.concatenate_terms = True
            self.enable_corruption = False
            self.history_length = 1
            self.flatten_history_dim = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    discriminator: AmpObsCfg = AmpObsCfg()
    depth: DepthCfg = DepthCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )

    add_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.02, 0.02),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="waist_yaw_link"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["waist_yaw_link"]),
            "mass_distribution_params": (-5, 5),
            "operation": "add",
        },
    )

    randomize_actuator_effort_limit = EventTerm(
        func=mdp.randomize_actuator_effort_limit,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "effort_limit_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    randomize_camera_intrinsics = EventTerm(
        func=mdp.randomize_camera_intrinsics,
        mode="startup",
        params={
            "sensor_cfg": SceneEntityCfg("raycaster_camera"),
            "intrinsic_range": {
                "f_x": (-0.3, 0.3),  # 焦距x的随机化范围（像素）
                "f_y": (-0.3, 0.3),  # 焦距y的随机化范围（像素）
                "c_x": (-0.5, 0.5),  # 主点x的随机化范围（像素）
                "c_y": (-0.5, 0.5),  # 主点y的随机化范围（像素）
            },
        },
    )

    randomize_camera_extrinsics = EventTerm(
        func=mdp.randomize_camera_extrinsics,
        mode="startup",
        params={
            "sensor_cfg": SceneEntityCfg("raycaster_camera"),
            "position_range": {
                "x": (-0.02, 0.02),  # x方向的随机化范围（米）
                "y": (-0.02, 0.02),  # y方向的随机化范围（米）
                "z": (-0.02, 0.02),  # z方向的随机化范围（米）
            },
            "orientation_range": {
                "roll": (-0.05, 0.05),  # 绕x轴旋转的随机化范围（弧度，约±2.9度）
                "pitch": (-0.05, 0.05),  # 绕y轴旋转的随机化范围（弧度，约±2.9度）
                "yaw": (-0.05, 0.05),  # 绕z轴旋转的随机化范围（弧度，约±2.9度）
            },
        },
    )

    reset_robot_states = EventTerm(
        func=mdp.reset_root_state_amp,
        mode="reset",
        params={
            "reference_state_initialization": True,
            "prob_rsi": 0.8,
            "root_pos_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "root_vel_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (-0.05, 0.05),
                "roll": (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
                "yaw": (-0.05, 0.05),
            },
            "joint_pos_range": (-0.2, 0.2),
            "joint_vel_range": (-1.0, 1.0),
            "default_motion_files": DEFAULT_RSI_MOTION_FILES,
            "terrain_specific_motion_files": {
                "hurdle": [HURDLE_MOTION_FILE],
            },
            "terrain_specific_prob_rsi": {
                "hurdle": 0.8,
            },
            "terrain_specific_joint_vel_range": {
                "hurdle": (-0.5, 0.5),
            },
            "terrain_specific_motion_level_range": (0, 1),
            "terrain_specific_reset_height_range": {
                "hurdle": (0.0, 0.0),
            },
            "terrain_reset_frames": {
                "hurdle": (0,),
            },
            "terrain_specific_rsi_excluded_motion_files": {
                "s_curve": [E1_RUNNING_MOTION_FILE],
                "stairs_up": [E1_RUNNING_MOTION_FILE],
                "spiral_stairs_up": [E1_RUNNING_MOTION_FILE],
                "stairs_down": [E1_RUNNING_MOTION_FILE],
                "spiral_stairs_down": [E1_RUNNING_MOTION_FILE],
                "thin_obstacles": [E1_RUNNING_MOTION_FILE],
                "square_ring_wall": [E1_RUNNING_MOTION_FILE],
                "braking_slope": [E1_RUNNING_MOTION_FILE],
                "l_curve": L_CURVE_RSI_EXCLUDED_MOTION_FILES,
                "thin_l_railed_platform": L_CURVE_RSI_EXCLUDED_MOTION_FILES,
            },
            "terrain_specific_yaw": {
                "hurdle": [0.0],
                "cross_stepping_stones3": [0.0, 1.57, 3.14, -1.57],  # 0°, 90°, 180°, -90°
                "cross_stepping_stones2": [0.0, 1.57, 3.14, -1.57],  # 0°, 90°, 180°, -90°
                "cross_stepping_stones1": [0.0, 1.57, 3.14, -1.57],  # 0°, 90°, 180°, -90°
                "beam": [0.0, 1.57, 3.14, -1.57],  # 0°, 90°, 180°, -90°
                "interlaced_corridor_wedges": (-0.5235987755982988, 0.5235987755982988),  # +/- 30 deg
                "interlaced_corridor_wedges_approach": (-0.2617993877991494, 0.2617993877991494),  # +/- 15 deg
                "interlaced_corridor_wedges_smooth_exit": (-0.2617993877991494, 0.2617993877991494),
                "interlaced_corridor_wedges_smooth_entry_exit": (-0.2617993877991494, 0.2617993877991494),
                "continuous_wave_slopes": (-0.2617993877991494, 0.2617993877991494),
                "symmetric_slope": (-0.17453292519943295, 0.17453292519943295),  # +/- 10 deg
                "braking_slope": (-0.17453292519943295, 0.17453292519943295),  # +/- 10 deg
                "s_curve": (2.9, 3.4),
                "l_curve": (-0.2617993877991494, 0.2617993877991494),
                "thin_l_railed_platform": (-0.2617993877991494, 0.2617993877991494),
            },
            "terrain_specific_pos_range": {
                "interlaced_corridor_wedges_smooth_entry_exit": {
                    "x": (-4.0, -3.2),
                    "y": (-0.75, -0.25),
                },
                "continuous_wave_slopes": {
                    "x": (-3.95, -3.85),
                    "y": (-0.60, 0.60),
                },
                "symmetric_slope": {
                    "x": (-1.85, -1.65),
                    "y": (-0.2, 0.2),
                },
                "braking_slope": {
                    "x": (-1.6, -1.4),
                    "y": (-0.2, 0.2),
                },
                "s_curve": {
                    "x": _S_CURVE_ENTRY_X_RANGE,
                    "y": _S_CURVE_ENTRY_Y_RANGE,
                },
                "l_curve": {
                    "x": (-0.1, 0.3),
                    "y": (-0.1, 0.1)
                },
                "thin_l_railed_platform": {
                    "x": (-0.1, 0.3),
                    "y": (-0.1, 0.1)
                },
            },
            "terrain_specific_heading_target": {
                "s_curve": {
                    "target_x": _S_CURVE_ENTRY_CENTER_XY[0],
                    "target_y_range": (
                        _S_CURVE_ENTRY_CENTER_XY[1],
                        _S_CURVE_ENTRY_CENTER_XY[1],
                    ),
                    "outside_spawn_y_range": _S_CURVE_ENTRY_CENTER_Y_RANGE,
                    "yaw_offset_range": (0.0, 0.0),
                },
            },
            "terrain_specific_spawn_z_offset": {
                "interlaced_corridor_wedges": -0.2,
                "interlaced_corridor_wedges_approach": -0.6,
                "interlaced_corridor_wedges_smooth_exit": -0.6,
                "interlaced_corridor_wedges_smooth_entry_exit": -0.6,
                "continuous_wave_slopes": -0.27,
            },
            "terrain_specific_figure8_spawn": {
                "spawn_mode": "center_cutout",
                "bridge_height": 0.1,
                "holes_depth": 0.0,
                "center_cutout_width_m": 0.8,
                "center_cutout_length_m": 0.8,
                "base_yaw_angles": (-1.57, 1.57),
                "yaw_jitter_deg": 30.0,
                "position_noise": 0.0,
            },
            "terrain_specific_l_curve_bidirectional_spawn": {
                terrain_key: {
                    "reverse_probability": 0.5,
                    "sub_terrain_key": terrain_key,
                    "longitudinal_offset_range": (-0.1, 0.3),
                    "lateral_offset_range": (-0.1, 0.1),
                    "yaw_jitter_range": (-0.2617993877991494, 0.2617993877991494),
                }
                for terrain_key in L_CURVE_TERRAIN_KEYS
            },
        },
    )

    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    register_virtual_obstacles = EventTerm(
        func=mdp.register_virtual_obstacle_to_sensor,
        mode="startup",
        params={
            "sensor_cfgs": [
                SceneEntityCfg("leg_volume_points"),
                SceneEntityCfg("hurdle_ankle_volume_points"),
            ],
        },
    )

    # continuous_disturbance = EventTerm(
    #     func=mdp.apply_continuous_disturbance,
    #     mode="step",
    #     params={
    #         "push_force_range": (25.0, 50.0),
    #         "push_torque_range": (25.0, 50.0),
    #         "disturbance_probability": 0.001,
    #         "disturbance_interval": (5, 15),
    #         "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
    #     },
    # )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity_terrain_specific,
        mode="interval",
        interval_range_s=(5.0, 15.0),
        params={
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.2, 0.2),
                "roll": (-0.4, 0.4),
                "pitch": (-0.4, 0.4),
                "yaw": (-0.4, 0.4),
            },
            "terrain_specific_velocity_range": {
                # # 第一阶段 hurdle 先不加 push robot
                "hurdle": {
                    "x": (-1.0, 1.0),
                    "y": (-1.0, 1.0),
                    "z": (-0.2, 0.2),
                    "roll": (-0.78, 0.78),
                    "pitch": (-0.78, 0.78),
                    "yaw": (-0.78, 0.78),
                },
                # "hurdle": {
                #     "x": (0.0, 0.0),
                #     "y": (0.0, 0.0),
                #     "z": (0.0, 0.0),
                #     "roll": (0.0, 0.0),
                #     "pitch": (0.0, 0.0),
                #     "yaw": (0.0, 0.0),
                # },
                "plane": {
                    "x": (-1.0, 1.0),
                    "y": (-1.0, 1.0),
                    "z": (-0.2, 0.2),
                    "roll": (-0.78, 0.78),
                    "pitch": (-0.78, 0.78),
                    "yaw": (-0.78, 0.78),
                },
                "noisy_slope_up": {
                    "x": (-1.0, 1.0),
                    "y": (-1.0, 1.0),
                    "z": (-0.2, 0.2),
                    "roll": (-0.78, 0.78),
                    "pitch": (-0.78, 0.78),
                    "yaw": (-0.78, 0.78),
                },
                "noisy_slope_down": {
                    "x": (-1.0, 1.0),
                    "y": (-1.0, 1.0),
                    "z": (-0.2, 0.2),
                    "roll": (-0.78, 0.78),
                    "pitch": (-0.78, 0.78),
                    "yaw": (-0.78, 0.78),
                },
                "stairs_up": {
                    "x": (-1.0, 1.0),
                    "y": (-1.0, 1.0),
                    "z": (-0.2, 0.2),
                    "roll": (-0.78, 0.78),
                    "pitch": (-0.78, 0.78),
                    "yaw": (-0.78, 0.78),
                },
                "spiral_stairs_up": {
                    "x": (-1.0, 1.0),
                    "y": (-1.0, 1.0),
                    "z": (-0.2, 0.2),
                    "roll": (-0.78, 0.78),
                    "pitch": (-0.78, 0.78),
                    "yaw": (-0.78, 0.78),
                },
                "stairs_down": {
                    "x": (-1.0, 1.0),
                    "y": (-1.0, 1.0),
                    "z": (-0.2, 0.2),
                    "roll": (-0.78, 0.78),
                    "pitch": (-0.78, 0.78),
                    "yaw": (-0.78, 0.78),
                },
                "spiral_stairs_down": {
                    "x": (-1.0, 1.0),
                    "y": (-1.0, 1.0),
                    "z": (-0.2, 0.2),
                    "roll": (-0.78, 0.78),
                    "pitch": (-0.78, 0.78),
                    "yaw": (-0.78, 0.78),
                },
                "squre_ring_wall": {
                    "x": (-1.0, 1.0),
                    "y": (-1.0, 1.0),
                    "z": (-0.2, 0.2),
                    "roll": (-0.78, 0.78),
                    "pitch": (-0.78, 0.78),
                    "yaw": (-0.78, 0.78),
                },
                "rough_flat": {
                    "x": (-1.0, 1.0),
                    "y": (-1.0, 1.0),
                    "z": (-0.2, 0.2),
                    "roll": (-0.78, 0.78),
                    "pitch": (-0.78, 0.78),
                    "yaw": (-0.78, 0.78),
                },
                "interlaced_corridor_wedges_smooth_entry_exit": {
                    "x": (-1.0, 1.0),
                    "y": (-1.0, 1.0),
                    "z": (-0.2, 0.2),
                    "roll": (-0.78, 0.78),
                    "pitch": (-0.78, 0.78),
                    "yaw": (-0.78, 0.78),
                },
                "symmetric_slope": {
                    "x": (-1.0, 1.0),
                    "y": (-1.0, 1.0),
                    "z": (-0.2, 0.2),
                    "roll": (-0.78, 0.78),
                    "pitch": (-0.78, 0.78),
                    "yaw": (-0.78, 0.78),
                },
                "braking_slope": {
                    "x": (-1.0, 1.0),
                    "y": (-1.0, 1.0),
                    "z": (-0.2, 0.2),
                    "roll": (-0.78, 0.78),
                    "pitch": (-0.78, 0.78),
                    "yaw": (-0.78, 0.78),
                },
            },
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    alive = RewTerm(func=mdp.is_alive, weight=0.2)

    # feet_air_time_variance = RewTerm(
    #     func=mdp.air_time_variance_penalty,
    #     weight=-1.0,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
    #     },
    # )

    # ------------------------------------------- Tracking Velocity -------------------------------------------
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.2,
        params={"std": 0.5, "skip_terrain_types": ("s_curve", *L_CURVE_TERRAIN_KEYS)},
    )
    track_lin_vel_x_linear = RewTerm(
        func=mdp.track_lin_x_speed_linear,
        weight=1.5,
        params={
            "skip_terrain_types": (
                "s_curve",
                *L_CURVE_TERRAIN_KEYS,
                "stairs_up",
                "spiral_stairs_up",
                "stairs_down",
                "spiral_stairs_down",
                "hurdle",
            ),
        },
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=1.0,
        params={
            "std": 0.5,
            "skip_terrain_types": (
                "figure8_bridge", "interlaced_corridor_wedges", "interlaced_corridor_wedges_approach",
                "interlaced_corridor_wedges_smooth_exit",
                "interlaced_corridor_wedges_smooth_entry_exit",
                "continuous_wave_slopes",
                "symmetric_slope"
            ),
            "zero_terrain_types": ("s_curve", *L_CURVE_TERRAIN_KEYS),
            "gate_high_speed_positive": True,
        },
    )
    low_speed = RewTerm(
        func=mdp.low_speed_reward,
        weight=0.5,
        params={
            "min_cmd_vel": 0.1,
            "skip_terrain_types": ("figure8_bridge",),
            "skip_high_speed_commands": True,
        },
    )
    high_speed_tracking_ratio = RewTerm(
        func=mdp.high_speed_tracking_ratio_reward,
        weight=1.0,
        params={
            "terrain_types": (
                "plane", "rough_flat", "noisy_slope_up", "noisy_slope_down", "interlaced_corridor_wedges_smooth_entry_exit",
                "continuous_wave_slopes", "symmetric_slope", "braking_slope"
            ),
            "command_name": "base_velocity",
            "min_command_speed": 1.5,
        },
    )
    delta_yaw = RewTerm(
        func=mdp.delta_yaw_reward_terrain_rescale,
        weight=-0.5,
        params={
            "terrain_types": (
                "figure8_bridge", "interlaced_corridor_wedges", "interlaced_corridor_wedges_approach",
                "interlaced_corridor_wedges_smooth_exit",
                "interlaced_corridor_wedges_smooth_entry_exit",
                "continuous_wave_slopes", "symmetric_slope", "braking_slope",
                "s_curve",
                *L_CURVE_TERRAIN_KEYS,
            ),
            "scale_on_terrain": 0.0,
        },
    )
    lin_vel_dir_reward = RewTerm(
        func=mdp.lin_vel_dir_reward,
        weight=-0.5,
        params={"skip_terrain_types": ("s_curve", *L_CURVE_TERRAIN_KEYS)},
    )

    # ------------------------------------------- stairs -------------------------------------------

    stairs_outward_progress = RewTerm(
        func=mdp.stairs_outward_progress_reward,
        weight=1.5,
        params={
            "terrain_types": (
                "stairs_up",
                "stairs_down",
                "spiral_stairs_up",
                "spiral_stairs_down",
            ),
            "min_forward_command": 0.1,
            "cap_by_command": True,
            "max_outward_speed": 1.5,
        },
    )

    stairs_down_large_step = RewTerm(
        func=mdp.stair_large_step_penalty,
        weight=-0.25,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"]),
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
            "terrain_types": ("stairs_down", "spiral_stairs_down"),
            "min_forward_command": 0.1,
            "stable_contact_time": 0.04,
            "max_radial_step_distance": 0.60,
            "min_support_height_drop": 0.03,
            "excess_distance_scale": 0.20,
        },
    )

    # ------------------------------------------- s_curve -------------------------------------------
    s_curve_entry_center_progress = RewTerm(
        func=mdp.curve_upgrade_entry_center_progress_reward,
        weight=1.0,
        params={
            "terrain_types": ("s_curve",),
            "local_x_range": _S_CURVE_ENTRY_X_RANGE,
            "local_y_range": _S_CURVE_ENTRY_Y_RANGE,
            "unguided_center_y_range": _S_CURVE_ENTRY_CENTER_Y_RANGE,
            "target_xy": _S_CURVE_ENTRY_CENTER_XY,
            "max_progress_speed": 2.0,
        },
    )
    s_curve_path_speed_linear = RewTerm(
        func=mdp.s_curve_path_speed_linear_reward,
        weight=1.5,
        params={
            "terrain_types": ("s_curve",),
            "high_speed_threshold": 1.0,
            "samples_per_semicircle": 64,
            "inward_tangent_bias_deg": 10.0,
            "entry_local_x_range": _S_CURVE_ENTRY_X_RANGE,
            "entry_local_y_range": _S_CURVE_ENTRY_Y_RANGE,
            "entry_center_y_range": _S_CURVE_ENTRY_CENTER_Y_RANGE,
            "entry_target_xy": _S_CURVE_ENTRY_CENTER_XY,
        },
    )
    s_curve_path_forward = RewTerm(
        func=mdp.s_curve_path_forward_reward,
        weight=1.2,
        params={
            "terrain_types": ("s_curve",),
            "std": 0.25,
            "min_cmd_speed": 0.0,
            "samples_per_semicircle": 64,
            "inward_tangent_bias_deg": 10.0,
        },
    )
    s_curve_heading_tangent = RewTerm(
        func=mdp.s_curve_heading_tangent_reward,
        weight=0.8,
        params={
            "terrain_types": ("s_curve",),
            "std": 0.5,
            "min_cmd_speed": 0.0,
            "samples_per_semicircle": 64,
            "inward_tangent_bias_deg": 10.0,
        },
    )
    s_curve_centerline = RewTerm(
        func=mdp.s_curve_centerline_reward,
        weight=1.0,
        params={
            "terrain_types": ("s_curve",),
            "std": 0.35,
            "edge_buffer_m": 0.05,
            "samples_per_semicircle": 64,
            "inward_centerline_offset_m": 0.05,
        },
    )
    s_curve_feet_impact = RewTerm(
        func=mdp.feet_impact,
        weight=-3.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
            "terrain_types": ("s_curve",),
        },
    )

    # ------------------------------------------- L curves -------------------------------------------
    l_curve_path_speed = RewTerm(
        func=mdp.l_curve_path_speed_reward,
        weight=1.2,
        params={
            "terrain_types": L_CURVE_TERRAIN_KEYS, "std": 0.25, "min_cmd_speed": 0.1,
            "turn_radius_ratio": 0.4, "corner_speed_cap_mps": 0.35,
            "corner_approach_distance_m": 1.1, "corner_recovery_distance_m": 0.3,
            "sub_terrain_key": "l_curve", "bidirectional": True,
        },
    )
    l_curve_heading_tangent = RewTerm(
        func=mdp.l_curve_heading_tangent_reward,
        weight=0.8,
        params={
            "terrain_types": L_CURVE_TERRAIN_KEYS, "std": 0.5, "min_cmd_speed": 0.1,
            "turn_radius_ratio": 0.4, "lookahead_base_m": 0.25,
            "lookahead_speed_scale_s": 0.25, "sub_terrain_key": "l_curve",
            "bidirectional": True,
        },
    )
    l_curve_centerline = RewTerm(
        func=mdp.l_curve_centerline_reward,
        weight=1.0,
        params={
            "terrain_types": L_CURVE_TERRAIN_KEYS, "std": 0.5, "edge_buffer_m": 0.05,
            "turn_radius_ratio": 0.4, "sub_terrain_key": "l_curve",
        },
    )
    l_curve_lateral_velocity = RewTerm(
        func=mdp.l_curve_lateral_velocity_penalty,
        weight=-0.2,
        params={
            "terrain_types": L_CURVE_TERRAIN_KEYS, "turn_radius_ratio": 0.4,
            "sub_terrain_key": "l_curve",
        },
    )
    l_curve_arm_contact = RewTerm(
        func=mdp.undesired_contacts_on_terrain,
        weight=-50.0,
        params={
            "threshold": 3.0,
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_arm_.*_link"),
            "terrain_types": L_CURVE_TERRAIN_KEYS,
        },
    )
    l_curve_corner_heading = RewTerm(
        func=mdp.l_curve_corner_heading_penalty,
        weight=-0.6,
        params={
            "terrain_types": L_CURVE_TERRAIN_KEYS, "min_cmd_speed": 0.1,
            "turn_radius_ratio": 0.4, "lookahead_base_m": 0.8,
            "lookahead_speed_scale_s": 0.1, "lookahead_max_m": 1.05,
            "preview_point_weight": 0.7, "corner_approach_distance_m": 1.1,
            "corner_recovery_distance_m": 0.3, "sub_terrain_key": "l_curve",
            "bidirectional": True,
        },
    )
    l_curve_corner_yaw_rate = RewTerm(
        func=mdp.l_curve_corner_yaw_rate_reward,
        weight=0.8,
        params={
            "terrain_types": L_CURVE_TERRAIN_KEYS, "std": 0.35, "min_cmd_speed": 0.1,
            "turn_radius_ratio": 0.4, "corner_speed_cap_mps": 0.35,
            "heading_feedback_gain": 0.8, "max_ref_ang_vel_rps": 1.2,
            "forward_gate_floor": 0.3, "lookahead_base_m": 0.8,
            "lookahead_speed_scale_s": 0.1, "lookahead_max_m": 1.05,
            "preview_point_weight": 0.7, "corner_approach_distance_m": 1.1,
            "corner_recovery_distance_m": 0.3, "sub_terrain_key": "l_curve",
            "bidirectional": True,
        },
    )
    l_curve_corner_progress = RewTerm(
        func=mdp.l_curve_corner_progress_reward,
        weight=0.3,
        params={
            "terrain_types": L_CURVE_TERRAIN_KEYS, "min_cmd_speed": 0.1,
            "turn_radius_ratio": 0.4, "corner_speed_cap_mps": 0.35,
            "lookahead_base_m": 0.8, "lookahead_speed_scale_s": 0.1,
            "lookahead_max_m": 1.05, "preview_point_weight": 0.7,
            "corner_approach_distance_m": 1.1, "corner_recovery_distance_m": 0.3,
            "sub_terrain_key": "l_curve", "bidirectional": True,
        },
    )

    # --------------------------- interlaced_corridor_wedges_smooth_entry_exit ---------------------------

    smooth_entry_exit_corridor_centerline = RewTerm(
        func=mdp.corridor_centerline_exp,
        weight=2.0,
        params={
            "std": 0.2,
            "terrain_types": ("interlaced_corridor_wedges_smooth_entry_exit",),
            "min_forward_command": 0.3,
            "local_x_range": (-2.8, 3.0),
        },
    )
    smooth_entry_exit_corridor_world_x_progress_guidance = RewTerm(
        func=mdp.corridor_world_x_progress_reward,
        weight=0.5,
        params={
            "terrain_types": ("interlaced_corridor_wedges_smooth_entry_exit",),
            "safe_half_width_m": 0.85,
            "max_forward_speed": 2.0,
            "cap_by_command": True,
            "min_forward_command": 0.3,
            "local_x_range": (-4.2, -2.8),
        },
    )
    smooth_entry_exit_corridor_world_x_progress = RewTerm(
        func=mdp.corridor_world_x_progress_reward,
        weight=0.5,
        params={
            "terrain_types": ("interlaced_corridor_wedges_smooth_entry_exit",),
            "safe_half_width_m": 0.7,
            "max_forward_speed": 2.0,
            "cap_by_command": True,
            "min_forward_command": 0.3,
            "local_x_range": (-2.8, 3.0),
        },
    )
    smooth_entry_exit_corridor_heading_threshold = RewTerm(
        func=mdp.corridor_heading_threshold_penalty,
        weight=-2.0,
        params={
            "terrain_types": ("interlaced_corridor_wedges_smooth_entry_exit",),
            "penalty_threshold": 0.5235987755982988,
            "penalty_limit": 1.0471975511965976,
            "target_heading": 0.0,
            "min_forward_command": 0.3,
            "local_x_range": (-4.2, -2.8),
        },
    )
    smooth_entry_exit_interlaced_ramp_step_up_lift = RewTerm(
        func=mdp.interlaced_corridor_ramp_step_up_lift_reward,
        weight=3.0,
        params={
            "contact_sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
            "left_foot_height_scanner_cfg": SceneEntityCfg("left_foot_height_scanner"),
            "right_foot_height_scanner_cfg": SceneEntityCfg("right_foot_height_scanner"),
            "offset": 0.06,
            "clearance_threshold": 0.15,
            "std": 0.05,
            "x_range": (-3.4, -2.8),
            "min_forward_command": 0.3,
            "min_episode_length": 10,
            "max_abs_local_y": 1.2,
            "terrain_types": ("interlaced_corridor_wedges_smooth_entry_exit",),
        },
    )
    smooth_entry_exit_corridor_lateral_threshold = RewTerm(
        func=mdp.corridor_lateral_threshold_penalty,
        weight=-3.0,
        params={
            "terrain_types": ("interlaced_corridor_wedges_smooth_entry_exit",),
            "penalty_threshold": 0.85,
            "penalty_limit": 1.5,
            "min_forward_command": 0.3,
            "local_x_range": (-4.2, 3.0),
        },
    )
    # smooth_entry_exit_corridor_lateral_exit_penalty = RewTerm(
    #     func=mdp.termination_term_indicator,
    #     weight=-50.0,
    #     params={"term_name": "smooth_entry_exit_corridor_lateral_exit"},
    # )

    # ------------------------------ continuous_wave_slopes ------------------------------
    continuous_wave_world_x_progress = RewTerm(
        func=mdp.corridor_world_x_progress_reward,
        weight=0.5,
        params={
            "terrain_types": ("continuous_wave_slopes",),
            "safe_half_width_m": 0.60,
            "max_forward_speed": 3.0,
            "cap_by_command": True,
            "min_forward_command": 0.3,
            "local_x_range": (-4.0, 3.8),
        },
    )
    continuous_wave_centerline = RewTerm(
        func=mdp.corridor_centerline_exp,
        weight=2.0,
        params={
            "std": 0.3,
            "terrain_types": ("continuous_wave_slopes",),
            "min_forward_command": 0.3,
            "local_x_range": (-4.0, 3.8),
        },
    )
    continuous_wave_outer_lateral = RewTerm(
        func=mdp.corridor_lateral_threshold_penalty,
        weight=-3.0,
        params={
            "terrain_types": ("continuous_wave_slopes",),
            "penalty_threshold": 0.60,
            "penalty_limit": 1.2,
            "min_forward_command": 0.3,
            "local_x_range": (-4.0, 3.8),
        },
    )
    continuous_wave_support_foot_edge = RewTerm(
        func=mdp.corridor_support_foot_lateral_penalty,
        weight=-1.0,
        params={
            "terrain_types": ("continuous_wave_slopes",),
            "penalty_threshold": 1.00,
            "penalty_limit": 1.20,
            "contact_force_threshold": 0.5,
            "min_forward_command": 0.3,
            "local_x_range": (-4.0, 3.8),
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["l_leg_ankle_roll_link", "r_leg_ankle_roll_link"]
            ),
            "contact_sensor_cfg": SceneEntityCfg(
                "contact_sensor", body_names=".*_ankle_roll_link"
            ),
        },
    )
    # continuous_wave_lateral_exit_penalty = RewTerm(
    #     func=mdp.termination_term_indicator,
    #     weight=-50.0,
    #     params={"term_name": "continuous_wave_slopes_lateral_exit"},
    # )

    # ------------------------------ symmetric_slope ------------------------------
    symmetric_slope_world_x_progress = RewTerm(
        func=mdp.corridor_world_x_progress_reward,
        weight=0.5,
        params={
            "terrain_types": ("symmetric_slope",),
            "safe_half_width_m": 0.6,
            "max_forward_speed": 3.0,
            "cap_by_command": True,
            "min_forward_command": 0.3,
            "local_x_range": (-1.85, 1.7),
        },
    )
    symmetric_slope_centerline = RewTerm(
        func=mdp.corridor_centerline_exp,
        weight=2.0,
        params={
            "std": 0.3,
            "terrain_types": ("symmetric_slope",),
            "min_forward_command": 0.3,
            "local_x_range": (-1.85, 1.7),
        },
    )

    # ------------------------------ braking_slope ------------------------------
    braking_slope_stairs_world_x_progress = RewTerm(
        func=mdp.braking_slope_world_x_progress_reward,
        weight=0.5,
        params={
            "terrain_types": ("braking_slope",),
            "section": "stairs",
            "safe_half_width_m": 0.6,
            "max_forward_speed": 1.5,
            "cap_by_command": True,
            "min_forward_command": 0.0,
        },
    )
    braking_slope_downhill_world_x_progress = RewTerm(
        func=mdp.braking_slope_world_x_progress_reward,
        weight=0.5,
        params={
            "terrain_types": ("braking_slope",),
            "section": "slope",
            "safe_half_width_m": 0.6,
            "max_forward_speed": 1.5,
            "cap_by_command": True,
            "min_forward_command": 0.0,
        },
    )
    braking_slope_centerline = RewTerm(
        func=mdp.corridor_centerline_exp,
        weight=2.0,
        params={
            "std": 0.3,
            "terrain_types": ("braking_slope",),
            "min_forward_command": 0.3,
            "local_x_range": (-1.6, 1.5),
        },
    )

    # ------------------------------------------------- Base --------------------------------------------------
    flat_orientation_exp = RewTerm(func=mdp.flat_orientation_exp, weight=0.5, params={"std": 0.5})
    # ------------------------------------------------- Feet --------------------------------------------------
    feet_contact = RewTerm(
        func=mdp.feet_contact,
        weight=0.5,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link")},
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    feet_impact = RewTerm(
        func=mdp.feet_impact,
        weight=-3.,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
            "skip_terrain_types": ("figure8_bridge",),
        },
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
            "threshold": 0.3,
            "gate_high_speed_positive": True,
        },
    )
    feet_unhold = RewTerm(
        func=mdp.feet_unhold_reward,
        weight=-2.5,
        params={
            "contact_sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
            "left_foot_height_scanner_cfg": SceneEntityCfg("left_foot_height_scanner"),
            "right_foot_height_scanner_cfg": SceneEntityCfg("right_foot_height_scanner"),
            "terrain_types": (
                "stairs_up", "stairs_down", "spiral_stairs_up", "spiral_stairs_down",
                "gap", "discrete_obstacles", "thin_obstacles", "square_ring_wall", "pillar_flat", "hurdle",
                "stepping_stones", "cross_stepping_stones2", "cross_stepping_stones1",
                "interlaced_corridor_wedges", "interlaced_corridor_wedges_approach",
                "interlaced_corridor_wedges_smooth_exit",
                "interlaced_corridor_wedges_smooth_entry_exit",
                "continuous_wave_slopes", "symmetric_slope", "braking_slope",
            ),
        },
    )
    imagined_foothold_guidance = RewTerm(
        func=mdp.imagined_foothold_guidance,
        weight=0.75,
        params={
            "left_foot_height_scanner_cfg": SceneEntityCfg("left_foot_height_scanner"),
            "patch_size": (0.225, 0.10),
            "patch_resolution": 0.05,
            "support_height_tolerance": 0.03,
            "reward_sigma": 0.25,
            "curriculum_level_threshold": 3,
            "terrain_types": (
                "stairs_up",
                "stairs_down",
                "spiral_stairs_up",
                "spiral_stairs_down",
            ),
            "downstairs_terrain_types": ("stairs_down", "spiral_stairs_down"),
            "downstairs_scale": 1.5,
        },
    )

    volume_points_penetration = RewTerm(
        func=mdp.volume_points_penetration,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("leg_volume_points"),
            "terrain_types": (
                "stairs_up", "stairs_down", "spiral_stairs_up", "spiral_stairs_down",
                "gap", "discrete_obstacles", "thin_obstacles", "pillar_flat",
                "stepping_stones", "cross_stepping_stones2", "cross_stepping_stones1",
                "interlaced_corridor_wedges", "interlaced_corridor_wedges_approach",
                "interlaced_corridor_wedges_smooth_exit",
                "interlaced_corridor_wedges_smooth_entry_exit",
                "s_curve",
                *L_CURVE_TERRAIN_KEYS,
                "continuous_wave_slopes", "symmetric_slope", "braking_slope"
            ),
        },
    )

    hurdle_ankle_volume_penetration = RewTerm(
        func=mdp.volume_points_penetration,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("hurdle_ankle_volume_points"),
            "tolerance": 0.001,
            "terrain_types": ("hurdle",),
        },
    )

    square_ring_wall_feet_impact = RewTerm(
        func=mdp.feet_impact,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
            "terrain_types": ("square_ring_wall",),
        },
    )

    volume_points_penetration_square_ring_wall_side_bands = RewTerm(
        func=mdp.volume_points_penetration_depth_exp,
        weight=-0.8,
        params={
            "sensor_names": ["leg_volume_points"],
            "tolerance": 0.0,
            "depth_scale": 0.01,
            "virtual_obstacle_keys": (
                "square_ring_wall_inner_bands",
                "square_ring_wall_outer_bands",
            ),
            "terrain_types": ("square_ring_wall",),
        },
    )

    feet_force = RewTerm(
        func=mdp.feet_force,
        weight=-1e-2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
            "threshold": 400,
            "max_reward": 600,
        },
    )

    feet_too_near = RewTerm(
        func=mdp.feet_too_near_humanoid_xy,
        weight=-8.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"]), "threshold": 0.15},
    )
    # 梅花桩等地形上、x 指令 > 0.3 且两脚都接触时惩罚两脚过近，鼓励每脚一桩
    # feet_too_near_in_stepping_terrain = RewTerm(
    #     func=mdp.feet_too_near_in_some_terrain_xy,
    #     weight=-5.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"]),
    #         "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
    #         "threshold": 0.2,
    #         "terrain_types": (
    #             "cross_stepping_stones3",
    #             "cross_stepping_stones2",
    #             "cross_stepping_stones1",
    #             "stepping_stones",
    #         ),
    #         "min_forward_command": 0.3,
    #     },
    # )

    feet_flat_ori = RewTerm(
        func=mdp.feet_orientation_contact,
        weight=-0.4,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "terrain_types": (
                "stairs_up", "stairs_down", "spiral_stairs_up", "spiral_stairs_down",
                "gap", "discrete_obstacles", "thin_obstacles", "square_ring_wall", "pillar_flat", "hurdle",
                "stepping_stones", "cross_stepping_stones2", "cross_stepping_stones1",
                "s_curve",
                *L_CURVE_TERRAIN_KEYS,
            ),
        },
    )

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-10,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor",
                body_names=[
                    r"^(?!l_leg_ankle_roll_link$)(?!r_leg_ankle_roll_link$).+$"
                ],
            ),
            "threshold": 1.0,
        },
    )

    undesired_contacts_height = RewTerm(
        func=mdp.undesired_contacts_height,
        weight=-10,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*_ankle_roll_link"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"]),
            "threshold": 1.0,
            "terrain_types": ["hurdle"],
            "height_threshold": 0.1,
        },
    )

    # ------------------------------------------------- Joint -------------------------------------------------
    joint_deviation_leg = RewTerm(
        func=mdp.joint_deviation_l1_with_zero_flag,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_ankle_roll_joint"],
            )
        },
    )
    joint_deviation_upbody = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.3,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_yaw_joint",
                    "waist_yaw_joint",
                ],
            )
        },
    )
    joint_deviation_standing = RewTerm(
        func=mdp.joint_deviation_l1_when_standing,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    r"^(?!l_leg_ankle_pitch_joint$)(?!r_leg_ankle_pitch_joint$)(?!l_leg_ankle_roll_joint$)(?!r_leg_ankle_roll_joint$).+$"
                ],
            )
        },
    )
    # ----------------------------------------------- Regulization -----------------------------------------------
    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0)
    energy = RewTerm(func=mdp.energy, weight=-1e-4)
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.5e-7,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]),
            # "skip_terrain_types": ("hurdle",),
        },
    )
    dof_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            # "skip_terrain_types": ("hurdle",),
        },
    )
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
        # params={"skip_terrain_types": ("hurdle",)},
    )
    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.1,
        params={"terrain_specific_scales": {"hurdle": 0.1}},
    )
    torque_limits = RewTerm(
        func=mdp.applied_torque_limits_by_ratio,
        weight=-0.01,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "limit_ratio": 0.8,
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Fall Down （Do not use in terrain）
    # fall_down = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.5})
    # (3) Bad Orientation
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.707})
    hurdle_bad_orientation = DoneTerm(
        func=mdp.bad_orientation_in_terrain,
        params={"limit_angle": 0.6, "terrain_types": ("hurdle",)},
    )
    hurdle_illegal_contact = DoneTerm(
        func=mdp.terrain_illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor",
                body_names=[r"^(?!l_leg_ankle_roll_link$)(?!r_leg_ankle_roll_link$).+$"],
            ),
            "threshold": 10,
            "episode_length_threshold": 150,
            "terrain_types": ("hurdle",),
        },
    )
    # (4) Fall Down in some terrain
    # fall_down_in_some_terrain = DoneTerm(
    #     func=mdp.fall_down_in_some_terrain,
    #     params={
    #         "minimum_height": 0.5,
    #         "terrain_types": [
    #             "cross_stepping_stones3",
    #             "cross_stepping_stones2",
    #             "cross_stepping_stones1",
    #             "stepping_stones",
    #             "beam",
    #             "gap",
    #         ]
    #     }
    # )
    # figure8_bridge_fall_off = DoneTerm(
    #     func=mdp.figure8_bridge_fall_off,
    #     params={
    #         "terrain_types": ["figure8_bridge"],
    #         "contact_sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
    #         "contact_force_threshold": 0.5,
    #         "root_drop_threshold_m": 0.06,
    #         # Allow a small outer margin and the vertical center transition used by the mesh.
    #         "annulus_outer_margin_m": 0.08,
    #         "transition_y_extend_m": 0.12,
    #         "transition_x_margin_m": 0.06,
    #     },
    # )
    s_curve_reach_end = DoneTerm(
        func=mdp.s_curve_reach_end_timeout,
        time_out=True,
        params={
            "terrain_types": ["s_curve"],
            "contact_sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
            "contact_force_threshold": 0.5,
            "start_region_xy_range": (1.0, 0.0, -3.8, -2.6),
            "end_region_xy_range": (0.0, -1.5, 2.6, 3.8),
        },
    )
    s_curve_invalid_contact = DoneTerm(
        func=mdp.s_curve_invalid_contact,
        params={
            "terrain_types": ["s_curve"],
            "contact_sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
            "contact_force_threshold": 0.5,
            "start_region_xy_range": (1.0, 0.0, -3.8, -2.6),
            "end_region_xy_range": (0.0, -1.5, 2.6, 3.8),
        },
    )
    l_curve_reach_end = DoneTerm(
        func=mdp.l_curve_reach_end_timeout,
        time_out=True,
        params={
            "terrain_types": L_CURVE_TERRAIN_KEYS,
            "contact_sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
            "contact_force_threshold": 0.5, "completion_margin_m": 0.05,
            "exit_lateral_tolerance_m": 0.15, "sub_terrain_key": "l_curve",
            "bidirectional": True,
        },
    )
    l_curve_invalid_foot_contact = DoneTerm(
        func=mdp.l_curve_invalid_foot_contact,
        params={
            "terrain_types": L_CURVE_TERRAIN_KEYS,
            "contact_sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
            "contact_force_threshold": 0.5, "edge_tolerance_m": 0.03,
            "start_region_tolerance_m": 0.15, "exit_region_tolerance_m": 0.15,
            "sub_terrain_key": "l_curve",
        },
    )
    # approach_corridor_lateral_exit = DoneTerm(
    #     func=mdp.interlaced_corridor_lateral_exit,
    #     params={
    #         "terrain_types": ("interlaced_corridor_wedges_approach",),
    #         "local_x_range": (-4.2, 3.0),
    #         "max_abs_local_y": 1.55,  # 1.5 m half-width plus the 0.06 m edge fade.
    #     },
    # )
    # smooth_exit_corridor_lateral_exit = DoneTerm(
    #     func=mdp.interlaced_corridor_lateral_exit,
    #     params={
    #         "terrain_types": ("interlaced_corridor_wedges_smooth_exit",),
    #         "local_x_range": (-4.2, 3.0),
    #         "max_abs_local_y": 1.55,
    #     },
    # )
    # smooth_entry_exit_corridor_lateral_exit = DoneTerm(
    #     func=mdp.interlaced_corridor_lateral_exit,
    #     params={
    #         "terrain_types": ("interlaced_corridor_wedges_smooth_entry_exit",),
    #         "local_x_range": (-4.2, 3.0),
    #         "max_abs_local_y": 1.55,
    #     },
    # )
    # continuous_wave_slopes_reach_end = DoneTerm(
    #     func=mdp.continuous_wave_slopes_reach_end_timeout,
    #     time_out=True,
    #     params={
    #         "terrain_types": ("continuous_wave_slopes",),
    #         "finish_x_m": 3.70,
    #         "max_abs_foot_y_m": 1.20,
    #         "asset_cfg": SceneEntityCfg(
    #             "robot", body_names=["l_leg_ankle_roll_link", "r_leg_ankle_roll_link"]
    #         ),
    #     },
    # )
    # continuous_wave_slopes_lateral_exit = DoneTerm(
    #     func=mdp.continuous_wave_slopes_lateral_exit,
    #     params={
    #         "terrain_types": ("continuous_wave_slopes",),
    #         "finish_x_m": 3.70,
    #         "max_abs_root_y_m": 1.20,
    #         "max_abs_contact_y_m": 1.25,
    #         "contact_force_threshold": 0.5,
    #         "asset_cfg": SceneEntityCfg(
    #             "robot", body_names=["l_leg_ankle_roll_link", "r_leg_ankle_roll_link"]
    #         ),
    #         "contact_sensor_cfg": SceneEntityCfg(
    #             "contact_sensor", body_names=".*_ankle_roll_link"
    #         ),
    #     },
    # )
    # continuous_wave_slopes_forbidden_contact = DoneTerm(
    #     func=mdp.continuous_wave_slopes_forbidden_contact,
    #     params={
    #         "terrain_types": ("continuous_wave_slopes",),
    #         "finish_x_m": 3.70,
    #         "max_abs_foot_y_m": 1.20,
    #         "threshold": 1.0,
    #         "asset_cfg": SceneEntityCfg(
    #             "robot", body_names=["l_leg_ankle_roll_link", "r_leg_ankle_roll_link"]
    #         ),
    #         "sensor_cfg": SceneEntityCfg(
    #             "contact_sensor",
    #             body_names=[r"^(?!l_leg_ankle_roll_link$)(?!r_leg_ankle_roll_link$).+$"],
    #         ),
    #     },
    # )
    # (5) Out of Bounds
    terrain_out_of_bounds = DoneTerm(func=mdp.terrain_out_of_bounds, time_out=True, params={"distance_buffer": 2.0})


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # terrain_levels = CurrTerm(func=mdp.terrain_levels_adaptive)
    # terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    terrain_levels = CurrTerm(
        func=mdp.terrain_levels_vel_with_termination,
        params={
            "s_curve_terrain_types": ("s_curve",),
            "s_curve_reach_end_term": "s_curve_reach_end",
            "s_curve_half_local_x": 0.0,
            "l_curve_terrain_types": L_CURVE_TERRAIN_KEYS,
            "l_curve_reach_end_term": "l_curve_reach_end",
            "l_curve_sub_terrain_key": "l_curve",
            "l_curve_turn_radius_ratio": 0.4,
            "l_curve_bidirectional": True,
            "continuous_wave_terrain_types": ("continuous_wave_slopes",),
            "continuous_wave_finish_x_m": 3.8,
            "symmetric_slope_terrain_types": ("symmetric_slope",),
            "symmetric_slope_finish_x_m": 1.7,
            "braking_slope_terrain_types": ("braking_slope",),
            "braking_slope_finish_x_m": 1.5,
        },
    )
    # pass


##
# Environment configuration
##


@configclass
class WalkRunDepthEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: E1SceneCfg = E1SceneCfg(num_envs=4096, env_spacing=3.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # viewer settings
        self.viewer.eye = (1.5, 1.5, 1.5)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # A 12 m terrain grid has 2.25x the height-field triangles of the former 8 m grid.
        # Keep the larger collision stack local to this dense WalkRun environment.
        self.sim.physx.gpu_collision_stack_size = 2**27

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.contact_sensor.update_period = self.sim.dt
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.left_foot_height_scanner is not None:
            self.scene.left_foot_height_scanner.update_period = self.sim.dt
        if self.scene.right_foot_height_scanner is not None:
            self.scene.right_foot_height_scanner.update_period = self.sim.dt
        if self.scene.raycaster_camera is not None:
            self.scene.raycaster_camera.update_period = 3 * self.sim.dt

        # ----------------------------------------------AMP----------------------------------------------
        self.motion_loader_name = "MotionLoaderE1"
        # self.motion_files = sorted(glob.glob(f"{ASSET_DIR}/datasets/walkrun/e1/*"))
        self.motion_files = sorted([*DEFAULT_RSI_MOTION_FILES, HURDLE_MOTION_FILE])
        self.reference_observation_horizon = 4
        self.num_preload_transitions = 200000
        self.observations.discriminator.history_length = self.reference_observation_horizon
