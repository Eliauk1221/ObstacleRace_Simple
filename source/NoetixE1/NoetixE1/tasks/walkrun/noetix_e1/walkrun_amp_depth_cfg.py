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
from NoetixE1.terrains import DEPTH_TERRAINS_CFG6, GreedyconcatEdgeCylinderCfg, TerrainImporterCfg, DEPTH_TERRAINS_CFG2, DEPTH_TERRAINS_CFG3, DEPTH_TERRAINS_CFG4, DEPTH_TERRAINS_CFG5

from . import mdp

##
# Pre-defined configs
##

from NoetixE1.assets.robots.e1.e1 import E1_23DOF_CFG, E1_23DOF_ACTION_SCALE  # isort:skip


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
                cylinder_radius=0.05,
                min_points=2,
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

    base_velocity = mdp.UniformVelocityWithZeroCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 10.0),
        rel_standing_envs=0.1,
        debug_vis=True,
        ranges=mdp.UniformVelocityWithZeroCommandCfg.Ranges(
            lin_vel_x=(-0.6, 1.5), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.0, 1.0), zero_prob=[0.4, 0.8, 0.4]
        ),
        terrain_specific_ranges={
            "cross_stepping_stones3": mdp.UniformVelocityWithZeroCommandCfg.Ranges(
                lin_vel_x=(0.0, 1.0),  # 在十字梅花桩地形上，x速度范围限制在 0.0 到 1.0
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(-0.0, 0.0),  # yaw角速度范围限制在 -0.5 到 0.5
                zero_prob=[0.5, 0.9, 0.5]  # zero概率：[x, y, yaw]
            ),
            "cross_stepping_stones2": mdp.UniformVelocityWithZeroCommandCfg.Ranges(
                lin_vel_x=(0.0, 1.0),  # 在十字梅花桩地形上，x速度范围限制在 0.0 到 1.0
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(-0.0, 0.0),  # yaw角速度范围限制在 -0.5 到 0.5
                zero_prob=[0.5, 0.9, 0.5]  # zero概率：[x, y, yaw]
            ),
            "cross_stepping_stones1": mdp.UniformVelocityWithZeroCommandCfg.Ranges(
                lin_vel_x=(0.0, 1.0),  # 在十字梅花桩地形上，x速度范围限制在 0.0 到 1.0
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(-0.0, 0.0),  # yaw角速度范围限制在 -0.5 到 0.5
                zero_prob=[0.5, 0.9, 0.5]  # zero概率：[x, y, yaw]
            ),
            "gap": mdp.UniformVelocityWithZeroCommandCfg.Ranges(
                lin_vel_x=(0.0, 1.5),  # 在十字梅花桩地形上，x速度范围限制在 0.0 到 1.0
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(-1.0, 1.0),  # yaw角速度范围限制在 -0.5 到 0.5
                zero_prob=[0.5, 0.9, 0.5]  # zero概率：[x, y, yaw]
            ),
            "stepping_stones": mdp.UniformVelocityWithZeroCommandCfg.Ranges(
                lin_vel_x=(0.0, 1.5),  # 在十字梅花桩地形上，x速度范围限制在 0.0 到 1.0
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(-1.0, 1.0),  # yaw角速度范围限制在 -0.5 到 0.5
                zero_prob=[0.5, 0.9, 0.5]  # zero概率：[x, y, yaw]
            ),
            "beam": mdp.UniformVelocityWithZeroCommandCfg.Ranges(
                lin_vel_x=(0.0, 1.5),  # 在十字梅花桩地形上，x速度范围限制在 0.0 到 1.0
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(0.0, 0.0),  # yaw角速度范围限制在 -0.5 到 0.5
                zero_prob=[0.5, 0.9, 0.5]  # zero概率：[x, y, yaw]
            ),
            "pit": mdp.UniformVelocityWithZeroCommandCfg.Ranges(
                lin_vel_x=(0.0, 1.5),  # 在十字梅花桩地形上，x速度范围限制在 0.0 到 1.0
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(-1.0, 1.0),  # yaw角速度范围限制在 -0.5 到 0.5
                zero_prob=[0.5, 0.9, 0.5]  # zero概率：[x, y, yaw]
            ),
            "interlaced_corridor_wedges": mdp.UniformVelocityWithZeroCommandCfg.Ranges(
                lin_vel_x=(0.3, 1.5),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(0.0, 0.0),
                zero_prob=[0.5, 0.9, 0.5],
            ),
            "figure8_bridge": mdp.UniformVelocityWithZeroCommandCfg.Ranges(
                lin_vel_x=(0.0, 2.5),
                lin_vel_y=(0.0, 0.5),
                ang_vel_z=(0.0, 0.0),
                zero_prob=[0.5, 0.9, 0.5],
            ),
            "s_curve": mdp.UniformVelocityWithZeroCommandCfg.Ranges(
                lin_vel_x=(0.0, 1.5),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(0.0, 0.0),
                zero_prob=[0.5, 0.9, 0.5],
            ),
        },
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
                "roll": (-0.025, 0.025),  # 绕x轴旋转的随机化范围（弧度，约±1.4度）
                "pitch": (-0.025, 0.025),  # 绕y轴旋转的随机化范围（弧度，约±1.4度）
                "yaw": (-0.025, 0.025),  # 绕z轴旋转的随机化范围（弧度，约±1.4度）
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
            "terrain_specific_yaw": {
                "cross_stepping_stones3": [0.0, 1.57, 3.14, -1.57],  # 0°, 90°, 180°, -90°
                "cross_stepping_stones2": [0.0, 1.57, 3.14, -1.57],  # 0°, 90°, 180°, -90°
                "cross_stepping_stones1": [0.0, 1.57, 3.14, -1.57],  # 0°, 90°, 180°, -90°
                "beam": [0.0, 1.57, 3.14, -1.57],  # 0°, 90°, 180°, -90°
                "interlaced_corridor_wedges": (-0.5235987755982988, 0.5235987755982988),  # +/- 30 deg
            },
            "terrain_specific_pos_range": {
                "cross_stepping_stones3": {
                    "x": (-0.3, 0.3),  # 在十字梅花桩地形上，x 范围限制在 -0.3 到 0.3
                    "y": (-0.3, 0.3),  # y 范围限制在 -0.3 到 0.3
                },
                "cross_stepping_stones2": {
                    "x": (-0.3, 0.3),  # 在十字梅花桩地形上，x 范围限制在 -0.3 到 0.3
                    "y": (-0.3, 0.3),  # y 范围限制在 -0.3 到 0.3
                },
                "cross_stepping_stones1": {
                    "x": (-0.1, 0.1),  # 在十字梅花桩地形上，x 范围限制在 -0.3 到 0.3
                    "y": (-0.1, 0.1),  # y 范围限制在 -0.3 到 0.3
                },
                "beam": {
                    "x": (-0.1, 0.1),  # 在十字梅花桩地形上，x 范围限制在 -0.3 到 0.3
                    "y": (-0.1, 0.1),  # y 范围限制在 -0.3 到 0.3
                },
                "interlaced_corridor_wedges": {
                    "x": (-2.25, -2.25),  # First 1.5 m segment center: -3.0 + 1.5 / 2.
                    "y": (0.0, 0.0),
                },
            },
            "terrain_specific_spawn_z_offset": {
                "interlaced_corridor_wedges": -0.2,
            },
            "terrain_specific_s_curve_spawn": {
                "path_fraction_range": (0.02, 0.15),
                "bridge_height": 0.1,
                "position_noise": 0.0,
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
            "sensor_cfgs": SceneEntityCfg("leg_volume_points"),
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
                "rough_flat": {
                    "x": (-1.0, 1.0),
                    "y": (-1.0, 1.0),
                    "z": (-0.2, 0.2),
                    "roll": (-0.78, 0.78),
                    "pitch": (-0.78, 0.78),
                    "yaw": (-0.78, 0.78),
                },
                "wave": {
                    "x": (-1.0, 1.0),
                    "y": (-1.0, 1.0),
                    "z": (-0.2, 0.2),
                    "roll": (-0.78, 0.78),
                    "pitch": (-0.78, 0.78),
                    "yaw": (-0.78, 0.78),
                },
                "beam": {
                    "x": (-0.0, 0.0),
                    "y": (-0.0, 0.0),
                    "z": (-0.2, 0.2),
                    "roll": (-0.78, 0.78),
                    "pitch": (-0.78, 0.78),
                    "yaw": (-0.0, 0.0),
                },
                "pit": {
                    "x": (-1.0, 1.0),
                    "y": (-1.0, 1.0),
                    "z": (-0.2, 0.2),
                    "roll": (-0.78, 0.78),
                    "pitch": (-0.78, 0.78),
                    "yaw": (-0.78, 0.78),
                },
                "interlaced_corridor_wedges": {
                    "x": (-0.8, 0.8),
                    "y": (-0.8, 0.8),
                    "z": (-0.2, 0.2),
                    "roll": (-0.5, 0.5),
                    "pitch": (-0.5, 0.5),
                    "yaw": (-0.5, 0.5),
                },
            },
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # ------------------------------------------- Tracking Velocity -------------------------------------------
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=1.2, params={"std": 0.5})
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=1.0,
        params={
            "std": 0.5,
            "skip_terrain_types": ("figure8_bridge", "s_curve", "interlaced_corridor_wedges"),
        },
    )
    low_speed = RewTerm(func=mdp.low_speed_reward, weight=0.5, params={"min_cmd_vel": 0.1})
    delta_yaw = RewTerm(
        func=mdp.delta_yaw_reward_terrain_rescale,
        weight=-0.5,
        params={
            "terrain_types": ("figure8_bridge", "s_curve", "interlaced_corridor_wedges"),
            "scale_on_terrain": 0.0,
        },
    )
    lin_vel_dir_reward = RewTerm(func=mdp.lin_vel_dir_reward, weight=-0.5)

    # ------------------------------------------- figure8_bridge -------------------------------------------
    figure8_commanded_idle = RewTerm(
        func=mdp.bridge_commanded_idle_penalty,
        weight=-1.0,
        params={
            "terrain_types": ("figure8_bridge",),
            "min_cmd_speed": 0.15,
            "min_body_speed": 0.08,
        },
    )
    figure8_lateral_offset = RewTerm(
        func=mdp.figure8_bridge_lateral_offset_penalty,
        weight=-1.0,
        params={
            "terrain_types": ("figure8_bridge",),
            "contact_sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
            "left_foot_height_scanner_cfg": SceneEntityCfg("left_foot_height_scanner"),
            "right_foot_height_scanner_cfg": SceneEntityCfg("right_foot_height_scanner"),
            "bridge_height": 0.1,
            "bridge_tol": 0.04,
            "edge_buffer_m": 0.05,
        },
    )
    figure8_feet_impact = RewTerm(
        func=mdp.feet_impact,
        weight=-10.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
            "terrain_types": ("figure8_bridge",),
        },
    )

    # ------------------------------------------- s_curve -------------------------------------------
    s_curve_commanded_idle = RewTerm(
        func=mdp.bridge_commanded_idle_penalty,
        weight=-1.0,
        params={
            "terrain_types": ("s_curve",),
            "min_cmd_speed": 0.15,
            "min_body_speed": 0.08,
        },
    )
    s_curve_lateral_offset = RewTerm(
        func=mdp.s_curve_lateral_offset_penalty,
        weight=-1.0,
        params={
            "terrain_types": ("s_curve",),
            "contact_sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
            "left_foot_height_scanner_cfg": SceneEntityCfg("left_foot_height_scanner"),
            "right_foot_height_scanner_cfg": SceneEntityCfg("right_foot_height_scanner"),
            "edge_buffer_m": 0.05,
        },
    )
    s_curve_path_forward = RewTerm(
        func=mdp.s_curve_path_forward_reward,
        weight=0.5,
        params={
            "terrain_types": ("s_curve",),
            "contact_sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
            "left_foot_height_scanner_cfg": SceneEntityCfg("left_foot_height_scanner"),
            "right_foot_height_scanner_cfg": SceneEntityCfg("right_foot_height_scanner"),
            "std": 0.25,
            "min_cmd_speed": 0.1,
        },
    )
    s_curve_path_progress = RewTerm(
        func=mdp.s_curve_path_progress_reward,
        weight=1.0,
        params={
            "terrain_types": ("s_curve",),
            "contact_sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
            "left_foot_height_scanner_cfg": SceneEntityCfg("left_foot_height_scanner"),
            "right_foot_height_scanner_cfg": SceneEntityCfg("right_foot_height_scanner"),
            "min_v_along": 0.05,
            "progress_scale": 1.0,
        },
    )
    s_curve_feet_impact = RewTerm(
        func=mdp.feet_impact,
        weight=-10.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
            "terrain_types": ("s_curve",),
        },
    )

    # ------------------------------------------- interlaced_corridor_wedges -------------------------------------------

    corridor_centerline = RewTerm(
        func=mdp.corridor_centerline_exp,
        weight=1.5,
        params={
            "std": 0.2,
            "terrain_types": ("interlaced_corridor_wedges",),
            "min_forward_command": 0.3,
            "local_x_range": (-3.0, 3.0),
        },
    )
    corridor_world_x_progress = RewTerm(
        func=mdp.corridor_world_x_progress_reward,
        weight=0.3,
        params={
            "terrain_types": ("interlaced_corridor_wedges",),
            "safe_half_width_m": 0.7,
            "max_forward_speed": 1.5,
            "min_forward_command": 0.3,
            "local_x_range": (-3.0, 3.0),
        },
    )
    interlaced_ramp_step_up_lift = RewTerm(
        func=mdp.interlaced_corridor_ramp_step_up_lift_reward,
        weight=6,
        params={
            "contact_sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
            "left_foot_height_scanner_cfg": SceneEntityCfg("left_foot_height_scanner"),
            "right_foot_height_scanner_cfg": SceneEntityCfg("right_foot_height_scanner"),
            "offset": 0.06,
            "clearance_threshold": 0.15,
            "std": 0.05,
            "x_range": (-3.4, -2.8),
            "min_forward_command": 0.3,
            "terrain_types": ("interlaced_corridor_wedges",),
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
            "skip_terrain_types": ("figure8_bridge", "s_curve"),
        },
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
            "threshold": 0.3,
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
                "stairs_up", "stairs_down",
                "gap", "discrete_obstacles", "hurdle",
                "stepping_stones", "cross_stepping_stones2", "cross_stepping_stones1",
                "interlaced_corridor_wedges",
            ),
        },
    )

    volume_points_penetration = RewTerm(
        func=mdp.volume_points_penetration,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("leg_volume_points"),
            "terrain_types": (
                "stairs_up", "stairs_down",
                "gap", "discrete_obstacles", "hurdle",
                "stepping_stones", "cross_stepping_stones2", "cross_stepping_stones1",
                "interlaced_corridor_wedges",
            ),
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
        weight=-1.0,
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
                "stairs_up", "stairs_down",
                "gap", "discrete_obstacles", "hurdle",
                "stepping_stones", "cross_stepping_stones2", "cross_stepping_stones1",
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

    # undesired_contacts_height = RewTerm(
    #     func=mdp.undesired_contacts_height,
    #     weight=-10,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*_ankle_roll_link"]),
    #         "asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"]),
    #         "threshold": 1.0,
    #         "terrain_types": ["hurdle"],
    #         "height_threshold": 0.1,
    #     },
    # )

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
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"])},
    )
    dof_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
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
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.707})  # 45deg
    # (4) Fall Down in some terrain
    fall_down_in_some_terrain = DoneTerm(
        func=mdp.fall_down_in_some_terrain,
        params={
            "minimum_height": 0.5,
            "terrain_types": [
                "cross_stepping_stones3",
                "cross_stepping_stones2",
                "cross_stepping_stones1",
                "stepping_stones",
                "beam",
                "gap",
            ]
        }
    )
    figure8_bridge_fall_off = DoneTerm(
        func=mdp.figure8_bridge_fall_off,
        params={
            "terrain_types": ["figure8_bridge"],
            "contact_sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
            "contact_force_threshold": 0.5,
            "root_drop_threshold_m": 0.06,
        },
    )
    s_curve_fall_off = DoneTerm(
        func=mdp.s_curve_fall_off,
        params={
            "terrain_types": ["s_curve"],
            "contact_sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
            "contact_force_threshold": 0.5,
            "root_drop_threshold_m": 0.06,
        },
    )
    # (5) Out of Bounds
    terrain_out_of_bounds = DoneTerm(func=mdp.terrain_out_of_bounds, time_out=True, params={"distance_buffer": 2.0})


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # terrain_levels = CurrTerm(func=mdp.terrain_levels_adaptive)
    # terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel_with_termination)
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
        self.motion_files = sorted([f"{ASSET_DIR}/datasets/walkrun/e1/e1_standing.txt",
                                    f"{ASSET_DIR}/datasets/walkrun/e1/e1_sidewalk.txt",
                                    f"{ASSET_DIR}/datasets/walkrun/e1/e1_walk_backward.txt",
                                    f"{ASSET_DIR}/datasets/walkrun/e1/e1_turning.txt",
                                    f"{ASSET_DIR}/datasets/walkrun/e1/e1_walk_forward.txt",
                                    ])
        self.reference_observation_horizon = 4
        self.num_preload_transitions = 200000
        self.observations.discriminator.history_length = self.reference_observation_horizon
