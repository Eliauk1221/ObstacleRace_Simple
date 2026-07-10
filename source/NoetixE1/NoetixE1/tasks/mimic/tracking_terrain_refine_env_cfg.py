# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.
from __future__ import annotations

from dataclasses import MISSING

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
from NoetixE1.terrains import TerrainImporterCfg
from NoetixE1.terrains.terrain_generator_cfg import FiledTerrainGeneratorCfg
from NoetixE1.terrains.trimesh import MotionMatchedTerrainCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from NoetixE1.terrains import MIMIC_TERRAINS_CFG, MIMIC_DEPTH_TERRAIN_REFINE_CFG

from . import mdp

##
# Pre-defined configs
##

##
# Scene definition
##

VELOCITY_RANGE = {
    "x": (0.0, 0.0),
    "y": (0.0, 0.0),
    "z": (0.0, 0.0),
    "roll": (0.0, 0.0),
    "pitch": (0.0, 0.0),
    "yaw": (0.0, 0.0),
}


@configclass
class E1SceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # # ground terrain
    # terrain = TerrainImporterCfg(
    #     prim_path="/World/ground",
    #     terrain_type="generator",  # plane generator
    #     terrain_generator=MIMIC_TERRAINS_CFG,
    #     max_init_terrain_level=5,
    #     collision_group=-1,
    #     physics_material=sim_utils.RigidBodyMaterialCfg(
    #         friction_combine_mode="multiply",
    #         restitution_combine_mode="multiply",
    #         static_friction=1.0,
    #         dynamic_friction=1.0,
    #     ),
    #     visual_material=sim_utils.MdlFileCfg(
    #         mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
    #         project_uvw=True,
    #         texture_scale=(0.25, 0.25),
    #     ),
    #     debug_vis=False,
    # )

    # 地形网格: num_rows x num_cols 个格子，每格用 motion_matched_terrain(difficulty, cfg) 生成。
    # metadata 中若有多个地形，terrain_idx = int(difficulty * len(terrains))，即 difficulty 决定用第几个地形。
    # - curriculum=True: difficulty 按行递增（行 0 最小、行 num_rows-1 最大），地形按行从易到难排布。
    # - curriculum=False（默认）: 每格 difficulty 随机，地形排布无规律。
    # 若希望“多地形有规律排布”，请设 curriculum=True 和 difficulty_range=(0, 1)。
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        # terrain_type="plane",
        # terrain_generator=None,
        terrain_type="hacked_generator",
        # 多种子地形（楼梯、gap、梅花桩等）按占比生成；可选 external_mesh 由 metadata.yaml 的 terrain_id 加载，难度微调位置/角度（运行时在 flat_env_cfg 中设置 path/metadata_yaml）
        terrain_generator=MIMIC_DEPTH_TERRAIN_REFINE_CFG,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
    )

    # robots
    robot: ArticulationCfg = MISSING

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
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        update_period=1 / 50,
        offset=RayCasterCfg.OffsetCfg(pos=(0, 0, 0.5)),
        mesh_prim_paths=["/World/ground"],
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(
            resolution=0.1, size=[1.6, 1.0]
        ),  # data = ((1.6/0.1)+1)*((1.0/0.1)+1) = 187
        debug_vis=True,
    )

##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    motion = mdp.MotionCommandCfg(
        asset_name="robot",
        motion_file="/home/user/NoetixE1-Lab/source/NoetixE1/NoetixE1/assets/datasets/mimic/dance1.npz",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
        pose_range={
            "x": (0.0, 0.0),  # m
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),  # rad
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        },
        # 可选：为特定 bin 指定不同的 pose_range。bin 索引为 0 到 bin_count-1（按时间分段）。
        # 每项为 (bin_spec, pose_range)：bin_spec 为 int（单个 bin）或 (start, end)（左闭右开区间）。
        velocity_range=VELOCITY_RANGE,
        joint_position_range=(0.0, 0.0),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    """如果要用HIM估计状态，需要保持policy和privileged观测的一致性，privileged多出来的观测项可以被HIM选择索引估计"""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={
                "sensor_cfg": SceneEntityCfg("height_scanner"),
            },
            clip=(-8.0, 8.0),
        )

    @configclass
    class PrivilegedCfg(ObsGroup):
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={
                "sensor_cfg": SceneEntityCfg("height_scanner"),
            },
            clip=(-8.0, 8.0),
        )

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    motion_global_anchor_pos = RewTerm(
        func=mdp.motion_global_anchor_position_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_global_anchor_ori = RewTerm(
        func=mdp.motion_global_anchor_orientation_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.4},
    )
    motion_body_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_body_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.4},
    )
    motion_body_lin_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 1.0},
    )
    motion_body_ang_vel = RewTerm(
        func=mdp.motion_global_body_angular_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 3.14},
    )

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor",
                body_names=[
                    r"^(?!l_leg_ankle_roll_link$)(?!r_leg_ankle_roll_link$)(?!l_leg_knee_link$)(?!r_leg_knee_link$)(?!l_arm_elbow_yaw_link$)(?!r_arm_elbow_yaw_link$).+$"
                ],
            ),
            "threshold": 1.0,
        },
    )
    # feet_slide = RewTerm(
    #     func=mdp.feet_slide,
    #     weight=-2,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
    #     },
    # )
    # ----------------------------------------------- Regulization -----------------------------------------------
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    contact_force = RewTerm(
        func=mdp.contact_force,
        weight=-1e-2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*_ankle_roll_link", ".*_knee_link"]),
            "threshold": 400,
            "max_reward": 600,
        },
    )
    torque_limits = RewTerm(
        func=mdp.applied_torque_limits_by_ratio,
        weight=-0.01,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "limit_ratio": 0.8,
        },
    )
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"])},
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    anchor_pos = DoneTerm(
        func=mdp.bad_anchor_pos_z_only,
        params={"command_name": "motion", "threshold": 0.25},
    )
    anchor_ori = DoneTerm(
        func=mdp.bad_anchor_ori,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "motion", "threshold": 0.8},
    )

    ee_body_pos = DoneTerm(
        func=mdp.bad_motion_body_pos_z_only,
        params={
            "command_name": "motion",
            "threshold": 0.4,
            "body_names": [
                "l_leg_ankle_roll_link",
                "r_leg_ankle_roll_link",
                "l_arm_elbow_yaw_link",
                "r_arm_elbow_yaw_link",
            ],
        },
    )
    illegal_reset_contact = DoneTerm(
        func=mdp.illegal_reset_contact,
        time_out=True,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor",
                body_names=[
                    r"^(?!l_leg_ankle_roll_link$)(?!r_leg_ankle_roll_link$)(?!l_arm_elbow_yaw_link$)(?!r_arm_elbow_yaw_link$).+$"
                ],
            ),
            "threshold": 500,
            "episode_length_threshold": 2,
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    pass


##
# Environment configuration
##


@configclass
class TrackingTerrainRefineEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: E1SceneCfg = E1SceneCfg(num_envs=4096, env_spacing=3.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 40.0
        # viewer settings
        self.viewer.eye = (1.5, 1.5, 1.5)
        self.viewer.origin_type = None
        self.viewer.asset_name = None
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.contact_sensor.update_period = self.sim.dt
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
