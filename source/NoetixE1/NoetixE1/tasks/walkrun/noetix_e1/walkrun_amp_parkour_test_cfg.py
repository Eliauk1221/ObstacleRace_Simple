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
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from NoetixE1.assets import ASSET_DIR
from NoetixE1.terrains import PARKOUR_TERRAINS_CFG

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
        terrain_generator=PARKOUR_TERRAINS_CFG,
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
            resolution=0.1, size=[1.6, 1.0]
        ),  # data = ((1.6/0.1)+1)*((1.0/0.1)+1) = 187
        debug_vis=True,
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
            lin_vel_x=(-0.6, 1.5), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.0, 1.0), zero_prob=[0.2, 0.8, 0.4]
        ),
        terrain_specific_ranges={
            "frame_platform": mdp.UniformVelocityWithZeroCommandCfg.Ranges(
                lin_vel_x=(0.0, 1.5),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(0.0, 0.0),
                zero_prob=[0.5, 0.9, 0.5]
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

        # observation terms (order preserved)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        euler_angles = ObsTerm(func=mdp.euler_angles)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
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
            clip=(-8.0, 8.0),
        )

        def __post_init__(self) -> None:
            self.concatenate_terms = True
            self.enable_corruption = False

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        euler_angles = ObsTerm(func=mdp.euler_angles)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
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
            clip=(-8.0, 8.0),
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

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    discriminator: AmpObsCfg = AmpObsCfg()


@configclass
class EventCfg:
    """Configuration for events."""

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
                    "x": (-0.3, 0.3),  # 在十字梅花桩地形上，x 范围限制在 -0.3 到 0.3
                    "y": (-0.3, 0.3),  # y 范围限制在 -0.3 到 0.3
                },
            },
            # 指定地形上 reset 时只从给定 motion 文件中采样姿态（路径需与 motion_files 完全一致）
            "terrain_specific_motion_files": {
                # 例：回字形高台地形仅从 parkour motion 采样
                "frame_platform": ["/home/sch/e1_parkour_45cmbox.json"],
                # 例：梅花桩地形可从 standing + walk 采样
                # "cross_stepping_stones1": [
                #     f"{ASSET_DIR}/datasets/walkrun/e1/e1_standing.txt",
                #     f"{ASSET_DIR}/datasets/walkrun/e1/e1_walk_forward.txt",
                # ],
            },
            # 仅在指定地形等级范围内启用上述“地形专属 motion”（闭区间）；None 表示全部等级启用，其他等级维持普通采样
            "terrain_specific_motion_level_range": (0, 1),  # 仅 0～1 级从专属 motion 采样，2 级及以上普通采样
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # ------------------------------------------- Tracking Velocity -------------------------------------------
    # 指定地形上 base 高度 ≥ 阈值(机器人高0.78)时速度跟踪奖励直接给满，否则与下式等价
    # track_lin_vel_xy_exp = RewTerm(
    #     func=mdp.track_lin_vel_xy_yaw_frame_exp_terrain_full,
    #     weight=1.2,
    #     params={
    #         "std": 0.5,
    #         "terrain_types": ("frame_platform",),
    #         "height_threshold": 0.9,
    #     },
    # )
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=1.2, params={"std": 0.5})
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=1.0, params={"std": 0.5})
    low_speed = RewTerm(func=mdp.low_speed_reward, weight=0.5, params={"min_cmd_vel": 0.1})
    # delta_yaw = RewTerm(func=mdp.delta_yaw_reward, weight=-0.5)
    lin_vel_dir_reward = RewTerm(func=mdp.lin_vel_dir_reward, weight=-0.5)
    delta_yaw_penalty_exp = RewTerm(
        func=mdp.delta_yaw_penalty_exp,
        weight=-2.0,
        params={"std": 1.0, "max_penalty": 10.0},  # 惩罚上限，防止角速度差很大时惩罚过大
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
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link")},
    )
    # feet_air_time = RewTerm(
    #     func=mdp.feet_air_time_positive_biped,
    #     weight=1.0,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
    #         "threshold": 0.3,
    #     },
    # )
    # feet_unhold = RewTerm(
    #     func=mdp.feet_unhold_reward,
    #     weight=-2.5,
    #     params={
    #         "contact_sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
    #         "left_foot_height_scanner_cfg": SceneEntityCfg("left_foot_height_scanner"),
    #         "right_foot_height_scanner_cfg": SceneEntityCfg("right_foot_height_scanner"),
    #     },
    # )

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
        func=mdp.feet_too_near_humanoid,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"]), "threshold": 0.2},
    )
    
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-10,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor",
                body_names=[
                    r"^(?!l_leg_ankle_roll_link$)(?!r_leg_ankle_roll_link$)(?!l_leg_knee_link$)(?!r_leg_knee_link$).+$"
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
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.1)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Fall Down （Do not use in terrain）
    # fall_down = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.5})
    # (3) Bad Orientation
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.98})  # 80deg, cos(10deg)=0.98
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
            ]
        }
    )

    # (5) Illegal Reset Contact
    illegal_reset_contact = DoneTerm(
        func=mdp.illegal_reset_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor",
                body_names=[
                    r"^(?!l_leg_ankle_roll_link$)(?!r_leg_ankle_roll_link$)(?!l_leg_knee_link$)(?!r_leg_knee_link$).+$"
                ],
            ),
            "threshold": 500,
            "episode_length_threshold": 2,
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # terrain_levels = CurrTerm(func=mdp.terrain_levels_adaptive)
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
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
        if self.scene.left_foot_height_scanner is not None:
            self.scene.left_foot_height_scanner.update_period = self.sim.dt
        if self.scene.right_foot_height_scanner is not None:
            self.scene.right_foot_height_scanner.update_period = self.sim.dt

        # ----------------------------------------------AMP----------------------------------------------
        self.motion_loader_name = "MotionLoaderE1"
        # self.motion_files = sorted(glob.glob(f"{ASSET_DIR}/datasets/walkrun/e1/*"))
        self.motion_files = sorted([
            f"{ASSET_DIR}/datasets/walkrun/e1/e1_standing.txt",
            f"{ASSET_DIR}/datasets/walkrun/e1/e1_turning.txt",
            f"{ASSET_DIR}/datasets/walkrun/e1/e1_walk_forward.txt",
            "/home/sch/e1_parkour_45cmbox.json",
        ])
        self.reference_observation_horizon = 4
        self.num_preload_transitions = 200000
        self.observations.discriminator.history_length = self.reference_observation_horizon
