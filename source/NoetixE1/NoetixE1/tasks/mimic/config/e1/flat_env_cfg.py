# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

import os

import yaml
from isaaclab.utils import configclass

from NoetixE1.assets import ASSET_DIR
from NoetixE1.assets.robots.e1.e1 import E1_24DOF_ACTION_SCALE, E1_24DOF_CFG, E1_23DOF_ACTION_SCALE, E1_23DOF_CFG
from NoetixE1.tasks.mimic.tracking_env_cfg import TrackingEnvCfg, Tracking23DoFEnvCfg
from NoetixE1.tasks.mimic.tracking_depth_env_cfg import TrackingDepthEnvCfg, Tracking23DoFDepthEnvCfg
from NoetixE1.tasks.mimic.tracking_terrain_refine_env_cfg import TrackingTerrainRefineEnvCfg


def _motion_file_from_metadata(motion_path: str) -> str | None:
    """Read first motion_file from metadata.yaml under motion_path. Returns full path or None."""
    meta_path = os.path.join(motion_path, "metadata.yaml")
    if not os.path.isfile(meta_path):
        return None
    with open(meta_path) as f:
        data = yaml.safe_load(f)
    motion_list = data.get("motion_files")
    if not motion_list or not isinstance(motion_list, list):
        return None
    first = motion_list[0]
    name = first.get("motion_file") if isinstance(first, dict) else None
    if not name:
        return None
    return os.path.join(motion_path, name)

@configclass
class E1FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = E1_24DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = E1_24DOF_ACTION_SCALE
        self.commands.motion.anchor_body_name = "waist_roll_link"
        self.commands.motion.motion_file = f"{ASSET_DIR}/datasets/mimic/135_02.npz"
        self.commands.motion.ankle_dof_names = [
            "l_leg_ankle_roll_joint",
            "r_leg_ankle_roll_joint",
            "l_leg_ankle_pitch_joint",
            "r_leg_ankle_pitch_joint",
        ]
        self.commands.motion.body_names = [
            "base_link",
            "waist_roll_link",
            "l_leg_hip_pitch_link",
            "r_leg_hip_pitch_link",
            "l_leg_knee_link",
            "r_leg_knee_link",
            "l_arm_shoulder_roll_link",
            "r_arm_shoulder_roll_link",
            "l_leg_ankle_roll_link",
            "r_leg_ankle_roll_link",
            "l_arm_elbow_pitch_link",
            "r_arm_elbow_pitch_link",
            "l_arm_elbow_yaw_link",
            "r_arm_elbow_yaw_link",
        ]


@configclass
class E1Flat23DoFEnvCfg(Tracking23DoFEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = E1_23DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = E1_23DOF_ACTION_SCALE
        self.commands.motion.anchor_body_name = "waist_yaw_link"
        self.commands.motion.motion_file = "/home/sch/motion_data/e1/23dof/pingpang/npz/pingpang1.npz"
        self.commands.motion.ankle_dof_names = [
            "l_leg_ankle_roll_joint",
            "r_leg_ankle_roll_joint",
            "l_leg_ankle_pitch_joint",
            "r_leg_ankle_pitch_joint",
        ]
        self.commands.motion.body_names = [
            "base_link",
            "waist_yaw_link",
            "l_leg_hip_pitch_link",
            "r_leg_hip_pitch_link",
            "l_leg_knee_link",
            "r_leg_knee_link",
            "l_arm_shoulder_roll_link",
            "r_arm_shoulder_roll_link",
            "l_leg_ankle_roll_link",
            "r_leg_ankle_roll_link",
            "l_arm_elbow_pitch_link",
            "r_arm_elbow_pitch_link",
            "l_arm_elbow_yaw_link",
            "r_arm_elbow_yaw_link",
        ]


@configclass
class E1Flat23DoFDepthEnvCfg(Tracking23DoFDepthEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = E1_23DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = E1_23DOF_ACTION_SCALE
        self.commands.motion.anchor_body_name = "waist_yaw_link"
        self.commands.motion.ankle_dof_names = [
            "l_leg_ankle_roll_joint",
            "r_leg_ankle_roll_joint",
            "l_leg_ankle_pitch_joint",
            "r_leg_ankle_pitch_joint",
        ]
        self.commands.motion.body_names = [
            "base_link",
            "waist_yaw_link",
            "l_leg_hip_pitch_link",
            "r_leg_hip_pitch_link",
            "l_leg_knee_link",
            "r_leg_knee_link",
            "l_arm_shoulder_roll_link",
            "r_arm_shoulder_roll_link",
            "l_leg_ankle_roll_link",
            "r_leg_ankle_roll_link",
            "l_arm_elbow_pitch_link",
            "r_arm_elbow_pitch_link",
            "l_arm_elbow_yaw_link",
            "r_arm_elbow_yaw_link",
        ]

        # terrain and motion: path and motion_file from MOTION_PATH and its metadata.yaml
        MOTION_PATH = "/home/sch/motion_data/e1/23dof/parkour/parkour_50cmbox"
        PLANE_TERRAIN = False
        if PLANE_TERRAIN:
            self.scene.terrain.terrain_type = "plane"
            self.scene.terrain.terrain_generator = None
        else:
            # 统一为所有「可加载外部 mesh」的子地形设置 path/metadata_yaml，与名称无关；具体 mesh 由 metadata 中 terrain_id 对应
            _metadata_yaml = os.path.join(MOTION_PATH, "metadata.yaml")
            for _name, _sub in self.scene.terrain.terrain_generator.sub_terrains.items():
                if hasattr(_sub, "path") and hasattr(_sub, "metadata_yaml"):
                    _sub.path = MOTION_PATH
                    _sub.metadata_yaml = _metadata_yaml
        self.commands.motion.motion_file = _motion_file_from_metadata(MOTION_PATH)

@configclass
class E1Flat23DoFDepthEnvCfg(Tracking23DoFDepthEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = E1_23DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = E1_23DOF_ACTION_SCALE
        self.commands.motion.anchor_body_name = "waist_yaw_link"
        self.commands.motion.ankle_dof_names = [
            "l_leg_ankle_roll_joint",
            "r_leg_ankle_roll_joint",
            "l_leg_ankle_pitch_joint",
            "r_leg_ankle_pitch_joint",
        ]
        self.commands.motion.body_names = [
            "base_link",
            "waist_yaw_link",
            "l_leg_hip_pitch_link",
            "r_leg_hip_pitch_link",
            "l_leg_knee_link",
            "r_leg_knee_link",
            "l_arm_shoulder_roll_link",
            "r_arm_shoulder_roll_link",
            "l_leg_ankle_roll_link",
            "r_leg_ankle_roll_link",
            "l_arm_elbow_pitch_link",
            "r_arm_elbow_pitch_link",
            "l_arm_elbow_yaw_link",
            "r_arm_elbow_yaw_link",
        ]

        # terrain and motion: path and motion_file from MOTION_PATH and its metadata.yaml
        MOTION_PATH = "/home/sch/motion_data/e1/23dof/parkour/parkour_50cmbox"
        PLANE_TERRAIN = False
        if PLANE_TERRAIN:
            self.scene.terrain.terrain_type = "plane"
            self.scene.terrain.terrain_generator = None
        else:
            _metadata_yaml = os.path.join(MOTION_PATH, "metadata.yaml")
            for _name, _sub in self.scene.terrain.terrain_generator.sub_terrains.items():
                if hasattr(_sub, "path") and hasattr(_sub, "metadata_yaml"):
                    _sub.path = MOTION_PATH
                    _sub.metadata_yaml = _metadata_yaml
        self.commands.motion.motion_file = _motion_file_from_metadata(MOTION_PATH)


@configclass
class E1Flat23DoFTerrainRefineEnvCfg(TrackingTerrainRefineEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = E1_23DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = E1_23DOF_ACTION_SCALE
        self.commands.motion.anchor_body_name = "waist_yaw_link"
        self.commands.motion.ankle_dof_names = [
            "l_leg_ankle_roll_joint",
            "r_leg_ankle_roll_joint",
            "l_leg_ankle_pitch_joint",
            "r_leg_ankle_pitch_joint",
        ]
        self.commands.motion.body_names = [
            "base_link",
            "waist_yaw_link",
            "l_leg_hip_pitch_link",
            "r_leg_hip_pitch_link",
            "l_leg_knee_link",
            "r_leg_knee_link",
            "l_arm_shoulder_roll_link",
            "r_arm_shoulder_roll_link",
            "l_leg_ankle_roll_link",
            "r_leg_ankle_roll_link",
            "l_arm_elbow_pitch_link",
            "r_arm_elbow_pitch_link",
            "l_arm_elbow_yaw_link",
            "r_arm_elbow_yaw_link",
        ]

        # terrain and motion: path and motion_file from MOTION_PATH and its metadata.yaml
        MOTION_PATH = "/home/sch/motion_data/e1/23dof/parkour/parkour_50cmbox"
        PLANE_TERRAIN = False
        if PLANE_TERRAIN:
            self.scene.terrain.terrain_type = "plane"
            self.scene.terrain.terrain_generator = None
        else:
            _metadata_yaml = os.path.join(MOTION_PATH, "metadata.yaml")
            for _name, _sub in self.scene.terrain.terrain_generator.sub_terrains.items():
                if hasattr(_sub, "path") and hasattr(_sub, "metadata_yaml"):
                    _sub.path = MOTION_PATH
                    _sub.metadata_yaml = _metadata_yaml
        self.commands.motion.motion_file = _motion_file_from_metadata(MOTION_PATH)
