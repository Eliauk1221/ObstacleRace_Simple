from isaaclab.utils import configclass
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm

from . import mdp
from .walkrun_amp_depth_cfg import (
    DEFAULT_RSI_MOTION_FILES,
    STAIR_DOWN_MOTION_FILE,
    STAIR_UP_MOTION_FILE,
    WalkRunDepthEnvCfg,
)


def _without_hurdle(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(value for value in values if value != "hurdle")


@configclass
class WalkRunDepthNoHurdleEnvCfg(WalkRunDepthEnvCfg):
    """WalkRun depth task with all hurdle-specific training configuration disabled."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # Keep the remaining walk/run and stair motions for AMP and RSI.
        self.motion_files = sorted(
            [
                *DEFAULT_RSI_MOTION_FILES,
                STAIR_DOWN_MOTION_FILE,
                STAIR_UP_MOTION_FILE,
            ]
        )

        # Keep downhill commands forward-only while allowing the requested speed up to 1.5 m/s.
        for terrain_type in ("stairs_down", "spiral_stairs_down"):
            downhill_ranges = self.commands.base_velocity.terrain_specific_ranges[terrain_type]
            downhill_ranges.lin_vel_x = (0.3, 1.5)
            downhill_ranges.lin_vel_y = (0.0, 0.0)
            downhill_ranges.ang_vel_z = (0.0, 0.0)
            downhill_ranges.zero_prob = [0.0, 1.0, 1.0]
        terrain_specific_yaw = self.events.reset_robot_states.params["terrain_specific_yaw"]
        terrain_specific_yaw["spiral_stairs_up"] = (0.5, 0.3)
        terrain_specific_yaw["spiral_stairs_down"] = (0.3, 0.5)
        terrain_specific_pos_range = self.events.reset_robot_states.params["terrain_specific_pos_range"]
        terrain_specific_pos_range["spiral_stairs_up"] = {"x": (0.0, 0.0), "y": (0.0, 0.0)}
        terrain_specific_pos_range["spiral_stairs_down"] = {"x": (0.0, 0.0), "y": (0.0, 0.0)}

        # Downstairs prioritizes stable footholds over matching the commanded speed.
        downhill_terrain_types = ("stairs_down", "spiral_stairs_down")
        self.rewards.low_speed.params["skip_terrain_types"] += downhill_terrain_types
        self.rewards.stairs_outward_progress.params["terrain_types"] = tuple(
            terrain_type
            for terrain_type in self.rewards.stairs_outward_progress.params["terrain_types"]
            if terrain_type not in downhill_terrain_types
        )
        self.rewards.stairs_down_safe_progress = RewTerm(
            func=mdp.stairs_outward_progress_reward,
            weight=0.5,
            params={
                "terrain_types": downhill_terrain_types,
                "min_forward_command": 0.1,
                "cap_by_command": False,
                "max_outward_speed": 0.5,
            },
        )
        self.rewards.stairs_down_stall = RewTerm(
            func=mdp.stairs_stall_penalty,
            weight=-1.0,
            params={
                "terrain_types": downhill_terrain_types,
                "min_forward_command": 0.1,
                "min_outward_speed": 0.05,
                "grace_steps": 50,
            },
        )
        self.rewards.imagined_foothold_guidance = None

        # Increase exposure to 4 m/s commands and strengthen only the high-speed tracking signal.
        for terrain_type in ("plane", "rough_flat"):
            high_speed_ranges = self.commands.base_velocity.terrain_specific_high_speed_ranges[terrain_type]
            high_speed_ranges.lin_vel_x = (1.5, 4.0)
            high_speed_ranges.probability = 0.5
        self.rewards.high_speed_tracking_ratio.weight = 2.0
        self.rewards.high_speed_undershoot = RewTerm(
            func=mdp.high_speed_undershoot_penalty,
            weight=-1.5,
            params={
                "terrain_types": ("plane", "rough_flat"),
                "command_name": "base_velocity",
                "min_command_speed": 2.5,
            },
        )
        self.curriculum.high_speed_tracking = CurrTerm(
            func=mdp.high_speed_tracking_curriculum,
            params={
                "command_name": "base_velocity",
                "terrain_types": ("plane", "rough_flat"),
                "max_speeds": (3.0, 3.5, 4.0),
                "promotion_thresholds": (0.9, 0.9),
                "min_samples": 256,
                "eval_bandwidth": 0.25,
                "ema_alpha": 0.1,
            },
        )
        # Relax regularization only for explicit high-speed samples.
        self.rewards.energy.params.update({"high_speed_scale": 0.5, "command_name": "base_velocity"})
        self.rewards.dof_torques_l2.func = mdp.joint_torques_l2_high_speed_rescaled
        self.rewards.dof_torques_l2.params.update(
            {"high_speed_scale": 0.5, "command_name": "base_velocity"}
        )
        self.rewards.action_rate_l2.func = mdp.action_rate_l2_high_speed_rescaled
        self.rewards.action_rate_l2.params = {
            "high_speed_scale": 0.5,
            "command_name": "base_velocity",
        }

        # Remove the hurdle terrain and its dedicated scene configuration.
        self.scene.terrain.terrain_generator.sub_terrains.pop("hurdle", None)
        edge_cfg = self.scene.terrain.virtual_obstacles["edges"]
        edge_cfg.terrain_specific_cylinder_radius.pop("hurdle", None)
        self.scene.hurdle_ankle_volume_points = None

        # Remove hurdle-specific command and reset behavior.
        self.commands.base_velocity.terrain_specific_ranges.pop("hurdle", None)
        reset_params = self.events.reset_robot_states.params
        for param_name in (
            "terrain_specific_motion_files",
            "terrain_specific_prob_rsi",
            "terrain_specific_joint_vel_range",
            "terrain_specific_reset_height_range",
            "terrain_reset_frames",
            "terrain_specific_yaw",
        ):
            reset_params[param_name].pop("hurdle", None)
        self.events.push_robot.params["terrain_specific_velocity_range"].pop("hurdle", None)
        self.events.register_virtual_obstacles.params["sensor_cfgs"] = [
            sensor_cfg
            for sensor_cfg in self.events.register_virtual_obstacles.params["sensor_cfgs"]
            if sensor_cfg.name != "hurdle_ankle_volume_points"
        ]

        # Disable hurdle-only rewards and terminations, while retaining all other terrain terms.
        self.rewards.hurdle_ankle_volume_penetration = None
        self.terminations.hurdle_bad_orientation = None
        self.terminations.hurdle_illegal_contact = None
        self.rewards.track_lin_vel_x_linear.params["skip_terrain_types"] = _without_hurdle(
            self.rewards.track_lin_vel_x_linear.params["skip_terrain_types"]
        )
        self.rewards.feet_unhold.params["terrain_types"] = _without_hurdle(
            self.rewards.feet_unhold.params["terrain_types"]
        )
        self.rewards.feet_flat_ori.params["terrain_types"] = _without_hurdle(
            self.rewards.feet_flat_ori.params["terrain_types"]
        )
