# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class UniformVelocityWithZeroCommand(CommandTerm):
    r"""Command generator that generates a velocity command in SE(2) from uniform distribution.

    The command comprises of a linear velocity in x and y direction and an angular velocity around
    the z-axis. It is given in the robot's base frame.
    """

    cfg: UniformVelocityWithZeroCommandCfg
    """The command generator configuration."""

    def __init__(self, cfg: UniformVelocityWithZeroCommandCfg, env: ManagerBasedRLEnv):
        """Initializes the command generator.

        Args:
            cfg: The command generator configuration.
            env: The environment.
        """
        super().__init__(cfg, env)
        # obtain the robot asset
        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]

        # crete buffers to store the command
        # -- command: x vel, y vel, yaw vel
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.is_standing_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # -- metrics
        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

        # create buffers for zero commands envs
        self.is_zero_vel_x_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_zero_vel_y_env = torch.zeros_like(self.is_zero_vel_x_env)
        self.is_zero_vel_yaw_env = torch.zeros_like(self.is_zero_vel_x_env)

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "NormalVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
        return self.vel_command_b

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # time for which the command was executed
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        # logs data
        self.metrics["error_vel_xy"] += (
            torch.norm(self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2], dim=-1) / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2]) / max_command_step
        )

    def _resample_command(self, env_ids):
        # sample velocity commands
        r = torch.empty(len(env_ids), device=self.device)
        
        # Initialize velocity commands and zero probabilities with default values
        vel_x = torch.zeros(len(env_ids), device=self.device)
        vel_y = torch.zeros(len(env_ids), device=self.device)
        vel_yaw = torch.zeros(len(env_ids), device=self.device)
        zero_prob_x = torch.full((len(env_ids),), self.cfg.ranges.zero_prob[0], device=self.device)
        zero_prob_y = torch.full((len(env_ids),), self.cfg.ranges.zero_prob[1], device=self.device)
        zero_prob_yaw = torch.full((len(env_ids),), self.cfg.ranges.zero_prob[2], device=self.device)
        
        # Track which environments have been processed with terrain-specific ranges
        processed_mask = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
        
        # Get terrain types for environments if terrain-specific ranges are configured
        if self.cfg.terrain_specific_ranges is not None and len(self.cfg.terrain_specific_ranges) > 0:
            from NoetixE1.tasks.walkrun.noetix_e1.mdp.curriculums import _check_terrain_types_from_positions
            
            # Use cached col_indices from step function if available, otherwise compute from positions
            col_indices_all = getattr(self._env.unwrapped, "_current_terrain_col_indices", None)
            if col_indices_all is not None:
                # Extract col_indices for the environments being resampled
                col_indices_subset = col_indices_all[env_ids]  # (len(env_ids),)
            else:
                # Fallback: compute from positions if cache is not available
                col_indices_subset = None
                asset = self._env.scene[self.cfg.asset_name]
                robot_positions_w = asset.data.root_pos_w[env_ids]  # (len(env_ids), 3)
            
            # Process each terrain type with specific ranges
            for terrain_type, terrain_ranges in self.cfg.terrain_specific_ranges.items():
                # Check if robots are on this terrain type (batch processing)
                if col_indices_subset is not None:
                    # Use cached col_indices
                    is_this_terrain = _check_terrain_types_from_positions(
                        self._env, None, (terrain_type,), col_indices=col_indices_subset
                    )
                else:
                    # Fallback: compute from positions
                    is_this_terrain = _check_terrain_types_from_positions(
                        self._env, robot_positions_w, (terrain_type,)
                    )
                
                if is_this_terrain.any():
                    num_matching_envs = is_this_terrain.sum().item()
                    
                    # Sample velocities for this terrain type
                    r_terrain = torch.empty(num_matching_envs, device=self.device)
                    vel_x[is_this_terrain] = r_terrain.uniform_(*terrain_ranges.lin_vel_x)
                    vel_y[is_this_terrain] = r_terrain.uniform_(*terrain_ranges.lin_vel_y)
                    vel_yaw[is_this_terrain] = r_terrain.uniform_(*terrain_ranges.ang_vel_z)
                    
                    # Set zero probabilities for this terrain type
                    zero_prob_x[is_this_terrain] = terrain_ranges.zero_prob[0]
                    zero_prob_y[is_this_terrain] = terrain_ranges.zero_prob[1]
                    zero_prob_yaw[is_this_terrain] = terrain_ranges.zero_prob[2]
                    
                    processed_mask = processed_mask | is_this_terrain
        
        # Fill remaining environments with default ranges
        not_processed = ~processed_mask
        if not_processed.any():
            r_default = torch.empty(not_processed.sum().item(), device=self.device)
            vel_x[not_processed] = r_default.uniform_(*self.cfg.ranges.lin_vel_x)
            vel_y[not_processed] = r_default.uniform_(*self.cfg.ranges.lin_vel_y)
            vel_yaw[not_processed] = r_default.uniform_(*self.cfg.ranges.ang_vel_z)
        
        # Set velocity commands
        self.vel_command_b[env_ids, 0] = vel_x
        self.vel_command_b[env_ids, 1] = vel_y
        self.vel_command_b[env_ids, 2] = vel_yaw

        # update element wise zero velocity command
        self.is_zero_vel_x_env[env_ids] = r.uniform_(0.0, 1.0) <= zero_prob_x
        self.is_zero_vel_y_env[env_ids] = r.uniform_(0.0, 1.0) <= zero_prob_y
        self.is_zero_vel_yaw_env[env_ids] = r.uniform_(0.0, 1.0) <= zero_prob_yaw

        # update standing envs
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    def _update_command(self):
        """Sets velocity command to zero for standing envs."""
        # Enforce standing (i.e., zero velocity command) for standing envs
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()  # TODO check if conversion is needed
        self.vel_command_b[standing_env_ids, :] = 0.0

        # Enforce zero velocity for individual elements
        # TODO: check if conversion is needed
        zero_vel_x_env_ids = self.is_zero_vel_x_env.nonzero(as_tuple=False).flatten()
        zero_vel_y_env_ids = self.is_zero_vel_y_env.nonzero(as_tuple=False).flatten()
        zero_vel_yaw_env_ids = self.is_zero_vel_yaw_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[zero_vel_x_env_ids, 0] = 0.0
        self.vel_command_b[zero_vel_y_env_ids, 1] = 0.0
        self.vel_command_b[zero_vel_yaw_env_ids, 2] = 0.0

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first time
            if not hasattr(self, "goal_vel_visualizer"):
                # -- goal
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                # -- current
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            # set their visibility to true
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # get marker location
        # -- base state
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        # -- resolve the scales and quaternions
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.command[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        # display markers
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    """
    Internal helpers.
    """

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        # convert everything back from base to world frame
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat


class TurnSequenceVelocityCommand(UniformVelocityWithZeroCommand):
    """Add a one-shot turn-then-hold sequence on selected terrains."""

    cfg: TurnSequenceVelocityCommandCfg

    def __init__(self, cfg: TurnSequenceVelocityCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        for name, limits in (
            ("turn_duration_range", cfg.turn_duration_range),
            ("yaw_speed_range", cfg.yaw_speed_range),
        ):
            lo, hi = map(float, limits)
            if not (0.0 < lo <= hi):
                raise ValueError(f"Invalid {name}: {limits}")
        if not (0.0 <= cfg.turn_sequence_probability <= 1.0):
            raise ValueError(f"Invalid turn_sequence_probability: {cfg.turn_sequence_probability}")

        self.is_turn_sequence_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_turn_sequence_turning = torch.zeros_like(self.is_turn_sequence_env)
        self.is_turn_sequence_holding = torch.zeros_like(self.is_turn_sequence_env)
        self.turn_sequence_time_left = torch.zeros(self.num_envs, device=self.device)
        self.turn_sequence_yaw_command = torch.zeros(self.num_envs, device=self.device)

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self.is_turn_sequence_env[env_ids] = False
        self.is_turn_sequence_turning[env_ids] = False
        self.is_turn_sequence_holding[env_ids] = False
        self.turn_sequence_time_left[env_ids] = 0.0
        self.turn_sequence_yaw_command[env_ids] = 0.0
        if not self.cfg.turn_terrain_types:
            return

        from NoetixE1.tasks.walkrun.noetix_e1.mdp.curriculums import _check_terrain_types_from_positions

        col_indices = getattr(self._env.unwrapped, "_current_terrain_col_indices", None)
        if col_indices is None:
            positions = self.robot.data.root_pos_w[env_ids]
            is_turn_terrain = _check_terrain_types_from_positions(
                self._env, positions, self.cfg.turn_terrain_types
            )
        else:
            is_turn_terrain = _check_terrain_types_from_positions(
                self._env, None, self.cfg.turn_terrain_types, col_indices=col_indices[env_ids]
            )

        # Select once at episode start. Periodic resampling must not accumulate more turn environments.
        episode_start = self.command_counter[env_ids] == 0
        selected = torch.rand(env_ids.numel(), device=self.device) < float(self.cfg.turn_sequence_probability)
        turn_env_ids = env_ids[is_turn_terrain & episode_start & selected]
        if turn_env_ids.numel() == 0:
            return
        self.is_turn_sequence_env[turn_env_ids] = True
        self.is_turn_sequence_turning[turn_env_ids] = True
        self.turn_sequence_time_left[turn_env_ids] = self._sample_phase_duration(
            turn_env_ids, self.cfg.turn_duration_range
        )
        self.turn_sequence_yaw_command[turn_env_ids] = self._sample_yaw_speed(turn_env_ids)
        self.vel_command_b[turn_env_ids, :] = 0.0
        self.vel_command_b[turn_env_ids, 2] = self.turn_sequence_yaw_command[turn_env_ids]
        self.time_left[turn_env_ids] = float("inf")

    def _sample_phase_duration(self, env_ids: torch.Tensor, limits: tuple[float, float]) -> torch.Tensor:
        lo, hi = map(float, limits)
        return lo + torch.rand(env_ids.numel(), device=self.device) * (hi - lo)

    def _sample_yaw_speed(self, env_ids: torch.Tensor) -> torch.Tensor:
        magnitude = self._sample_phase_duration(env_ids, self.cfg.yaw_speed_range)
        direction = torch.where(
            torch.rand(env_ids.numel(), device=self.device) < 0.5,
            -torch.ones_like(magnitude),
            torch.ones_like(magnitude),
        )
        return magnitude * direction

    def _update_command(self):
        super()._update_command()
        turn_env_ids = self.is_turn_sequence_env.nonzero(as_tuple=False).flatten()
        if turn_env_ids.numel() == 0:
            return

        turning = turn_env_ids[self.is_turn_sequence_turning[turn_env_ids]]
        self.turn_sequence_time_left[turning] -= self._env.step_dt
        finished_turning = turning[self.turn_sequence_time_left[turning] <= 0.0]
        if finished_turning.numel() > 0:
            self.is_turn_sequence_turning[finished_turning] = False
            self.is_turn_sequence_holding[finished_turning] = True
            self.turn_sequence_time_left[finished_turning] = 0.0

        self.vel_command_b[turn_env_ids, :] = 0.0
        turning = turn_env_ids[self.is_turn_sequence_turning[turn_env_ids]]
        self.vel_command_b[turning, 2] = self.turn_sequence_yaw_command[turning]


class KeyboardVelocityCommand(UniformVelocityWithZeroCommand):
    """Command generator that exposes manual overrides for base velocity commands."""

    def __init__(self, cfg: UniformVelocityWithZeroCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.manual_command = torch.zeros_like(self.vel_command_b)

    def set_manual_command(self, command: torch.Tensor):
        """Set the manual command (shape (num_envs, 3) or (1, 3))."""
        if command.ndim == 1:
            command = command.unsqueeze(0)
        if command.shape[0] == 1:
            command = command.repeat(self.num_envs, 1)
        if command.shape != self.manual_command.shape:
            raise ValueError(
                f"Manual command shape {command.shape} does not match ({self.num_envs}, 3)."
            )
        self.manual_command.copy_(command.to(self.device))
        self._clamp_manual_command()
        self._sync_manual_command()

    def adjust_manual_command_component(self, index: int, delta: float):
        """Adjust a single component of the manual command by ``delta``."""
        if index < 0 or index >= self.manual_command.shape[1]:
            raise IndexError(f"Invalid command index {index}. Expected 0, 1, or 2.")
        self.manual_command[:, index].add_(delta)
        self._clamp_manual_command()
        self._sync_manual_command()

    def reset_manual_command(self):
        """Reset the manual command to zero."""
        self.manual_command.zero_()
        self._sync_manual_command()

    def get_manual_command(self) -> torch.Tensor:
        """Return a copy of the manual command tensor."""
        return self.manual_command.clone()

    def _clamp_manual_command(self):
        ranges = self.cfg.ranges
        self.manual_command[:, 0].clamp_(ranges.lin_vel_x[0], ranges.lin_vel_x[1])
        self.manual_command[:, 1].clamp_(ranges.lin_vel_y[0], ranges.lin_vel_y[1])
        self.manual_command[:, 2].clamp_(ranges.ang_vel_z[0], ranges.ang_vel_z[1])

    def _sync_manual_command(self):
        self.vel_command_b.copy_(self.manual_command)

    def _resample_command(self, env_ids):
        self.vel_command_b[env_ids] = self.manual_command[env_ids]
        self.time_left[env_ids] = float("inf")

    def _update_command(self):
        self.vel_command_b.copy_(self.manual_command)

@configclass
class UniformVelocityWithZeroCommandCfg(CommandTermCfg):
    """Configuration for the normal velocity command generator."""

    class_type: type = UniformVelocityWithZeroCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    rel_standing_envs: float = 0.0
    """The sampled probability of environments that should be standing still. Defaults to 0.0."""

    @configclass
    class Ranges:
        """Uniform distribution ranges for the velocity commands."""

        lin_vel_x: tuple[float, float] = MISSING
        """Range for the linear-x velocity command (in m/s)."""

        lin_vel_y: tuple[float, float] = MISSING
        """Range for the linear-y velocity command (in m/s)."""

        ang_vel_z: tuple[float, float] = MISSING
        """Range for the angular-z velocity command (in rad/s)."""

        zero_prob: tuple[float, float, float] = MISSING
        """Probability of zero velocity for the normal distribution.

        The tuple contains the probability of zero linear-x, linear-y, and angular-z velocity.
        """

    ranges: Ranges = MISSING
    """Distribution ranges for the velocity commands."""

    terrain_specific_ranges: dict[str, Ranges] | None = None
    """Terrain-specific velocity ranges and zero probabilities.
    
    Dictionary mapping terrain type names to Ranges configurations.
    Example: {"cross_stepping_stones": Ranges(lin_vel_x=(0.0, 1.0), ...)}
    If None, uses the default ranges for all terrains.
    """

    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    """The configuration for the goal velocity visualization marker. Defaults to GREEN_ARROW_X_MARKER_CFG."""

    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
    """The configuration for the current velocity visualization marker. Defaults to BLUE_ARROW_X_MARKER_CFG."""

    # Set the scale of the visualization markers to (0.5, 0.5, 0.5)
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)


@configclass
class TurnSequenceVelocityCommandCfg(UniformVelocityWithZeroCommandCfg):
    """Configuration for an isolated terrain-specific turn-then-hold sequence."""

    class_type: type = TurnSequenceVelocityCommand
    turn_terrain_types: tuple[str, ...] = ()
    turn_sequence_probability: float = 0.0
    turn_duration_range: tuple[float, float] = (1.0, 2.5)
    yaw_speed_range: tuple[float, float] = (0.2, 0.5)
