# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

from __future__ import annotations

import gymnasium as gym
import torch
from isaaclab.assets.articulation import Articulation
from isaaclab.envs.common import VecEnvStepReturn
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat
from NoetixE1.managers import MyObservationManager
from NoetixE1.tasks.walkrun.noetix_e1.walkrun_amp_cfg import WalkRunFlatEnvCfg

from rsl_rl.utils.motion_loader import *


class ManagerBasedRLAmpEnv(ManagerBasedRLEnv, gym.Env):
    def __init__(self, cfg: WalkRunFlatEnvCfg, render_mode: str | None = None, **kwargs):
        self.motion_loader = None
        # initialize the base class to setup the scene.
        super().__init__(cfg=cfg, render_mode=render_mode)

        self.cfg: WalkRunFlatEnvCfg

        self.observation_manager = MyObservationManager(self.cfg.observations, self)
        self.robot: Articulation = self.scene["robot"]
        self.feet_both_contact_time = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        # Cache for terrain type (col_indices) computed in step function
        self._current_terrain_col_indices: torch.Tensor | None = None
        self._foothold_body_ids = torch.tensor(
            self.robot.find_bodies(
                ["l_leg_ankle_roll_link", "r_leg_ankle_roll_link"], preserve_order=True
            )[0],
            device=self.device,
            dtype=torch.long,
        )
        contact_sensor = self.scene.sensors["contact_sensor"]
        contact_body_names = list(contact_sensor.body_names)
        self._foothold_contact_body_ids = torch.tensor(
            [
                contact_body_names.index("l_leg_ankle_roll_link"),
                contact_body_names.index("r_leg_ankle_roll_link"),
            ],
            device=self.device,
            dtype=torch.long,
        )
        self._imagined_foothold_mu = torch.zeros(self.num_envs, 2, 2, device=self.device)
        self._imagined_foothold_sigma = torch.full((self.num_envs, 2), 0.15, device=self.device)
        self._foothold_prediction_valid = False
        self._foothold_decision_base_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._foothold_decision_base_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        self._foothold_decision_contact = torch.ones(
            self.num_envs, 2, device=self.device, dtype=torch.bool
        )
        # ---------------------- Initial Motion Loader ----------------------
        self.motion_loader_class: MotionLoaderE1 = eval(self.cfg.motion_loader_name)
        self.motion_loader: MotionLoaderE1 = self.motion_loader_class(
            device=self.device,
            time_between_frames=self.step_dt,
            reference_observation_horizon=self.cfg.reference_observation_horizon,
            num_preload_transitions=self.cfg.num_preload_transitions,
            motion_files=self.cfg.motion_files,
            sim_joint_names=self.robot.joint_names,
            sim_key_pos_names=self.robot.body_names,
        )

    """
    Properties
    """

    def visualize_motion(self, time, traj_idx):
        """
        Update the robot simulation state based on the AMP motion capture data at a given time.

        This function sets the joint positions and velocities, root position and orientation,
        and linear/angular velocities according to the AMP motion frame at the specified time,
        then steps the simulation and updates the scene.

        Args:
            time (float): The time (in seconds) at which to fetch the AMP motion frame.

        Returns:
            None
        """
        assert self.num_envs == 1

        visual_motion_frame = self.motion_loader.get_full_frame_at_time(traj_idx, time)
        device = self.device

        dof_pos = self.motion_loader_class.get_joint_pose(visual_motion_frame)[
            ..., self.motion_loader.joint_mapping["motion2sim"]
        ]
        dof_vel = self.motion_loader_class.get_joint_vel(visual_motion_frame)[
            ..., self.motion_loader.joint_mapping["motion2sim"]
        ]

        self.robot.write_joint_position_to_sim(dof_pos)
        self.robot.write_joint_velocity_to_sim(dof_vel)

        env_ids = torch.arange(self.num_envs, device=device)

        root_pos = self.motion_loader_class.get_root_pos(visual_motion_frame)

        quat_xyzw = self.motion_loader_class.get_root_rot(visual_motion_frame)
        quat_wxyz = torch.tensor(
            [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=torch.float, device=device
        )

        base_lin_vel = self.motion_loader_class.get_linear_vel(visual_motion_frame)
        base_ang_vel = self.motion_loader_class.get_angular_vel(visual_motion_frame)

        lin_vel = quat_apply(quat_wxyz, base_lin_vel)
        ang_vel = quat_apply(quat_wxyz, base_ang_vel)

        # root state: [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
        root_state = torch.zeros((self.num_envs, 13), device=device)
        root_state[:, 0:3] = torch.tile(root_pos.unsqueeze(0), (self.num_envs, 1))
        root_state[:, 3:7] = torch.tile(quat_wxyz.unsqueeze(0), (self.num_envs, 1))
        root_state[:, 7:10] = torch.tile(lin_vel.unsqueeze(0), (self.num_envs, 1))
        root_state[:, 10:13] = torch.tile(ang_vel.unsqueeze(0), (self.num_envs, 1))

        self.robot.write_root_state_to_sim(root_state, env_ids)
        self.sim.render()
        self.scene.update(dt=self.step_dt)

        key_pos = self.motion_loader_class.get_key_pos_local(visual_motion_frame)
        key_pos = key_pos.view(-1, 3)[self.motion_loader.key_pos_mapping["motion2sim"]] + root_pos
        return key_pos

    """
    Helper methods.
    """

    def set_imagined_footholds(self, mu: torch.Tensor, sigma: torch.Tensor) -> None:
        """Cache policy-side imagined contacts for this decision step."""
        if mu.shape != self._imagined_foothold_mu.shape or sigma.shape != self._imagined_foothold_sigma.shape:
            raise ValueError(
                f"Expected foothold mu/sigma shapes {self._imagined_foothold_mu.shape}/"
                f"{self._imagined_foothold_sigma.shape}, got {mu.shape}/{sigma.shape}."
            )
        self._imagined_foothold_mu.copy_(mu.detach().to(self.device))
        self._imagined_foothold_sigma.copy_(sigma.detach().to(self.device))
        self._foothold_prediction_valid = True

    def _foothold_contact_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        contact_sensor = self.scene.sensors["contact_sensor"]
        forces_history = contact_sensor.data.net_forces_w_history[
            :, :, self._foothold_contact_body_ids, :
        ]
        contact = torch.amax(torch.linalg.vector_norm(forces_history, dim=-1), dim=1) > 0.5
        current_forces = contact_sensor.data.net_forces_w[:, self._foothold_contact_body_ids, :]
        return contact, current_forces

    @staticmethod
    def _yaw_from_quat(quat_w: torch.Tensor) -> torch.Tensor:
        w, x, y, z = quat_w.unbind(dim=-1)
        return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def _compute_terrain_statistics(self):
        """Compute terrain statistics and add them to extras["log"].

        This method retrieves pre-computed statistics from the adaptive terrain state
        (computed in curriculums._update_terrain_state) to avoid redundant calculations.
        Statistics include:
        - Mean terrain level per terrain type
        - Max terrain level per terrain type
        - Number of environments per terrain type

        The statistics are added to self.extras["log"] with keys:
        - Terrain_Level_Mean/{terrain_type_name}
        - Terrain_Level_Max/{terrain_type_name}
        - Terrain_Num_Envs/{terrain_type_name}
        """
        if "log" not in self.extras:
            self.extras["log"] = {}

        # Get terrain information if available
        if not (
            hasattr(self.scene, "terrain")
            and hasattr(self.scene.terrain, "terrain_levels")
            and hasattr(self.scene.terrain, "terrain_types")
        ):
            return

        # Try to get pre-computed statistics from adaptive terrain state
        from NoetixE1.tasks.walkrun.noetix_e1.mdp.curriculums import _adaptive_terrain_state

        env_id_str = str(id(self))
        if env_id_str in _adaptive_terrain_state:
            state = _adaptive_terrain_state[env_id_str]
            # Use pre-computed statistics from _update_terrain_state
            for terrain_type_name in state["terrain_type_names"]:
                self.extras["log"][f"Terrain_Level_Mean/{terrain_type_name}"] = state["terrain_mean_level"].get(
                    terrain_type_name, 0.0
                )
                self.extras["log"][f"Terrain_Level_Max/{terrain_type_name}"] = state["terrain_max_level"].get(
                    terrain_type_name, 0.0
                )
                self.extras["log"][f"Terrain_Num_Envs/{terrain_type_name}"] = state["terrain_env_nums"].get(
                    terrain_type_name, 0.0
                )
            return

        # Fallback: compute statistics manually if adaptive state is not available
        terrain = self.scene.terrain
        terrain_levels = terrain.terrain_levels  # Shape: (num_envs,)
        terrain_types = terrain.terrain_types  # Shape: (num_envs,) - column indices

        # Build column-to-terrain-type mapping
        col_to_terrain_type_map = self._build_col_to_terrain_type_map(terrain)
        if col_to_terrain_type_map is None:
            return

        # Group environments by terrain type name and compute statistics using tensor operations
        terrain_levels_float = terrain_levels.float()

        # Get unique terrain type names and create a reverse mapping (terrain_type_name -> index)
        terrain_type_names = list(set(col_to_terrain_type_map.values()))
        terrain_type_name_to_idx = {name: idx for idx, name in enumerate(terrain_type_names)}

        # Create a mapping tensor: col_idx -> terrain_type_idx
        # Find max col_idx to determine tensor size
        max_col_idx = max(col_to_terrain_type_map.keys()) if col_to_terrain_type_map else -1
        if max_col_idx < 0:
            return

        # Create mapping tensor: col_idx -> terrain_type_idx (-1 for invalid col_idx)
        col_to_type_idx = torch.full((max_col_idx + 1,), -1, dtype=torch.long, device=self.device)
        for col_idx, terrain_type_name in col_to_terrain_type_map.items():
            terrain_type_idx = terrain_type_name_to_idx[terrain_type_name]
            col_to_type_idx[col_idx] = terrain_type_idx

        # Map terrain_types to terrain_type_indices using tensor indexing
        # Filter valid column indices (within range and in mapping)
        valid_col_mask = (terrain_types >= 0) & (terrain_types <= max_col_idx)
        terrain_type_indices = torch.full_like(terrain_types, -1, dtype=torch.long)
        if valid_col_mask.any():
            terrain_type_indices[valid_col_mask] = col_to_type_idx[terrain_types[valid_col_mask]]

        # Filter out invalid mappings (-1 means col_idx not in col_to_terrain_type_map)
        valid_mask = terrain_type_indices >= 0

        # Compute statistics for each terrain type using tensor operations
        for terrain_type_name, terrain_type_idx in terrain_type_name_to_idx.items():
            # Create mask for environments of this terrain type
            type_mask = (terrain_type_indices == terrain_type_idx) & valid_mask

            if type_mask.any():
                # Extract terrain levels for this terrain type
                type_levels = terrain_levels_float[type_mask]

                # Compute statistics
                self.extras["log"][f"Terrain_Level_Mean/{terrain_type_name}"] = torch.mean(type_levels).item()
                self.extras["log"][f"Terrain_Level_Max/{terrain_type_name}"] = torch.max(type_levels).item()
                self.extras["log"][f"Terrain_Num_Envs/{terrain_type_name}"] = float(type_mask.sum().item())

    def _compute_high_speed_statistics(self):
        """Log terrain-specific command tracking for the explicit high-speed sampling mode."""
        command_term = self.command_manager.get_term("base_velocity")
        high_speed_ranges = getattr(command_term.cfg, "terrain_specific_high_speed_ranges", None)
        high_speed_mask = getattr(command_term, "high_speed_mask", None)
        if not high_speed_ranges or high_speed_mask is None:
            return

        from NoetixE1.tasks.walkrun.noetix_e1.mdp.curriculums import _check_terrain_types_from_positions

        command = command_term.command
        gate_fn = getattr(command_term, "high_speed_reward_gate", None)
        reward_gate = gate_fn() if gate_fn is not None else torch.ones(self.num_envs, device=self.device)
        actual_velocity_yaw = quat_apply_inverse(
            yaw_quat(self.robot.data.root_quat_w), self.robot.data.root_lin_vel_w
        )
        col_indices = self._current_terrain_col_indices
        robot_positions_w = self.robot.data.root_pos_w

        for terrain_type, ranges in high_speed_ranges.items():
            type_mask = _check_terrain_types_from_positions(
                self,
                robot_positions_w,
                (terrain_type,),
                col_indices=col_indices,
            )
            selected = type_mask & high_speed_mask
            effective = (
                selected
                & (command[:, 0] >= float(ranges.lin_vel_x[0]))
                & (command[:, 0] <= float(ranges.lin_vel_x[1]))
                & (torch.abs(command[:, 1]) <= 1e-6)
            )

            type_count = torch.clamp(type_mask.float().sum(), min=1.0)
            selected_count = torch.clamp(selected.float().sum(), min=1.0)
            selected_float = selected.float()
            mean_command_x = torch.sum(command[:, 0] * selected_float) / selected_count
            mean_actual_vx = torch.sum(actual_velocity_yaw[:, 0] * selected_float) / selected_count
            mean_forward_deficit = torch.sum(
                torch.clamp(command[:, 0] - actual_velocity_yaw[:, 0], min=0.0) * selected_float
            ) / selected_count
            mean_reward_gate = torch.sum(reward_gate * selected_float) / selected_count

            self.extras["log"][f"Speed/High_Selected_Rate/{terrain_type}"] = (
                selected.float().sum() / type_count
            ).detach()
            self.extras["log"][f"Speed/High_Effective_Rate/{terrain_type}"] = (
                effective.float().sum() / type_count
            ).detach()
            self.extras["log"][f"Speed/Command_X/{terrain_type}/high"] = mean_command_x.detach()
            self.extras["log"][f"Speed/Actual_Vx/{terrain_type}/high"] = mean_actual_vx.detach()
            self.extras["log"][f"Speed/Forward_Deficit/{terrain_type}/high"] = mean_forward_deficit.detach()
            self.extras["log"][f"Speed/Reward_Gate/{terrain_type}/high"] = mean_reward_gate.detach()

    def _build_col_to_terrain_type_map(self, terrain) -> dict[int, str] | None:
        """Build a mapping from column index to terrain type name.

        This method replicates the logic from TerrainGenerator._generate_curriculum_terrains
        to determine which terrain type each column corresponds to based on the proportion
        configuration.

        Args:
            terrain: The TerrainImporter instance.

        Returns:
            A dictionary mapping column indices to terrain type names, or None if the mapping
            cannot be determined.
        """
        if not (hasattr(terrain, "cfg") and hasattr(terrain.cfg, "terrain_generator") and terrain.cfg.terrain_generator is not None):
            return None

        terrain_gen_cfg = terrain.cfg.terrain_generator
        if not hasattr(terrain_gen_cfg, "sub_terrains"):
            return None

        terrain_type_names = list(terrain_gen_cfg.sub_terrains.keys())
        if len(terrain_type_names) == 0:
            return None

        col_to_terrain_type_map = {}

        # Build mapping based on curriculum or random mode
        if terrain_gen_cfg.curriculum and hasattr(terrain_gen_cfg, "num_cols"):
            # Curriculum mode: use proportion-based mapping
            import numpy as np

            proportions = np.array([sub_cfg.proportion for sub_cfg in terrain_gen_cfg.sub_terrains.values()])
            proportions /= np.sum(proportions)
            # Find the sub-terrain index for each column (same logic as TerrainGenerator)
            for col_idx in range(terrain_gen_cfg.num_cols):
                sub_index = np.min(np.where(col_idx / terrain_gen_cfg.num_cols + 0.001 < np.cumsum(proportions))[0])
                col_to_terrain_type_map[col_idx] = terrain_type_names[sub_index]
        elif hasattr(terrain_gen_cfg, "num_cols"):
            # Non-curriculum mode: assume uniform distribution
            # Note: This is a fallback and may not be accurate for random mode
            num_cols = terrain_gen_cfg.num_cols
            num_terrain_types = len(terrain_type_names)
            for col_idx in range(num_cols):
                terrain_type_idx = col_idx % num_terrain_types
                col_to_terrain_type_map[col_idx] = terrain_type_names[terrain_type_idx]
        else:
            return None

        return col_to_terrain_type_map

    def _compute_terrain_termination_statistics(self, reset_env_ids: torch.Tensor):
        """Compute termination statistics per terrain type for current step and add them to extras["log"].

        This method calculates the percentage of each termination type for each terrain type
        based on the current reset environments (not cumulative).

        Args:
            reset_env_ids: Environment IDs that are being reset in the current step.

        The statistics are added to self.extras["log"] with keys:
        - Terrain_Termination/{terrain_type_name}/{term_name}
        """
        if "log" not in self.extras:
            self.extras["log"] = {}

        if len(reset_env_ids) == 0:
            return

        # Get terrain information if available
        if not (
            hasattr(self.scene, "terrain")
            and hasattr(self.scene.terrain, "terrain_types")
        ):
            return

        # Get terrain types for reset environments (batch processing)
        terrain_types = self.scene.terrain.terrain_types[reset_env_ids]  # Shape: (len(reset_env_ids),)

        # Build column-to-terrain-type mapping
        col_to_terrain_type_map = self._build_col_to_terrain_type_map(self.scene.terrain)
        if col_to_terrain_type_map is None:
            return

        # Get termination term names and their values for reset environments (batch processing)
        term_names = self.termination_manager.active_terms
        num_terms = len(term_names)
        term_dones = self.termination_manager._term_dones[reset_env_ids]  # Shape: (len(reset_env_ids), num_terms)

        # Create mapping tensor: col_idx -> terrain_type_name index
        max_col_idx = max(col_to_terrain_type_map.keys()) if col_to_terrain_type_map else -1
        if max_col_idx < 0:
            return

        # Get unique terrain type names
        terrain_type_names = list(set(col_to_terrain_type_map.values()))
        terrain_type_name_to_idx = {name: idx for idx, name in enumerate(terrain_type_names)}

        # Create mapping tensor: col_idx -> terrain_type_idx (-1 for invalid col_idx)
        col_to_type_idx = torch.full((max_col_idx + 1,), -1, dtype=torch.long, device=self.device)
        for col_idx, terrain_type_name in col_to_terrain_type_map.items():
            terrain_type_idx = terrain_type_name_to_idx[terrain_type_name]
            col_to_type_idx[col_idx] = terrain_type_idx

        # Map terrain_types to terrain_type_indices using tensor indexing (batch processing)
        valid_col_mask = (terrain_types >= 0) & (terrain_types <= max_col_idx)
        terrain_type_indices = torch.full_like(terrain_types, -1, dtype=torch.long)
        if valid_col_mask.any():
            terrain_type_indices[valid_col_mask] = col_to_type_idx[terrain_types[valid_col_mask]]

        # Filter out invalid mappings
        valid_mask = terrain_type_indices >= 0

        # Find which termination term triggered for each environment (batch processing)
        # For each environment, find the first True termination term
        # term_dones shape: (len(reset_env_ids), num_terms)
        # We want to find the first True value in each row
        term_indices = torch.argmax(term_dones.float(), dim=1)  # Shape: (len(reset_env_ids),)
        # If no termination term is True, argmax will return 0, so we need to check if any term is True
        has_termination = term_dones.any(dim=1)  # Shape: (len(reset_env_ids),)
        # Set term_indices to -1 for environments without termination
        term_indices = torch.where(has_termination, term_indices, torch.full_like(term_indices, -1))

        # Compute statistics for each terrain type using batch operations
        # Store in hierarchical structure: Terrain_Termination/{term_name}/{terrain_type_name} = percentage
        # This matches the structure of Terrain_Level_Mean/{terrain_type_name}
        for terrain_type_name, terrain_type_idx in terrain_type_name_to_idx.items():
            # Create mask for environments of this terrain type with valid termination
            type_mask = (terrain_type_indices == terrain_type_idx) & valid_mask & has_termination

            if not type_mask.any():
                # If no environments of this terrain type, set all termination percentages to 0
                for term_name in term_names:
                    self.extras["log"][f"Termination_{term_name}/{terrain_type_name}"] = 0.0
                continue

            # Get termination term indices for this terrain type
            type_term_indices = term_indices[type_mask]  # Shape: (num_envs_of_this_type,)

            # Count occurrences of each termination term using bincount
            term_counts = torch.bincount(type_term_indices, minlength=num_terms).float()  # Shape: (num_terms,)
            total_count = type_mask.sum().float().item()

            if total_count == 0:
                # If total count is 0, set all termination percentages to 0
                for term_name in term_names:
                    self.extras["log"][f"Termination_{term_name}/{terrain_type_name}"] = 0.0
                continue

            # Calculate percentage for each termination term
            for term_idx, term_name in enumerate(term_names):
                count = term_counts[term_idx].item()
                percentage = (count / total_count)
                self.extras["log"][f"Termination_{term_name}/{terrain_type_name}"] = percentage

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Execute one time-step of the environment's dynamics and reset terminated environments.

        Unlike the :class:`ManagerBasedEnv.step` class, the function performs the following operations:

        1. Process the actions.
        2. Perform physics stepping.
        3. Perform rendering if gui is enabled.
        4. Update the environment counters and compute the rewards and terminations.
        5. Reset the environments that terminated.
        6. Compute the observations.
        7. Return the observations, rewards, resets and extras.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """
        # Cache the exact decision frame used by the imagined-contact model.
        self._foothold_decision_base_pos_w.copy_(self.robot.data.root_pos_w)
        self._foothold_decision_base_quat_w.copy_(self.robot.data.root_quat_w)
        decision_contact, _ = self._foothold_contact_state()
        self._foothold_decision_contact.copy_(decision_contact)
        contact_sensor = self.scene.sensors["contact_sensor"]
        decision_air_time = contact_sensor.data.current_air_time[
            :, self._foothold_contact_body_ids
        ].clone()

        # process actions
        self.action_manager.process_action(action.to(self.device))

        self.recorder_manager.record_pre_step()

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self.action_manager.apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # -- compute terrain type (col_indices) for all robots based on current positions (batch processing)
        # This is computed once per step and cached for use by reward/termination/command functions
        from NoetixE1.tasks.walkrun.noetix_e1.mdp.curriculums import (
            _check_terrain_types_from_positions,
            _get_terrain_type_from_position,
        )
        robot_positions_w = self.robot.data.root_pos_w  # (num_envs, 3)
        self._current_terrain_col_indices = _get_terrain_type_from_position(self, robot_positions_w)

        # Detect first valid contact before terminated environments are reset.
        post_contact, post_forces = self._foothold_contact_state()
        touchdown = post_contact & ~decision_contact
        stumble = torch.linalg.vector_norm(post_forces[..., :2], dim=-1) > (
            4.0 * torch.abs(post_forces[..., 2])
        )
        touchdown_valid = touchdown & (decision_air_time >= 0.04) & ~stumble
        touchdown_xy_w = self.robot.data.body_pos_w[:, self._foothold_body_ids, :2].clone()
        stair_direction = torch.full(
            (self.num_envs,), -1, device=self.device, dtype=torch.int8
        )
        up_mask = _check_terrain_types_from_positions(
            self,
            None,
            ("stairs_up", "spiral_stairs_up"),
            col_indices=self._current_terrain_col_indices,
        )
        down_mask = _check_terrain_types_from_positions(
            self,
            None,
            ("stairs_down", "spiral_stairs_down"),
            col_indices=self._current_terrain_col_indices,
        )
        stair_direction[up_mask] = 0
        stair_direction[down_mask] = 1
        self.extras["foothold_events"] = {
            "decision_swing": ~decision_contact,
            "decision_base_xy": self._foothold_decision_base_pos_w[:, :2].clone(),
            "decision_base_yaw": self._yaw_from_quat(self._foothold_decision_base_quat_w).clone(),
            "touchdown": touchdown,
            "touchdown_valid": touchdown_valid,
            "touchdown_xy_w": touchdown_xy_w,
            "stair_direction": stair_direction,
        }

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)
        # -- check terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        # -- reward computation
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
        command_term = self.command_manager.get_term("base_velocity")
        gate_fn = getattr(command_term, "high_speed_reward_gate", None)
        self.extras["high_speed_reward_gate"] = (
            gate_fn().clone() if gate_fn is not None else torch.ones(self.num_envs, device=self.device)
        )

        if len(self.recorder_manager.active_terms) > 0:
            # update observations for recording if needed
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        terminal_env_ids = reset_env_ids.clone()
        terminal_actor_obs = self.obs_buf["policy"][terminal_env_ids].clone()
        terminal_critic_obs = self.obs_buf["critic"][terminal_env_ids].clone()
        terminal_amp_obs = self.obs_buf["discriminator"][terminal_env_ids].clone()

        if len(reset_env_ids) > 0:
            # trigger recorder terms for pre-reset calls
            self.recorder_manager.record_pre_reset(reset_env_ids)

            self._reset_idx(reset_env_ids)
            # reset buffer
            self.feet_both_contact_time[reset_env_ids] = 0.0
            # update articulation kinematics
            self.scene.write_data_to_sim()
            self.sim.forward()

            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

            # trigger recorder terms for post-reset calls
            self.recorder_manager.record_post_reset(reset_env_ids)

        # -- update command
        self.command_manager.compute(dt=self.step_dt)
        # -- step interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        # -- step mode events (called every step)
        if "step" in self.event_manager.available_modes:
            self.event_manager.apply(mode="step")
        # -- compute observations
        # note: done after reset to get the correct observations for reset envs
        self.obs_buf = self.observation_manager.compute(update_history=True)

        self.extras["terminal_observations"] = {
            "env_ids": terminal_env_ids,
            "policy": terminal_actor_obs,
            "critic": terminal_critic_obs,
            "amp": terminal_amp_obs,
        }

        # -- Compute terrain statistics and add to extras["log"]
        self._compute_terrain_statistics()
        self._compute_high_speed_statistics()
        # -- Compute terrain-specific termination statistics for current step
        if len(reset_env_ids) > 0:
            self._compute_terrain_termination_statistics(reset_env_ids)

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
