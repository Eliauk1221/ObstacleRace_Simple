# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

"""Termination functions for the MDP."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch
from isaaclab.assets import Articulation, RigidObject

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.managers import ManagerTermBase, ManagerTermBaseCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor

def fall_down_in_some_terrain(
    env: ManagerBasedRLEnv,
    minimum_height: float = 0.5,
    terrain_types: list[str] | tuple[str, ...] | None = None,
    asset_cfg: str = "robot",
) -> torch.Tensor:
    """Terminate if robot falls below minimum height in specific terrain types.

    This function checks if the robot's root height is below a minimum threshold,
    but only terminates if the robot is currently on one of the specified terrain types.

    Args:
        env: The learning environment.
        minimum_height: Minimum height threshold for the root link (in m). Defaults to 0.5.
        terrain_types: List of terrain type names to check. If None or empty, checks all terrains.
                      Defaults to None.
        asset_cfg: Name of the robot asset. Defaults to "robot".

    Returns:
        A boolean tensor of shape (num_envs,) indicating which environments should terminate.
        True means the robot has fallen below the minimum height in the specified terrain types.
    """
    # Get robot asset
    asset: Articulation = env.scene[asset_cfg]
    
    # Get root height (z-coordinate of root link position)
    root_height = asset.data.root_link_pos_w[:, 2]  # (num_envs,)
    
    # Check if height is below minimum threshold
    is_below_minimum = root_height < minimum_height
    
    # If no terrain types specified, terminate based on height alone
    if not terrain_types or len(terrain_types) == 0:
        return is_below_minimum
    
    # Get terrain type for each environment (batch processing)
    # Use cached col_indices from step function if available, otherwise compute from positions
    from NoetixE1.tasks.walkrun.noetix_e1.mdp.curriculums import _check_terrain_types_from_positions
    
    col_indices = getattr(env.unwrapped, "_current_terrain_col_indices", None)
    if col_indices is None:
        # Fallback: compute from positions if cache is not available
        robot_positions_w = asset.data.root_link_pos_w  # (num_envs, 3)
        is_desired_terrain = _check_terrain_types_from_positions(env, robot_positions_w, terrain_types)
    else:
        # Use cached col_indices (batch processing)
        is_desired_terrain = _check_terrain_types_from_positions(env, None, terrain_types, col_indices=col_indices)
    
    # Terminate if height is below minimum AND on desired terrain
    return is_below_minimum & is_desired_terrain


def terrain_out_of_bounds(
    env: ManagerBasedRLEnv,
    asset_cfg: str = "robot",
    distance_buffer: float = 3.0,
    print_reason: bool = False,
) -> torch.Tensor:
    """Terminate when the actor move too close to the edge of the terrain.

    If the actor moves too close to the edge of the terrain, the termination is activated. The distance
    to the edge of the terrain is calculated based on the size of the terrain and the distance buffer.
    """
    if env.scene.cfg.terrain.terrain_type == "plane":
        return torch.zeros(
            (env.num_envs,), device=env.device, dtype=torch.bool
        )  # we have infinite terrain because it is a plane
    elif env.scene.cfg.terrain.terrain_type == "generator":
        # obtain the size of the sub-terrains
        terrain_gen_cfg = env.scene.terrain.cfg.terrain_generator
        grid_width, grid_length = terrain_gen_cfg.size
        n_rows, n_cols = terrain_gen_cfg.num_rows, terrain_gen_cfg.num_cols
        border_width = terrain_gen_cfg.border_width
        # compute the size of the map
        map_width = n_rows * grid_width + 2 * border_width
        map_height = n_cols * grid_length + 2 * border_width

        # extract the used quantities (to enable type-hinting)
        asset: RigidObject = env.scene[asset_cfg]

        # check if the agent is out of bounds
        x_out_of_bounds = torch.abs(asset.data.root_pos_w[:, 0]) > 0.5 * map_width - distance_buffer
        y_out_of_bounds = torch.abs(asset.data.root_pos_w[:, 1]) > 0.5 * map_height - distance_buffer
        return_ = torch.logical_or(x_out_of_bounds, y_out_of_bounds)
        if print_reason and return_.any():
            print("The base is out of the terrain border:", return_.sum())
        return return_
    else:
        raise ValueError("Received unsupported terrain type, must be either 'plane' or 'generator'.")


class illegal_reset_contact(ManagerTermBase):
    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.threshold = cfg.params["threshold"]
        self.sensor_cfg = cfg.params["sensor_cfg"]
        self.print_reason = cfg.params.get("print_reason", False)
        self.episode_length_threshold = cfg.params.get("episode_length_threshold", 1)
        self.illegal_contact_counter = torch.zeros(env.num_envs, device=env.device, dtype=torch.int)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        threshold: float,
        sensor_cfg: SceneEntityCfg,
        print_reason: bool = False,
        episode_length_threshold: int = 1,
    ) -> torch.Tensor:
        """Timeout if the robot is reset with some undesired penetration with the environment.
        within the first episode_length_threshold steps.
        """
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        net_contact_forces = contact_sensor.data.net_forces_w_history
        contacts = torch.any(
            torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold, dim=1
        )
        self.illegal_contact_counter += contacts.int()
        return_ = torch.logical_and(
            self.illegal_contact_counter >= episode_length_threshold,
            env.episode_length_buf <= episode_length_threshold,
        )
        if return_.any() and print_reason:
            print(f"illegal_reset_contact: {return_.sum()} envs")
        return return_

    def reset(self, env_ids: Sequence[int] | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.illegal_contact_counter[env_ids] = 0
