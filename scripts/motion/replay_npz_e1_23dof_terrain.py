"""Replay motion (npz) for E1-23DoF and load matched terrain mesh (stl) from metadata.yaml.

This script is based on `scripts/motion/replay_npz_e1_23dof.py`, with an added terrain importer:
it reads `metadata.yaml` under `--motion_path` to obtain:
  - the motion file (npz) path
  - the terrain mesh (stl) path (used by motion_matched terrain generator)

The terrain is imported via the same motion-matched mesh pipeline used in Mimic tasks.

.. code-block:: bash

    # Minimal usage: directory contains metadata.yaml, *.npz, *.stl
    python scripts/motion/replay_npz_e1_23dof_terrain.py --motion_path /path/to/dataset_dir

    # Override motion file explicitly (still loads terrain from metadata.yaml)
    python scripts/motion/replay_npz_e1_23dof_terrain.py --motion_path /path/to/dataset_dir --motion_file /path/to/motion.npz
"""

from __future__ import annotations

import argparse
import os

import torch
import yaml

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay converted motions for E1-23DoF with matched terrain mesh.")
parser.add_argument(
    "--motion_path",
    type=str,
    required=True,
    help="Dataset directory containing metadata.yaml and referenced motion/terrain files.",
)
parser.add_argument(
    "--motion_file",
    type=str,
    default=None,
    help="Optional override: absolute/relative path to a local npz motion file. If not set, uses metadata.yaml.",
)
parser.add_argument(
    "--terrain_id",
    type=int,
    default=None,
    help="Optional: select a terrain entry by terrain_id from metadata.yaml. "
    "If not set, uses the first terrain entry.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from NoetixE1.assets.robots.e1.e1 import E1_23DOF_CFG
from NoetixE1.tasks.mimic.mdp import MotionLoader
from NoetixE1.terrains import TerrainImporterCfg
from NoetixE1.terrains.terrain_generator_cfg import FiledTerrainGeneratorCfg
from NoetixE1.terrains.trimesh.mesh_terrains_cfg import MotionMatchedTerrainCfg


def _load_metadata(motion_path: str) -> dict:
    meta_path = os.path.join(motion_path, "metadata.yaml")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"metadata.yaml not found under motion_path: {meta_path}")
    with open(meta_path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid metadata.yaml content: expected dict, got {type(data).__name__}")
    return data


def _select_terrain_entry(metadata: dict, terrain_id: int | None) -> dict:
    terrains = metadata.get("terrains")
    if not terrains or not isinstance(terrains, list):
        raise ValueError("metadata.yaml must contain a non-empty 'terrains' list.")
    if terrain_id is None:
        entry = terrains[0]
        if not isinstance(entry, dict):
            raise ValueError("metadata.yaml 'terrains' entries must be dicts.")
        return entry
    for entry in terrains:
        if isinstance(entry, dict) and entry.get("terrain_id") == terrain_id:
            return entry
    raise ValueError(f"terrain_id={terrain_id} not found in metadata.yaml terrains list.")


def _select_motion_file(metadata: dict) -> str:
    motion_files = metadata.get("motion_files")
    if not motion_files or not isinstance(motion_files, list):
        raise ValueError("metadata.yaml must contain a non-empty 'motion_files' list.")
    entry = motion_files[0]
    if not isinstance(entry, dict) or not entry.get("motion_file"):
        raise ValueError("metadata.yaml motion_files[0] must be a dict with key 'motion_file'.")
    return entry["motion_file"]


@configclass
class ReplayMotionsWithTerrainSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene with motion-matched terrain."""

    # terrain (motion_matched mesh generator)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="hacked_generator",
        terrain_generator=FiledTerrainGeneratorCfg(
            # single cell is enough for replay/visualization
            size=(9, 12),
            border_width=0.0,
            border_height=0.0,
            num_rows=1,
            num_cols=1,
            sub_terrains={
                "motion_matched": MotionMatchedTerrainCfg(proportion=1.0),
            },
        ),
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

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation
    robot: ArticulationCfg = E1_23DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, motion_file: str):
    # Extract scene entities
    robot: Articulation = scene["robot"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()

    # MotionLoader requires: motion_file, body_indexes (torch.Tensor), device
    # body_indexes=[0] means using the root body (base_link)
    motion = MotionLoader(
        motion_file,
        torch.tensor([0], dtype=torch.long, device=sim.device),
        sim.device,
    )
    time_steps = torch.zeros(scene.num_envs, dtype=torch.long, device=sim.device)

    # Simulation loop
    while simulation_app.is_running():
        time_steps += 1
        reset_ids = time_steps >= motion.time_step_total
        time_steps[reset_ids] = 0

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion.body_pos_w[time_steps][:, 0] + scene.env_origins[:, None, :]
        root_states[:, 3:7] = motion.body_quat_w[time_steps][:, 0]
        root_states[:, 7:10] = motion.body_lin_vel_w[time_steps][:, 0]
        root_states[:, 10:] = motion.body_ang_vel_w[time_steps][:, 0]

        robot.write_root_state_to_sim(root_states)
        robot.write_joint_state_to_sim(motion.joint_pos[time_steps], motion.joint_vel[time_steps])
        scene.write_data_to_sim()
        sim.render()  # visualization only; no physics step
        scene.update(sim_dt)


def main():
    motion_path = args_cli.motion_path
    metadata = _load_metadata(motion_path)

    # Configure motion_matched dataset path/metadata for terrain generation
    scene_cfg = ReplayMotionsWithTerrainSceneCfg(num_envs=1, env_spacing=2.0)
    scene_cfg.terrain.terrain_generator.sub_terrains["motion_matched"].path = motion_path
    scene_cfg.terrain.terrain_generator.sub_terrains["motion_matched"].metadata_yaml = os.path.join(
        motion_path, "metadata.yaml"
    )

    # Resolve motion file (override > metadata)
    if args_cli.motion_file is not None:
        motion_file = args_cli.motion_file
    else:
        motion_file = os.path.join(motion_path, _select_motion_file(metadata))

    # Optional: validate terrain entry exists (and allow users to specify terrain_id)
    _select_terrain_entry(metadata, args_cli.terrain_id)

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    run_simulator(sim, scene, motion_file)


if __name__ == "__main__":
    main()
    simulation_app.close()
