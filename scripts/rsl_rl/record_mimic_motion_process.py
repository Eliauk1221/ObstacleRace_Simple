# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import json
import sys
import threading
try:
    from pynput import keyboard
except ImportError:
    keyboard = None
from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import NoetixE1.tasks  # noqa: F401
import torch
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from tqdm import tqdm

from rsl_rl.runners import *
from rsl_rl.utils.exporter import attach_onnx_metadata, export_policy_as_onnx

mujoco2isaaclab_joint = [0, 6, 12, 1, 7, 13, 18, 2, 8, 14, 19, 3, 9, 15, 20, 4, 10, 16, 21, 5, 11, 17, 22]
isaaclab2mujoco_joint = [0, 3, 7, 11, 15, 19, 1, 4, 8, 12, 16, 20, 2, 5, 9, 13, 17, 21, 6, 10, 14, 18, 22]

mujoco2isaaclab_keypose = [0, 1, 7, 13, 2, 8, 14, 19, 3, 9, 15, 20, 4, 10, 16, 21, 5, 11, 17, 22, 6, 12, 18, 23]
isaaclab2mujoco_keypose = [0, 1, 4, 8, 12, 16, 20, 2, 5, 9, 13, 17, 21, 3, 6, 10, 14, 18, 22, 7, 11, 15, 19, 23]

# 左右脚 body：与 walkrun_amp_depth_cfg 一致，用正则 .*_ankle_roll_link
FEET_BODY_NAMES = ".*_ankle_roll_link"
contact_id = [6, 12]

# 是否记录全身所有 link 的 contact（Motion KeyPosNames 顺序）。默认 False，与原逻辑保持兼容。
RECORD_ALL_BODY_CONTACT = True
# 是否额外记录 height_scan，与 obs.height_scan 对齐（大致形状一致）
RECORD_HEIGHT_SCAN = True


def _contact_mask_from_sensor(
    env, num_envs: int, device: torch.device, force_threshold: float = 0.5, record_all: bool = False
):
    """从 contact sensor 读取接触信息。

    - record_all=False: 返回 (num_envs, 2) 的左右脚接触（与原逻辑一致，顺序由 contact_id 决定）。
    - record_all=True:  返回 (num_envs, N_keypose) 的全身接触，顺序为motion顺序一致。
    """
    try:
        contact_sensor = env.scene.sensors["contact_sensor"]
        net_contact_forces = contact_sensor.data.net_forces_w_history
        forces = torch.norm(net_contact_forces, dim=-1)  # [H, N, B]
        max_forces = torch.max(forces, dim=1)[0]  # [N, B]
        contact_bool = max_forces > force_threshold  # [N, B]

        if not record_all:
            # 保持原行为：只取左右踝，返回 (N, 2)
            return contact_bool[:, contact_id].to(dtype=torch.float32)

        return contact_bool.to(dtype=torch.float32)

    except Exception:
        if record_all:
            return torch.zeros((num_envs, len(isaaclab2mujoco_keypose)), device=device, dtype=torch.float32)
        return torch.zeros((num_envs, 2), device=device, dtype=torch.float32)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # set the play environment config
    env_cfg.episode_length_s = 100.0
    env_cfg.scene.num_envs = 1
    env_cfg.scene.env_spacing = 2.5

    env_cfg.observations.policy.enable_corruption = False
    env_cfg.viewer.origin_type = None
    env_cfg.viewer.asset_name = None

    env_cfg.events.physics_material = None
    env_cfg.events.add_joint_default_pos = None
    env_cfg.events.base_com = None
    env_cfg.events.randomize_joint_parameters = None
    env_cfg.events.randomize_actuator_gains = None
    env_cfg.events.push_robot = None

    # env_cfg.scene.terrain.terrain_type = "plane"
    # env_cfg.scene.terrain.terrain_generator = None

    # if env_cfg.scene.terrain.terrain_generator is not None:
    #     env_cfg.scene.terrain.terrain_generator.num_rows = 5
    #     env_cfg.scene.terrain.terrain_generator.num_cols = 5
    #     env_cfg.scene.terrain.terrain_generator.curriculum = False
    #     env_cfg.scene.terrain.terrain_generator.difficulty_range = (0.4, 0.4)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    runner_class: OnPolicyRunner | AmpHimOnPolicyRunner = eval(agent_cfg.class_name)
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # # export policy to onnx/jit
    # export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    # export_policy_as_onnx(policy_nn, path=export_model_dir, filename="policy.onnx")
    # attach_onnx_metadata(env=env.unwrapped, path=export_model_dir)

    # ---- force motion start from frame 0 (deterministic) ----
    motion_cmd = None
    try:
        motion_cmd = env.unwrapped.command_manager.get_term("motion")
    except Exception:
        motion_cmd = None
    if motion_cmd is not None and hasattr(motion_cmd, "_adaptive_sampling") and hasattr(motion_cmd, "time_steps"):
        # MotionCommand._resample_command() 会先调用 _adaptive_sampling(env_ids) 来设置 time_steps。
        # 这里替换为固定从 0 开始，确保 reset 时落在第 0 帧。
        def _adaptive_sampling_start0(env_ids):
            if env_ids is None:
                return
            try:
                if len(env_ids) == 0:
                    return
            except TypeError:
                # slice / scalar
                pass
            motion_cmd.time_steps[env_ids] = 0

        motion_cmd._adaptive_sampling = _adaptive_sampling_start0

    dt = float(env.unwrapped.step_dt)
    motion_total = int(getattr(getattr(motion_cmd, "motion", None), "time_step_total", 0) or 0)

    # reset environment
    obs = env.get_observations()
    motion_frames = []
    # simulate environment: record from frame 0 to last frame, then stop/save/exit.
    # 说明：MotionCommand 每步 time_steps += 1；当 time_steps==last 时就结束，避免下一步 wrap/resample。
    max_steps = motion_total if motion_total > 0 else 10_000_000
    for i in tqdm(range(max_steps)):
        if not simulation_app.is_running():
            break

        # 如果已经到最后一帧，则保存退出（确保 last frame 已被记录）
        if motion_cmd is not None and motion_total > 0:
            cur_ts = int(motion_cmd.time_steps[0].item())
            if cur_ts >= motion_total - 1 and len(motion_frames) > 0:
                break

        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)

            root_pos = env.unwrapped.robot.data.root_pos_w.clone()
            root_quat = env.unwrapped.robot.data.root_quat_w.clone()
            root_quat = torch.cat((root_quat[:, 1:], root_quat[:, 0].unsqueeze(1)), dim=1)  #  wxyz -> xyzw
            dof_pos = env.unwrapped.robot.data.joint_pos[:, isaaclab2mujoco_joint]
            key_pos = (env.unwrapped.robot.data.body_pos_w[:, isaaclab2mujoco_keypose, :] - root_pos.unsqueeze(1)).flatten(1, 2)
            root_vel = env.unwrapped.robot.data.root_lin_vel_b.clone()
            root_ang_vel = env.unwrapped.robot.data.root_ang_vel_b.clone()
            dof_vel = env.unwrapped.robot.data.joint_vel[:, isaaclab2mujoco_joint]
            # contact：可选仅记录左右脚，或记录全身所有 link 的 contact（Motion KeyPosNames 顺序）
            contact_mask = _contact_mask_from_sensor(
                env.unwrapped,
                root_pos.shape[0],
                root_pos.device,
                record_all=RECORD_ALL_BODY_CONTACT,
            )
            # 可选：记录 height_scan（base_link 射线高度图），按 RayCaster 的 ray_hits_w 展平成一维
            if RECORD_HEIGHT_SCAN:
                try:
                    height_scanner = env.unwrapped.scene.sensors["height_scanner"]
                    ray_hits_w = height_scanner.data.ray_hits_w  # [..., num_rays, 3]
                    heights = height_scanner.data.pos_w[:, 2] - ray_hits_w[..., 2] - 0.5
                    if heights.ndim == 3:
                        # [H, N, R] -> 取最后一帧 H-1
                        heights_last = heights[-1]  # [N, R]
                    elif heights.ndim == 2:
                        heights_last = heights  # [N, R]
                    else:
                        heights_last = torch.zeros((root_pos.shape[0], 0), device=root_pos.device)
                    height_scan_vec = heights_last.reshape(root_pos.shape[0], -1)
                except Exception:
                    height_scan_vec = torch.zeros((root_pos.shape[0], 0), device=root_pos.device)
            else:
                height_scan_vec = torch.zeros((root_pos.shape[0], 0), device=root_pos.device)

            cur_frames_tensor = torch.cat(
                (root_pos, root_quat, dof_pos, key_pos, root_vel, root_ang_vel, dof_vel, contact_mask, height_scan_vec),
                dim=-1,
            )
            motion_frames.append(cur_frames_tensor[0].tolist())

    motion_data = {}
    motion_data["FrameDuration"] = dt
    motion_data["JointNames"] = [env.unwrapped.robot.joint_names[i] for i in isaaclab2mujoco_joint]
    motion_data["KeyPosNames"] = [env.unwrapped.robot.body_names[i] for i in isaaclab2mujoco_keypose]
    motion_data["MotionWeight"] = 0.5
    motion_data["Frames"] = motion_frames
    # export motion capture to json format
    out_path = "/home/sch/motion_data/e1/23dof_contact/e1_parkour_45cmbox.json"
    with open(out_path, "w") as f:
        json.dump(motion_data, f, indent=1, separators=(",", ":"))
    print("Generating " + out_path)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
