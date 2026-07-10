# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import threading

try:
    from pynput import keyboard
except ImportError:
    keyboard = None

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip
import onnxruntime as ort

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--visualize_depth", action="store_true", default=False, help="Visualize depth map for the first environment.")
parser.add_argument("--depth_visual_interval", type=int, default=5, help="Visual depth image every N steps (default: 30 for better performance).")
parser.add_argument("--visualize_moe", action="store_true", default=False, help="Print and visualize MoE gate weights online.")
parser.add_argument("--moe_visual_interval", type=int, default=10, help="Print/refresh MoE weights every N steps.")
parser.add_argument("--moe_env_idx", type=int, default=0, help="Environment index used for MoE weight visualization.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

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
import NoetixE1.terrains as terrain_gen  # 用于指定地形类型（取消注释下方代码时使用）
import torch

# 深度图可视化相关导入
plt = None
if args_cli.visualize_depth or args_cli.visualize_moe:
    try:
        import matplotlib

        # 尝试使用交互式后端进行在线可视化
        # 如果失败，会回退到 Agg 后端
        try:
            # 尝试使用 TkAgg（需要 X11）
            matplotlib.use("TkAgg")
        except Exception:
            try:
                # 尝试使用 Qt5Agg
                matplotlib.use("Qt5Agg")
            except Exception:
                # 回退到 Agg（非交互式，但可以保存）
                matplotlib.use("Agg")
                print("[WARN] Using non-interactive backend. For online visualization, ensure X11 forwarding is enabled.")

        import matplotlib.pyplot as plt

        # 设置交互模式
        plt.ion()  # 开启交互模式
    except ImportError:
        print("[WARN] matplotlib not installed. Depth visualization disabled. Install via: pip install matplotlib")
        args_cli.visualize_depth = False
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import rsl_rl.runners as rsl_rl_runners
from rsl_rl.utils.exporter import attach_onnx_metadata, export_policy_as_onnx

from NoetixE1.tasks.walkrun.noetix_e1.mdp.commands import KeyboardVelocityCommand


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
    env_cfg.episode_length_s = 600.0
    env_cfg.scene.num_envs = 10
    env_cfg.scene.env_spacing = 2.5

    # env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    # env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
    # env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    # env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    # env_cfg.commands.base_velocity.ranges.zero_prob = [0.0, 0.0, 0.0]
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.viewer.origin_type = None
    env_cfg.viewer.asset_name = None

    env_cfg.events.physics_material = None
    env_cfg.events.add_joint_default_pos = None
    env_cfg.events.base_com = None
    env_cfg.events.randomize_joint_parameters = None
    env_cfg.events.randomize_actuator_gains = None
    env_cfg.events.push_robot = None
    env_cfg.events.continuous_disturbance = None

    if env_cfg.scene.terrain.terrain_type == "hacked_generator":
        env_cfg.scene.terrain.terrain_generator.num_rows = 1
        env_cfg.scene.terrain.terrain_generator.num_cols = 1
    elif env_cfg.scene.terrain.terrain_type == "generator":
        env_cfg.scene.terrain.terrain_generator.num_rows = 1
        env_cfg.scene.terrain.terrain_generator.num_cols = 1
        env_cfg.scene.terrain.terrain_generator.curriculum = False
        env_cfg.scene.terrain.terrain_generator.difficulty_range = (0.1, 0.1)

        # 测试带噪声的四种地形
        # 取消注释下面的配置来测试不同的地形（每次只测试一种）

        # 选项3: 带噪声的上楼梯
        # env_cfg.scene.terrain.terrain_generator.sub_terrains = {
        #     "frame_platform": terrain_gen.HfDoubleFramePlatformTerrainCfg(
        #         proportion=0.1,
        #         height_range_1=(0.3, 0.5), width_range_1=(0.5, 0.5), distance_range_1=(1.25, 1.25),
        #         height_range_2=(0.3, 0.5), width_range_2=(1.0, 1.0), distance_range_2=(3.0, 3.0),
        #     )
        # }
        
        # # pit
        # env_cfg.scene.terrain.terrain_generator.sub_terrains = {
        #     "pit": terrain_gen.MeshRailsTerrainCfg(
        #         proportion=0.1,
        #         rail_thickness_range=(0.5, 1.0),
        #         rail_height_range=(0.3, 0.3),
        #         platform_width=2.0,
        #     )
        # }

        # # 选项0: 平地
        # env_cfg.scene.terrain.terrain_generator.sub_terrains = {
        #     "plane": terrain_gen.HfRandomUniformTerrainCfg(
        #         proportion=0.1,
        #         noise_range=(0.0, 0.0),
        #         noise_step=0.02,
        #     )
        # }

        # # 选项1: 带噪声的下斜坡
        # env_cfg.scene.terrain.terrain_generator.sub_terrains = {
        #     "noisy_pyramid_slop_down": terrain_gen.HfNoisyPyramidSlopedTerrainCfg(
        #         proportion=0.1,
        #         slope_range=(0.1, 0.4),
        #         noise_range=(0.01, 0.03), noise_step=0.01,
        #         num_obstacles=3, obstacle_size=(0.2, 0.4), obstacle_height=(0.5, 1.5)
        #     )
        # }

        # # 选项2: 带噪声的上斜坡
        # env_cfg.scene.terrain.terrain_generator.sub_terrains = {
        #     "noisy_pyramid_slop_up": terrain_gen.HfNoisyInvertedPyramidSlopedTerrainCfg(
        #         proportion=0.1,
        #         slope_range=(0.1, 0.4),
        #         noise_range=(0.01, 0.03), noise_step=0.01,
        #         num_obstacles=0, obstacle_size=(0.2, 0.4), obstacle_height=(0.5, 1.5)
        #     )
        # }

        # 选项3: 带噪声的上楼梯
        env_cfg.scene.terrain.terrain_generator.sub_terrains = {
            "noisy_pyramid_stairs_up": terrain_gen.HfNoisyInvertedPyramidStairsTerrainCfg(
                proportion=0.1,
                step_height_range=(0.05, 0.25),
                step_width_pixels_range=(10, 5),
                platform_width=2.0,
                border_width=0.5,
                noise_range=(0.00, 0.00),
                noise_step=0.02,
            )
        }

        # # 选项4: 带噪声的下楼梯
        # env_cfg.scene.terrain.terrain_generator.sub_terrains = {
        #     "noisy_pyramid_stairs_down": terrain_gen.HfNoisyPyramidStairsTerrainCfg(
        #         proportion=0.1,
        #         step_height_range=(0.05, 0.25),
        #         step_width_pixels_range=(10, 5),
        #         platform_width=2.0,
        #         border_width=0.5,
        #         noise_range=(0.0, 0.0),
        #         noise_step=0.02,
        #     )
        # }
        
        # # 选项5: 多gap地形（可配置深度，支持多个gap）
        # env_cfg.scene.terrain.terrain_generator.sub_terrains = {
        #     "multi_gap": terrain_gen.HfMultiGapTerrainCfg(
        #         proportion=1.0,
        #         platform_width=2.0,
        #         gap_width_pixels_range=(10, 10),  # gap宽度随难度从0.1到0.5米
        #         gap_depth_range=(0.2, 1.0),  # gap深度1.0米
        #         step_height_range=(0.03, 0.1),  # 台阶高度随难度从0.05到0.2米
        #     )
        # }
        
        # env_cfg.scene.terrain.terrain_generator.sub_terrains = {
        #     "single_gap": terrain_gen.HfSingleGapTerrainCfg(
        #         proportion=1.0,
        #         platform_width=2.0,
        #         gap_width_pixels_range=(20, 20),  # gap宽度随难度从0.1到0.5米
        #         gap_depth_range=(0.2, 1.0),  # gap深度1.0米
        #         step_height_range=(0.03, 0.1),  # 台阶高度随难度从0.05到0.2米
        #     )
        # }
        
        # 选线6：step stones 地形
        # env_cfg.scene.terrain.terrain_generator.sub_terrains = {
        #     "stepping_stones": terrain_gen.HfSteppingStonesTerrainCfg(
        #         proportion=1.0,
        #         stone_height_max=0.05,
        #         stone_width_range=(0.3, 0.5),
        #         stone_distance_range=(0.1, 0.3),
        #         holes_depth=(-0.3, -0.5),
        #         border_width=0.5,
        #         platform_width=2.0,
        #     )
        # }
        
        # # 选项6.5：cross step stones 地形（十字梅花桩）
        # env_cfg.scene.terrain.terrain_generator.sub_terrains = {
        #     "cross_stepping_stones2": terrain_gen.HfCrossSteppingStonesTerrainCfg(
        #         proportion=1.0,
        #         stone_height_max=0.05,
        #         stone_width_range=(0.3, 0.5),
        #         stone_distance_range=(0.1, 0.3),
        #         holes_depth=(-0.3, -0.5),
        #         num_stone_rows=2,
        #         platform_width=1.5,
        #     )
        # }
        
        # # 选项7：beam 地形
        # env_cfg.scene.terrain.terrain_generator.sub_terrains = {
        #     "beam": terrain_gen.MeshStarTerrainCfg(
        #         proportion=1.0,
        #         num_bars=4,
        #         bar_width_range=(0.1, 0.1),
        #         bar_height_range=(0.3, 1.0),
        #         platform_width=2.0,
        #     )
        # }
        # # 选项8：离散障碍物地形
        # env_cfg.scene.terrain.terrain_generator.sub_terrains = {
        #     "discrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
        #         proportion=1.0, obstacle_width_range=(0.3, 1.0), obstacle_height_range=(0.05, 0.30), num_obstacles=30
        #     )
        # }
        
        # # 选项9：跨栏地形（固定3个回形跨栏，均匀分布）
        # env_cfg.scene.terrain.terrain_generator.sub_terrains = {
        #     "hurdle": terrain_gen.HfHurdleTerrainCfg(
        #         proportion=1.0,
        #         platform_width=1.5,
        #         hurdle_height_range=(0.05, 0.35),  # 跨栏高度随难度从0.05m增加到0.3m
        #         hurdle_width_pixels=(1, 2, 3),  # 所有3个栏的宽度都是1, 2, 3像素
        #     )
        # }

    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    if CONTROL_ROBOT:
        base_velocity_cfg = getattr(getattr(env_cfg, "commands", None), "base_velocity", None)
        if base_velocity_cfg is None:
            print("[WARN] Keyboard control requested but base_velocity command not found in env config.")
        else:
            base_velocity_cfg.class_type = KeyboardVelocityCommand
            base_velocity_cfg.resampling_time_range = (1e9, 1e9)

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
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # 深度图可视化设置（将在policy_nn加载后初始化）
    depth_fig = None
    depth_axes = None
    depth_im = None
    depth_cbar = None
    attn_axes = None
    attn_im = None
    attn_cbar = None
    attn_depth_bg = None  # 注意力权重图的深度图背景
    depth_vmin = 0.0
    depth_vmax = 2.0
    moe_fig = None
    moe_axes = None
    moe_bars = None

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    command_listener = None
    command_lock = threading.Lock()
    reset_env_flag = threading.Event()
    command_term = None
    if CONTROL_ROBOT:
        try:
            command_term = env.unwrapped.command_manager.get_term("base_velocity")
        except KeyError:
            command_term = None
            print("[WARN] base_velocity term not found in environment; keyboard control disabled.")
        if command_term is not None and not isinstance(command_term, KeyboardVelocityCommand):
            print("[WARN] base_velocity term is not KeyboardVelocityCommand; keyboard control disabled.")
            command_term = None

        if command_term is not None:
            if keyboard is None:
                raise ImportError("Keyboard control requires 'pynput'. Install via 'pip install pynput'.")
            step_size = 0.1
            command_term.reset_manual_command()

            print("[INFO] Keyboard control enabled (W/S forward/back, A/D left/right, Q/E yaw, R reset command, T reset robot).")

            def _print_command():
                cmd = command_term.get_manual_command()[0].detach().cpu().tolist()
                print(f"[CMD] Current command (x, y, yaw): {cmd}")

            def _adjust(idx: int, delta: float):
                with command_lock:
                    command_term.adjust_manual_command_component(idx, delta)
                _print_command()

            def _reset():
                with command_lock:
                    command_term.reset_manual_command()
                _print_command()

            def _on_press(key_event):
                try:
                    char = key_event.char.lower()
                except AttributeError:
                    return
                if char == "w":
                    _adjust(0, step_size)
                elif char == "s":
                    _adjust(0, -step_size)
                elif char == "d":
                    _adjust(1, -step_size)
                elif char == "a":
                    _adjust(1, step_size)
                elif char == "q":
                    _adjust(2, step_size)
                elif char == "e":
                    _adjust(2, -step_size)
                elif char == "r":
                    _reset()
                elif char == "t":
                    reset_env_flag.set()
                    print("[INFO] Robot reset requested (T key pressed).")

            command_listener = keyboard.Listener(on_press=_on_press)
            command_listener.daemon = True
            command_listener.start()

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    try:
        runner_class = getattr(rsl_rl_runners, agent_cfg.class_name)
    except AttributeError as exc:
        raise ValueError(f"Runner class '{agent_cfg.class_name}' not found in rsl_rl.runners.") from exc
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

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None
    cmoe_task = args_cli.task is not None and "CMoE" in args_cli.task
    cmoe_policy = "CMoE" in type(policy_nn).__name__
    can_visualize_moe = hasattr(policy_nn, "estimator") and hasattr(policy_nn, "actor")
    if args_cli.visualize_moe and (not can_visualize_moe or not (cmoe_task or cmoe_policy)):
        print("[WARN] Current policy/task is not CMoE-compatible. MoE visualization disabled.")
        args_cli.visualize_moe = False
    elif args_cli.visualize_moe:
        print(f"[INFO] CMoE policy detected: task={args_cli.task}, policy_class={type(policy_nn).__name__}")
        if args_cli.moe_env_idx < 0 or args_cli.moe_env_idx >= env.num_envs:
            print(f"[WARN] moe_env_idx={args_cli.moe_env_idx} out of range, falling back to 0.")
            args_cli.moe_env_idx = 0
        if plt is None:
            print("[WARN] matplotlib unavailable. MoE weights will be printed to terminal only.")
    
    # 检查raycaster_camera是否存在（用于3D可视化）
    raycaster_camera = None
    if args_cli.visualize_depth and hasattr(env.unwrapped.scene, "sensors") and "raycaster_camera" in env.unwrapped.scene.sensors:
        raycaster_camera = env.unwrapped.scene.sensors["raycaster_camera"]
        if hasattr(raycaster_camera, "ray_visualizer") and raycaster_camera.ray_visualizer is not None:
            num_marker_types = len(raycaster_camera.ray_visualizer.cfg.markers)
            print(f"[INFO] raycaster_camera.ray_visualizer has {num_marker_types} marker types.")
    
    # Check if using AttnDepthEncoder or CnnAttnDepthEncoder for attention weight visualization
    use_attn_visualization = False
    attn_depth_encoder = None
    if args_cli.visualize_depth and hasattr(policy_nn, "depth_encoder"):
        from rsl_rl.modules.attn_depth_encoder import AttnDepthEncoder
        from rsl_rl.modules.cnn_attn_depth_encoder import CnnAttnDepthEncoder
        if isinstance(policy_nn.depth_encoder, (AttnDepthEncoder, CnnAttnDepthEncoder)):
            use_attn_visualization = True
            attn_depth_encoder = policy_nn.depth_encoder
            encoder_type = "AttnDepthEncoder" if isinstance(policy_nn.depth_encoder, AttnDepthEncoder) else "CnnAttnDepthEncoder"
            print(f"[INFO] {encoder_type} detected. Attention weights will be visualized.")
    
    # 创建深度图可视化窗口
    if args_cli.visualize_depth:
        if use_attn_visualization:
            depth_fig, (depth_axes, attn_axes) = plt.subplots(1, 2, figsize=(16, 6))
            depth_fig.suptitle("Depth Map & Attention Weights Visualization (Environment 0)", fontsize=14)
            plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, wspace=0.3)
        else:
            depth_fig, depth_axes = plt.subplots(1, 1, figsize=(8, 6))
            depth_fig.suptitle("Depth Map Visualization (Environment 0)", fontsize=14)
            plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        plt.show(block=False)
        plt.pause(0.1)
        print("[INFO] Depth visualization window created.")

    if args_cli.visualize_moe and plt is not None:
        moe_fig, moe_axes = plt.subplots(1, 1, figsize=(8, 4))
        moe_fig.suptitle(f"MoE Gate Weights (Environment {args_cli.moe_env_idx})", fontsize=14)
        try:
            num_experts = int(policy_nn.actor.num_experts)
        except Exception:
            num_experts = 8
            print("[WARN] Failed to infer num_experts from actor, fallback to 8.")
        moe_bars = moe_axes.bar(range(num_experts), [0.0] * num_experts, color="tab:orange")
        moe_axes.set_ylim(0.0, 1.0)
        moe_axes.set_xlabel("Expert")
        moe_axes.set_ylabel("Weight")
        moe_axes.set_title("Current gate softmax")
        plt.show(block=False)
        plt.pause(0.1)
        print("[INFO] MoE visualization window created.")

    # export policy to onnx/jit
    if EXPORT_POLICY:
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")
        attach_onnx_metadata(env=env.unwrapped, path=export_model_dir)

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0

    # 深度图可视化函数（在线更新，优化性能）
    def visualize_depth_online(
        depth_data, step, fig, axes, im_ref, cbar_ref, vmin_fixed, vmax_fixed,
        attn_axes=None, attn_im_ref=None, attn_cbar_ref=None, attn_weights=None,
        attn_depth_encoder=None, attn_depth_bg_ref=None
    ):
        """在线可视化深度图（第一个环境），更新现有窗口（性能优化版本）
        
        Args:
            depth_data: 深度图数据
            step: 当前步数
            fig: matplotlib figure对象
            axes: matplotlib axes对象（深度图）
            im_ref: 深度图图像对象引用
            cbar_ref: 深度图colorbar引用
            vmin_fixed: 深度图最小值
            vmax_fixed: 深度图最大值
            attn_axes: 注意力权重axes对象（可选）
            attn_im_ref: 注意力权重图像对象引用（可选）
            attn_cbar_ref: 注意力权重colorbar引用（可选）
            attn_weights: 注意力权重数据（可选）
        """
        if depth_data is None or fig is None or axes is None:
            return None, None, None, None

        # 转换为numpy数组（使用共享内存，避免复制）
        if isinstance(depth_data, torch.Tensor):
            depth_np = depth_data[0].cpu().numpy() + 1.0
        else:
            depth_np = depth_data[0] + 1.0

        # 处理维度：可能是 (H, W), (H, W, 1), (1, H, W) 等
        if depth_np.ndim == 3:
            if depth_np.shape[-1] == 1:
                depth_np = depth_np.squeeze(-1)
            elif depth_np.shape[0] == 1:
                depth_np = depth_np.squeeze(0)

        if depth_np.ndim != 2:
            if step == 0:  # 只在第一次打印警告
                print(f"[WARN] Unexpected depth shape: {depth_np.shape}, skipping visualization")
            return im_ref, cbar_ref, attn_im_ref, attn_cbar_ref, attn_depth_bg_ref

        # 使用固定的颜色范围（从配置中读取）
        vmin = vmin_fixed
        vmax = vmax_fixed

        # 更新图像数据（如果图像对象已存在）
        if im_ref is not None:
            # 更新现有图像数据（最快的方式）
            im_ref.set_data(depth_np)
            # 不更新 colorbar，使用固定范围
            # im_ref.set_clim(vmin, vmax)  # 固定范围，不需要更新
        else:
            # 首次创建图像
            axes.clear()
            # 使用最近邻插值更快，关闭抗锯齿，设置 aspect="auto" 避免重新计算
            im_ref = axes.imshow(depth_np, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest", aspect="auto")
            axes.set_title("Depth Map")
            axes.set_xlabel("Width")
            axes.set_ylabel("Height")
            # 关闭坐标轴刻度以提高性能（可选）
            # axes.set_xticks([])
            # axes.set_yticks([])
            cbar_ref = plt.colorbar(im_ref, ax=axes, label="Depth (m)")
            # 只在首次创建时调用 tight_layout，后续不再调用
            plt.tight_layout()

        # 可视化注意力权重（如果提供）
        if attn_weights is not None and attn_axes is not None and attn_depth_encoder is not None:
            # 处理注意力权重：可能是 [B, num_heads, 1, num_patches] 或 [B, 1, num_patches]
            attn_np = attn_weights[0].cpu().numpy()  # 取第一个batch
            
            if step == 0:
                print(f"[DEBUG] Raw attention weights shape: {attn_np.shape}")
            
            # 如果是多头注意力，平均所有头
            if attn_np.ndim == 3:  # [num_heads, 1, num_patches]
                attn_np = attn_np.mean(axis=0)  # 平均所有头 -> [1, num_patches]
            elif attn_np.ndim == 2:  # [1, num_patches]
                pass  # 已经是单头或已处理
            
            # 移除维度1（query维度）
            if attn_np.ndim == 2 and attn_np.shape[0] == 1:
                attn_np = attn_np.squeeze(0)  # [num_patches]
            
            if step == 0:
                print(f"[DEBUG] After processing, attention weights shape: {attn_np.shape}, size: {attn_np.size}")
            
            # 将注意力权重重塑为空间维度
            # 需要知道patch_size和原始深度图尺寸
            # AttnDepthEncoder uses patch_embed, CnnAttnDepthEncoder uses cnn
            if hasattr(attn_depth_encoder, 'patch_embed'):
                # AttnDepthEncoder: patch-based
                patch_size = attn_depth_encoder.patch_embed.kernel_size[0]
                H_orig, W_orig = depth_np.shape
                H_patches = H_orig // patch_size
                W_patches = W_orig // patch_size
                if step == 0:
                    print(f"[DEBUG] AttnDepthEncoder: patch_size={patch_size}, H={H_orig}, W={W_orig}, H_patches={H_patches}, W_patches={W_patches}, expected patches={H_patches * W_patches}")
            elif hasattr(attn_depth_encoder, 'cnn'):
                # CnnAttnDepthEncoder: CNN-based, output is 6x8 patches
                H_orig, W_orig = depth_np.shape
                H_patches = 8  # CNN output height
                W_patches = 12  # CNN output width
                if step == 0:
                    print(f"[DEBUG] CnnAttnDepthEncoder: H={H_orig}, W={W_orig}, H_patches={H_patches}, W_patches={W_patches}, expected patches={H_patches * W_patches}")
            else:
                if step == 0:
                    print("[WARN] Unknown encoder type, cannot determine patch size")
                attn_np = None
            
            if attn_np is not None and attn_np.size == H_patches * W_patches:
                attn_np = attn_np.reshape(H_patches, W_patches)
                if step == 0:
                    print(f"[DEBUG] Reshaped attention weights to: {attn_np.shape}")
                # 上采样到原始深度图尺寸
                try:
                    import scipy.ndimage
                    zoom_factor = (H_orig / H_patches, W_orig / W_patches)
                    attn_np = scipy.ndimage.zoom(attn_np, zoom_factor, order=1)
                    if step == 0:
                        print(f"[DEBUG] After zoom, attention weights shape: {attn_np.shape}")
                except ImportError:
                    # Fallback: 使用简单的重复上采样
                    import numpy as np
                    zoom_h = H_orig // H_patches
                    zoom_w = W_orig // W_patches
                    attn_np = np.repeat(np.repeat(attn_np, zoom_h, axis=0), zoom_w, axis=1)
                    # 裁剪到正确尺寸
                    attn_np = attn_np[:H_orig, :W_orig]
                    if step == 0:
                        print(f"[DEBUG] After repeat upsampling, attention weights shape: {attn_np.shape}")
            elif attn_np is not None:
                if step == 0:
                    print(f"[WARN] Attention weights size {attn_np.size} doesn't match expected {H_patches * W_patches}")
                attn_np = None
            
            if attn_np is not None:
                # 将注意力权重叠加在深度图上
                # 归一化注意力权重到 [0, 1] 范围
                attn_norm = (attn_np - attn_np.min()) / (attn_np.max() - attn_np.min() + 1e-8)
                
                # 更新或创建融合图像
                if attn_im_ref is not None and attn_depth_bg_ref is not None:
                    # 更新图像：同时更新深度图背景和注意力权重层
                    depth_for_attn = depth_np.copy()
                    depth_for_attn_norm = (depth_for_attn - depth_for_attn.min()) / (depth_for_attn.max() - depth_for_attn.min() + 1e-8)
                    attn_depth_bg_ref.set_data(depth_for_attn_norm)
                    attn_im_ref.set_data(attn_norm)
                    if step == 0:
                        print("[DEBUG] Updated existing attention overlay image")
                else:
                    attn_axes.clear()
                    # 先显示深度图作为背景（灰度）
                    depth_for_attn = depth_np.copy()
                    # 归一化深度图到 [0, 1] 用于显示
                    depth_for_attn_norm = (depth_for_attn - depth_for_attn.min()) / (depth_for_attn.max() - depth_for_attn.min() + 1e-8)
                    # 显示深度图背景
                    attn_depth_bg_ref = attn_axes.imshow(
                        depth_for_attn_norm,
                        cmap="gray",
                        interpolation="bilinear",
                        aspect="auto",
                        alpha=0.6  # 背景透明度
                    )
                    
                    # 叠加注意力权重（热力图，带透明度）
                    attn_im_ref = attn_axes.imshow(
                        attn_norm,
                        cmap="hot",
                        interpolation="bilinear",
                        aspect="auto",
                        alpha=0.6,  # 前景透明度，可以调整
                        vmin=0.0,
                        vmax=1.0
                    )
                    attn_axes.set_title("Depth Map + Attention Weights Overlay")
                    attn_axes.set_xlabel("Width")
                    attn_axes.set_ylabel("Height")
                    attn_cbar_ref = plt.colorbar(attn_im_ref, ax=attn_axes, label="Attention Weight")
                    plt.tight_layout()
                    if step == 0:
                        print("[DEBUG] Created new attention overlay image")
            elif step == 0:
                print("[WARN] Failed to process attention weights for visualization")
        elif step == 0:
            if attn_weights is None:
                print("[DEBUG] attn_weights is None")
            if attn_axes is None:
                print("[DEBUG] attn_axes is None")
            if attn_depth_encoder is None:
                print("[DEBUG] attn_depth_encoder is None in visualize_depth_online")

        # 优化绘制：减少不必要的操作
        # 使用 draw_idle 而不是 draw，更高效（延迟绘制，批量更新）
        fig.canvas.draw_idle()
        # 减少 flush_events 调用频率，只在更新间隔时刷新
        if step % args_cli.depth_visual_interval == 0:
            fig.canvas.flush_events()

        return im_ref, cbar_ref, attn_im_ref, attn_cbar_ref, attn_depth_bg_ref

    def visualize_moe_online(gate_scores, step, env_idx, fig, axes, bars):
        """Print and optionally visualize MoE gate weights for one environment."""
        if gate_scores is None:
            return bars
        weights = gate_scores[env_idx].detach().cpu().tolist()
        weight_str = ", ".join(f"{value:.4f}" for value in weights)
        print(f"[MOE][step={step}][env={env_idx}] [{weight_str}]")

        if fig is None or axes is None or bars is None:
            return bars

        for bar, value in zip(bars, weights):
            bar.set_height(value)
        axes.set_title(f"Current gate softmax (step {step})")
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        return bars

    def get_cmoe_gate_scores(policy_module, obs_dict):
        """Extract gate weights via estimator for CMoE actor-critic."""
        if not hasattr(policy_module, "estimator"):
            return None
        try:
            depth_obs = policy_module.get_depth_obs(obs_dict)
            proprio_obs = policy_module.get_actor_obs(obs_dict)
            if hasattr(policy_module, "actor_obs_normalizer"):
                proprio_obs = policy_module.actor_obs_normalizer(proprio_obs)
            _, _, gate = policy_module.estimator(proprio_obs, depth_obs)
            return gate
        except Exception as exc:
            print(f"[WARN] Failed to extract CMoE gate scores: {exc}")
            return None

    # simulate environment
    use_onnx_inference = INTERFACE_ONNX and not args_cli.visualize_moe
    session = None
    if use_onnx_inference:
        load_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        session = ort.InferenceSession(os.path.join(load_model_dir, "policy.onnx"))

        def onnx_policy(obs, depth):
            in_obs_name = session.get_inputs()[0].name
            in_depth_name = session.get_inputs()[1].name
            out_action_name = session.get_outputs()[0].name
            result = session.run([out_action_name], {in_obs_name: obs.detach().cpu().numpy(), in_depth_name: depth.detach().cpu().numpy()})
            return torch.from_numpy(result[0]).to(env.device)
    elif args_cli.visualize_moe and INTERFACE_ONNX:
        print("[INFO] MoE visualization requires PyTorch inference. ONNX inference disabled for this run.")

    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # Check for environment reset request (before inference_mode)
            if reset_env_flag.is_set():
                reset_env_flag.clear()
                print("[INFO] Resetting environment...")
                obs, _ = env.reset()
                if command_term is not None:
                    command_term.reset_manual_command()
                    print("[INFO] Command reset to zero.")
            # onnx interface
            if use_onnx_inference:
                actions = onnx_policy(obs["policy"], obs["depth"].view(1, -1))
            else:
                actions = policy(obs)
                if args_cli.visualize_moe and timestep % args_cli.moe_visual_interval == 0:
                    gate_scores = get_cmoe_gate_scores(policy_nn, obs)
                    moe_bars = visualize_moe_online(
                        gate_scores,
                        timestep,
                        args_cli.moe_env_idx,
                        moe_fig,
                        moe_axes,
                        moe_bars,
                    )
            # env stepping
            obs, _, _, _ = env.step(actions)

            # 在线可视化深度图（如果需要）
            if args_cli.visualize_depth and timestep % args_cli.depth_visual_interval == 0:
                if "depth" in obs:
                    # 计算注意力权重（如果使用AttnDepthEncoder）
                    attn_weights = None
                    if use_attn_visualization and attn_depth_encoder is not None:
                        with torch.no_grad():
                            # 获取proprio, vel, latent（与update_distribution中相同）
                            proprio_obs = policy_nn.get_actor_obs(obs)
                            if hasattr(policy_nn, "actor_obs_normalizer"):
                                proprio_obs = policy_nn.actor_obs_normalizer(proprio_obs)
                            # vel, _ = policy_nn.him_estimator(proprio_obs)
                            if hasattr(policy_nn, "him_estimator"):
                                vel, _ = policy_nn.him_estimator(proprio_obs)
                            else:
                                depth = policy_nn.get_depth_obs(obs)
                                vel, _, _ = policy_nn.estimator(proprio_obs, depth)
                            
                            # 获取第一个环境的输入（用于可视化）
                            # 深度图形状为 [B, H, W, 1]
                            depth_batch = policy_nn.get_depth_obs(obs)[:1]  # [1, H, W, 1]
                            proprio_batch = proprio_obs[:1, -policy_nn.num_one_step_obs:]
                            vel_batch = vel[:1]
                            
                            attn_weights = attn_depth_encoder.get_attention_weights(
                                depth_batch, proprio_batch, vel_batch
                            )
                    
                    # 可视化深度图（无论是否有注意力权重）
                    depth_im, depth_cbar, attn_im, attn_cbar, attn_depth_bg = visualize_depth_online(
                        obs["depth"], timestep, depth_fig, depth_axes, depth_im, depth_cbar, depth_vmin, depth_vmax,
                        attn_axes, attn_im, attn_cbar, attn_weights, attn_depth_encoder, attn_depth_bg
                    )
                    
                    # 3D可视化注意力权重：直接使用raycaster_camera的ray_visualizer（仅在存在注意力权重时）
                    if attn_weights is not None and raycaster_camera is not None:
                        # 获取第一个环境的注意力权重
                        attn_weights_env0 = attn_weights[0, 0].cpu()  # [1, 1, num_patches]
                        
                        # 获取ray hits（第一个环境）
                        ray_hits_w = raycaster_camera.ray_hits_w[0].cpu()  # [num_rays, 3]
                        
                        # 获取深度图尺寸（用于reshape注意力权重）
                        H, W = raycaster_camera.image_shape  # (height, width)
                        
                        # 将注意力权重reshape为与深度图对应的格式
                        patch_size = 2  # 默认值
                        
                        # 使用patch_size计算patches的H和W
                        patch_H = H // patch_size
                        patch_W = W // patch_size
                        
                        # 上采样到原始深度图尺寸
                        attn_2d = attn_weights_env0.view(patch_H, patch_W)
                        # 使用简单的重复上采样
                        attn_2d_upsampled = attn_2d.repeat_interleave(patch_size, dim=0).repeat_interleave(patch_size, dim=1)
                        attn_flat = attn_2d_upsampled[:H, :W].flatten()
                        
                        # 归一化注意力权重到 [0, 1]
                        attn_norm = (attn_flat - attn_flat.min()) / (attn_flat.max() - attn_flat.min() + 1e-8)
                        
                        # 获取marker类型数量
                        num_marker_types = len(raycaster_camera.ray_visualizer.cfg.markers)
                        
                        # 将注意力权重映射到marker类型索引（0到num_marker_types-1）
                        marker_indices = (attn_norm * (num_marker_types - 1)).long().clamp(0, num_marker_types - 1)
                        
                        # 过滤无效的ray hits（inf或nan）
                        valid_mask = ~torch.any(torch.isinf(ray_hits_w) | torch.isnan(ray_hits_w), dim=1)
                        ray_hits_valid = ray_hits_w[valid_mask]
                        marker_indices_valid = marker_indices[valid_mask]
                        
                        # 使用raycaster_camera的visualizer可视化注意力权重
                        raycaster_camera.ray_visualizer.visualize(translations=ray_hits_valid, marker_indices=marker_indices_valid)
                        
            # print(f"Command: {env.unwrapped.command_manager.get_command('base_velocity').cpu().numpy()[0]}, Current Velocity: {env.unwrapped.robot.data.root_lin_vel_b.cpu().numpy()[0]}")
        timestep += 1
        if args_cli.video:
            # Exit the play loop after recording one video
            print(f"[INFO] Recording video: {timestep}/{args_cli.video_length}")
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()
    if command_listener is not None:
        command_listener.stop()

    # 关闭深度图窗口
    if args_cli.visualize_depth and depth_fig is not None:
        plt.close(depth_fig)
        print("[INFO] Depth visualization window closed.")
    if args_cli.visualize_moe and moe_fig is not None:
        plt.close(moe_fig)
        print("[INFO] MoE visualization window closed.")


if __name__ == "__main__":
    EXPORT_POLICY = True
    INTERFACE_ONNX = False
    CONTROL_ROBOT = True
    # run the main function
    main()
    # close sim app
    simulation_app.close()
