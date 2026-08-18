# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (  # noqa:F401
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)
from NoetixE1.assets.robots.e1.e1 import E1_23DOF_JOINT_NAMES


@configclass
class RslRlAmpPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name = "AMPOldHIMPPO"  # AMPHIMPPO
    value_loss_coef = 1.0
    use_clipped_value_loss = True
    clip_param = 0.2
    entropy_coef = 0.01
    num_learning_epochs = 5
    num_mini_batches = 4
    learning_rate = 1.0e-3
    schedule = "adaptive"
    gamma = 0.99
    lam = 0.95
    desired_kl = 0.01
    max_grad_norm = 1.0
    discriminator_learning_rate = 5e-6
    discriminator_gradient_penalty_coef = 5
    discriminator_num_mini_batches = 80
    discriminator_loss_function = "WassersteinLoss"  # BCEWithLogitsLoss, MSELoss, WassersteinLoss
    amp_replay_buffer_size = 200000
    normalize_advantage_per_mini_batch = False
    foothold_imagination_enabled = True
    foothold_learning_rate = 1.0e-3
    foothold_hidden_dims = (512, 256, 128)
    foothold_min_sigma = 0.02
    foothold_max_sigma = 0.50
    foothold_replay_buffer_size = 50000
    foothold_pending_steps = 16
    foothold_batch_size = 2048
    foothold_updates_per_iteration = 4
    symmetry_cfg = RslRlSymmetryCfg(
        use_data_augmentation=True,
        use_mirror_loss=False,
        data_augmentation_func="NoetixE1.tasks.walkrun.noetix_e1.mdp.symmetry.e1_old_him:data_augmentation_func_e1",
        mirror_loss_coeff=1.0,
    )
    rnd_cfg = None


@configclass
class RslRlHimActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name = "OldHimDepthMoEActorCritic"
    init_noise_std = 1.0
    actor_obs_normalization = False
    critic_obs_normalization = False
    actor_hidden_dims = [256, 128, 64]
    critic_hidden_dims = [256, 128, 64]
    num_moe_experts = 4
    depth_class_name = "CnnAttnDepthEncoder"
    depth_latent_dim = 32
    depth_stack = 1
    command_dim = 3
    estimate_dim = 3
    activation = "elu"


@configclass
class AmpPpoDepthRunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = "AmpOldHimOnPolicyRunner"
    seed = 42
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 500
    policy = RslRlHimActorCriticCfg()
    algorithm = RslRlAmpPpoAlgorithmCfg()
    clip_actions = 18.0

    experiment_name = "walkrun_depth"
    run_name = ""
    logger = "tensorboard"
    neptune_project = "walkrun_depth"
    wandb_project = "walkrun_depth"
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"

    # amp discriminator parameter
    normalize_style_reward = False
    amp_reward_coef = 0.8  # motion:(0.4, 0.5) policy:(0.8, 0.8)
    amp_reward_lerp = 0.8  # reward = reward_coef * (1 - reward_lerp) * style_reward + reward_lerp * task_reward
    style_reward_function = "wasserstein_mapping"  # log_mapping, quad_mapping, wasserstein_mapping
    discriminator_shape = [1024, 512]
    # masking configuration
    """
        AMP观测结构（单帧，从JOINT_POSE开始的126维）：
        ┌──────────────────┬──────────┬────────────┐
        │ part             │ range    │ size       │
        ├──────────────────┼──────────┼────────────┤
        │ Joint pos        │ [0:23]   │ 23         │
        │ Key pos          │ [23:95]  │ 72         │
        │ Linear velocity  │ [95:98]  │ 3 (XYZ)    │
        │ Angular velocity │ [98:101] │ 3 (RPY)    │
        │ Joint velocity   │ [101:124]│ 23         │
        │ Contact mask     │ [124:126]│ 2          │
        └──────────────────┴──────────┴────────────┘
        Mask Example:
        discriminator_mask_joint_names = [
            "l_leg_ankle_pitch_joint",
            "r_leg_ankle_pitch_joint",
            "l_leg_ankle_roll_joint",
            "r_leg_ankle_roll_joint",
        ]
        discriminator_mask_dims = [
            (95, 98),    # Linear velocity
            (98, 101),   # Angular velocity
        ]
    """
    joint_names = E1_23DOF_JOINT_NAMES
    discriminator_mask_joint_names = None
    discriminator_mask_dims = None  # None or list of tuples
    discriminator_height_scan_encoder_enable = False
    discriminator_height_scan_start_idx = 148
    discriminator_height_scan_input_dim = 187
    discriminator_height_scan_latent_dim = 16
    discriminator_height_scan_encoder_shape = (64, 32)


@configclass
class AmpPpoDoubleDepthRunnerCfg(AmpPpoDepthRunnerCfg):
    policy = RslRlHimActorCriticCfg()
    policy.class_name = "OldHimDoubleDepthMoEActorCritic"


@configclass
class RslRlHimParkourActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name = "OldHimDepthMoEActorCritic"
    init_noise_std = 1.0
    actor_obs_normalization = False
    critic_obs_normalization = False
    actor_hidden_dims = [256, 128, 64]
    critic_hidden_dims = [256, 128, 64]
    num_moe_experts = 4
    depth_class_name = "AdaptLightDepthEncoder"
    depth_latent_dim = 32
    depth_stack = 1
    command_dim = 3
    estimate_dim = 3
    activation = "elu"


@configclass
class AmpPpoParkourDepthRunnerCfg(AmpPpoDepthRunnerCfg):
    policy = RslRlHimParkourActorCriticCfg()
    experiment_name = "parkour_depth"


@configclass
class AmpExpandPpoParkourDepthRunnerCfg(AmpPpoParkourDepthRunnerCfg):
    # masking configuration
    """
        AMP观测结构（单帧，从JOINT_POSE开始的335维）：
        ┌──────────────────┬──────────┬────────────┐
        │ part             │ range    │ size       │
        ├──────────────────┼──────────┼────────────┤
        │ Joint pos        │ [0:23]   │ 23         │
        │ Key pos          │ [23:95]  │ 72         │
        │ Linear velocity  │ [95:98]  │ 3 (XYZ)    │
        │ Angular velocity │ [98:101] │ 3 (RPY)    │
        │ Joint velocity   │ [101:124]│ 23         │
        │ Contact mask     │ [124:148]│ 24         │
        │ Height scan      │ [148:335]│ 187        │
        └──────────────────┴──────────┴────────────┘
        Mask Example:
        discriminator_mask_joint_names = [
            "l_leg_ankle_pitch_joint",
            "r_leg_ankle_pitch_joint",
            "l_leg_ankle_roll_joint",
            "r_leg_ankle_roll_joint",
        ]
        discriminator_mask_dims = [
            (95, 98),    # Linear velocity
            (98, 101),   # Angular velocity
        ]
    """
    discriminator_mask_joint_names = None
    discriminator_mask_dims = None  # None or list of tuples
    discriminator_height_scan_encoder_enable = True
    discriminator_height_scan_start_idx = 148
    discriminator_height_scan_input_dim = 187
    discriminator_height_scan_latent_dim = 16
    discriminator_height_scan_encoder_shape = (64, 32)
