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
    class_name = "AMPPIMPPO"  # AMPPIMPPO
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
    symmetry_cfg = RslRlSymmetryCfg(
        use_data_augmentation=True,
        use_mirror_loss=False,
        data_augmentation_func="NoetixE1.tasks.walkrun.noetix_e1.mdp.symmetry.e1_old_him:data_augmentation_func_e1",
        mirror_loss_coeff=1.0,
    )
    rnd_cfg = None


@configclass
class RslRlPimActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name = "PIMDepthActorCritic"
    init_noise_std = 1.0
    actor_obs_normalization = False
    critic_obs_normalization = False
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    estimator_class_name = "PIMEstimator"
    encoder_hidden_dims = [256, 64, 16]
    projector_hidden_dims = [256, 64]
    depth_class_name = "LightDepthEncoder"
    depth_latent_dim = 32
    depth_stack = 1
    activation = "elu"


@configclass
class AmpPpoDepthRunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = "AmpPimOnPolicyRunner"
    seed = 42
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 500
    policy = RslRlPimActorCriticCfg()
    algorithm = RslRlAmpPpoAlgorithmCfg()
    clip_actions = 18.0

    experiment_name = "walkrun_pim"
    run_name = ""
    logger = "tensorboard"
    neptune_project = "walkrun_pim"
    wandb_project = "walkrun_pim"
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


@configclass
class RslRlPimActorCriticMoECfg(RslRlPpoActorCriticCfg):
    class_name = "PIMDepthMoEActorCritic"
    init_noise_std = 1.0
    actor_obs_normalization = False
    critic_obs_normalization = False
    actor_hidden_dims = [256, 128, 64]
    critic_hidden_dims = [256, 128, 64]
    estimator_class_name = "PIMEstimator2"
    encoder_hidden_dims = [256, 64, 32]
    projector_hidden_dims = [256, 64]
    num_moe_experts = 4
    depth_class_name = "CnnAttnDepthEncoder"
    depth_latent_dim = 32
    activation = "elu"


@configclass
class AmpPpoDepthMoERunnerCfg(AmpPpoDepthRunnerCfg):
    policy = RslRlPimActorCriticMoECfg()


@configclass
class RslRlGatePimDepthActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name = "GatePIMActorCriticCMoE"
    init_noise_std = 1.0
    actor_obs_normalization = False
    critic_obs_normalization = False
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    estimator_class_name = "GatePIMEstimatorDepth"
    encoder_hidden_dims = [256, 64, 16]
    projector_hidden_dims = [256, 64]
    num_moe_experts = 4
    depth_class_name = "CnnAttnDepthEncoder"
    depth_latent_dim = 32
    depth_stack = 1
    activation = "elu"


@configclass
class RslRlAmpGatePIMPpoAlgorithmCfg(RslRlAmpPpoAlgorithmCfg):
    class_name = "AMPGatePIMPPO"  # AMPGatePIMPPO


@configclass
class AmpPpoDepthCMoERunnerCfg(AmpPpoDepthRunnerCfg):
    policy = RslRlGatePimDepthActorCriticCfg()
    algorithm = RslRlAmpGatePIMPpoAlgorithmCfg()
    experiment_name = "walkrun_gate_pim"


@configclass
class RslRlGatePimScanDotsActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name = "GatePIMActorCriticCMoE"
    init_noise_std = 1.0
    actor_obs_normalization = False
    critic_obs_normalization = False
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    estimator_class_name = "GatePIMEstimatorScanDots"
    encoder_hidden_dims = [256, 64, 16]
    projector_hidden_dims = [256, 64]
    num_moe_experts = 4
    depth_class_name = "CnnAttnDepthEncoder"
    depth_latent_dim = 32
    depth_stack = 1
    activation = "elu"


@configclass
class AmpPpoScanDotsCMoERunnerCfg(AmpPpoDepthRunnerCfg):
    policy = RslRlGatePimScanDotsActorCriticCfg()
    algorithm = RslRlAmpGatePIMPpoAlgorithmCfg()
    experiment_name = "walkrun_gate_pim"
