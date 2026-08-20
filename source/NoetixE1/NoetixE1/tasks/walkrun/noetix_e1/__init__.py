# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="WalkRun-E1-AMP-v0",
    entry_point="NoetixE1.tasks.walkrun.noetix_e1.env:ManagerBasedRLAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walkrun_amp_cfg:WalkRunFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_ppo_cfg:AmpPpoRunnerCfg",
    },
)

gym.register(
    id="WalkRun-E1-AMP-Depth",
    entry_point="NoetixE1.tasks.walkrun.noetix_e1.env:ManagerBasedRLAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walkrun_amp_depth_cfg:WalkRunDepthEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_depth_ppo_cfg:AmpPpoDepthRunnerCfg",
    },
)

gym.register(
    id="WalkRun-E1-AMP-Double-Depth",
    entry_point="NoetixE1.tasks.walkrun.noetix_e1.env:ManagerBasedRLAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walkrun_amp_double_depth_cfg:WalkRunDepthEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_depth_ppo_cfg:AmpPpoDoubleDepthRunnerCfg",
    },
)

gym.register(
    id="WalkRun-E1-AMP-Old-Him-Depth",
    entry_point="NoetixE1.tasks.walkrun.noetix_e1.env:ManagerBasedRLAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walkrun_amp_depth_cfg:WalkRunDepthEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_old_him_depth_ppo_cfg:AmpPpoDepthRunnerCfg",
    },
)

gym.register(
    id="WalkRun-E1-AMP-Old-Him-Double-Depth",
    entry_point="NoetixE1.tasks.walkrun.noetix_e1.env:ManagerBasedRLAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walkrun_amp_double_depth_cfg:WalkRunDepthEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_old_him_depth_ppo_cfg:AmpPpoDoubleDepthRunnerCfg",
    },
)

gym.register(
    id="WalkRun-E1-AMP-Old-Him-Depth-MoE",
    entry_point="NoetixE1.tasks.walkrun.noetix_e1.env:ManagerBasedRLAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walkrun_amp_depth_cfg:WalkRunDepthEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_old_him_depth_moe_ppo_cfg:AmpPpoDepthRunnerCfg",
    },
)

gym.register(
    id="WalkRun-E1-AMP-Old-Him-Depth-MoE-NoHurdle",
    entry_point="NoetixE1.tasks.walkrun.noetix_e1.env:ManagerBasedRLAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.walkrun_amp_depth_no_hurdle_cfg:WalkRunDepthNoHurdleEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_old_him_depth_moe_no_hurdle_ppo_cfg:"
            "AmpPpoNoHurdleDepthRunnerCfg"
        ),
    },
)

gym.register(
    id="WalkRun-E1-AMP-Old-Him-Double-Depth-MoE",
    entry_point="NoetixE1.tasks.walkrun.noetix_e1.env:ManagerBasedRLAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walkrun_amp_double_depth_cfg:WalkRunDepthEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_old_him_depth_moe_ppo_cfg:AmpPpoDoubleDepthRunnerCfg",
    },
)

gym.register(
    id="WalkRun-E1-AMP-PIM",
    entry_point="NoetixE1.tasks.walkrun.noetix_e1.env:ManagerBasedRLAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walkrun_amp_depth_cfg:WalkRunDepthEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_pim_ppo_cfg:AmpPpoDepthRunnerCfg",
    },
)

gym.register(
    id="WalkRun-E1-AMP-PIM-Depth",
    entry_point="NoetixE1.tasks.walkrun.noetix_e1.env:ManagerBasedRLAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walkrun_amp_depth_cfg:WalkRunDepthEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_pim_depth_ppo_cfg:AmpPpoDepthRunnerCfg",
    },
)

gym.register(
    id="WalkRun-E1-AMP-PIM-Depth-MoE",
    entry_point="NoetixE1.tasks.walkrun.noetix_e1.env:ManagerBasedRLAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walkrun_amp_depth_cfg:WalkRunDepthEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_pim_depth_ppo_cfg:AmpPpoDepthMoERunnerCfg",
    },
)

gym.register(
    id="WalkRun-E1-AMP-Gate-PIM-Depth-CMoE",
    entry_point="NoetixE1.tasks.walkrun.noetix_e1.env:ManagerBasedRLAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walkrun_amp_depth_cfg:WalkRunDepthEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_pim_depth_ppo_cfg:AmpPpoDepthCMoERunnerCfg",
    },
)

gym.register(
    id="WalkRun-E1-AMP-Gate-PIM-ScanDots-CMoE",
    entry_point="NoetixE1.tasks.walkrun.noetix_e1.env:ManagerBasedRLAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walkrun_amp_depth_cfg:WalkRunDepthEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_pim_depth_ppo_cfg:AmpPpoScanDotsCMoERunnerCfg",
    },
)

gym.register(
    id="WalkRun-E1-AMP-Refine-Motion",
    entry_point="NoetixE1.tasks.walkrun.noetix_e1.env:ManagerBasedRLAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walkrun_amp_refine_motion_cfg:WalkRunDepthEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_ppo_refine_motion_cfg:AmpPpoRunnerCfg",
    },
)

gym.register(
    id="WalkRun-E1-AMP-Parkour-Test",
    entry_point="NoetixE1.tasks.walkrun.noetix_e1.env:ManagerBasedRLAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walkrun_amp_parkour_test_cfg:WalkRunDepthEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_ppo_refine_motion_cfg:AmpPpoRunnerCfg",
    },
)

gym.register(
    id="WalkRun-E1-AMP-Parkour-Depth",
    entry_point="NoetixE1.tasks.walkrun.noetix_e1.env:ManagerBasedRLAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walkrun_amp_parkour_depth_cfg:WalkRunDepthEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_old_him_depth_moe_ppo_cfg:AmpPpoParkourDepthRunnerCfg",
    },
)

gym.register(
    id="WalkRun-E1-AMP-Expand-Parkour-Depth",
    entry_point="NoetixE1.tasks.walkrun.noetix_e1.env:ManagerBasedRLAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walkrun_amp_parkour_depth_cfg:WalkRunAMPExpandDepthEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_old_him_depth_moe_ppo_cfg:AmpExpandPpoParkourDepthRunnerCfg",
    },
)