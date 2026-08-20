from isaaclab.utils import configclass

from .rsl_rl_amp_old_him_depth_moe_ppo_cfg import AmpPpoDepthRunnerCfg, RslRlAmpPpoAlgorithmCfg


@configclass
class AmpPpoNoHurdleDepthRunnerCfg(AmpPpoDepthRunnerCfg):
    """Old-HIM depth MoE runner with an isolated no-hurdle experiment log."""

    algorithm = RslRlAmpPpoAlgorithmCfg()
    algorithm.symmetry_cfg.use_mirror_loss = True
    algorithm.symmetry_cfg.mirror_loss_coeff = 2.0
    algorithm.foothold_imagination_enabled = False
    experiment_name = "walkrun_depth_no_hurdle"
