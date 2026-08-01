# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Definitions for neural-network components for RL-agents."""

from .actor_critic import ActorCritic
from .actor_critic_recurrent import ActorCriticRecurrent
from .discriminator import Discriminator
from .foothold_imagination import FootholdImagination
from .him_actor_critic import HimActorCritic
from .pim_actor_critic import PIMActorCritic
from .pim_depth_actor_critic import PIMDepthActorCritic
from .pim_depth_moe_actor_critic import PIMDepthMoEActorCritic
from .gate_pim_actor_critic_cmoe import GatePIMActorCriticCMoE
from .him_depth_actor_critic import HimDepthActorCritic
from .him_double_depth_actor_critic import HimDoubleDepthActorCritic
from .old_him_depth_actor_critic import OldHimDepthActorCritic
from .old_him_double_depth_actor_critic import OldHimDoubleDepthActorCritic
from .old_him_depth_moe_actor_critic import OldHimDepthMoEActorCritic
from .old_him_double_depth_moe_actor_critic import OldHimDoubleDepthMoEActorCritic

from .rnd import *
from .student_teacher import StudentTeacher
from .student_teacher_recurrent import StudentTeacherRecurrent
from .symmetry import *

__all__ = [
    "ActorCritic",
    "ActorCriticRecurrent",
    "HimActorCritic",
    "PIMActorCritic",
    "PIMDepthActorCritic",
    "PIMDepthMoEActorCritic",
    "GatePIMActorCriticCMoE",
    "HimDepthActorCritic",
    "HimDoubleDepthActorCritic",
    "OldHimDepthActorCritic",
    "OldHimDoubleDepthActorCritic",
    "OldHimDepthMoEActorCritic",
    "OldHimDoubleDepthMoEActorCritic",
    "StudentTeacher",
    "StudentTeacherRecurrent",
    "Discriminator",
    "FootholdImagination",
]
