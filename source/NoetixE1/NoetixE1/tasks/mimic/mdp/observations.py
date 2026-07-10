# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat, quat_apply, quat_inv, quat_mul, subtract_frame_transforms
from NoetixE1.tasks.mimic.mdp.commands import MotionCommand
from isaaclab.sensors import ContactSensor, RayCasterCamera
from NoetixE1.sensors import MultiMeshRayCasterCamera
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class RandomRectMaskGPU:
    """随机矩形mask生成器，支持时间持续性。

    在深度图上随机生成矩形mask区域，这些mask会持续一定步数，模拟深度传感器的随机失效。
    """

    def __init__(
        self,
        p: float = 0.5,
        num_masks: tuple[int, int] = (1, 3),
        rect_ratio: tuple[float, float] = (0.1, 0.25),
        persist_steps: int = 5,
    ):
        """初始化随机矩形mask生成器。

        Args:
            p: 启用mask的概率
            num_masks: 每个样本mask数量范围 (min, max)
            rect_ratio: mask占比范围 (min, max)，相对于图像尺寸的比例
            persist_steps: mask持续步数
        """
        self.p = p
        self.num_masks = num_masks
        self.rect_ratio = rect_ratio
        self.persist_steps = persist_steps
        self._mask = None
        self._counter = 0

    @torch.no_grad()
    def __call__(self, depth: torch.Tensor) -> torch.Tensor:
        """对深度图应用随机矩形mask。

        Args:
            depth: 深度图，形状为 [B, 1, H, W]

        Returns:
            应用mask后的深度图，形状为 [B, 1, H, W]
        """
        B, _, H, W = depth.shape
        device = depth.device

        # 复用旧mask（如果还在持续期内）
        if self._mask is not None and self._counter < self.persist_steps:
            self._counter += 1
            return depth * self._mask

        # 生成新mask
        mask = torch.ones((B, 1, H, W), device=device)

        # 决定哪些样本启用mask
        apply_mask = torch.rand(B, device=device) < self.p
        if not apply_mask.any():
            self._mask = mask
            self._counter = 1
            return depth

        # 为每个样本生成mask数量
        num_masks = torch.randint(self.num_masks[0], self.num_masks[1] + 1, (B,), device=device)

        # 最大mask数量
        max_masks = num_masks.max().item()

        # 生成随机矩形参数
        rh = (torch.rand(B, max_masks, device=device) * (self.rect_ratio[1] - self.rect_ratio[0]) + self.rect_ratio[0]) * H
        rw = (torch.rand(B, max_masks, device=device) * (self.rect_ratio[1] - self.rect_ratio[0]) + self.rect_ratio[0]) * W
        rh = rh.long().clamp(min=1, max=H)
        rw = rw.long().clamp(min=1, max=W)

        y0 = (torch.rand(B, max_masks, device=device) * (H - rh.float())).long()
        x0 = (torch.rand(B, max_masks, device=device) * (W - rw.float())).long()

        # 构建mask
        yy = torch.arange(H, device=device).view(1, 1, H, 1)
        xx = torch.arange(W, device=device).view(1, 1, 1, W)

        for m in range(max_masks):
            # 哪些batch含有第m个mask
            valid = (m < num_masks) & apply_mask
            if not valid.any():
                continue

            y_min = y0[:, m].view(B, 1, 1, 1)
            y_max = (y0[:, m] + rh[:, m]).view(B, 1, 1, 1)
            x_min = x0[:, m].view(B, 1, 1, 1)
            x_max = (x0[:, m] + rw[:, m]).view(B, 1, 1, 1)

            rect = ((yy >= y_min) & (yy < y_max) & (xx >= x_min) & (xx < x_max))
            rect = rect & valid.view(B, 1, 1, 1)
            mask[rect] = 0.0

        self._mask = mask
        self._counter = 1
        return depth * mask


class HighReflectionNoiseGPU:
    """高反噪声生成器，支持时间持续性。

    在深度图上随机生成高反矩形区域，这些区域内的深度值会乘以一个大于1的系数，
    模拟高反射表面导致的深度测量异常。
    """

    def __init__(
        self,
        p: float = 0.5,
        num_patches: tuple[int, int] = (1, 3),
        rect_ratio: tuple[float, float] = (0.1, 0.25),
        coefficient_range: tuple[float, float] = (1.1, 1.5),
        persist_steps: int = 5,
    ):
        """初始化高反噪声生成器。

        Args:
            p: 启用高反噪声的概率
            num_patches: 每个样本高反块数量范围 (min, max)
            rect_ratio: 高反块占比范围 (min, max)，相对于图像尺寸的比例
            coefficient_range: 高反系数范围 (min, max)，每个高反块内的系数相同
            persist_steps: 高反块持续步数
        """
        self.p = p
        self.num_patches = num_patches
        self.rect_ratio = rect_ratio
        self.coefficient_range = coefficient_range
        self.persist_steps = persist_steps
        self._coefficient_map = None  # 存储每个像素的系数（1.0表示无高反）
        self._counter = 0

    @torch.no_grad()
    def __call__(self, depth: torch.Tensor) -> torch.Tensor:
        """对深度图应用高反噪声。

        Args:
            depth: 深度图，形状为 [B, 1, H, W]

        Returns:
            应用高反噪声后的深度图，形状为 [B, 1, H, W]
        """
        B, _, H, W = depth.shape
        device = depth.device

        # 复用旧系数映射（如果还在持续期内）
        if self._coefficient_map is not None and self._counter < self.persist_steps:
            self._counter += 1
            return depth * self._coefficient_map

        # 生成新系数映射（初始为1.0，表示无高反）
        coefficient_map = torch.ones((B, 1, H, W), device=device)

        # 决定哪些样本启用高反噪声
        apply_noise = torch.rand(B, device=device) < self.p
        if not apply_noise.any():
            self._coefficient_map = coefficient_map
            self._counter = 1
            return depth

        # 为每个样本生成高反块数量
        num_patches = torch.randint(self.num_patches[0], self.num_patches[1] + 1, (B,), device=device)

        # 最大高反块数量
        max_patches = num_patches.max().item()

        # 生成随机矩形参数
        rh = (torch.rand(B, max_patches, device=device) * (self.rect_ratio[1] - self.rect_ratio[0]) + self.rect_ratio[0]) * H
        rw = (torch.rand(B, max_patches, device=device) * (self.rect_ratio[1] - self.rect_ratio[0]) + self.rect_ratio[0]) * W
        rh = rh.long().clamp(min=1, max=H)
        rw = rw.long().clamp(min=1, max=W)

        y0 = (torch.rand(B, max_patches, device=device) * (H - rh.float())).long()
        x0 = (torch.rand(B, max_patches, device=device) * (W - rw.float())).long()

        # 为每个高反块生成系数（每个块内的系数相同）
        coefficients = torch.rand(B, max_patches, device=device) * (self.coefficient_range[1] - self.coefficient_range[0]) + self.coefficient_range[0]

        # 构建系数映射
        yy = torch.arange(H, device=device).view(1, 1, H, 1)
        xx = torch.arange(W, device=device).view(1, 1, 1, W)

        for p in range(max_patches):
            # 哪些batch含有第p个高反块
            valid = (p < num_patches) & apply_noise
            if not valid.any():
                continue

            y_min = y0[:, p].view(B, 1, 1, 1)
            y_max = (y0[:, p] + rh[:, p]).view(B, 1, 1, 1)
            x_min = x0[:, p].view(B, 1, 1, 1)
            x_max = (x0[:, p] + rw[:, p]).view(B, 1, 1, 1)

            # 矩形区域
            rect = ((yy >= y_min) & (yy < y_max) & (xx >= x_min) & (xx < x_max))
            rect = rect & valid.view(B, 1, 1, 1)

            # 将矩形区域内的系数设置为对应的高反系数
            # 使用广播：coefficients[:, p] 的形状是 (B,)，会自动广播到 rect 选中的区域
            coeff_values = coefficients[:, p].view(B, 1, 1, 1)
            coefficient_map = torch.where(rect, coeff_values, coefficient_map)

        self._coefficient_map = coefficient_map
        self._counter = 1
        return depth * coefficient_map


class FullImageMaskGPU:
    """整张图片mask生成器，支持时间持续性。

    在深度图上随机生成整张图片的mask，模拟深度传感器完全失效的情况。
    """

    def __init__(
        self,
        p: float = 0.5,
        persist_steps: int = 5,
    ):
        """初始化整张图片mask生成器。

        Args:
            p: 启用mask的概率
            persist_steps: mask持续步数
        """
        self.p = p
        self.persist_steps = persist_steps
        self._mask = None
        self._counter = 0

    @torch.no_grad()
    def __call__(self, depth: torch.Tensor) -> torch.Tensor:
        """对深度图应用整张图片mask。

        Args:
            depth: 深度图，形状为 [B, 1, H, W]

        Returns:
            应用mask后的深度图，形状为 [B, 1, H, W]
        """
        B, _, H, W = depth.shape
        device = depth.device

        # 复用旧mask（如果还在持续期内）
        if self._mask is not None and self._counter < self.persist_steps:
            self._counter += 1
            return depth * self._mask

        # 生成新mask
        mask = torch.ones((B, 1, H, W), device=device)

        # 决定哪些样本启用mask
        apply_mask = torch.rand(B, device=device) < self.p
        if apply_mask.any():
            # 对启用的样本，将整张图片mask掉（设为0）
            mask[apply_mask] = 0.0

        self._mask = mask
        self._counter = 1
        return depth * mask


def robot_anchor_ori_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    mat = matrix_from_quat(command.robot_anchor_quat_w)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_anchor_lin_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, :3].view(env.num_envs, -1)


def robot_anchor_ang_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, 3:6].view(env.num_envs, -1)


def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )

    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)


def generated_commands_multi_frame(
    env: ManagerBasedRLEnv,
    command_name: str,
    num_frames: int = 5,
    frame_interval: int = 1,
) -> torch.Tensor:
    """观测未来多帧的 reference motion（joint_pos + joint_vel），越界时取最后一帧不循环。

    与 generated_commands 仅“未来一帧”不同，这里返回 [t+1, t+1+interval, t+1+2*interval, ...]
    共 num_frames 帧的 (joint_pos, joint_vel) 拼接；若某帧索引超出 motion 末尾则 clamp 到最后一帧。

    Args:
        env: 环境
        command_name: MotionCommand 名称，如 "motion"
        num_frames: 未来帧数（默认 5）
        frame_interval: 帧间间隔（步数），1 表示连续帧（默认 1）

    Returns:
        (num_envs, num_frames * (num_joints * 2)) 的 Tensor
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    motion = command.motion
    total = motion.time_step_total
    device = command.time_steps.device

    # 未来帧索引：(num_envs, num_frames)，超出末尾时 clamp 到最后一帧，不从头循环
    # 第 i 帧对应 time_steps + i * frame_interval
    step_offsets = torch.arange(num_frames, device=device, dtype=torch.long) * frame_interval
    indices = (command.time_steps.unsqueeze(1) + step_offsets.unsqueeze(0)).clamp(max=total - 1)

    idx_flat = indices.reshape(-1)
    joint_pos_future = motion.joint_pos[idx_flat].reshape(env.num_envs, num_frames, -1)
    joint_vel_future = motion.joint_vel[idx_flat].reshape(env.num_envs, num_frames, -1)
    # 与单帧 command 一致：每帧 [joint_pos, joint_vel]，再展平
    return torch.cat([joint_pos_future, joint_vel_future], dim=-1).reshape(env.num_envs, -1)


def position_ref_multi_frame(
    env: ManagerBasedRLEnv,
    command_name: str,
    num_frames: int = 5,
    frame_interval: int = 1,
    anchor_frame: str = "reference",
) -> torch.Tensor:
    """未来多帧 motion anchor 相对于锚点的位置差（在锚点系下），越界 clamp 到最后一帧。

    对应 InstinctLab position_b_ref_command (anchor_frame=reference) 与 position_ref_command (anchor_frame=robot)。
    - anchor_frame="reference"（用于 policy）：未来 motion anchor 相对**当前 motion anchor** 的位置差，在**当前 motion 系**下。
    - anchor_frame="robot"（用于 privilege）：未来 motion anchor 相对**当前真实 robot anchor** 的位置差，在**robot 系**下。

    Returns:
        (num_envs, num_frames * 3)，仅 anchor 一个 body 的多帧位置差。
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    motion = command.motion
    total = motion.time_step_total
    device = command.time_steps.device
    anchor_body_raw_idx = motion._body_indexes[command.motion_anchor_body_index]

    step_offsets = torch.arange(num_frames, device=device, dtype=torch.long) * frame_interval
    indices = (command.time_steps.unsqueeze(1) + step_offsets.unsqueeze(0)).clamp(max=total - 1)
    idx_flat = indices.reshape(-1)

    # 未来多帧 motion anchor 世界系位置 (num_envs, num_frames, 3)
    future_anchor_pos_w = motion._body_pos_w[idx_flat, anchor_body_raw_idx, :].reshape(env.num_envs, num_frames, 3)
    future_anchor_pos_w = future_anchor_pos_w + env.scene.env_origins.unsqueeze(1)

    if anchor_frame == "reference":
        # 当前 motion anchor 作为锚点：位置差在当前 motion 系下
        anchor_pos_w = command.anchor_pos_w
        anchor_quat_w = command.anchor_quat_w
    else:
        # 当前真实 robot anchor 作为锚点：位置差在 robot 系下
        anchor_pos_w = command.robot_anchor_pos_w
        anchor_quat_w = command.robot_anchor_quat_w

    delta_w = future_anchor_pos_w - anchor_pos_w.unsqueeze(1)
    # quat_apply 要求 quat (N,4) 与 vec (N,3) 的 N 一致，展平为 (num_envs*num_frames,)
    quat_flat = quat_inv(anchor_quat_w).unsqueeze(1).expand(-1, num_frames, -1).reshape(-1, 4)
    delta_flat = delta_w.reshape(-1, 3)
    pos_b = quat_apply(quat_flat, delta_flat)
    return pos_b.reshape(env.num_envs, num_frames * 3)


def motion_anchor_ori_b_multi_frame(
    env: ManagerBasedRLEnv,
    command_name: str,
    num_frames: int = 5,
    frame_interval: int = 1,
) -> torch.Tensor:
    """未来多帧 motion anchor 相对于锚点的朝向（在锚点系下，6D），越界 clamp 到最后一帧。

    - anchor_frame="reference"（policy）：未来 motion anchor 相对**当前 motion anchor** 的朝向，在当前 motion 系下。
    - anchor_frame="robot"（privilege）：未来 motion anchor 相对**当前真实 robot anchor** 的朝向，在 robot 系下。

    Returns:
        (num_envs, num_frames * 6)
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    motion = command.motion
    total = motion.time_step_total
    device = command.time_steps.device
    anchor_body_raw_idx = motion._body_indexes[command.motion_anchor_body_index]

    step_offsets = torch.arange(num_frames, device=device, dtype=torch.long) * frame_interval
    indices = (command.time_steps.unsqueeze(1) + step_offsets.unsqueeze(0)).clamp(max=total - 1)
    idx_flat = indices.reshape(-1)

    future_anchor_quat_w = motion._body_quat_w[idx_flat, anchor_body_raw_idx, :].reshape(
        env.num_envs, num_frames, 4
    )

    anchor_quat_w = command.robot_anchor_quat_w

    # quat_mul 要求两四元数 shape 一致，expand 为 (num_envs, num_frames, 4)
    quat_inv_expand = quat_inv(anchor_quat_w).unsqueeze(1).expand(-1, num_frames, -1)
    ori_b = quat_mul(quat_inv_expand, future_anchor_quat_w)
    mat = matrix_from_quat(ori_b)
    rot_6d = mat[..., :2].reshape(env.num_envs, num_frames, -1)
    return rot_6d.reshape(env.num_envs, -1)


def rotation_ref_multi_frame(
    env: ManagerBasedRLEnv,
    command_name: str,
    num_frames: int = 5,
    frame_interval: int = 1,
) -> torch.Tensor:
    """未来多帧 reference motion 的 body 旋转（6D：旋转矩阵前两列），越界 clamp 到最后一帧。

    与 InstinctLab rotation_ref 对应，支持可配置帧数与帧间隔。
    Returns:
        (num_envs, num_frames * num_bodies * 6)
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    motion = command.motion
    total = motion.time_step_total
    device = command.time_steps.device
    body_indexes = motion._body_indexes

    step_offsets = torch.arange(num_frames, device=device, dtype=torch.long) * frame_interval
    indices = (command.time_steps.unsqueeze(1) + step_offsets.unsqueeze(0)).clamp(max=total - 1)
    idx_flat = indices.reshape(-1)

    quat_w = motion._body_quat_w[idx_flat][:, body_indexes, :]
    mat = matrix_from_quat(quat_w)
    rot_6d = mat[..., :2].reshape(quat_w.shape[0], -1)
    return rot_6d.reshape(env.num_envs, num_frames, -1).reshape(env.num_envs, -1)


# 全局字典，用于存储每个环境的masker实例（以环境对象id为key）
_depth_maskers: dict[int, RandomRectMaskGPU] = {}
_full_image_maskers: dict[int, FullImageMaskGPU] = {}
_high_reflection_noisers: dict[int, HighReflectionNoiseGPU] = {}


def add_depth_noise(
    depth: torch.Tensor,
    distance_noise: float = 0.0,
    gaussian_std: float = 0.0,
    salt_pepper_prob: float = 0.0,
    stripe_prob: float = 0.0,
    edge_drag_prob: float = 0.0,
    min_range: float = 0.0,
    max_range: float = 2.0,
    high_reflection_prob: float = 0.0,
    high_reflection_num_patches: tuple[int, int] = (1, 3),
    high_reflection_rect_ratio: tuple[float, float] = (0.1, 0.25),
    high_reflection_coefficient_range: tuple[float, float] = (1.1, 1.5),
    high_reflection_persist_steps: int = 3,
    mask_prob: float = 0.0,
    mask_num_masks: tuple[int, int] = (1, 3),
    mask_rect_ratio: tuple[float, float] = (0.1, 0.25),
    mask_persist_steps: int = 3,
    full_image_mask_prob: float = 0.0,
    full_image_mask_persist_steps: int = 3,
    full_image_random_prob: float = 0.0,
    env_id: int | None = None,
) -> torch.Tensor:
    """为深度图添加各种噪声以仿真真实的深度传感器。

    Args:
        depth: 原始深度图，形状为 (B, H, W), (B, 1, H, W) 或 (B, H, W, 1)
        distance_noise: 随距离增加的噪声系数
        gaussian_std: 高斯噪声的标准差系数（相对于深度值）
        salt_pepper_prob: 椒盐噪声的概率
        stripe_prob: 条纹失效的概率（随机整行或整列为0）
        edge_drag_prob: 边缘拖影的概率
        min_range: 深度最小值
        max_range: 深度最大值
        high_reflection_prob: 高反噪声的概率
        high_reflection_num_patches: 每个样本高反块数量范围 (min, max)
        high_reflection_rect_ratio: 高反块占比范围 (min, max)，相对于图像尺寸的比例
        high_reflection_coefficient_range: 高反系数范围 (min, max)，每个高反块内的系数相同
        high_reflection_persist_steps: 高反块持续步数
        mask_prob: 随机矩形mask的概率
        mask_num_masks: 每个样本mask数量范围 (min, max)
        mask_rect_ratio: mask占比范围 (min, max)，相对于图像尺寸的比例
        mask_persist_steps: mask持续步数
        full_image_mask_prob: 整张图片mask的概率
        full_image_mask_persist_steps: 整张图片mask持续步数
        full_image_random_prob: 整张图片随机噪声的概率
        env_id: 环境对象id，用于保存masker状态（如果为None，则不使用持久化mask）

    Returns:
        添加噪声后的深度图，保持原始形状
    """
    noisy = depth.clone()
    original_shape = noisy.shape
    original_dim = noisy.dim()

    # 统一转换为 (B, 1, H, W) 格式进行处理
    if original_dim == 3:
        # (B, H, W) -> (B, 1, H, W)
        noisy = noisy.unsqueeze(1)
    elif original_dim == 4:
        if original_shape[-1] == 1:
            # (B, H, W, 1) -> (B, 1, H, W)
            noisy = noisy.permute(0, 3, 1, 2)
        # else: 已经是 (B, C, H, W) 格式，保持不变

    B, C, H, W = noisy.shape
    device = noisy.device

    # 1. 随距离增加的噪声
    if distance_noise > 0:
        noise = (torch.rand_like(noisy) * 2 - 1) * (distance_noise * noisy.clamp(min=1e-3))
        noisy += noise

    # 2. 高斯噪声
    if gaussian_std > 0:
        noise = torch.randn_like(noisy) * (gaussian_std * noisy.clamp(min=1e-3))
        noisy += noise

    # 3. 椒盐噪声
    if salt_pepper_prob > 0:
        rand = torch.rand_like(noisy)
        noisy[rand < (salt_pepper_prob / 2)] = 0.0
        noisy[(rand >= salt_pepper_prob / 2) & (rand < salt_pepper_prob)] = max_range

    # 4. 条纹失效：随机整行或整列为0
    if stripe_prob > 0:
        # 随机水平/垂直条纹 mask
        horizontal = torch.rand(B, 1, H, 1, device=device) < stripe_prob
        vertical = torch.rand(B, 1, 1, W, device=device) < stripe_prob
        mask = horizontal | vertical
        noisy = torch.where(mask, torch.zeros_like(noisy), noisy)

    # 5. 边缘拖影
    if edge_drag_prob > 0:
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
        grad_x = F.conv2d(noisy, sobel_x, padding=1)
        grad_y = F.conv2d(noisy, sobel_y, padding=1)
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2)
        edge_mask = grad_mag > grad_mag.mean(dim=[2, 3], keepdim=True) * 2.0
        rand_drag = (torch.rand_like(noisy) < edge_drag_prob) & edge_mask
        shift = torch.roll(noisy, shifts=1, dims=3)
        noisy = torch.where(rand_drag, shift, noisy)

    # 6. 高反噪声（带时间持续性，在mask之前应用）
    if high_reflection_prob > 0:
        if env_id is not None:
            # 使用持久化noiser
            if env_id not in _high_reflection_noisers:
                _high_reflection_noisers[env_id] = HighReflectionNoiseGPU(
                    p=high_reflection_prob,
                    num_patches=high_reflection_num_patches,
                    rect_ratio=high_reflection_rect_ratio,
                    coefficient_range=high_reflection_coefficient_range,
                    persist_steps=high_reflection_persist_steps,
                )
            noisy = _high_reflection_noisers[env_id](noisy)
        else:
            # 不使用持久化，每次随机生成（用于向后兼容）
            noiser = HighReflectionNoiseGPU(
                p=high_reflection_prob,
                num_patches=high_reflection_num_patches,
                rect_ratio=high_reflection_rect_ratio,
                coefficient_range=high_reflection_coefficient_range,
                persist_steps=1,  # 不持久化
            )
            noisy = noiser(noisy)

    # 7. 随机矩形mask（带时间持续性）
    if mask_prob > 0:
        if env_id is not None:
            # 使用持久化masker
            if env_id not in _depth_maskers:
                _depth_maskers[env_id] = RandomRectMaskGPU(
                    p=mask_prob,
                    num_masks=mask_num_masks,
                    rect_ratio=mask_rect_ratio,
                    persist_steps=mask_persist_steps,
                )
            noisy = _depth_maskers[env_id](noisy)
        else:
            # 不使用持久化，每次随机生成（用于向后兼容）
            masker = RandomRectMaskGPU(
                p=mask_prob,
                num_masks=mask_num_masks,
                rect_ratio=mask_rect_ratio,
                persist_steps=1,  # 不持久化
            )
            noisy = masker(noisy)

    # 7. 整张图片mask（带时间持续性）
    if full_image_mask_prob > 0:
        if env_id is not None:
            # 使用持久化masker
            if env_id not in _full_image_maskers:
                _full_image_maskers[env_id] = FullImageMaskGPU(
                    p=full_image_mask_prob,
                    persist_steps=full_image_mask_persist_steps,
                )
            noisy = _full_image_maskers[env_id](noisy)
        else:
            # 不使用持久化，每次随机生成（用于向后兼容）
            masker = FullImageMaskGPU(
                p=full_image_mask_prob,
                persist_steps=1,  # 不持久化
            )
            noisy = masker(noisy)

    # 8. 随机噪声
    if full_image_random_prob > 0:
        mask = torch.rand(B, 1, 1, 1, device=device) < full_image_random_prob
        random_noise = torch.rand_like(noisy) * (max_range - min_range) + min_range
        noisy = torch.where(mask, random_noise, noisy)

    # 9. 范围过滤
    mask_valid = (noisy >= min_range) & (noisy <= max_range)
    noisy[~mask_valid] = 0.0

    # 恢复原始维度格式
    if original_dim == 3:
        # (B, 1, H, W) -> (B, H, W)
        noisy = noisy.squeeze(1)
    elif original_dim == 4:
        if original_shape[-1] == 1:
            # (B, 1, H, W) -> (B, H, W, 1)
            noisy = noisy.permute(0, 2, 3, 1)
        # else: 保持 (B, C, H, W) 格式

    return noisy


def raycaster_depth(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("raycaster_camera"),
    distance_noise: float = 0.0,
    gaussian_std: float = 0.0,
    salt_pepper_prob: float = 0.0,
    stripe_prob: float = 0.0,
    edge_drag_prob: float = 0.0,
    min_range: float | None = None,
    max_range: float | None = None,
    high_reflection_prob: float = 0.0,
    high_reflection_num_patches: tuple[int, int] = (1, 3),
    high_reflection_rect_ratio: tuple[float, float] = (0.1, 0.25),
    high_reflection_coefficient_range: tuple[float, float] = (1.1, 1.5),
    high_reflection_persist_steps: int = 3,
    mask_prob: float = 0.0,
    mask_num_masks: tuple[int, int] = (1, 3),
    mask_rect_ratio: tuple[float, float] = (0.1, 0.25),
    mask_persist_steps: int = 3,
    full_image_mask_prob: float = 0.0,
    full_image_mask_persist_steps: int = 3,
    full_image_random_prob: float = 0.0,
) -> torch.Tensor:
    """获取深度图，可选择性地添加噪声以仿真真实的深度传感器。

    Args:
        env: 环境实例
        sensor_cfg: 传感器配置
        distance_noise: 随距离增加的噪声系数
        gaussian_std: 高斯噪声的标准差系数（相对于深度值）
        salt_pepper_prob: 椒盐噪声的概率
        stripe_prob: 条纹失效的概率（随机整行或整列为0）
        edge_drag_prob: 边缘拖影的概率
        min_range: 深度最小值，如果为None则从配置中读取（默认0.0）
        max_range: 深度最大值，如果为None则从相机配置的max_distance中读取
        high_reflection_prob: 高反噪声的概率
        high_reflection_num_patches: 每个样本高反块数量范围 (min, max)
        high_reflection_rect_ratio: 高反块占比范围 (min, max)，相对于图像尺寸的比例
        high_reflection_coefficient_range: 高反系数范围 (min, max)，每个高反块内的系数相同
        high_reflection_persist_steps: 高反块持续步数
        mask_prob: 随机矩形mask的概率
        mask_num_masks: 每个样本mask数量范围 (min, max)
        mask_rect_ratio: mask占比范围 (min, max)，相对于图像尺寸的比例
        mask_persist_steps: mask持续步数
        full_image_mask_prob: 整张图片mask的概率
        full_image_mask_persist_steps: 整张图片mask持续步数
        full_image_random_prob: 整张图片随机噪声的概率

    Returns:
        深度图，形状为 (num_envs, height, width)
    """
    # extract the used quantities (to enable type-hinting)
    camera: RayCasterCamera | MultiMeshRayCasterCamera = env.scene.sensors[sensor_cfg.name]
    depth = camera.data.output["distance_to_image_plane"]

    # 从相机配置中读取 max_distance（如果 max_range 未指定）
    if max_range is None:
        max_range = camera.cfg.max_distance

    # 设置默认 min_range
    if min_range is None:
        min_range = 0.0

    # 获取环境对象id，用于masker状态持久化
    env_id = id(env)

    # 应用噪声（如果有任何噪声参数大于0）
    if any([distance_noise > 0, gaussian_std > 0, salt_pepper_prob > 0, stripe_prob > 0, edge_drag_prob > 0, high_reflection_prob > 0, mask_prob > 0, full_image_mask_prob > 0, full_image_random_prob > 0]):
        depth = add_depth_noise(
            depth,
            distance_noise=distance_noise,
            gaussian_std=gaussian_std,
            salt_pepper_prob=salt_pepper_prob,
            stripe_prob=stripe_prob,
            edge_drag_prob=edge_drag_prob,
            min_range=min_range,
            max_range=max_range,
            high_reflection_prob=high_reflection_prob,
            high_reflection_num_patches=high_reflection_num_patches,
            high_reflection_rect_ratio=high_reflection_rect_ratio,
            high_reflection_coefficient_range=high_reflection_coefficient_range,
            high_reflection_persist_steps=high_reflection_persist_steps,
            mask_prob=mask_prob,
            mask_num_masks=mask_num_masks,
            mask_rect_ratio=mask_rect_ratio,
            mask_persist_steps=mask_persist_steps,
            full_image_mask_prob=full_image_mask_prob,
            full_image_mask_persist_steps=full_image_mask_persist_steps,
            full_image_random_prob=full_image_random_prob,
            env_id=env_id,
        )

    depth -= 2.0
    return depth
