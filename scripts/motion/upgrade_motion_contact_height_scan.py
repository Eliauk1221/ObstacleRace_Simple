#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import argparse
import os
from copy import deepcopy

import numpy as np


# 和训练/record 一致的 keypose 映射
mujoco2isaaclab_keypose = [0, 1, 7, 13, 2, 8, 14, 19, 3, 9, 15, 20, 4, 10, 16, 21, 5, 11, 17, 22, 6, 12, 18, 23]
isaaclab2mujoco_keypose = [0, 1, 4, 8, 12, 16, 20, 2, 5, 9, 13, 17, 21, 3, 6, 10, 14, 18, 22, 7, 11, 15, 19, 23]

# 旧 json 中「双脚 contact」在帧向量中的末尾长度（与 MotionLoaderE1.CONTACT_MASK_SIZE 一致）
OLD_CONTACT_DIMS = [6, 12]
# 新 json 中「全身 contact」长度：等于 KeyPosNames 数量；这里假定与 isaaclab2mujoco_keypose 一致
NEW_CONTACT_SIZE = len(isaaclab2mujoco_keypose)

# height_scan 维度：与某个任务中的 height_scanner 射线数量一致
# 例如 walkrun_amp_depth_cfg 里：(resolution=0.1, size=[1.6, 1.0]) -> 187
HEIGHT_SCAN_DIM = 187
HEIGHT_SCAN_VALUE = 0.28412


def upgrade_motion_file(
    input_path: str,
    output_path: str,
):
    """
    将旧格式 motion json 升级为新格式：
    - contact: 2 -> NEW_CONTACT_SIZE，双脚原位置映射到新向量对应的两个元素，其余补 0
    - height_scan: 追加长度 HEIGHT_SCAN_DIM、值为 HEIGHT_SCAN_VALUE 的向量
    """
    with open(input_path, "r") as f:
        motion_data = json.load(f)

    frames = np.asarray(motion_data["Frames"], dtype=np.float32)
    num_frames, old_dim = frames.shape

    if old_dim < len(OLD_CONTACT_DIMS):
        raise ValueError(f"Frame dim ({old_dim}) < OLD_CONTACT_SIZE ({OLD_CONTACT_SIZE}), 无法解析 contact。")

    # 拆分：前面的 pose+vel / 末尾的 2 维 contact
    pose_dim = old_dim - len(OLD_CONTACT_DIMS)
    pose_part = frames[:, :pose_dim]           # [T, pose_dim]
    contact_old = frames[:, pose_dim:]         # [T, 2]

    # 构造新的全身 contact：shape [T, NEW_CONTACT_SIZE]
    contact_new = np.zeros((num_frames, NEW_CONTACT_SIZE), dtype=np.float32)

    # 简单策略：假定旧的 2 维 contact = [左脚, 右脚]
    # 并且它们在 KeyPosNames 中已经有位置；如果知道确切的 index，可在这里填写
    # 下面示例：直接放在向量的最后两维（也可以根据你的新 loader 要求调整下标）
    left_idx_new = OLD_CONTACT_DIMS[0]
    right_idx_new = OLD_CONTACT_DIMS[1]

    contact_new[:, left_idx_new] = contact_old[:, 0]
    contact_new[:, right_idx_new] = contact_old[:, 1]

    # 构造 height_scan：常数向量
    height_scan = np.full((num_frames, HEIGHT_SCAN_DIM), HEIGHT_SCAN_VALUE, dtype=np.float32)

    # 拼接新帧：pose_part + contact_new + height_scan
    frames_new = np.concatenate([pose_part, contact_new, height_scan], axis=1)

    motion_data_new = deepcopy(motion_data)
    motion_data_new["Frames"] = frames_new.tolist()

    # 你也可以在这里新增元数据字段，标明新 contact/height_scan 的尺寸
    motion_data_new["ContactDim"] = int(NEW_CONTACT_SIZE)
    motion_data_new["HeightScanDim"] = int(HEIGHT_SCAN_DIM)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(motion_data_new, f, indent=1, separators=(",", ":"))

    print(f"Upgraded motion saved to: {output_path}")
    print(f"Old frame dim: {old_dim}, new frame dim: {frames_new.shape[1]}")


def main():
    parser = argparse.ArgumentParser(description="Upgrade motion json: 2-d contact -> full-body contact + height_scan.")
    parser.add_argument("input", type=str, help="Path to original motion json file.")
    parser.add_argument(
        "-o", "--output", type=str, default=None, help="Path to upgraded motion json file. Default: <input>_upgraded.json"
    )
    args = parser.parse_args()

    input_path = args.input
    if args.output is None:
        root, ext = os.path.splitext(input_path)
        output_path = root + "_upgraded" + ext
    else:
        output_path = args.output

    upgrade_motion_file(input_path, output_path)


if __name__ == "__main__":
    main()