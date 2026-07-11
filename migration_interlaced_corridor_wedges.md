# 交叉斜坡（`interlaced_corridor_wedges`）迁移指南

> **用途**：将源仓库 `Noetix_e1_perception`（分支 `dzx-develop`）中与 **纵向交错交叉斜坡** 相关的全部改动，迁移到一份 **新仓库 / 新分支**。
>
> **源仓库路径（参考）**：`/home/user/perceptionnew/Noetix_e1_perception`
>
> **关键 Git 提交（按时间顺序）**：
> | Commit | 说明 |
> |--------|------|
> | `9658ce4` | 交叉斜坡地形实现 + DEPTH_TERRAINS_CFG6 启用 + play showcase |
> | `ae7f5c8` | `terrain_specific_spawn_z_offset` 基础设施（events.py，可能被其他地形共用） |
> | `7433c78` | 初始化（spawn/yaw）、`corridor_centerline` 奖励、速度指令收紧 |
> | `28d541a` | `interlaced_ramp_step_up_lift` 抬腿上坡奖励、spawn 低侧出生 |

---

## 1. 功能概述

### 1.1 地形：`interlaced_corridor_wedges`

沿 **+x** 方向 4 段连续交叉斜坡（之字形），几何参数：

| 参数 | 值 | 含义 |
|------|-----|------|
| `corridor_half_width_m` | 1.5 | 走廊半宽，总宽 3 m（y ∈ [-1.5, 1.5]） |
| `segment_length_m` | 1.5 | 每段长度 |
| `num_segments` | 4 | 段数，总长 6 m |
| `slope_angle_range` | (15°, 15°) | 固定 15° 坡 |
| `pattern_origin_x_m` | **-3.0** | 图案起点；覆盖 x ∈ **[-3, +3]** |
| `edge_blend_m` | 0.02 | 横向棱边软过渡 |
| `corridor_edge_fade_m` | 0.06 | \|y\| > 1.5 外侧 fade 到 0 |

**高度参考（terrain-cell 坐标系，`local_xy = root_pos_w.xy - env_origins.xy`）**：
- 中心线 y=0：高度约 0.40 m（段内变化）
- 边缘 y=±1.5：峰值约 0.79 m
- 段 0 低侧 y ≈ -0.6：高度约 0.16–0.23 m（适合 ~0.25 m 抬腿上坡）

### 1.2 训练任务流（设计意图）

1. **Spawn**：在 pattern 起点前 **平地**（x < -3），低侧 y ≈ -0.6
2. **Step-up**：在 x ≈ -3 坡沿处抬腿 ~0.2–0.3 m 踏上斜坡
3. **Traverse**：沿 +x 走在走廊横向中线 y→0，穿越 4 段交叉坡

### 1.3 坐标系约定

所有 spawn / reward 的 x、y 均指 **terrain-cell 局部坐标**：
```python
local_xy = root_pos_w[:, :2] - env.scene.env_origins[:, :2]
```
与 `play.py` 中 **P 键** 打印的 `local_xy` 一致。

---

## 2. 必须迁移的文件清单

### 2.1 核心代码（必须）

| 文件 | 改动类型 | 说明 |
|------|----------|------|
| `source/NoetixE1/NoetixE1/terrains/custom_terrain/custom_terrains.py` | 新增函数 | `interlaced_corridor_wedges_terrain()`（约 L1502–1565）；依赖已有 `_smoothstep01()` |
| `source/NoetixE1/NoetixE1/terrains/custom_terrain/custom_terrains_cfg.py` | 新增类 | `HfInterlacedCorridorWedgesTerrainCfg` |
| `source/NoetixE1/NoetixE1/terrains/custom_terrain/__init__.py` | 导出 | 加入 `HfInterlacedCorridorWedgesTerrainCfg` |
| `source/NoetixE1/NoetixE1/terrains/terrain_generator_cfg.py` | 注册子地形 | import + `DEPTH_TERRAINS_CFG6.sub_terrains["interlaced_corridor_wedges"]` |
| `source/NoetixE1/NoetixE1/tasks/walkrun/noetix_e1/mdp/rewards.py` | 新增 2 函数 | `corridor_centerline_exp`, `interlaced_corridor_ramp_step_up_lift_reward` |
| `source/NoetixE1/NoetixE1/tasks/walkrun/noetix_e1/mdp/events.py` | 扩展 | `terrain_specific_yaw` 支持 `(min,max)` 元组；`_apply_terrain_specific_spawn_z_offset()` |
| `source/NoetixE1/NoetixE1/tasks/walkrun/noetix_e1/walkrun_amp_depth_cfg.py` | 配置 | 见 §3 逐项清单 |
| `scripts/rsl_rl/play.py` | 调试 | `--showcase_terrain interlaced_corridor_wedges` + P 键打印位置 |

### 2.2 不要迁移（除非明确需要）

- `hurdle_checkpoint/**`、`walkrun_depth/**`：训练日志与 checkpoint
- `policy077.onnx` 等导出模型：与地形迁移无关
- commit `9658ce4` 中大量 `walkrun_depth/` 实验目录

### 2.3 前置依赖（目标仓库须已具备）

以下 **不是** 本次新增，但交叉斜坡功能 **依赖** 它们：

| 模块 | 用途 |
|------|------|
| `FiledTerrainGenerator` / `FiledTerrainGeneratorCfg` | 记录子地形类型名，供 reward / reset 按 terrain type 过滤 |
| `mdp/curriculums.py` 中 `_adaptive_terrain_state`、`_check_terrain_types_from_positions` | 奖励与 reset 判断当前 env 是否在指定子地形 |
| `reset_root_state_amp()` in `events.py` | AMP RSI reset；`terrain_specific_pos_range` / `terrain_specific_yaw` |
| `left_foot_height_scanner` / `right_foot_height_scanner` RayCaster | 抬腿奖励需要 |
| `contact_sensor` on `*_ankle_roll_link` | 抬腿奖励判断 swing phase |
| `DEPTH_TERRAINS_CFG6` 及 `E1SceneCfg.terrain.terrain_generator` | 场景引用该 terrain generator |

---

## 3. `walkrun_amp_depth_cfg.py` 逐项改动

在目标仓库的同名文件中，按下列位置 **新增或修改**（键名必须完全一致：`"interlaced_corridor_wedges"`）。

### 3.1 场景地形

```python
# E1SceneCfg.terrain
terrain_generator=DEPTH_TERRAINS_CFG6,  # 须含 interlaced_corridor_wedges 子地形
```

### 3.2 速度指令 `CommandsCfg.base_velocity.terrain_specific_ranges`

```python
"interlaced_corridor_wedges": mdp.UniformVelocityWithZeroCommandCfg.Ranges(
    lin_vel_x=(0.3, 1.5),      # 强制前进，避免零速站桩
    lin_vel_y=(-0.2, 0.2),
    ang_vel_z=(-0.5, 0.5),
    zero_prob=[0.5, 0.9, 0.5],
),
```

### 3.3 Reset 事件 `EventCfg.reset_robot_states`

**`terrain_specific_yaw`** — 面朝 +x，±30°：
```python
"interlaced_corridor_wedges": (-0.5235987755982988, 0.5235987755982988),  # ±π/6
```

**`terrain_specific_pos_range`** — 平地低侧出生（**以源仓库当前值为准**）：
```python
"interlaced_corridor_wedges": {
    "x": (-4.0, -3.2),    # pattern 起点 x=-3 前的平地
    "y": (-0.75, -0.25),  # 段 0 低侧，坡沿高度 ~0.16–0.23 m
},
```

**`terrain_specific_spawn_z_offset`** — AMP motion z 补偿：
```python
"terrain_specific_spawn_z_offset": {
    "interlaced_corridor_wedges": -0.6,
},
```
> 说明：RSI 使用 `z = motion_z + env_origin_z + offset`，**不做** terrain height 查询补偿。

### 3.4 Push 扰动 `EventCfg.push_robot`

在 `terrain_specific_velocity_range` 中加入：
```python
"interlaced_corridor_wedges": {
    "x": (-0.8, 0.8),
    "y": (-0.8, 0.8),
    "z": (-0.2, 0.2),
    "roll": (-0.5, 0.5),
    "pitch": (-0.5, 0.5),
    "yaw": (-0.5, 0.5),
},
```

### 3.5 奖励 `RewardsCfg`

**A. 走廊中线奖励**
```python
corridor_centerline = RewTerm(
    func=mdp.corridor_centerline_exp,
    weight=0.5,
    params={
        "std": 0.2,
        "terrain_types": ("interlaced_corridor_wedges",),
        "min_forward_command": 0.3,
        "local_x_range": (-3.0, 3.0),  # 仅在 pattern 区域内生效
    },
)
```

**B. 坡沿抬腿上坡奖励**
```python
interlaced_ramp_step_up_lift = RewTerm(
    func=mdp.interlaced_corridor_ramp_step_up_lift_reward,
    weight=6,
    params={
        "contact_sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
        "left_foot_height_scanner_cfg": SceneEntityCfg("left_foot_height_scanner"),
        "right_foot_height_scanner_cfg": SceneEntityCfg("right_foot_height_scanner"),
        "offset": 0.06,
        "clearance_threshold": 0.15,
        "std": 0.05,
        "x_range": (-3.4, -2.8),  # pattern_origin_x_m=-3 坡沿前后
        "min_forward_command": 0.3,
        "terrain_types": ("interlaced_corridor_wedges",),
    },
)
```

**C. 已有惩罚项中加入 terrain_types**（若目标仓库已有这些 RewTerm，追加 `"interlaced_corridor_wedges"`）：
- `feet_unhold.terrain_types`
- `feet_flat_ori.terrain_types`

---

## 4. `events.py` 关键改动摘要

### 4.1 `terrain_specific_yaw` 类型扩展

```python
terrain_specific_yaw: dict[str, list[float] | tuple[float, float]] | None = None
```

采样逻辑：若值为 `tuple` 且 `len==2`，则 `uniform(lo, hi)`；否则从离散列表随机选。

### 4.2 `_apply_terrain_specific_spawn_z_offset()`

在 `reset_root_state_amp()` 末尾、`write_root_pose_to_sim` 之前调用：
```python
_apply_terrain_specific_spawn_z_offset(
    env, env_ids, positions, terrain_specific_spawn_z_offset, asset.device
)
```

---

## 5. `terrain_generator_cfg.py` 注册块

```python
from .custom_terrain import (
    ...
    HfInterlacedCorridorWedgesTerrainCfg,
    ...
)

# DEPTH_TERRAINS_CFG6.sub_terrains 内：
"interlaced_corridor_wedges": HfInterlacedCorridorWedgesTerrainCfg(
    proportion=0.1,
    corridor_half_width_m=1.5,
    segment_length_m=1.5,
    num_segments=4,
    slope_angle_range=(15.0, 15.0),
    edge_blend_m=0.02,
    corridor_edge_fade_m=0.06,
    pattern_origin_x_m=-3.0,
    flat_patch_sampling=dict(_DEPTH6_FLAT_PATCH_TARGET),
),
```

---

## 6. `play.py` 调试支持

### 6.1 CLI showcase

确保 `--showcase_terrain` choices 含 `"interlaced_corridor_wedges"`，并实现分支：

```python
elif st == "interlaced_corridor_wedges":
    env_cfg.scene.terrain.terrain_generator.sub_terrains = {
        "interlaced_corridor_wedges": terrain_gen.HfInterlacedCorridorWedgesTerrainCfg(
            proportion=1.0,
            corridor_half_width_m=1.5,
            segment_length_m=1.5,
            num_segments=4,
            slope_angle_range=(15.0, 15.0),
            edge_blend_m=0.02,
            corridor_edge_fade_m=0.06,
            pattern_origin_x_m=-3.0,
            flat_patch_sampling=_flat_patch_target_for_pose_command(),
        )
    }
```

### 6.2 P 键打印 local_xy

键盘控制模式下，`P` 键打印 terrain-cell 坐标，用于校准 spawn / reward 的 x、y 范围。

---

## 7. 推荐迁移步骤（给 Agent 的执行顺序）

```
Step 1  对比目标仓库与源仓库，确认 FiledTerrainGenerator、foot height scanner、reset_root_state_amp 已存在
Step 2  复制/合并 custom_terrains.py 中 interlaced_corridor_wedges_terrain 函数
Step 3  复制 HfInterlacedCorridorWedgesTerrainCfg + __init__.py 导出
Step 4  在 terrain_generator_cfg.py 注册子地形（DEPTH_TERRAINS_CFG6 或等价 config）
Step 5  合并 events.py：yaw tuple 支持 + spawn_z_offset（若目标仓库尚无）
Step 6  合并 rewards.py：两个新 reward 函数
Step 7  合并 walkrun_amp_depth_cfg.py：§3 全部配置项
Step 8  合并 play.py showcase + P 键（可选但强烈建议）
Step 9  运行验证（§8）
```

### 快捷 Git 方式（若两仓库历史相关）

```bash
# 在目标仓库
git remote add source <源仓库 URL>
git fetch source

# 按 commit cherry-pick（仅代码文件，跳过 checkpoint/logs）
git cherry-pick 9658ce4 --no-commit
git cherry-pick 7433c78 --no-commit
git cherry-pick 28d541a --no-commit

# 只 stage 交叉斜坡相关文件
git add source/NoetixE1/NoetixE1/terrains/custom_terrain/
git add source/NoetixE1/NoetixE1/terrains/terrain_generator_cfg.py
git add source/NoetixE1/NoetixE1/tasks/walkrun/noetix_e1/mdp/rewards.py
git add source/NoetixE1/NoetixE1/tasks/walkrun/noetix_e1/mdp/events.py
git add source/NoetixE1/NoetixE1/tasks/walkrun/noetix_e1/walkrun_amp_depth_cfg.py
git add scripts/rsl_rl/play.py

# 解决冲突后 commit
```

> Cherry-pick 时 `walkrun_amp_depth_cfg.py` 可能与目标分支其他改动冲突（如 `DEPTH_TERRAINS_CFG` vs `DEPTH_TERRAINS_CFG6`），**手动保留 interlaced 相关块**。

---

## 8. 验证清单

### 8.1 地形生成

```bash
python scripts/rsl_rl/play.py --task <WalkRun-AMP-Depth-Task-ID> \
  --showcase_terrain interlaced_corridor_wedges --num_envs 1
```

- [ ] 地形在 x∈[-3,3] 出现 4 段交叉坡，y=±1.5 为边界
- [ ] x < -3 区域为平地
- [ ] 按 **P** 打印 `local_xy`，确认 spawn 范围合理

### 8.2 训练配置

- [ ] `grep interlaced_corridor_wedges walkrun_amp_depth_cfg.py` 覆盖 commands / events / rewards
- [ ] reward 函数可被 `from NoetixE1.tasks.walkrun.noetix_e1 import mdp` 导入（`mdp/__init__.py` 已 `from .rewards import *`）

### 8.3 行为观察

- [ ] Reset 后 robot 落在 x∈[-4,-3.2]、y∈[-0.75,-0.25] 附近（平地低侧）
- [ ] 收到 cmd_x > 0.3 时，在 x∈[-3.4,-2.8] 摆动脚 clearance > 0.15 m 有 `interlaced_ramp_step_up_lift` 奖励
- [ ] 进入 x∈[-3,3] 后 `corridor_centerline` 将 y 拉向 0

---

## 9. 已知调参注意点

| 项 | 说明 |
|----|------|
| `corridor_centerline` vs step-up | 低侧 spawn（y<0）时，centerline 奖励可能与 step-up 阶段 **冲突**；可考虑 `local_x_range` 改为 `(-2.5, 3.0)` 延后再启用 |
| `interlaced_ramp_step_up_lift.weight` | 当前 6，偏高；可降至 0.6–2.0 视训练稳定性调整 |
| `terrain_specific_spawn_z_offset: -0.6` | 若 robot 出生埋地/悬空，在 play 中观察后微调 |
| `clearance_threshold: 0.15` | 对应 ~0.25 m 抬腿目标的下限；可按实际 foot scanner 标定调整 |

---

## 10. 源文件精确锚点（便于 diff）

| 符号 | 源仓库位置 |
|------|-----------|
| `interlaced_corridor_wedges_terrain` | `custom_terrains.py` L1502–1565 |
| `HfInterlacedCorridorWedgesTerrainCfg` | `custom_terrains_cfg.py` L571–603 |
| `DEPTH_TERRAINS_CFG6` 注册 | `terrain_generator_cfg.py` L524–535 |
| `corridor_centerline_exp` | `rewards.py` L462–497 |
| `interlaced_corridor_ramp_step_up_lift_reward` | `rewards.py` L500–577 |
| `_apply_terrain_specific_spawn_z_offset` | `events.py` L24–49 |
| `reset_root_state_amp` yaw tuple | `events.py` L114–130 |
| `CommandsCfg` interlaced ranges | `walkrun_amp_depth_cfg.py` L333–338 |
| `EventCfg` spawn/yaw/z_offset | `walkrun_amp_depth_cfg.py` L654–696 |
| `RewardsCfg` 两个新 RewTerm | `walkrun_amp_depth_cfg.py` L891–919 |
| `play.py` showcase | L491–504 |
| `play.py` P 键 | L633–638, L672–673 |

---

## 11. Agent 任务 Prompt 模板（可直接复制）

```
请将源仓库 Noetix_e1_perception（分支 dzx-develop）中 interlaced_corridor_wedges（纵向交错交叉斜坡）相关改动迁移到本仓库。

请阅读 docs/migration_interlaced_corridor_wedges.md，按 §7 步骤执行：
1. 迁移地形实现（custom_terrains.py / custom_terrains_cfg.py / terrain_generator_cfg.py）
2. 迁移 events.py 扩展（yaw tuple、spawn_z_offset）
3. 迁移 rewards.py 两个函数
4. 迁移 walkrun_amp_depth_cfg.py 中所有 interlaced_corridor_wedges 配置
5. 迁移 play.py showcase 与 P 键调试
6. 完成 §8 验证清单

约束：
- 不要迁移 checkpoint、logs、onnx 模型
- 子地形键名必须为 "interlaced_corridor_wedges"
- 坐标系使用 terrain-cell：local_xy = root_pos_w.xy - env_origins.xy
- 若与目标仓库现有 cfg 冲突，只合并 interlaced 相关块，不破坏其他地形
- 迁移完成后列出改动文件清单与验证结果
```
