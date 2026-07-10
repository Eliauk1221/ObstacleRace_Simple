#!/usr/bin/env python3
"""Generate STL mesh from multiple boxes with configurable size and pose.

Requirements: trimesh; for --config YAML: PyYAML.

Boxes can be defined in script (edit DEFAULT_BOXES) or via a YAML config file. Each box has:
  - size: [x, y, z] (extents in meters)
  - position: [x, y, z]  (center of the box)
  - euler_deg: [roll, pitch, yaw] in degrees (optional, default [0,0,0])
  - euler_rad: [roll, pitch, yaw] in radians (optional, use instead of euler_deg)

.. code-block:: bash

    # In-script boxes, output to default path
    python scripts/motion/boxes_to_stl.py

    # Custom output path
    python scripts/motion/boxes_to_stl.py --output my_terrain.stl

    # From YAML config
    python scripts/motion/boxes_to_stl.py --config boxes_config_example.yaml --output out.stl
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np
import trimesh


def euler_to_rotation_matrix(euler_rad: list[float] | tuple[float, float, float]) -> np.ndarray:
    """Build 4x4 rotation matrix from Euler angles (roll, pitch, yaw) in radians."""
    r, p, y = euler_rad
    cx, sx = math.cos(r), math.sin(r)
    cy, sy = math.cos(p), math.sin(p)
    cz, sz = math.cos(y), math.sin(y)
    R = np.array([
        [cy * cz, -cy * sz, sy],
        [cx * sz + sx * sy * cz, cx * cz - sx * sy * sz, -sx * cy],
        [sx * sz - cx * sy * cz, sx * cz + cx * sy * sz, cx * cy],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    return T


def box_to_mesh(
    size: list[float] | tuple[float, float, float],
    position: list[float] | tuple[float, float, float] = (0.0, 0.0, 0.0),
    euler_deg: list[float] | tuple[float, float, float] | None = None,
    euler_rad: list[float] | tuple[float, float, float] | None = None,
) -> trimesh.Trimesh:
    """Create a box mesh with given size and pose.

    Args:
        size: [x, y, z] extents (full side lengths) in meters.
        position: [x, y, z] center position.
        euler_deg: [roll, pitch, yaw] in degrees (optional).
        euler_rad: [roll, pitch, yaw] in radians (optional). Ignored if euler_deg is set.

    Returns:
        Trimesh instance.
    """
    extents = tuple(float(x) for x in size)
    box = trimesh.creation.box(extents=extents)
    # Center the box at origin, then apply rotation then translation
    if euler_deg is not None:
        euler_rad = [math.radians(x) for x in euler_deg]
    if euler_rad is not None:
        rot = euler_to_rotation_matrix(tuple(euler_rad))
        box.apply_transform(rot)
    trans = trimesh.transformations.translation_matrix(tuple(position))
    box.apply_transform(trans)
    return box


def boxes_to_stl(
    boxes: list[dict],
    output_path: str,
) -> None:
    """Create a single STL mesh from a list of box definitions and export it.

    Each item in `boxes` can have:
      - size: [x, y, z] (required)
      - position: [x, y, z] (default [0,0,0])
      - euler_deg: [roll, pitch, yaw] in degrees (optional)
      - euler_rad: [roll, pitch, yaw] in radians (optional)
    """
    meshes = []
    for i, b in enumerate(boxes):
        size = b.get("size")
        if size is None:
            raise ValueError(f"Box {i}: missing 'size' (required).")
        position = b.get("position", [0.0, 0.0, 0.0])
        euler_deg = b.get("euler_deg")
        euler_rad = b.get("euler_rad")
        mesh = box_to_mesh(
            size=size,
            position=position,
            euler_deg=euler_deg,
            euler_rad=euler_rad,
        )
        meshes.append(mesh)

    combined = trimesh.util.concatenate(meshes)
    combined.export(output_path)
    print(f"Exported {len(meshes)} box(es) to {os.path.abspath(output_path)}")


# -----------------------------------------------------------------------------
# Example boxes (used when --config is not provided)
# -----------------------------------------------------------------------------
DEFAULT_BOXES = [
    {"size": [1.0, 1.0, 0.1], "position": [0.0, 0.0, 0.05]},
    {"size": [0.3, 0.3, 0.2], "position": [0.2, 0.0, 0.2], "euler_deg": [0, 0, 15]},
]


def main():
    parser = argparse.ArgumentParser(
        description="Generate STL mesh from multiple boxes (size + pose)."
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="boxes_terrain.stl",
        help="Output STL file path (default: boxes_terrain.stl)",
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="YAML file defining 'boxes' list (each: size, position, optional euler_deg or euler_rad)",
    )
    args = parser.parse_args()

    if args.config:
        import yaml
        with open(args.config) as f:
            data = yaml.safe_load(f)
        boxes = data.get("boxes")
        if not boxes:
            raise SystemExit("Config must contain a 'boxes' list.")
    else:
        boxes = DEFAULT_BOXES

    boxes_to_stl(boxes, args.output)


if __name__ == "__main__":
    main()
