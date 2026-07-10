# Template for Noetix-E1 Isaac Lab Projects

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.0.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.2.1-silver)](https://isaac-sim.github.io/IsaacLab)
[![RSL_RK](https://img.shields.io/badge/RSL_RL-3.1.1-silver)](https://github.com/leggedrobotics/rsl_rl)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)

## Overview

This project/repository serves as a template for building projects or extensions based on Isaac Lab.
It allows you to develop in an isolated environment, outside of the core Isaac Lab repository.

**Key Features:**

- `Isolation` Work outside the core Isaac Lab repository, ensuring that your development efforts remain self-contained.
- `Flexibility` This template is set up to allow your code to be run as an extension in Omniverse.

**Keywords:** extension, template, isaaclab

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
- ***Note that the Isaacsim version must be 5.0.0!!!***
- We recommend using the conda installation as it simplifies calling Python scripts from the terminal.

- Clone or copy this project/repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

- Using a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    # use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python -m pip install -e source/NoetixE1
    python -m pip install -e rsl_rl
- Verify that the extension is correctly installed by:

    - Listing the available tasks:

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/list_envs.py
        ```

    - Running a task:

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/rsl_rl/train.py --task=<TASK_NAME>
        ```

## Usage

### Visualize motion

Visualize the motion by updating the simulation with data from e1/datasets/motion_visualization.

```bash
python scripts/rsl_rl/play_animation.py --task=WalkRun-E1-v0
```


### Train

Train the policy using AMP expert data from e1/datasets/motion_amp_expert.

```bash
python scripts/rsl_rl/train.py --task=WalkRun-E1-v0 --headless --num_envs=4096
```

### Play

Run the trained policy.

```bash
python scripts/rsl_rl/play.py --task=WalkRun-E1-v0 --num_envs=1
```

### Sim2Sim(MuJoCo)

Evaluate the trained policy in MuJoCo to perform cross-simulation validation.

Exported_policy/ contains pretrained policies provided by the project. When using the play script, trained policy is exported automatically and saved to path like logs/run/[timestamp]/exported/policy.pt.
```bash
python scripts/sim2sim.py --policy Exported_policy/walk.onnx --duration 100
```

### Train depth Mimic with terrain
Train the depth mimic policy.

```bash
python scripts/rsl_rl/train.py --task=Mimic-E1-23dof-depth --headless --num_envs=4096
```
set `MOTION_PATH` in /tasks/mimic/config/e1/flat_env_cfg.py to the path of the dataset directory.

#### motion file struct
`Motion_PATH` should include `metadata.yaml` file.
`metadata.yaml` should be set as follows:
```yaml
motion_files:
- motion_file: parkour1.npz
  terrain_id: 0
terrains:
- terrain_file: box_50_50_150.stl
  terrain_id: 0
  origin_offset: [0,-1.5,0.2] # set the offset of the terrain in the world frame
```

#### manual generate boxes mesh
```
python scripts/motion/boxes_to_stl.py --config boxes_config_example.yaml --output out.stl
```

#### replay mimic motion with terrain
```
python scripts/motion/replay_npz_e1_23dof_terrain.py --motion_path /path/to/your/dataset_dir
```
`/path/to/your/dataset_dir` should include `metadata.yaml` file

### Tensorboard
```bash
tensorboard --logdir=logs/walkrun
```

### Set up IDE (Optional)

To setup the IDE, please follow these instructions:

- Run VSCode Tasks, by pressing `Ctrl+Shift+P`, selecting `Tasks: Run Task` and running the `setup_python_env` in the drop down menu.
  When running this task, you will be prompted to add the absolute path to your Isaac Sim installation.

If everything executes correctly, it should create a file .python.env in the `.vscode` directory.
The file contains the python paths to all the extensions provided by Isaac Sim and Omniverse.
This helps in indexing all the python modules for intelligent suggestions while writing code.


## Code formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```

## Troubleshooting

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing.
In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/NoetixE1"
        "<path-to-rsl-rl>/rsl_rl",
        "<path-to-isaac-lab>/IsaacLab/source/isaaclab_tasks",
        "<path-to-isaac-lab>/IsaacLab/source/isaaclab_mimic",
        "<path-to-isaac-lab>/IsaacLab/source/extensions",
        "<path-to-isaac-lab>/IsaacLab/source/isaaclab_assets",
        "<path-to-isaac-lab>/IsaacLab/source/isaaclab_rl",
        "<path-to-isaac-lab>/IsaacLab/source/isaaclab",
    ]
}
```

### Pylance Crash

If you encounter a crash in `pylance`, it is probable that too many files are indexed and you run out of memory.
A possible solution is to exclude some of omniverse packages that are not used in your project.
To do so, modify `.vscode/settings.json` and comment out packages under the key `"python.analysis.extraPaths"`
Some examples of packages that can likely be excluded are:

```json
"<path-to-isaac-sim>/extscache/omni.anim.*"         // Animation packages
"<path-to-isaac-sim>/extscache/omni.kit.*"          // Kit UI tools
"<path-to-isaac-sim>/extscache/omni.graph.*"        // Graph UI tools
"<path-to-isaac-sim>/extscache/omni.services.*"     // Services tools
...
```
