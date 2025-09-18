# 🦿St Gym Project

<p align="left">
  <a href="https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html">
    <img alt="IsaacSim" src="https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg">
  </a>
  <a href="https://isaac-sim.github.io/IsaacLab">
    <img alt="IsaacLab" src="https://img.shields.io/badge/IsaacLab-2.1.0-silver">
  </a>
  <a href="https://docs.python.org/3/whatsnew/3.10.html">
    <img alt="Python" src="https://img.shields.io/badge/Python-3.10-blue.svg">
  </a>
  <a href="https://developer.nvidia.com/cuda-toolkit">
    <img alt="CUDA" src="https://img.shields.io/badge/CUDA-12.8-green?logo=nvidia">
  </a>
  <a href="https://docs.docker.com/">
    <img alt="Docker" src="https://img.shields.io/badge/Docker-24.0+-blue?logo=docker">
  </a>
  <a href="https://pre-commit.com/">
    <img alt="pre-commit" src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white">
  </a>
  <a href="LICENSE">
    <img alt="License" src="https://img.shields.io/badge/license-MIT-yellow.svg">
  </a>
</p>

St Gym is a simulation and reinforcement learning framework for developing locomotion policies on legged robots—particularly tailored for the Lus2 humanoid robot.This project builds upon [Isaac Lab](https://github.com/NVIDIA-Omniverse/IsaacLab) and provides a modular interface for training and evaluating imitation-based controllers in both Isaac Sim and MuJoCo environments.It allows users to develop, train, and test policies in a flexible, extendable setting, isolated from the core Isaac Sim framework.

🛠️ This project relies on **Python 3.10**, **CUDA 12.8**, **Isaac Sim 4.5.0**, and **Isaac Lab 2.1.0**.

# Installation

## Prerequisites

Make sure your system meets the following requirements:

- Ubuntu 20.04 or 22.04
- Python 3.10
- CUDA 12.8
- [Isaac Sim 4.5.0](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
- [Isaac Lab 2.1.0](https://isaac-sim.github.io/IsaacLab)

## Repository Setup

First, create a workspace directory and set an environment variable for convenience:

```bash
mkdir -p $HOME/workspace/lumos_ws
export WK="$HOME/workspace/lumos_ws"
cd $WK
```

Next, clone the required repositories into the workspace:

- Clone `st_gym`

```bash
git clone http://git.lumosbot.tech/lumosbot/locomotion/st_gym.git
```

- Clone st_rl

```bash
git clone http://git.lumosbot.tech/lumosbot/locomotion/st_rl.git
```

- Clone `lumos_rl_gym`

Make sure to clone the correct feature branch (`feature/add_lus1_v3`):

```bash
git clone --branch feature/add_lus1_v3 --single-branch http://git.lumosbot.tech/lumosbot/locomotion/lumos_rl_gym.git
```

Project Directory Structure

```
lumos_ws/
├── lumos_rl_gym/        
├── st_gym/ 
├── st_rl/   
```

## Dependencies

Activate the Isaac Lab conda environment:

```bash
conda activate env_isaaclab
```

Install the reinforcement learning algorithm components in editable mode:

```bash
cd st_gym/exts/legged_robots/ && pip install -e .
```

### Robot Resources Installation and Model Generation

Run the following command to convert the URDF robot model to USD format and generate robot assets.

**Note:** Replace `YOUR_WORKSPACE` with the root path of your workspace directory.

```bash
python $YOUR_WORKSPACE/IsaacLab/scripts/tools/convert_urdf.py \
    $YOUR_WORKSPACE/lumos_rl_gym/resources/robots/lus2/urdf/lus2_joint27.urdf \
    $YOUR_WORKSPACE/lumos_rl_gym/resources/robots/lus2/usd/lus2_joint27.usd \
    --merge-joints --joint-stiffness 10000 --joint-damping 0.0 --rendering_mode quality
```

Download and prepare **PyBullet** utility modules:

```bash
cd ~/workspace/lumos_ws/st_rl/st_rl/
mkdir pybullet_utils
git clone https://github.com/bulletphysics/bullet3.git
cp -r bullet3/examples/pybullet/gym/pybullet_utils ~/workspace/lumos_ws/st_rl/st_rl/
```

### Set Python Path Environment Variables

Add the following project directories to your `PYTHONPATH` to ensure proper module resolution.

**Note:** Replace `YOUR_WORKSPACE` with your workspace root path.

```bash
export PYTHONPATH=$YOUR_WORKSPACE/st_rl:$PYTHONPATH
export PYTHONPATH=$YOUR_WORKSPACE/st_gym/exts/legged_robots:$PYTHONPATH
export PYTHONPATH=$YOUR_WORKSPACE/st_rl/st_rl/pybullet_utils:$PYTHONPATH
export PYTHONPATH=$YOUR_WORKSPACE/st_gym/exts/legged_robots:$PYTHONPATH
```

# Data

## Asset Location

The USD robot asset files used in this project are located at:

```swift
lumos_rl_gym/resources/robots/lus2/usd/
```

These assets are programmatically generated or modified by the following script:

```python
st_gym/exts/legged_robots/legged_robots/assets/lumos.py
```

The file `lumos.py` defines the full articulation and actuation settings for the Lus2 robot, including USD paths, joint initialization states, and group-wise actuator parameters (e.g., stiffness, damping, limits). 

## Motion Data (`motion_files`)

The motion data used for AMP-based imitation is stored in `.pkl` format under:

```swift
humanoid_demo_retarget/data/lus2_joint27/pkl/
```

- Each `.pkl` file contains a dictionary with the following keys:

```python
dict_keys([
    'LoopMode',
    'FrameDuration',
    'EnableCycleOffsetPosition',
    'EnableCycleOffsetRotation',
    'MotionWeight',
    'Fields',
    'Frames'
])
```

These files define a motion sequence used as a demonstration trajectory. To specify which motion file is used during training or playback, update the `motion_files` field in:

```python
st_gym/exts/legged_robots/legged_robots/tasks/locomotion/velocity/config/lus2/amp_data_cfg.py
```

Example:

```python
motion_files = glob.glob(os.getenv("HOME") + "/workspace/lumos_ws/humanoid_demo_retarget/data/lus2_joint27/pkl/CMU_CMU_12_12_04_poses.pkl")

```

## Motion Data (`teacher_ac_path`)

The `teacher_ac_path` points to a trained AMP policy (`.pt` file) that serves as the imitation target in distillation mode, or is used directly for demo playback via ` ./scripts/st_rl/play.py`.

You can specify the path to the trained checkpoint as follows: (e.g.)

```python
teacher_ac_path = os.getenv("HOME") + "/workspace/lumos_ws/st_gym/logs/st_rl/lus2_flat/2025-06-05_15-16-48/model_400.pt"
```

Both `motion_files` and `teacher_ac_path` are required to enable demo-based playback.

# Training 

## Quick Start

Supported terrains: **flat** and **rough** surfaces. Run the following commands to start training:

```bash
# Flat terrain training
python ./scripts/st_rl/train.py --task Flat-Lus2 --headless
```

```bash
# Rough terrain training
python ./scripts/st_rl/train.py --task Rough-Lus2 --headless
```

## Configuration

Configuration files for different environments and algorithms are located in:

```text
exts/legged_robots/tasks/locomotion/velocity/config/
└── lus2/
    ├── agents/
    │   └── st_rl_ppo_cfg.py      # PPO training parameters
    ├── flat_env_cfg.py           # Flat terrain configuration
    ├── rough_env_cfg.py          # Rough terrain configuration
    └── amp_data_cfg.py           # Motion data and AMP configuration
```

To modify motion sources or environment properties, edit the corresponding files above.

# Playing

## Evaluate by Isaac Lab 

This runs the trained policy in Isaac Sim using the specified checkpoint. 

Replace the example log folder `2025-06-05_15-16-48` with the actual one under: `st_gym/logs/st_rl/lus2_flat/`

```bash
python ./scripts/st_rl/play.py --task Flat-Lus2 --load_run 2025-06-05_15-16-48 --checkpoint model_400.pt
```

![Peek 2025-06-12 16-12](./docs/png.gif)

## Sim2Sim by Mujoco

Replay the same trained policy in **MuJoCo** using:

```bash
./run.sh -n Flat-Lus2 -s -l 2025-06-05_15-16-48
```

 `2025-06-05_15-16-48` is just an example log folder. Check your actual run logs in `st_gym/logs/st_rl/lus2_flat/`.

## Script Arguments and Shortcuts

The `run.sh` script provides a convenient interface for launching training, playback, simulation, and evaluation tasks with minimal configuration.

```bash
# Train the Flat-Lus2 task
./run.sh -t
# Equivalent to:
python ./scripts/st_rl/train.py --task Flat-Lus2 --headless
```

```bash
# Play policy from a specific run and checkpoint
./run.sh -n Flat-Lus2 -p -l 2025-06-05_15-16-48 -c model_400.pt
# Equivalent to:
python ./scripts/st_rl/play.py --task Flat-Lus2 --load_run 2025-06-05_15-16-48 --checkpoint model_400.pt
```

| Option | Description                                               |
| ------ | :-------------------------------------------------------- |
| `-e`   | Specify the experiment name (default: `Flat-Lus2`)        |
| `-t`   | Set run mode to **training** (`train`)                    |
| `-p`   | Set run mode to **playback** (`play`)                     |
| `-s`   | Set run mode to **simulation only** (`sim2mujoco`)        |
| `-a`   | Set run mode to **evaluation** (`assess`)                 |
| `-l`   | Load a previous run (implies `--resume=True`)             |
| `-c`   | Load a specific checkpoint by index                       |
| `-d`   | Enable demo trajectory playback (`--play_demo_traj`)      |
| `-r`   | Export the trained model to RKNN format (`--export_rknn`) |

# Troubleshooting

- Data Error: Insufficient Frames

  You may see this error if the trajectory data is too short:

```yaml
AssertionError: required frame num 1000 should less than the loaded trajtectory frame number [xxx.]
```

The RL environment expects **at least 1000 frames** per trajectory (defined by the max episode length).

If your motion data has fewer frames (e.g., 200), you can either repeat the sequence multiple times or interpolate between keyframes to generate smooth transitions and reach the required 1000 frames.

# References

- [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/)
- [RL Games Framework](https://github.com/Denys88/rl_games)
- [Kepler Robot Specifications](https://www.kepler.com/robotics)

# Acknowledgements
