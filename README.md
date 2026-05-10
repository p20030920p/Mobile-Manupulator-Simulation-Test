# NMOMA Reproduction MVP

This repository is a practical reproduction scaffold for the paper
**Primitive-based Truncated Diffusion for Efficient Trajectory Generation of Differential Drive Mobile Manipulators**.

The current implementation focuses on a runnable MVP:

- Pure Python/numpy core modules for keypoint extraction, synthetic scenes, primitive libraries, PTDM-style sampling, losses, post-processing, metrics, and dataset I/O.
- ROS 2 Jazzy interface scaffolding for `/nmoma/plan`.
- MuJoCo MJCF model for a differential-drive mobile base with a 7-DoF arm.
- Tests that run without ROS 2, MuJoCo, PyTorch, or internet access.

## Quick Check

```bash
python -m unittest discover -s tests
python scripts/run_core_planner_demo.py
```

On this Codex workspace, use the bundled Python executable if `python` is not associated correctly:

```powershell
& 'C:\Users\qzl\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -X utf8 -m unittest discover -s tests
```

## Repository Layout

- `nmoma_repro/`: Core reproduction package.
- `tests/`: Unit and scaffold contract tests.
- `scripts/`: Dataset generation, core planner demo, MuJoCo smoke demo.
- `mujoco/ddmoma_7dof.xml`: MuJoCo robot model scaffold.
- `ros2_ws/src/nmoma_msgs`: ROS 2 action interface package.
- `ros2_ws/src/nmoma_ros2`: ROS 2 Python action server wrapper.
- `docs/REPRODUCTION_ZH.md`: Chinese reproduction guide.

## Current Fidelity

This is not yet a paper-number reproduction. The official NMOMA repository did not include the full training/evaluation implementation when the plan was created, so this repo implements a transparent MVP that preserves the public contracts and algorithmic shape. The next step is replacing the numpy sampler with the full PyTorch PTDM and paper-scale data generation.
