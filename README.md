# NMOMA Reproduction MVP

Version: `0.2.0`

This repository is a practical reproduction scaffold for the paper
**Primitive-based Truncated Diffusion for Efficient Trajectory Generation of Differential Drive Mobile Manipulators**.

The current implementation focuses on a runnable MVP:

- Pure Python/numpy core modules for keypoint extraction, synthetic scenes, primitive libraries, PTDM-style sampling, losses, post-processing, metrics, and dataset I/O.
- ROS 2 Jazzy interface scaffolding for `/nmoma/plan`.
- MuJoCo MJCF model for a differential-drive mobile base with a 7-DoF arm.
- Tests and smoke checks that keep each version maintainable.

## Quick Check

```bash
python -m unittest discover -s tests
python scripts/run_core_planner_demo.py
python scripts/run_mujoco_demo.py
python scripts/run_benchmark.py --task-count 5 --sample-count 3
```

On this Codex workspace, use the bundled Python executable if `python` is not associated correctly:

```powershell
& 'C:\Users\qzl\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -X utf8 -m unittest discover -s tests
```

## Repository Layout

- `nmoma_repro/`: Core reproduction package.
- `tests/`: Unit and scaffold contract tests.
- `scripts/`: Dataset generation, core planner demo, MuJoCo smoke demo.
- `mujoco_models/ddmoma_7dof.xml`: MuJoCo robot model scaffold.
- `ros2_ws/src/nmoma_msgs`: ROS 2 action interface package.
- `ros2_ws/src/nmoma_ros2`: ROS 2 Python action server wrapper.
- `docs/REPRODUCTION_ZH.md`: Chinese reproduction guide.
- `docs/ROADMAP.md`: staged reproduction roadmap.
- `docs/RELEASE_CHECKLIST.md`: release and GitHub upload checklist.

## MVP Training And Benchmark Flow

```bash
python scripts/generate_mvp_dataset.py --count 100 --scene cuboids --out datasets/mvp_cuboids
python scripts/fit_primitives.py --dataset datasets/mvp_cuboids --out artifacts/primitives_mvp.npz --primitive-count 32
python scripts/run_benchmark.py --scenes cuboids,mixed --task-count 100 --sample-count 4 --primitive-library artifacts/primitives_mvp.npz
```

The benchmark report uses the paper's metric names: `S.R.`, `T.P./ms`, `T.D./s`, and `D.S.`.

## Maintenance Flow

Every public version should pass:

```bash
python -m unittest discover -s tests
python scripts/run_core_planner_demo.py
python scripts/run_mujoco_demo.py
python scripts/run_benchmark.py --task-count 3 --sample-count 3 --scenes cuboids
python -m compileall nmoma_repro scripts ros2_ws/src/nmoma_ros2/nmoma_ros2
```

Before uploading to GitHub, follow `docs/RELEASE_CHECKLIST.md`. The repository includes GitHub Actions CI under `.github/workflows/ci.yml`, plus PR and issue templates for maintainable iteration.

Typical upload flow:

```bash
git add .
git commit -m "chore: prepare v0.2.0 maintenance release"
git push origin master
```

## Current Fidelity

This is not yet a paper-number reproduction. The official NMOMA repository did not include the full training/evaluation implementation when the plan was created, so this repo implements a transparent MVP that preserves the public contracts and algorithmic shape. The next step is replacing the numpy sampler with the full PyTorch PTDM and paper-scale data generation.
