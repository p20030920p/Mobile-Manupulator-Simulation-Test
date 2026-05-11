# Changelog

All notable changes to this repository are tracked here.

## 0.2.1 - 2026-05-12

- Added visual MuJoCo demo modes: `--viewer` for an interactive window and `--render-dir` for frame export.
- Added a pure standard-library PNG frame writer so VM/headless users can inspect rendered frames without Pillow/OpenCV/imageio.
- Clarified that the default MuJoCo command is a stability smoke test, not a visual demo.

## 0.2.0 - 2026-05-11

- Fixed MuJoCo instability by replacing the 6DoF `base_free` scaffold with a planar `base_x/base_y/base_yaw` base that matches the paper state space.
- Renamed the local MuJoCo model directory to `mujoco_models/` so it does not shadow the official `mujoco` Python package.
- Added primitive fitting and benchmark scaffolds with paper-style metric names: `S.R.`, `T.P./ms`, `T.D./s`, and `D.S.`.
- Added GitHub Actions CI, pull request template, issue templates, release checklist, roadmap, and contribution guide.

## 0.1.0 - 2026-05-10

- Added the initial NMOMA MVP reproduction scaffold.
- Implemented numpy core modules for KSE, scenes/SDF, primitive libraries, diffusion formula, losses, post-processing, planner API, metrics, and dataset I/O.
- Added ROS 2 action wrapper scaffolding and the first MuJoCo robot model scaffold.
