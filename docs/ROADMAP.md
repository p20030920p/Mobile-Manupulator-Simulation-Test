# Roadmap

## v0.2.x - Maintainable MVP

- Keep CI green on Ubuntu 24.04.
- Stabilize MuJoCo model and ROS 2 action contracts.
- Add small benchmark and primitive-fitting smoke workflows.
- Keep all generated data out of git.

## v0.3.0 - Dataset And Baseline Expansion

- Add reproducible Cuboids/Mixed dataset generation presets.
- Add baseline comparison runners for linear MVP and primitive-conditioned MVP.
- Export benchmark reports to JSON/CSV.

## v0.4.0 - PyTorch PTDM Skeleton

- Add PyTorch modules for KSE encoder, point-cloud encoder, attention fusion, primitive classifier, and denoiser.
- Keep `PlannerModel.forward(pointcloud, start_state, goal_state, sample_count)` compatible.
- Add CPU-scale training smoke tests before GPU training.

## v0.5.0 - ROS 2 + MuJoCo Closed Loop

- Add action client demo.
- Publish planned base path and arm trajectory into MuJoCo execution.
- Record execution logs for replayable benchmark cases.

## v1.0.0 - Paper-Level Reproduction Candidate

- Run large-scale benchmark on Cuboids, Mixed, and Replica-style scenes.
- Report `S.R.`, `T.P./ms`, `T.D./s`, and `D.S.` with fixed seeds.
- Compare PTDM, vanilla DDPM-style sampler, and classical baseline.
