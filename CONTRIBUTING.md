# Contributing

This repository is maintained as a staged paper reproduction. Keep every version runnable, tested, and documented.

## Development Rules

- Preserve the public planner API unless the change includes tests and documentation updates.
- Do not commit generated datasets, model checkpoints, ROS build output, or benchmark artifacts.
- Add or update tests before changing behavior.
- Keep smoke checks small enough for GitHub Actions.
- Document reproduction gaps honestly; do not report paper-level parity until benchmark evidence exists.

## Required Local Checks

```bash
python -m unittest discover -s tests
python scripts/run_core_planner_demo.py
python scripts/run_mujoco_demo.py
python scripts/run_benchmark.py --task-count 3 --sample-count 3 --scenes cuboids
```

## Versioning

Use semantic versioning for the MVP:

- Patch: bug fixes and docs-only maintenance.
- Minor: new reproduction capability, benchmark scaffold, ROS/MuJoCo interface improvement.
- Major: breaking public API or dataset contract changes.

Update `pyproject.toml`, `CHANGELOG.md`, and `docs/RELEASE_CHECKLIST.md` when preparing a release.
