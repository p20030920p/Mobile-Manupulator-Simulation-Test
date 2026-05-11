# Release Checklist

Use this checklist before tagging or pushing a public GitHub version.

## 1. Verify Working Tree

```bash
git status --short
```

Confirm that only intended files are changed.

## 2. Run Required Checks

```bash
python -m unittest discover -s tests
python scripts/run_core_planner_demo.py
python scripts/run_mujoco_demo.py
python scripts/run_benchmark.py --task-count 3 --sample-count 3 --scenes cuboids
python -m compileall nmoma_repro scripts ros2_ws/src/nmoma_ros2/nmoma_ros2
```

## 3. Update Version Notes

- Update `pyproject.toml`.
- Update `CHANGELOG.md`.
- Update `README.md` and `docs/REPRODUCTION_ZH.md` if behavior changed.

## 4. Commit

```bash
git add .
git commit -m "chore: prepare v0.2.0 maintenance release"
```

## 5. Push To GitHub

```bash
git push origin master
```

If the default branch is renamed:

```bash
git branch -M main
git push -u origin main
```

## 6. Optional Tag

```bash
git tag v0.2.0
git push origin v0.2.0
```

## 7. GitHub Review

- Confirm CI passes.
- Confirm README renders correctly.
- Confirm no datasets, checkpoints, `.venv`, ROS build output, or generated artifacts were uploaded.
