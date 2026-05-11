# Architecture

## Core Data Contract

Robot state:

```text
[x, y, theta, q1, q2, q3, q4, q5, q6, q7]
```

Path point:

```text
[x, y, cos(theta), sin(theta), q1, q2, q3, q4, q5, q6, q7]
```

Each candidate path has shape:

```text
[64, 11]
```

The planner batch output has shape:

```text
[sample_count, 64, 11]
```

## Module Boundaries

- `kinematics.py`: robot spec, state/path conversion, KSE keypoint sequence.
- `scene.py`: synthetic obstacle generation, point clouds, SDF queries.
- `primitives.py`: K-Means primitive library.
- `diffusion.py`: truncated primitive diffusion formula.
- `planner.py`: MVP `PlannerModel.forward` API.
- `losses.py`: training loss components used by the future PyTorch model.
- `optimizer.py`: lightweight path pruning and resampling post-processing.
- `metrics.py`: reproduction metrics helpers.
- `data.py`: dataset sample file contract.
- `training.py`: primitive fitting from MVP dataset samples.
- `benchmark.py`: paper metric contract for `S.R.`, `T.P./ms`, `T.D./s`, and `D.S.`.

## ROS 2 Interface

The fixed action endpoint is `/nmoma/plan`, backed by `nmoma_msgs/action/Plan.action`.

Request:

```text
float64[] start_state
float64[] goal_state
sensor_msgs/PointCloud2 scene
int32 sample_count
```

Result:

```text
nav_msgs/Path base_path
trajectory_msgs/JointTrajectory arm_trajectory
bool success
string message
```

Feedback:

```text
float32 progress
string stage
```

## Upgrade Path

The intended replacement seam is `PlannerModel.forward`. Keep the same input/output contract and replace the numpy sampler with a PyTorch PTDM model. ROS 2, MuJoCo, data, metrics, and benchmark code should continue to call the same planner API.
