# NMOMA 论文复现说明

## 当前实现状态

这个工作区已经落地了一个可运行的 MVP，而不是只停留在计划层：

- `KSE`：`nmoma_repro/kinematics.py` 实现差速移动机械臂状态到关键点序列的映射。
- `Scene/SDF`：`nmoma_repro/scene.py` 支持 `cuboids` 和 `mixed` 随机场景、点云采样和 SDF 查询。
- `Primitive Library`：`nmoma_repro/primitives.py` 用 numpy 实现 K-Means primitive 聚类和最近 primitive 查询。
- `PTDM-style sampler`：`nmoma_repro/planner.py` 暴露论文计划中的 `PlannerModel.forward(pointcloud, start_state, goal_state, sample_count)` API。
- `Truncated diffusion`：`nmoma_repro/diffusion.py` 实现线性 diffusion schedule 和 primitive 截断加噪公式。
- `Loss/Post-processing`：`nmoma_repro/losses.py` 和 `nmoma_repro/optimizer.py` 实现重建、平滑、均匀采样、安全损失接口，以及路径剪枝/重采样。
- `ROS 2`：`ros2_ws/src/nmoma_msgs/action/Plan.action` 固定 `/nmoma/plan` action 接口，`nmoma_ros2` 提供 action server 包装。
- `MuJoCo`：`mujoco/ddmoma_7dof.xml` 提供差速底盘 + 7DoF 机械臂模型脚手架。

## Ubuntu 24.04 复现环境

推荐系统：

```bash
sudo apt update
sudo apt install -y git python3-pip python3-venv build-essential cmake
sudo apt install -y ros-jazzy-desktop python3-colcon-common-extensions python3-rosdep
```

Python 依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-core.txt
pip install -r requirements-ubuntu24-ros2-mujoco.txt
pip install -e .
```

ROS 2 构建：

```bash
source /opt/ros/jazzy/setup.bash
cd ros2_ws
colcon build --symlink-install
source install/setup.bash
ros2 run nmoma_ros2 nmoma_planner_node
```

MuJoCo smoke test：

```bash
python scripts/run_mujoco_demo.py
```

核心算法 smoke test：

```bash
python -m unittest discover -s tests
python scripts/run_core_planner_demo.py
```

## 如何继续推进到论文级复现

1. 先用 `scripts/generate_mvp_dataset.py --count 1000 --scene cuboids` 生成小规模数据，验证数据格式和训练管线。
2. 把 `PlannerModel` 替换成 PyTorch 模型：PointNet/Point Transformer 点云 encoder、KSE MLP encoder、cross-attention fusion、primitive classifier、DiT denoiser。
3. 用 `DiffusionSchedule.linear(total_steps=1200)` 和 `truncated_step=50` 对齐论文的 `50/1200` 训练截断设置。
4. 推理阶段保留 `sample_count=3..5`，实现 2-step DDIM denoising，并把输出接入 `optimizer.prune_path` 和后续轨迹优化。
5. 将当前简化 post-processing 替换为 TopAY 风格 arc length-yaw piecewise polynomial optimization。
6. Benchmark 指标固定为 `S.R.`、`T.P.`、`T.D.`、`D.S.`，先在 `cuboids/mixed` 上复现实验趋势，再加入 ReplicaCAD。

## 当前限制

- 当前没有 PyTorch 训练网络，只实现了可测试的 numpy MVP 和接口契约。
- 当前 MuJoCo 模型是结构脚手架，不等价于论文作者的完整机器人动力学参数。
- 当前 ROS 2 action 使用 MVP planner，尚未接入 GPU 模型、SDF 服务器或全量轨迹优化器。
- 官方 `nmoma/nmoma` 完整代码发布后，应优先对齐其模型、权重、数据集脚本和评测协议。
