from __future__ import annotations

import argparse
import struct
import time
import zlib
from pathlib import Path


ARM_JOINT_NAMES = tuple(f"arm_joint_{index}" for index in range(1, 8))
PLANAR_BASE_JOINT_NAMES = ("base_x", "base_y", "base_yaw")


def assert_finite_state(data, step: int, max_abs_qacc: float = 1.0e6) -> None:
    import numpy as np

    checks = {
        "qpos": data.qpos,
        "qvel": data.qvel,
        "qacc": data.qacc,
    }
    for name, values in checks.items():
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"Simulation became unstable at step {step}: non-finite {name}")
    if float(np.max(np.abs(data.qacc))) > max_abs_qacc:
        raise RuntimeError(
            f"Simulation became unstable at step {step}: qacc exceeded {max_abs_qacc:g}"
        )


def demo_joint_targets(phase: float) -> dict[str, float]:
    import math

    targets = {
        "base_x": 0.55 * math.sin(2.0 * math.pi * phase),
        "base_y": 0.35 * math.sin(4.0 * math.pi * phase + math.pi / 5.0),
        "base_yaw": 0.65 * math.sin(2.0 * math.pi * phase + math.pi / 3.0),
    }
    for index, name in enumerate(ARM_JOINT_NAMES, start=1):
        targets[name] = 0.45 * math.sin(2.0 * math.pi * phase + index * 0.45)
    return targets


def apply_demo_pose(model, data, phase: float) -> None:
    import mujoco

    for joint_name, value in demo_joint_targets(phase).items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise RuntimeError(f"Joint not found in MuJoCo model: {joint_name}")
        qpos_address = model.jnt_qposadr[joint_id]
        data.qpos[qpos_address] = value
    mujoco.mj_forward(model, data)


def write_ppm_frame(path: Path, image) -> None:
    import numpy as np

    frame = np.asarray(image, dtype=np.uint8)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("PPM frame must have shape (height, width, 3)")
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width, _ = frame.shape
    with path.open("wb") as handle:
        handle.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        handle.write(frame.tobytes())


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    checksum = zlib.crc32(kind + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)


def write_png_frame(path: Path, image) -> None:
    import numpy as np

    frame = np.asarray(image, dtype=np.uint8)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("PNG frame must have shape (height, width, 3)")
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width, _ = frame.shape
    raw_scanlines = b"".join(b"\x00" + frame[row].tobytes() for row in range(height))
    png = b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
            _png_chunk(b"IDAT", zlib.compress(raw_scanlines, level=6)),
            _png_chunk(b"IEND", b""),
        ]
    )
    path.write_bytes(png)


def write_frame(path: Path, image, frame_format: str) -> None:
    if frame_format == "png":
        write_png_frame(path.with_suffix(".png"), image)
    elif frame_format == "ppm":
        write_ppm_frame(path.with_suffix(".ppm"), image)
    else:
        raise ValueError(f"unsupported frame format: {frame_format}")


def render_demo_frames(
    mujoco,
    model,
    data,
    render_dir: Path,
    steps: int,
    render_every: int,
    width: int,
    height: int,
    frame_format: str,
) -> int:
    renderer = mujoco.Renderer(model, height=height, width=width)
    frame_count = 0
    for step in range(steps):
        phase = step / max(steps - 1, 1)
        apply_demo_pose(model, data, phase)
        assert_finite_state(data, step)
        if step % render_every == 0 or step == steps - 1:
            renderer.update_scene(data)
            image = renderer.render()
            write_frame(render_dir / f"frame_{frame_count:04d}", image, frame_format)
            frame_count += 1
    renderer.close()
    return frame_count


def run_viewer_demo(mujoco, model, data, steps: int, realtime: bool) -> None:
    import mujoco.viewer

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for step in range(steps):
            phase = step / max(steps - 1, 1)
            apply_demo_pose(model, data, phase)
            assert_finite_state(data, step)
            viewer.sync()
            if realtime:
                time.sleep(float(model.opt.timestep))


def run_stability_smoke(mujoco, model, data, steps: int) -> None:
    for step in range(steps):
        mujoco.mj_step(model, data)
        assert_finite_state(data, step)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or visualize the NMOMA MuJoCo model.")
    parser.add_argument("--steps", type=int, default=500, help="Number of simulation/demo steps.")
    parser.add_argument("--viewer", action="store_true", help="Open an interactive MuJoCo viewer window.")
    parser.add_argument("--render-dir", type=Path, default=None, help="Directory for exported PPM demo frames.")
    parser.add_argument("--render-every", type=int, default=20, help="Render every N steps when --render-dir is set.")
    parser.add_argument("--frame-format", choices=["png", "ppm"], default="png", help="Frame format for --render-dir.")
    parser.add_argument("--width", type=int, default=960, help="Rendered frame width.")
    parser.add_argument("--height", type=int, default=540, help="Rendered frame height.")
    parser.add_argument("--no-realtime", action="store_true", help="Do not sleep between viewer frames.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import mujoco
    except ImportError as exc:
        raise SystemExit("MuJoCo is not installed. On Ubuntu 24.04 run: pip install mujoco") from exc

    model_path = Path(__file__).resolve().parents[1] / "mujoco_models" / "ddmoma_7dof.xml"
    xml = model_path.read_text(encoding="utf-8")
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    if args.render_every < 1:
        raise SystemExit("--render-every must be at least 1")

    rendered_frames = 0
    if args.viewer:
        run_viewer_demo(mujoco, model, data, args.steps, realtime=not args.no_realtime)
    elif args.render_dir is not None:
        rendered_frames = render_demo_frames(
            mujoco,
            model,
            data,
            args.render_dir,
            args.steps,
            args.render_every,
            args.width,
            args.height,
            args.frame_format,
        )
    else:
        run_stability_smoke(mujoco, model, data, args.steps)

    print(
        f"Loaded {model_path.name}: nq={model.nq}, nv={model.nv}, "
        f"nbody={model.nbody}, stable_steps={args.steps}"
    )
    if args.render_dir is not None:
        print(f"Rendered {rendered_frames} demo frames to {args.render_dir}")
    elif not args.viewer:
        print("This was a smoke test only. Use --viewer or --render-dir artifacts/mujoco_demo_frames for a visual demo.")


if __name__ == "__main__":
    main()
