from __future__ import annotations

from pathlib import Path


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


def main() -> None:
    try:
        import mujoco
    except ImportError as exc:
        raise SystemExit("MuJoCo is not installed. On Ubuntu 24.04 run: pip install mujoco") from exc

    model_path = Path(__file__).resolve().parents[1] / "mujoco_models" / "ddmoma_7dof.xml"
    xml = model_path.read_text(encoding="utf-8")
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    for step in range(500):
        mujoco.mj_step(model, data)
        assert_finite_state(data, step)
    print(
        f"Loaded {model_path.name}: nq={model.nq}, nv={model.nv}, "
        f"nbody={model.nbody}, stable_steps=500"
    )


if __name__ == "__main__":
    main()
