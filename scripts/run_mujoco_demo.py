from __future__ import annotations

from pathlib import Path


def main() -> None:
    try:
        import mujoco
    except ImportError as exc:
        raise SystemExit("MuJoCo is not installed. On Ubuntu 24.04 run: pip install mujoco") from exc

    model_path = Path(__file__).resolve().parents[1] / "mujoco" / "ddmoma_7dof.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    for _ in range(100):
        mujoco.mj_step(model, data)
    print(f"Loaded {model_path.name}: nq={model.nq}, nv={model.nv}, nbody={model.nbody}")


if __name__ == "__main__":
    main()
