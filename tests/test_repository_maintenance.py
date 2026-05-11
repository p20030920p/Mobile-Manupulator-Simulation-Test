import tomllib
import unittest
from pathlib import Path

import nmoma_repro


ROOT = Path(__file__).resolve().parents[1]


class RepositoryMaintenanceTest(unittest.TestCase):
    def test_project_version_is_declared_once_in_pyproject(self):
        data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

        self.assertEqual(data["project"]["version"], "0.2.0")
        self.assertEqual(nmoma_repro.__version__, "0.2.0")

    def test_github_ci_runs_core_reproduction_checks(self):
        workflow_path = ROOT / ".github" / "workflows" / "ci.yml"
        steps = workflow_path.read_text(encoding="utf-8")

        self.assertIn("python -m unittest discover -s tests", steps)
        self.assertIn("python scripts/run_core_planner_demo.py", steps)
        self.assertIn("python scripts/run_mujoco_demo.py", steps)
        self.assertIn("python scripts/run_benchmark.py --task-count 3 --sample-count 3 --scenes cuboids", steps)

    def test_maintenance_documents_exist_and_reference_release_flow(self):
        required = [
            ROOT / "CHANGELOG.md",
            ROOT / "CONTRIBUTING.md",
            ROOT / "docs" / "RELEASE_CHECKLIST.md",
            ROOT / "docs" / "ROADMAP.md",
            ROOT / ".github" / "pull_request_template.md",
        ]
        for path in required:
            self.assertTrue(path.exists(), f"missing {path}")

        changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        checklist = (ROOT / "docs" / "RELEASE_CHECKLIST.md").read_text(encoding="utf-8")
        self.assertIn("0.2.0", changelog)
        self.assertIn("python -m unittest discover -s tests", checklist)
        self.assertIn("git push origin", checklist)


if __name__ == "__main__":
    unittest.main()
