from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class EntrypointContractTests(unittest.TestCase):
    def test_stage_directories_have_only_requested_entrypoints(self) -> None:
        stages = [path for path in ROOT.iterdir() if path.is_dir() and path.name[:2].isdigit()]
        self.assertEqual({path.name[:2] for path in stages}, {f"{index:02d}" for index in range(8)})
        for stage in stages:
            files = {path.name for path in stage.iterdir() if path.is_file()}
            expected = {"run.py", "run.sh"} if stage.name[:2] in {"01", "06"} else {"run.py"}
            self.assertEqual(files, expected, stage.name)

    def test_only_gpu_stages_have_slurm_shells(self) -> None:
        shells = {path.parent.name[:2]: path for path in ROOT.glob("[0-9][0-9]_*/run.sh")}
        self.assertEqual(set(shells), {"01", "06"})
        for stage, path in shells.items():
            text = path.read_text(encoding="utf-8")
            self.assertEqual(text.splitlines()[0], "#!/bin/bash")
            for required in ("#SBATCH", "--array=", "05:45:00", "gpu:A6000:1", "FINAL_JOB_ID="):
                self.assertIn(required, text, f"stage {stage}: {required}")

    def test_cpu_stage_entrypoints_are_plain_sequential_python(self) -> None:
        for stage in ("00", "02", "03", "04", "05", "07"):
            path = next(ROOT.glob(f"{stage}_*/run.py"))
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("ThreadPoolExecutor", text, stage)
            self.assertNotIn("run_parallel", text, stage)
            self.assertNotIn("run_single", text, stage)
            self.assertNotIn("run_cpu_stage", text, stage)
            self.assertNotIn("print(", text, stage)

    def test_aggregate_cpu_runner_is_removed(self) -> None:
        self.assertFalse((ROOT / "run_cpu_stage.py").exists())

    def test_stages_do_not_require_a_shared_python_package(self) -> None:
        self.assertFalse((ROOT / "common").exists())
        for path in ROOT.glob("[0-9][0-9]_*/run.py"):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("exp8_assumption_embedding_pilot.common", text, path.parent.name)
            self.assertNotIn("sys.path", text, path.parent.name)


if __name__ == "__main__":
    unittest.main()
