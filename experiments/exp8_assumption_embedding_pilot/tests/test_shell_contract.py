from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class EntrypointContractTests(unittest.TestCase):
    def test_only_gpu_stage_shell_files_remain(self) -> None:
        observed = {path.name for path in ROOT.glob("[0-9][0-9]_*.sh")}
        self.assertEqual(observed, {"01_cache_embeddings.sh", "06_run_exp05_fusion.sh"})
        self.assertFalse((ROOT / "submit_all.sh").exists())
        self.assertFalse((ROOT / "run_smoke.sh").exists())

    def test_gpu_stages_have_array_and_direct_sbatch_contracts(self) -> None:
        for name in ("01_cache_embeddings.sh", "06_run_exp05_fusion.sh"):
            path = ROOT / name
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines[0], "#!/bin/bash", name)
            header = "\n".join(lines[:12])
            for directive in (
                "#SBATCH --job-name=",
                "#SBATCH --output=_log/",
                "#SBATCH --partition=cpu",
                "#SBATCH --time=00:15:00",
                "#SBATCH --nodes=1",
                "#SBATCH --mem=2G",
                "#SBATCH --chdir=.",
            ):
                self.assertIn(directive, header, f"{name}: {directive}")
            text = path.read_text(encoding="utf-8")
            for required in ("MODE=", "DRY_RUN", "LOCAL", "--array=", "05:45:00", "gpu:A6000:1", "FINAL_JOB_ID="):
                self.assertIn(required, text, f"{name}: {required}")

    def test_cpu_stages_are_exposed_by_python_cli(self) -> None:
        text = (ROOT / "run_cpu_stage.py").read_text(encoding="utf-8")
        for stage in (
            '"prepare"',
            '"exp01"',
            '"exp02"',
            '"exp03"',
            '"exp04"',
            '"exp06-sample"',
            '"exp06-summarize"',
            '"pilot-summary"',
        ):
            self.assertIn(stage, text)
        self.assertNotIn("sbatch", text)
        self.assertIn("ThreadPoolExecutor", text)

    def test_result_directories_remain_isolated(self) -> None:
        cpu_runner = (ROOT / "run_cpu_stage.py").read_text(encoding="utf-8")
        for result in ("exp01_results", "exp02_results", "exp03_results", "exp04_results", "exp06_results"):
            self.assertIn(result, cpu_runner)
        self.assertIn("exp05_results", (ROOT / "06_run_exp05_fusion.sh").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
