from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ShellContractTests(unittest.TestCase):
    def test_all_declared_stages_exist(self) -> None:
        names = [
            "00_prepare_data.sh",
            "01_cache_embeddings.sh",
            "02_run_exp01_audit.sh",
            "03_run_exp02_retrieval.sh",
            "04_run_exp03_residual.sh",
            "05_run_exp04_controls.sh",
            "06_run_exp05_fusion.sh",
            "07_run_exp06_sample.sh",
            "08_run_exp06_summarize.sh",
            "09_summarize_pilot.sh",
            "submit_all.sh",
        ]
        for name in names:
            self.assertTrue((ROOT / name).is_file(), name)

    def test_heavy_stages_have_array_dry_run_local_and_time_limit(self) -> None:
        heavy = (
            "00_prepare_data.sh",
            "01_cache_embeddings.sh",
            "03_run_exp02_retrieval.sh",
            "04_run_exp03_residual.sh",
            "05_run_exp04_controls.sh",
            "06_run_exp05_fusion.sh",
        )
        for name in heavy:
            text = (ROOT / name).read_text(encoding="utf-8")
            for required in ("MODE=", "DRY_RUN", "LOCAL", "--array=", "05:45:00", "FINAL_JOB_ID="):
                self.assertIn(required, text, f"{name}: {required}")

    def test_stage_scripts_have_direct_sbatch_headers(self) -> None:
        stages = sorted(ROOT.glob("[0-9][0-9]_*.sh"))
        self.assertEqual(len(stages), 10)
        for path in stages:
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines[0], "#!/bin/bash", path.name)
            header = "\n".join(lines[:12])
            for directive in (
                "#SBATCH --job-name=",
                "#SBATCH --output=_log/",
                "#SBATCH --partition=",
                "#SBATCH --time=",
                "#SBATCH --nodes=1",
                "#SBATCH --mem=",
                "#SBATCH --chdir=.",
            ):
                self.assertIn(directive, header, f"{path.name}: {directive}")

    def test_result_directories_are_isolated(self) -> None:
        mapping = {
            "02_run_exp01_audit.sh": "exp01_results",
            "03_run_exp02_retrieval.sh": "exp02_results",
            "04_run_exp03_residual.sh": "exp03_results",
            "05_run_exp04_controls.sh": "exp04_results",
            "06_run_exp05_fusion.sh": "exp05_results",
            "07_run_exp06_sample.sh": "exp06_results",
            "08_run_exp06_summarize.sh": "exp06_results",
        }
        for name, expected in mapping.items():
            self.assertIn(expected, (ROOT / name).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
