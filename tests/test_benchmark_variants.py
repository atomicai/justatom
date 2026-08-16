import os
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parents[1]


class BenchmarkVariantTests(unittest.TestCase):
    @unittest.skipIf(os.name == "nt", "benchmark shell tests require a Unix environment")
    def test_all_variants_are_vanilla_atom_gate_and_atomic(self):
        with TemporaryDirectory() as tmpdir:
            bench_root = Path(tmpdir) / "bench"
            result = subprocess.run(
                [
                    "bash",
                    "scripts/run_benchmark.sh",
                    "--dataset-ids",
                    "justatom",
                    "--variants",
                    "all",
                    "--bench-root",
                    str(bench_root),
                    "--dry-run",
                ],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Variants: vanilla atom_gate atomic", result.stdout)
            self.assertNotIn("bank_only", result.stdout)

            commands = (bench_root / "COMMANDS.md").read_text()
            self.assertIn("## vanilla", commands)
            self.assertIn("## atom_gate", commands)
            self.assertIn("## atomic", commands)
            self.assertNotIn("## bank_only", commands)

    @unittest.skipIf(os.name == "nt", "benchmark shell tests require a Unix environment")
    def test_atomic_variant_delegates_canonical_defaults_to_method_profile(self):
        with TemporaryDirectory() as tmpdir:
            bench_root = Path(tmpdir) / "bench"
            result = subprocess.run(
                [
                    "bash",
                    "scripts/run_benchmark.sh",
                    "--dataset-ids",
                    "justatom",
                    "--variants",
                    "atomic",
                    "--bench-root",
                    str(bench_root),
                    "--dry-run",
                ],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)

            commands = (bench_root / "COMMANDS.md").read_text()
            self.assertIn("## atomic", commands)
            self.assertIn("--method atomic", commands)
            self.assertNotIn("--recipe", commands)
            self.assertNotIn("--memory-bank-size", commands)

    @unittest.skipIf(os.name == "nt", "benchmark shell tests require a Unix environment")
    def test_benchmark_forwards_normalized_memory_mass_controls(self):
        with TemporaryDirectory() as tmpdir:
            bench_root = Path(tmpdir) / "bench"
            result = subprocess.run(
                [
                    "bash",
                    "scripts/run_benchmark.sh",
                    "--dataset-ids",
                    "justatom",
                    "--variants",
                    "atomic",
                    "--bench-root",
                    str(bench_root),
                    "--memory-bank-mass-ratio",
                    "0.5",
                    "--memory-bank-mass-ramp-steps",
                    "20",
                    "--dry-run",
                ],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            commands = (bench_root / "COMMANDS.md").read_text()
            self.assertIn("--memory-bank-mass-ratio 0.5", commands)
            self.assertIn("--memory-bank-mass-ramp-steps 20", commands)

    @unittest.skipIf(os.name == "nt", "benchmark shell tests require a Unix environment")
    def test_benchmark_forwards_train_config_and_auxiliary_gradient_controls(self):
        with TemporaryDirectory() as tmpdir:
            bench_root = Path(tmpdir) / "bench"
            result = subprocess.run(
                [
                    "bash",
                    "scripts/run_benchmark.sh",
                    "--dataset-ids",
                    "justatom",
                    "--variants",
                    "atom_gate",
                    "--bench-root",
                    str(bench_root),
                    "--train-config",
                    "configs/experiments/qwen3-06b-lora-alpha-gradient-safe.yaml",
                    "--aux-gradient-mode",
                    "safe",
                    "--aux-gradient-max-norm-ratio",
                    "0.25",
                    "--aux-gradient-eps",
                    "1e-12",
                    "--dry-run",
                ],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            commands = (bench_root / "COMMANDS.md").read_text()
            self.assertIn(
                "--train-config configs/experiments/qwen3-06b-lora-alpha-gradient-safe.yaml",
                commands,
            )
            self.assertIn("--aux-gradient-mode safe", commands)
            self.assertIn("--aux-gradient-max-norm-ratio 0.25", commands)
            self.assertIn("--aux-gradient-eps 1e-12", commands)

    @unittest.skipIf(os.name == "nt", "pipeline shell tests require a Unix environment")
    def test_pipeline_rejects_missing_train_config_before_creating_run_root(self):
        with TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "runs"
            result = subprocess.run(
                [
                    "bash",
                    "scripts/run_pipeline.sh",
                    "--dataset-ids",
                    "justatom",
                    "--output-root",
                    str(output_root),
                    "--train-config",
                    "configs/experiments/missing.yaml",
                    "--eval-only",
                ],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("training config does not exist", result.stderr)
            self.assertFalse(output_root.exists())

    @unittest.skipIf(os.name == "nt", "benchmark shell tests require a Unix environment")
    def test_retired_bank_variant_points_to_atomic(self):
        result = subprocess.run(
            [
                "bash",
                "scripts/run_benchmark.sh",
                "--dataset-ids",
                "justatom",
                "--variants",
                "atom_gate_bank",
                "--dry-run",
            ],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid variant", result.stderr)
        self.assertIn("vanilla,atom_gate,atomic", result.stderr)

    def test_pipeline_uses_dev_dataset_preset_for_eval_when_available(self):
        script = (REPO_ROOT / "scripts" / "run_pipeline.sh").read_text()

        self.assertIn("resolve_eval_dataset_id()", script)
        self.assertIn('eval_config_id="$(resolve_eval_dataset_id "$dataset_id")"', script)
        self.assertIn('"$dataset_dir/tuned_eval" "$eval_config_id"', script)

    def test_benchmark_uses_current_shell_for_pipeline(self):
        script = (REPO_ROOT / "scripts" / "run_benchmark.sh").read_text()

        self.assertIn('PIPELINE_SHELL="${PIPELINE_SHELL:-${BASH:-bash}}"', script)
        self.assertIn('command=("$PIPELINE_SHELL" "$REPO_ROOT/scripts/run_pipeline.sh")', script)
        self.assertNotIn('command=(bash "$REPO_ROOT/scripts/run_pipeline.sh")', script)
        self.assertIn("requires Bash >= 4", script)


if __name__ == "__main__":
    unittest.main()
