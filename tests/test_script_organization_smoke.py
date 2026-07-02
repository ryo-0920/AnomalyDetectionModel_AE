import importlib
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

LEGACY_CLI_SCRIPTS = [
    "1_transformer/train_transformer_autoencoder.py",
    "1_transformer/train_score_csv.py",
    "1_transformer/plot_timechart.py",
    "1_transformer/eval_score_csv.py",
]

NEW_MAINSTREAM_MODULES = [
    "gofumi_ae",
    "gofumi_ae.cli.train",
    "gofumi_ae.cli.score",
    "gofumi_ae.cli.evaluate",
    "gofumi_ae.cli.plot_timechart",
    "gofumi_ae.models",
    "gofumi_ae.training",
    "gofumi_ae.inference",
    "gofumi_ae.evaluation",
    "gofumi_ae.visualization",
    "gofumi_ae.datasets",
    "gofumi_ae.ui",
]

EXPECTED_EXPERIMENT_SCRIPTS = [
    "collect_intentional_accel.py",
    "gofumi_accel_keyboard.py",
    "pngtovideo.py",
    "export_results_excel.py",
]


def run_python(*args: str) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory(prefix="mplconfig-") as mpl_dir:
        env = os.environ.copy()
        env["MPLCONFIGDIR"] = mpl_dir
        return subprocess.run(
            ["python", *args],
            cwd=PROJECT_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )


class ScriptOrganizationSmokeTest(unittest.TestCase):
    def test_gofumi_ae_package_and_mainstream_modules_are_importable(self) -> None:
        for module_name in NEW_MAINSTREAM_MODULES:
            with self.subTest(module=module_name):
                try:
                    importlib.import_module(module_name)
                except Exception as exc:  # pragma: no cover
                    self.fail(f"expected importable module '{module_name}': {exc}")

    def test_legacy_cli_help_commands_succeed(self) -> None:
        for script_path in LEGACY_CLI_SCRIPTS:
            with self.subTest(script=script_path):
                result = run_python(script_path, "--help")
                if result.returncode != 0:
                    self.fail(
                        f"{script_path} --help exited with {result.returncode}\n"
                        f"stdout:\n{result.stdout}\n"
                        f"stderr:\n{result.stderr}"
                    )
                combined_output = f"{result.stdout}\n{result.stderr}".lower()
                self.assertIn("help", combined_output)

    def test_experimental_scripts_are_relocated_under_experiments(self) -> None:
        experiments_dir = PROJECT_ROOT / "experiments"
        self.assertTrue(
            experiments_dir.is_dir(),
            "expected experiments/ directory for experimental helper scripts",
        )

        existing_relpaths = {
            path.relative_to(PROJECT_ROOT).as_posix()
            for path in experiments_dir.rglob("*.py")
        }

        for script_name in EXPECTED_EXPERIMENT_SCRIPTS:
            with self.subTest(script=script_name):
                self.assertTrue(
                    any(relpath.endswith(f"/{script_name}") for relpath in existing_relpaths),
                    f"expected {script_name} to live under experiments/",
                )


if __name__ == "__main__":
    unittest.main()
