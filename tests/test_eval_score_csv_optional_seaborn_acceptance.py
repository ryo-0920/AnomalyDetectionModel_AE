import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class EvalScoreCsvOptionalSeabornAcceptanceTest(unittest.TestCase):
    def test_legacy_cli_help_starts_when_seaborn_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory(prefix="no-seaborn-") as blocker_dir:
            blocker_path = Path(blocker_dir)
            self.write_seaborn_blocker(blocker_path)
            env = self.child_env(blocker_path)

            blocked_import = self.run_python(
                "-c",
                "import seaborn",
                env=env,
            )
            self.assertNotEqual(blocked_import.returncode, 0)
            self.assertIn("No module named 'seaborn'", blocked_import.stderr)

            help_result = self.run_python(
                "1_transformer/eval_score_csv.py",
                "--help",
                env=env,
            )

        self.assertEqual(
            help_result.returncode,
            0,
            f"help failed\nstdout:\n{help_result.stdout}\nstderr:\n{help_result.stderr}",
        )
        combined_output = f"{help_result.stdout}\n{help_result.stderr}"
        self.assertIn("--config", combined_output)
        self.assertIn("--config_on", combined_output)
        self.assertIn("--config_off", combined_output)
        self.assertNotIn("ModuleNotFoundError", combined_output)
        self.assertNotIn("No module named 'seaborn'", combined_output)

    @staticmethod
    def write_seaborn_blocker(path: Path) -> None:
        sitecustomize = path / "sitecustomize.py"
        sitecustomize.write_text(
            textwrap.dedent(
                """
                import importlib.abc
                import sys


                class BlockSeaborn(importlib.abc.MetaPathFinder):
                    def find_spec(self, fullname, path=None, target=None):
                        if fullname == "seaborn" or fullname.startswith("seaborn."):
                            raise ModuleNotFoundError("No module named 'seaborn'")
                        return None


                sys.meta_path.insert(0, BlockSeaborn())
                """
            ),
            encoding="utf-8",
        )

    @staticmethod
    def child_env(blocker_path: Path) -> dict[str, str]:
        env = os.environ.copy()
        env["MPLCONFIGDIR"] = str(blocker_path / "mplconfig")
        existing_pythonpath = env.get("PYTHONPATH")
        pythonpath_parts = [str(blocker_path)]
        if existing_pythonpath:
            pythonpath_parts.append(existing_pythonpath)
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
        return env

    @staticmethod
    def run_python(*args: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, *args],
            cwd=PROJECT_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )


if __name__ == "__main__":
    unittest.main()
