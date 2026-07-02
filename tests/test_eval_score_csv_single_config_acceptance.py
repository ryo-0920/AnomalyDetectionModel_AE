import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

MPLCONFIGDIR = Path(tempfile.gettempdir()) / "gofumi_ae_test_mplconfig"
MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

if "seaborn" not in sys.modules:
    seaborn_stub = types.ModuleType("seaborn")
    seaborn_stub.boxplot = mock.Mock(name="seaborn.boxplot")
    seaborn_stub.heatmap = mock.Mock(name="seaborn.heatmap")
    sys.modules["seaborn"] = seaborn_stub

from gofumi_ae.evaluation import standard


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_python(*args: str) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory(prefix="mplconfig-") as mpl_dir:
        env = os.environ.copy()
        env["MPLCONFIGDIR"] = mpl_dir
        return subprocess.run(
            [sys.executable, *args],
            cwd=PROJECT_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )


class EvalScoreCsvSingleConfigAcceptanceTest(unittest.TestCase):
    def test_legacy_cli_help_shows_new_and_legacy_config_options(self) -> None:
        result = run_python("1_transformer/eval_score_csv.py", "--help")

        self.assertEqual(
            result.returncode,
            0,
            f"help failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
        combined = f"{result.stdout}\n{result.stderr}"
        self.assertRegex(
            combined,
            r"(?<![A-Za-z0-9_-])--config(?![A-Za-z0-9_-])",
            "expected --config as a standalone CLI option, not only as a prefix of legacy options",
        )
        for option in ("--config_on", "--config_off"):
            with self.subTest(option=option):
                self.assertIn(option, combined)

    def test_single_config_routes_same_json_to_on_and_off_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            single_config = tmp_path / "single_config.json"
            self.write_json(single_config, self.single_config())

            calls = self.run_standard_main_with_eval_mocks(
                "--config",
                str(single_config),
                "--on_dir",
                str(tmp_path / "on"),
                "--off_dir",
                str(tmp_path / "off"),
                "--out_dir",
                str(tmp_path / "out"),
            )

        self.assertEqual(calls["label_cfg"]["path"], "label.xlsx")
        self.assertEqual(calls["ledger_cfg"]["path"], "normal.xlsx")
        self.assertEqual(calls["on_accel"], "shared_accel")
        self.assertEqual(calls["off_accel"], "shared_accel")

    def test_legacy_two_config_mode_still_routes_on_and_off_separately(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            on_config = tmp_path / "on_config.json"
            off_config = tmp_path / "off_config.json"
            self.write_json(on_config, self.on_config())
            self.write_json(off_config, self.off_config())

            calls = self.run_standard_main_with_eval_mocks(
                "--config_on",
                str(on_config),
                "--config_off",
                str(off_config),
                "--on_dir",
                str(tmp_path / "on"),
                "--off_dir",
                str(tmp_path / "off"),
                "--out_dir",
                str(tmp_path / "out"),
            )

        self.assertEqual(calls["label_cfg"]["path"], "on_label.xlsx")
        self.assertEqual(calls["ledger_cfg"]["path"], "off_normal.xlsx")
        self.assertEqual(calls["on_accel"], "on_accel")
        self.assertEqual(calls["off_accel"], "off_accel")

    def test_mixed_single_and_legacy_config_mode_errors_before_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            single_config = tmp_path / "single_config.json"
            on_config = tmp_path / "on_config.json"
            off_config = tmp_path / "off_config.json"
            self.write_json(single_config, self.single_config())
            self.write_json(on_config, self.on_config())
            self.write_json(off_config, self.off_config())

            for extra_args in (
                ("--config_on", str(on_config)),
                ("--config_off", str(off_config)),
                ("--config_on", str(on_config), "--config_off", str(off_config)),
            ):
                with self.subTest(extra_args=extra_args):
                    error_text, load_label = self.run_standard_main_expect_config_error(
                        "--config",
                        str(single_config),
                        *extra_args,
                        "--on_dir",
                        str(tmp_path / "on"),
                        "--off_dir",
                        str(tmp_path / "off"),
                    )
                    self.assertIn("--config", error_text)
                    self.assertTrue(
                        "--config_on" in error_text or "--config_off" in error_text,
                        error_text,
                    )
                    load_label.assert_not_called()

    def test_lone_legacy_config_option_errors_before_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            on_config = tmp_path / "on_config.json"
            off_config = tmp_path / "off_config.json"
            self.write_json(on_config, self.on_config())
            self.write_json(off_config, self.off_config())

            cases = (
                (("--config_on", str(on_config)), "--config_off"),
                (("--config_off", str(off_config)), "--config_on"),
            )
            for args, missing_option in cases:
                with self.subTest(missing_option=missing_option):
                    error_text, load_label = self.run_standard_main_expect_config_error(
                        *args,
                        "--on_dir",
                        str(tmp_path / "on"),
                        "--off_dir",
                        str(tmp_path / "off"),
                    )
                    self.assertIn(missing_option, error_text)
                    load_label.assert_not_called()

    def test_missing_config_path_reports_target_path_before_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            missing_config = tmp_path / "missing_config.json"

            error_text, load_label = self.run_standard_main_expect_config_error(
                "--config",
                str(missing_config),
                "--on_dir",
                str(tmp_path / "on"),
                "--off_dir",
                str(tmp_path / "off"),
            )

        self.assertIn(str(missing_config), error_text)
        load_label.assert_not_called()

    def test_missing_required_single_config_keys_report_path_and_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cases = {
                "missing_label.json": (
                    {
                        "evaluation": {
                            "normal_ledger_sheet": {
                                "path": "normal.xlsx",
                                "file_column_excel_index": 3,
                            }
                        }
                    },
                    "evaluation.label_review_sheet",
                ),
                "missing_ledger.json": (
                    {
                        "evaluation": {
                            "label_review_sheet": {
                                "path": "label.xlsx",
                                "file_column_excel_index": 19,
                                "a1_override_col": 9,
                                "k_col": 11,
                                "m_col": 13,
                                "n_col": 15,
                            }
                        }
                    },
                    "evaluation.normal_ledger_sheet",
                ),
                "missing_evaluation.json": ({}, "evaluation"),
            }

            for filename, (payload, missing_key) in cases.items():
                config_path = tmp_path / filename
                self.write_json(config_path, payload)
                with self.subTest(missing_key=missing_key):
                    error_text, load_label = self.run_standard_main_expect_config_error(
                        "--config",
                        str(config_path),
                        "--on_dir",
                        str(tmp_path / "on"),
                        "--off_dir",
                        str(tmp_path / "off"),
                    )
                    self.assertIn(str(config_path), error_text)
                    self.assertIn(missing_key, error_text)
                    load_label.assert_not_called()

    def test_run_dir_is_not_used_as_on_or_off_dir_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            single_config = tmp_path / "single_config.json"
            self.write_json(single_config, self.single_config())

            error_text, load_label = self.run_standard_main_expect_config_error(
                "--config",
                str(single_config),
            )

        self.assertIn("--on_dir", error_text)
        self.assertIn("--off_dir", error_text)
        load_label.assert_not_called()

    def run_standard_main_with_eval_mocks(self, *args: str) -> dict:
        calls: dict = {}
        argv = ["eval_score_csv.py", *args]
        on_dir = Path(args[args.index("--on_dir") + 1])
        off_dir = Path(args[args.index("--off_dir") + 1])
        on_dir.mkdir(parents=True, exist_ok=True)
        off_dir.mkdir(parents=True, exist_ok=True)

        def load_label_intervals(label_cfg: dict) -> pd.DataFrame:
            calls["label_cfg"] = label_cfg
            return pd.DataFrame([{"basename": "on_file"}])

        def load_normal_basenames_from_ledger(ledger_cfg: dict) -> list[str]:
            calls["ledger_cfg"] = ledger_cfg
            return ["off_file"]

        def build_on_summary(_result_dir: Path, _label_df: pd.DataFrame, accel_col_name: str) -> pd.DataFrame:
            calls["on_accel"] = accel_col_name
            return self.summary_row(label=1, basename="on_file")

        def build_off_summary(
            _result_dir: Path,
            _normal_basenames: list[str],
            accel_col_name: str,
            verbose: bool = False,
        ) -> pd.DataFrame:
            calls["off_accel"] = accel_col_name
            calls["verbose"] = verbose
            return self.summary_row(label=0, basename="off_file")

        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(sys, "argv", argv))
            stack.enter_context(mock.patch.object(standard, "load_label_intervals", side_effect=load_label_intervals))
            stack.enter_context(
                mock.patch.object(
                    standard,
                    "load_normal_basenames_from_ledger",
                    side_effect=load_normal_basenames_from_ledger,
                )
            )
            stack.enter_context(mock.patch.object(standard, "build_per_file_summary_from_dir", side_effect=build_on_summary))
            stack.enter_context(
                mock.patch.object(
                    standard,
                    "build_per_file_summary_normal_from_dir",
                    side_effect=build_off_summary,
                )
            )
            self.patch_downstream_evaluation(stack)
            with contextlib.redirect_stdout(io.StringIO()):
                standard.main()
        return calls

    def run_standard_main_expect_config_error(self, *args: str) -> tuple[str, mock.Mock]:
        argv = ["eval_score_csv.py", *args]
        load_label = mock.Mock(name="load_label_intervals")
        stdout = io.StringIO()
        stderr = io.StringIO()
        with mock.patch.object(sys, "argv", argv):
            with mock.patch.object(standard, "load_label_intervals", load_label):
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    with self.assertRaises((SystemExit, ValueError, FileNotFoundError)) as raised:
                        standard.main()
        error_text = f"{stdout.getvalue()}\n{stderr.getvalue()}\n{raised.exception}"
        return error_text, load_label

    def patch_downstream_evaluation(self, stack: contextlib.ExitStack) -> None:
        stack.enter_context(mock.patch.object(standard, "add_segment_columns", side_effect=lambda df: df))
        stack.enter_context(mock.patch.object(standard, "ensure_anomaly_rate_column", side_effect=lambda df: df))
        stack.enter_context(
            mock.patch.object(
                standard,
                "compute_confusion_matrices",
                return_value={
                    "A2_to_A3start": {"TP": 0},
                    "A2_to_A3end": {"TP": 0},
                    "lenient": {"TP": 0},
                    "strict": {"TP": 0},
                },
            )
        )
        stack.enter_context(mock.patch.object(standard, "compute_margin_stats", return_value={}))
        stack.enter_context(mock.patch.object(standard, "aggregate_by_segments", return_value=pd.DataFrame([{"R_phase_A2": 0.0}])))
        stack.enter_context(
            mock.patch.object(
                standard,
                "compute_threshold_curve_onoff_a2",
                return_value=pd.DataFrame([{"threshold": 0.0, "fpr": 0.0, "tpr": 0.0}]),
            )
        )
        stack.enter_context(mock.patch.object(standard, "plot_roc_onoff_a2", return_value=None))
        stack.enter_context(mock.patch.object(standard, "plot_first_phase_distribution", return_value=None))
        stack.enter_context(mock.patch.object(standard, "plot_margin_boxplots", return_value=None))
        stack.enter_context(mock.patch.object(standard, "plot_macro_heatmap", return_value=None))
        stack.enter_context(mock.patch.object(standard, "tpr_at_fpr_targets_onoff", return_value={1e-4: 0.0}))
        stack.enter_context(mock.patch.object(standard, "plot_tpr_vs_fpr_targets", return_value=None))

    @staticmethod
    def write_json(path: Path, payload: dict) -> None:
        path.write_text(json.dumps(payload), encoding="utf-8")

    @staticmethod
    def single_config() -> dict:
        return {
            "evaluation": {
                "label_review_sheet": {
                    "path": "label.xlsx",
                    "sheet_name": "labels",
                    "file_column_excel_index": 19,
                    "a1_override_col": 9,
                    "k_col": 11,
                    "m_col": 13,
                    "n_col": 15,
                },
                "normal_ledger_sheet": {
                    "path": "normal.xlsx",
                    "sheet_name": 0,
                    "file_column_excel_index": 3,
                },
                "accel_column_name": "shared_accel",
                "run_dir": "must_not_be_used_as_cli_input",
            }
        }

    @staticmethod
    def on_config() -> dict:
        return {
            "evaluation": {
                "label_review_sheet": {
                    "path": "on_label.xlsx",
                    "sheet_name": "labels",
                    "file_column_excel_index": 19,
                    "a1_override_col": 9,
                    "k_col": 11,
                    "m_col": 13,
                    "n_col": 15,
                },
                "accel_column_name": "on_accel",
            }
        }

    @staticmethod
    def off_config() -> dict:
        return {
            "evaluation": {
                "normal_ledger_sheet": {
                    "path": "off_normal.xlsx",
                    "sheet_name": 0,
                    "file_column_excel_index": 3,
                },
                "accel_column_name": "off_accel",
            }
        }

    @staticmethod
    def summary_row(label: int, basename: str) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "basename": basename,
                    "label": label,
                    "detected": bool(label),
                    "first_detect_phase": "A2" if label else "OTHER",
                    "first_detection_time": 1.0 if label else float("nan"),
                    "pre_collision_detected": bool(label),
                    "collision_time": 3.0 if label else float("nan"),
                    "A1_start": 0.0 if label else float("nan"),
                    "A2_start": 1.0 if label else float("nan"),
                    "A3_start": 3.0 if label else float("nan"),
                    "A3_end": 4.0 if label else float("nan"),
                    "accel_A2_max": 10.0 if label else float("nan"),
                    "accel_A2_bin": "0-100%" if label else "unknown",
                    "shift_at_collision": "unknown",
                    "anomaly_rate": 1.0 if label else 0.0,
                }
            ]
        )


if __name__ == "__main__":
    unittest.main()
