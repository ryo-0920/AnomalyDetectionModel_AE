# スクリプト整理 テストケース定義

## TC-01 新本流パッケージ import smoke

- 目的: `gofumi_ae` パッケージと主要モジュールが repo root から import できることを確認する。
- 自動テスト: `tests/test_script_organization_smoke.py`
- 主な対象:
  - `gofumi_ae`
  - `gofumi_ae.cli.train`
  - `gofumi_ae.cli.score`
  - `gofumi_ae.cli.evaluate`
  - `gofumi_ae.cli.plot_timechart`
  - `gofumi_ae.models`
  - `gofumi_ae.training`
  - `gofumi_ae.inference`
  - `gofumi_ae.evaluation`
  - `gofumi_ae.visualization`
  - `gofumi_ae.datasets`
  - `gofumi_ae.ui`
- 対応: AC-07

## TC-02 旧 CLI help smoke

- 目的: 旧公開 CLI パスの互換性を確認する。
- 自動テスト: `tests/test_script_organization_smoke.py`
- 対象コマンド:
  - `python 1_transformer/train_transformer_autoencoder.py --help`
  - `python 1_transformer/train_score_csv.py --help`
  - `python 1_transformer/plot_timechart.py --help`
  - `python 1_transformer/eval_score_csv.py --help`
- 対応: AC-01, AC-02, AC-03, AC-04

## TC-03 root/config/output path check

- 目的: 旧 CLI wrapper から起動しても、既定 config、artifacts、output、result の基準が project root からずれないことを確認する。
- 確認対象:
  - `config/hyperparams_common.json`
  - `config/tagged_dataset_filter_train.json`
  - `config/tagged_dataset_filter_infer.json`
  - `config/inference_ground_truth.csv`
  - `artifacts/transformer_ae`
  - `output/Valid_results`
  - `result/OFF_pa99`
- 対応: AC-05, AC-06

## TC-04 非TTY behavior

- 目的: 対話前提の CLI が、非TTY実行時に明示エラーを返すことを確認する。
- 対象候補:
  - `1_transformer/train_score_csv.py`
  - `1_transformer/plot_timechart.py`
  - `1_transformer/train_transformer_autoencoder.py`
- 対応: AC-01, AC-02, AC-03

## TC-05 推論出力契約 smoke

- 目的: `train_score_csv.py` が従来の legacy CSV と TF 互換フォルダ出力契約を維持することを確認する。
- 方針: 実装後、最小 fixture または既存 artifacts の扱いを決めてから自動化する。
- 対応: AC-06

## TC-06 実験用スクリプト再配置期待

- 目的: CARLA/動画/Excel 補助スクリプトが `experiments/` 配下へ整理される期待を固定する。
- 自動テスト: `tests/test_script_organization_smoke.py`
- 対象ファイル:
  - `collect_intentional_accel.py`
  - `gofumi_accel_keyboard.py`
  - `pngtovideo.py`
  - `export_results_excel.py`
- 対応: AC-09

## TC-07 記録確認

- 目的: wrapper 方針、実験用スコープ判断、cache/output Git 管理方針が標準成果物に記録されていることを確認する。
- 確認対象:
  - `docs/specs/overview.md`
  - `docs/specs/requirements.md`
  - `docs/specs/design.md`
  - `docs/specs/decision_log.md`
- 対応: AC-08, AC-09

## 対象外

- CARLA 系スクリプトは外部環境依存が強く、かつ実験用と確定したため、本流の必須受け入れテスト対象に含めない。
- `app/pngtovideo.py` と `app/export_results_excel.py` は実験用と確定したため、本流の必須受け入れテスト対象に含めない。
