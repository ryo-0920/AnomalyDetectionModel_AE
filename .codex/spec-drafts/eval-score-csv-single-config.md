# eval_score_csv 単一 config 対応 仕様草案

## ステータス

- 2026-06-22 に承認済み仕様へ昇格済み。
- 昇格先は `docs/specs/overview.md`, `docs/specs/requirements.md`, `docs/specs/design.md`, `docs/specs/decision_log.md`。
- test-designer は本草案をテスト可能かつ昇格 OK と判断し、人間判断待ちの未確定事項は残っていない。
- 本草案フェーズでは本体コード、テスト、依存定義、README は編集しない。

## 適用される承認済み仕様

- `docs/specs/overview.md`
  - 旧 CLI パスと既存の公開起動方法は当面維持する。
  - スクリプト整理後も、既存 CLI 互換、既存設定参照、既存出力契約は維持対象とする。
- `docs/specs/requirements.md`
  - R-011: 既存の公開 CLI パスと主要な起動方法は当面維持する。
  - R-012: 通常版と Nstep 版の切替により、既存の学習・推論・評価フローの公開契約を不必要に変更しない。
  - 非機能要件: 既存 CLI 引数、設定参照先、出力契約を可能な限り維持する。
- `docs/specs/design.md`
  - `1_transformer/eval_score_csv.py` は旧 CLI 互換 wrapper として本流実装へ委譲する方針。

## 現行挙動

- 旧 CLI 入口 `1_transformer/eval_score_csv.py` は `gofumi_ae.cli.evaluate.main()` へ委譲する。
- `src/gofumi_ae/cli/evaluate.py` は `--config_on`, `--config_off`, `--on_dir`, `--off_dir`, `--out_dir`, `--fpr_targets`, `--score_col`, `--verbose` を CLI 引数として受理し、実処理を `gofumi_ae.evaluation.standard` へ委譲する。
- `src/gofumi_ae/evaluation/standard.py` は現時点で `--config_on` と `--config_off` を required としている。
- ON 側設定から `evaluation.label_review_sheet` を読み、OFF 側設定から `evaluation.normal_ledger_sheet` を読む。
- 既存の設定例は次の 2 ファイルに分かれている。
  - `config/tagged_dataset_filter_eval_ON.json`
  - `config/tagged_dataset_filter_eval_OFF.json`

## 要求

`eval_score_csv` を、ON/OFF 評価設定を 1 つの共有 config JSON で指定できるようにする。既存の `--config_on` / `--config_off` による 2 ファイル指定は、可能な限り後方互換として維持する。

## 仕様案

### S-001: 単一 config 指定

- 評価 CLI は新しい任意引数 `--config` を受理する。
- `--config` に指定した JSON は、ON 側設定と OFF 側設定の両方として読み込まれる。
- 単一 config JSON はトップレベルに `evaluation` セクションを持つ。
- 単一 config JSON の `evaluation` セクションは、少なくとも次の両方を含む。
  - `label_review_sheet`
  - `normal_ledger_sheet`
- `--config` 実行時、ON 側のラベル区間生成は同一 config の `evaluation.label_review_sheet` を使う。
- `--config` 実行時、OFF 側の正常台帳照合は同一 config の `evaluation.normal_ledger_sheet` を使う。
- `evaluation.accel_column_name` は ON/OFF の両方に同じ値を適用する。未指定時は現行どおり `accelpedalangle` を使う。

### S-002: 既存 2 config 指定の維持

- 既存の `--config_on` と `--config_off` は削除しない。
- `--config` を使わない場合、`--config_on` と `--config_off` の両方を必須とする現行の起動契約を維持する。
- 2 config 指定時の読み込み元は現行どおり次の対応とする。
  - ON 側: `--config_on` の `evaluation.label_review_sheet`
  - OFF 側: `--config_off` の `evaluation.normal_ledger_sheet`
- 2 config 指定時の `accel_column_name` は現行どおり ON/OFF それぞれの config から読み、未指定時はそれぞれ `accelpedalangle` を使う。

### S-003: CLI 指定の妥当性

- CLI は次のどちらか一方の指定形式を受け付ける。
  - 単一 config 形式: `--config <path>`
  - 既存 2 config 形式: `--config_on <path> --config_off <path>`
- `--config` と `--config_on` / `--config_off` の混在指定は、どちらを優先するかが曖昧になるため usage error とする。
- `--config` がなく、`--config_on` または `--config_off` の片方だけが指定された場合は usage error とする。
- 指定された config ファイルが存在しない場合は、対象 path を示して失敗する。
- 必須キーがない場合は、対象 path と不足キーを示して失敗する。

### S-004: 変更しない挙動

- `--on_dir`, `--off_dir`, `--out_dir`, `--fpr_targets`, `--score_col`, `--verbose` の意味は変更しない。
- `--on_dir` と `--off_dir` は引き続き CLI 引数として指定する。`evaluation.run_dir` を、この変更だけを理由に入力ディレクトリの既定値として採用しない。
- per-file summary、混同行列、ROC、区間別集計、plot の算出ロジックと出力ファイル名は変更しない。
- `config/tagged_dataset_filter_eval_ON.json` と `config/tagged_dataset_filter_eval_OFF.json` は、この変更だけを理由に削除しない。
- 依存管理、Docker、CUDA、OS 要件は変更しない。

## 単一 config JSON 例

```json
{
  "enabled": true,
  "evaluation": {
    "label_review_sheet": {
      "path": "path/to/label_review_sheet.xlsx",
      "sheet_name": "label_review_sheet",
      "file_column_excel_index": 19,
      "a1_override_col": 9,
      "k_col": 11,
      "m_col": 13,
      "n_col": 15,
      "a1_delta_seconds": 5.0
    },
    "normal_ledger_sheet": {
      "path": "path/to/normal_ledger.xlsx",
      "sheet_name": 0,
      "file_column_excel_index": 3
    },
    "accel_column_name": "accelpedalangle",
    "target_fpr": 0.01,
    "a1_level": 1,
    "file_prob_fallback": {
      "enabled": false
    },
    "group_regex": "^([^_]+)",
    "output": {
      "evaluation_subdir": "evaluation",
      "save_plots": true
    }
  }
}
```

## 非目標

- 評価計算の意味を変えない。
- `--on_dir` / `--off_dir` を config ファイルから自動補完する機能は追加しない。
- ON/OFF で別々の `accel_column_name` を 1 ファイル内に持つ新スキーマは追加しない。
- 既存の ON/OFF 分割 config ファイルを統合・削除・リネームしない。
- `docs/specs/**` への昇格は test-designer レビュー前には行わない。

## 受け入れ条件案

- AC-001: `python 1_transformer/eval_score_csv.py --help` で `--config`、`--config_on`、`--config_off` が確認できる。
- AC-002: `--config <path>` だけを指定した場合、同一 JSON から `evaluation.label_review_sheet` と `evaluation.normal_ledger_sheet` が読み込まれる。
- AC-003: `--config_on <on_path> --config_off <off_path>` を指定した場合、現行どおり ON config から `label_review_sheet`、OFF config から `normal_ledger_sheet` が読み込まれる。
- AC-004: `--config` と `--config_on` / `--config_off` を混在指定した場合、usage error になり、評価処理を開始しない。
- AC-005: `--config` がなく `--config_on` または `--config_off` の片方だけを指定した場合、usage error になり、評価処理を開始しない。
- AC-006: 単一 config に `evaluation.label_review_sheet` または `evaluation.normal_ledger_sheet` が不足する場合、対象 path と不足キーを示して失敗する。
- AC-007: 存在しない config path を指定した場合、対象 path を示して失敗する。
- AC-008: 2 config 形式で既存の `config/tagged_dataset_filter_eval_ON.json` と `config/tagged_dataset_filter_eval_OFF.json` を指定する起動方法は引き続き受理される。
- AC-009: 単一 config 形式でも 2 config 形式でも、`--on_dir`, `--off_dir`, `--out_dir`, `--fpr_targets`, `--score_col`, `--verbose` の意味と出力契約は変わらない。

## test-designer へ渡す確認観点

- `argparse` レベルで、単一 config 形式、2 config 形式、混在指定、片側欠落を観測可能なテストへ落とせるか。
- config 読み込みを小さな関数に分離する場合、同一 JSON を ON/OFF 両側へ割り当てる contract を fixture で確認できるか。
- `evaluation.run_dir` を入力ディレクトリ既定値として使わない非目標を、差分レビューまたは CLI 引数検査で確認できるか。
- 既存 2 config 形式の後方互換性を最小 smoke で確認できるか。

## 昇格可否

- test-designer の昇格 OK 判定に基づき、2026-06-22 に `docs/specs/**` へ昇格済み。
- 人間判断待ちの未確定仕様はない。
- 承認済み仕様、テスト計画、テストケース定義、受け入れ条件対応表が揃ったため、`UNLOCK:IMPLEMENT` に基づく実装許可待ちへ移行可能。
