# eval_score_csv 単一 config 対応 テストケース定義

## 共通 fixture 方針

テストでは実データ評価を走らせず、最小の一時 JSON config と config 解決 contract を検査する。

単一 config の代表入力:

```json
{
  "evaluation": {
    "label_review_sheet": {
      "path": "label.xlsx",
      "sheet_name": "labels",
      "file_column_excel_index": 19,
      "a1_override_col": 9,
      "k_col": 11,
      "m_col": 13,
      "n_col": 15
    },
    "normal_ledger_sheet": {
      "path": "normal.xlsx",
      "sheet_name": 0,
      "file_column_excel_index": 3
    },
    "accel_column_name": "shared_accel",
    "run_dir": "must_not_be_used_as_cli_input"
  }
}
```

2 config 形式の代表入力:

- ON config: `evaluation.label_review_sheet` と `evaluation.accel_column_name = "on_accel"` を持つ。
- OFF config: `evaluation.normal_ledger_sheet` と `evaluation.accel_column_name = "off_accel"` を持つ。

期待される resolved contract:

- 単一 config 形式:
  - `label_cfg` は単一 config の `evaluation.label_review_sheet`
  - `ledger_cfg` は単一 config の `evaluation.normal_ledger_sheet`
  - `accel_col_on == "shared_accel"`
  - `accel_col_off == "shared_accel"`
- 2 config 形式:
  - `label_cfg` は ON config の `evaluation.label_review_sheet`
  - `ledger_cfg` は OFF config の `evaluation.normal_ledger_sheet`
  - `accel_col_on == "on_accel"`
  - `accel_col_off == "off_accel"`

## TC-01 CLI help に --config が表示される

- 目的: 旧 CLI 入口の help で新旧 config 引数が観測できることを確認する。
- 対応: AC-001
- 入力:
  - `python 1_transformer/eval_score_csv.py --help`
- 期待値:
  - 終了コードが 0。
  - stdout または stderr に `--config`、`--config_on`、`--config_off` が含まれる。
- 判定方法:
  - `subprocess.run(..., capture_output=True, check=False)` の戻り値と出力文字列を検査する。

## TC-02 単一 config 形式の config 解決

- 目的: `--config <path>` だけで ON/OFF 両方の評価設定が同一 JSON から読み込まれることを確認する。
- 対応: AC-002
- 入力:
  - 共通 fixture の単一 config JSON
  - CLI args 相当: `--config <single_config> --on_dir <on_dir> --off_dir <off_dir>`
- 前提:
  - `<on_dir>` と `<off_dir>` は一時ディレクトリとして存在させる。
  - 実 Excel 読み込みや評価計算は実行せず、config 解決結果を検査する。
- 期待値:
  - `label_cfg["path"] == "label.xlsx"`
  - `ledger_cfg["path"] == "normal.xlsx"`
  - `accel_col_on == "shared_accel"`
  - `accel_col_off == "shared_accel"`
  - `config_on_path` と `config_off_path` の役割が同一 path として扱われる。
- 判定方法:
  - 実装フェーズで追加される config 解決 helper、または `standard.main()` 周辺を mock した受け入れテストで検査する。

## TC-03 既存 2 config 形式の後方互換

- 目的: `--config_on` と `--config_off` の既存 contract を維持することを確認する。
- 対応: AC-003, AC-007
- 入力:
  - 共通 fixture の ON config JSON
  - 共通 fixture の OFF config JSON
  - CLI args 相当: `--config_on <on_config> --config_off <off_config> --on_dir <on_dir> --off_dir <off_dir>`
  - 既存 config path smoke 候補:
    - `config/tagged_dataset_filter_eval_ON.json`
    - `config/tagged_dataset_filter_eval_OFF.json`
- 期待値:
  - `label_cfg` は ON config 由来。
  - `ledger_cfg` は OFF config 由来。
  - `accel_col_on == "on_accel"`
  - `accel_col_off == "off_accel"`
  - 既存 config path を指定した CLI 引数構成が argparse/config validation レベルで受理される。
- 判定方法:
  - 一時 JSON で config 解決結果を検査する。
  - 既存 config path はファイル存在と required key の smoke を検査する。実評価処理は実データ依存のため最小テストでは実行しない。

## TC-04 --config と分割 config の混在指定を拒否する

- 目的: 優先順位が曖昧な指定形式を usage error として拒否し、評価処理を開始しないことを確認する。
- 対応: AC-004
- 入力:
  - `--config <single_config> --config_on <on_config> --on_dir <on_dir> --off_dir <off_dir>`
  - `--config <single_config> --config_off <off_config> --on_dir <on_dir> --off_dir <off_dir>`
  - `--config <single_config> --config_on <on_config> --config_off <off_config> --on_dir <on_dir> --off_dir <off_dir>`
- 期待値:
  - いずれも usage error。
  - 終了コードは非 0、または config 解決 helper が `SystemExit` / `ValueError` を送出する。
  - error message に `--config` と `--config_on` または `--config_off` の混在が分かる文言が含まれる。
  - `load_label_intervals`、`load_normal_basenames_from_ledger`、per-file summary builder は呼ばれない。
- 判定方法:
  - `standard.main()` を mock 付きで呼ぶ、または config 解決 helper の例外を検査する。

## TC-05 分割 config の片側欠落を拒否する

- 目的: `--config` がない場合に `--config_on` / `--config_off` の片方だけでは評価を開始しないことを確認する。
- 対応: AC-005
- 入力:
  - `--config_on <on_config> --on_dir <on_dir> --off_dir <off_dir>`
  - `--config_off <off_config> --on_dir <on_dir> --off_dir <off_dir>`
- 期待値:
  - いずれも usage error。
  - error message に不足している `--config_on` または `--config_off` が分かる文言が含まれる。
  - 評価処理関数は呼ばれない。
- 判定方法:
  - `standard.main()` を mock 付きで呼ぶ、または config 解決 helper の例外を検査する。

## TC-06 config path と必須 key のエラー

- 目的: 存在しない config path と必須 key 欠落が、対象 path と不足 key を示して失敗することを確認する。
- 対応: AC-006, S-003-path
- 入力:
  - 存在しない path を `--config <missing_path>` に指定。
  - 単一 config で `evaluation.label_review_sheet` を欠落させた JSON。
  - 単一 config で `evaluation.normal_ledger_sheet` を欠落させた JSON。
  - `evaluation` セクション自体を欠落させた JSON。
- 期待値:
  - 存在しない path では error message に missing path が含まれる。
  - key 欠落では error message に config path と不足 key が含まれる。
  - 評価処理関数は呼ばれない。
- 判定方法:
  - config 解決 helper の例外、または CLI main の stderr/stdout を検査する。

## TC-07 既存 config ファイル contract smoke

- 目的: 既存の ON/OFF 分割 config ファイルが削除・リネームされず、2 config 形式の required key を満たすことを確認する。
- 対応: AC-007
- 入力:
  - `config/tagged_dataset_filter_eval_ON.json`
  - `config/tagged_dataset_filter_eval_OFF.json`
- 期待値:
  - 両ファイルが存在する。
  - ON config に `evaluation.label_review_sheet` が存在する。
  - OFF config に `evaluation.normal_ledger_sheet` が存在する。
- 判定方法:
  - `json.loads(Path(...).read_text(encoding="utf-8"))` で読み、key を検査する。

## TC-08 既存 CLI 引数の意味を維持する smoke

- 目的: config 指定方式の追加により、既存 CLI 引数が削除・意味変更されていないことを確認する。
- 対応: AC-008
- 入力:
  - `python 1_transformer/eval_score_csv.py --help`
  - help 上で確認する引数:
    - `--on_dir`
    - `--off_dir`
    - `--out_dir`
    - `--fpr_targets`
    - `--score_col`
    - `--verbose`
- 期待値:
  - 各引数が help に残っている。
  - `--on_dir` と `--off_dir` は config の `evaluation.run_dir` ではなく CLI 引数として扱われる。
- 判定方法:
  - help 出力検査。
  - `--config <single_config>` のみで `--on_dir` / `--off_dir` を省略した場合に usage error になることを検査する。

## TC-09 出力契約の非変更レビュー

- 目的: per-file summary、混同行列、ROC、区間別集計、plot の算出ロジックと出力ファイル名を変更していないことを確認する。
- 対応: AC-008
- 入力:
  - 実装差分
  - 既存 smoke:
    - `python -m unittest tests.test_script_organization_smoke`
- 期待値:
  - config 解決と CLI validation 以外の評価計算ロジックが変更されていない。
  - `per_file_summary_on.csv`, `per_file_summary_off.csv`, `per_file_summary_all.csv`, `confusion_matrices.txt` などの既存出力名が変更されていない。
  - 依存管理、Docker、CUDA、OS 要件に変更がない。
- 判定方法:
  - validator の差分レビューと既存 smoke を組み合わせる。

## 実装前 red 想定

`tests/test_eval_score_csv_single_config_acceptance.py` を追加した直後、本体実装前は以下のいずれかで red になる想定である。

- `python 1_transformer/eval_score_csv.py --help` に `--config` が含まれない。
- `gofumi_ae.evaluation.standard` が `--config` を受け付けず、単一 config 形式を解決できない。
- `--config_on` / `--config_off` が required のままで、単一 config 形式の argparse が通らない。
- 混在指定または片側欠落が仕様どおりの usage error にならない。

最小 red 確認コマンド:

```sh
python -m unittest tests.test_eval_score_csv_single_config_acceptance
```

## 対象外

- 実 Excel 台帳と実 anomaly CSV を用いた full evaluation は最小受け入れテストでは実施しない。
- GPU/CUDA 固有検証は、この CLI/config contract 変更の必須受け入れ条件に含めない。
