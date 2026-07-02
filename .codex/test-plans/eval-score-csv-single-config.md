# eval_score_csv 単一 config 対応 テスト計画

## 対象

spec-designer 最新出力 `.codex/spec-drafts/eval-score-csv-single-config.md` に基づき、旧 CLI 入口 `1_transformer/eval_score_csv.py` と本流 `gofumi_ae.evaluation.standard` が、新しい `--config` 単一 config 形式を受け付けつつ、既存の `--config_on` / `--config_off` 形式を維持することを確認する。

## 前提

- `UNLOCK:IMPLEMENT` は有効だが、今回の immediate task はテスト設計成果物の作成であり、本体ソースと `tests/**` は編集しない。
- spec-designer の最新出力は `.codex/spec-drafts/eval-score-csv-single-config.md`、未確定事項は `.codex/open-issues/eval-score-csv-single-config.md`、作業メモは `.codex/work-notes/eval-score-csv-single-config.md` とする。
- `.codex/open-issues/eval-score-csv-single-config.md` には人間判断待ちの未解決仕様項目はない。
- 既存テストは `unittest` ベースの `tests/test_script_organization_smoke.py` のみ確認できる。`pyproject.toml` と pytest 設定は確認できないため、実行可能受け入れテストは追加依存なしの `python -m unittest` を第一候補とする。
- Docker は不要。通常はホストのプロジェクトローカル Python 環境で検証する。
- `1_transformer/eval_score_csv.py` は旧 CLI 互換 wrapper として `gofumi_ae.cli.evaluate.main()` に委譲するため、help smoke は旧入口で確認し、config 解決 contract は本流評価 module で確認する。

## テスト可能性判定

仕様はテスト可能であり、テスト設計側からは仕様昇格 OK と判断する。

- `--config`、`--config_on`、`--config_off` は CLI help と usage error の戻り値で観測可能である。
- 単一 config の ON/OFF 両側割り当ては、同一 JSON から `evaluation.label_review_sheet` と `evaluation.normal_ledger_sheet` が読み出される config 解決 contract として観測可能である。
- 2 config 形式の後方互換性は、ON 側 path から `label_review_sheet`、OFF 側 path から `normal_ledger_sheet` が読み出されることで観測可能である。
- 混在指定と片側欠落は、評価処理を開始しない usage error として観測可能である。
- 必須 key 欠落と存在しない path は、例外または CLI エラーのメッセージに対象 path と不足 key が含まれることで観測可能である。
- `--on_dir` / `--off_dir` を config の `evaluation.run_dir` から補完しない非目標は、CLI 引数なしでは usage error になることと、差分レビューで確認可能である。

## 受け入れ条件対応表

| ID | 受け入れ条件 | 検証方法 | 自動化 |
| --- | --- | --- | --- |
| AC-001 | `python 1_transformer/eval_score_csv.py --help` で `--config`、`--config_on`、`--config_off` が確認できる。 | TC-01 | 実装フェーズで自動化 |
| AC-002 | `--config <path>` だけを指定した場合、同一 JSON から `evaluation.label_review_sheet` と `evaluation.normal_ledger_sheet` が読み込まれる。 | TC-02 | 実装フェーズで自動化 |
| AC-003 | `--config_on <on_path> --config_off <off_path>` を指定した場合、現行どおり ON config から `label_review_sheet`、OFF config から `normal_ledger_sheet` が読み込まれる。 | TC-03 | 実装フェーズで自動化 |
| AC-004 | `--config` と `--config_on` / `--config_off` を混在指定した場合、usage error になり、評価処理を開始しない。 | TC-04 | 実装フェーズで自動化 |
| AC-005 | `--config` がなく `--config_on` または `--config_off` の片方だけを指定した場合、usage error になり、評価処理を開始しない。 | TC-05 | 実装フェーズで自動化 |
| AC-006 | 単一 config に `evaluation.label_review_sheet` または `evaluation.normal_ledger_sheet` が不足する場合、対象 path と不足 key を示して失敗する。 | TC-06 | 実装フェーズで自動化 |
| AC-007 | 2 config 形式で既存の `config/tagged_dataset_filter_eval_ON.json` と `config/tagged_dataset_filter_eval_OFF.json` を指定する起動方法は引き続き受理される。 | TC-03, TC-07 | 実装フェーズで自動化 |
| AC-008 | 単一 config 形式でも 2 config 形式でも、`--on_dir`, `--off_dir`, `--out_dir`, `--fpr_targets`, `--score_col`, `--verbose` の意味と出力契約は変わらない。 | TC-08, TC-09 | 一部自動、一部差分レビュー |
| S-003-path | 指定された config ファイルが存在しない場合は、対象 path を示して失敗する。 | TC-06 | 実装フェーズで自動化 |

## 境界条件

- `--config` 単独指定は有効で、`--config_on` / `--config_off` は不要である。
- `--config_on` と `--config_off` の両方指定は有効で、`--config` は不要である。
- `--config` と `--config_on` の混在、`--config` と `--config_off` の混在、3 引数すべての混在はいずれも usage error である。
- `--config_on` のみ、または `--config_off` のみは usage error である。
- 単一 config では `evaluation.label_review_sheet` と `evaluation.normal_ledger_sheet` の両方が必須である。
- 2 config 形式では ON config の `label_review_sheet` と OFF config の `normal_ledger_sheet` が必須であり、逆側の key に依存しない。
- `evaluation.accel_column_name` は単一 config では ON/OFF 同一値、未指定時は `accelpedalangle` である。
- 2 config 形式では ON/OFF それぞれの `accel_column_name` が独立し、未指定時はそれぞれ `accelpedalangle` である。
- `evaluation.run_dir` が config に存在しても、`--on_dir` / `--off_dir` の省略を許可しない。

## 実装フェーズの Red/Green 方針

1. 次の受け入れテストを追加する: `tests/test_eval_score_csv_single_config_acceptance.py`
2. 本体実装前に `python -m unittest tests.test_eval_score_csv_single_config_acceptance` を実行し、少なくとも `--config` help 欠落、単一 config 形式未対応、または混在/片側欠落の usage error 未実装で red になることを validator が確認する。
3. 実装では、評価計算や出力生成を変えず、CLI 引数と config 解決 contract を green にする。
4. 実装後は同じコマンドで green を確認し、既存 smoke と組み合わせて旧 CLI 互換を確認する。

## 検証コマンド案

最小 red/green 確認:

```sh
python -m unittest tests.test_eval_score_csv_single_config_acceptance
```

既存 smoke を含む範囲確認:

```sh
python -m unittest tests.test_eval_score_csv_single_config_acceptance tests.test_script_organization_smoke
```

pytest/JUnit XML が validator に必要になった場合の候補:

```sh
uv run pytest tests/test_eval_score_csv_single_config_acceptance.py --junit-xml=.codex/validation/artifacts/eval-score-csv-single-config/pytest.xml
```

ただし現時点で pytest 依存と設定は確認できないため、pytest/JUnit XML は追加依存または既存環境確認後に採用する。

## 対象外と未自動化理由

- 実データと Excel 台帳を使うフル評価実行は、時間、データ配置、Excel fixture の保守コストが大きいため、最小受け入れテストでは直接実行しない。config 解決 contract と CLI usage error を自動化し、評価計算と出力ファイル名の不変性は既存 smoke と差分レビューで確認する。
- plot、混同行列、ROC、区間別集計の数値回帰は、この仕様が入力 config 指定方式だけを変更するため、最小受け入れテストの直接対象にしない。
- Docker/CUDA 検証は不要。config JSON の解決 contract は GPU 有無に依存しない。

## 未解決事項

テスト設計側から仕様へ差し戻すべき blocker はない。

補足として、仕様本文 S-003 には「存在しない config path」の期待値があるが、受け入れ条件案には独立 AC として列挙されていない。本テスト計画では `S-003-path` として対応付けるため promotion blocker ではないが、承認済み仕様へ昇格する際に AC として明記すると追跡性がさらに明確になる。
