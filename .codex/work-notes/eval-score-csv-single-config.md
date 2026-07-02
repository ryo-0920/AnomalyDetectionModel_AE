# eval_score_csv 単一 config 対応 作業メモ

## 依頼

`eval_score_csv` を、次の 2 ファイル指定を必須にせず、ON/OFF 評価設定を 1 つの共有 config JSON で指定できるようにする。

- `config/tagged_dataset_filter_eval_OFF.json`
- `config/tagged_dataset_filter_eval_ON.json`

後方互換として、既存の `--config_on` / `--config_off` 形式は可能な限り維持する。

## 確認した現行挙動

- `1_transformer/eval_score_csv.py` は wrapper であり、`gofumi_ae.cli.evaluate.main()` に委譲する。
- `src/gofumi_ae/cli/evaluate.py` は `--config_on` と `--config_off` を CLI help に出してから、`gofumi_ae.evaluation.standard.main()` へ委譲する。
- `src/gofumi_ae/evaluation/standard.py` は `--config_on` と `--config_off` を required として定義している。
- ON 側は `evaluation.label_review_sheet` を使う。
- OFF 側は `evaluation.normal_ledger_sheet` を使う。
- `--on_dir` と `--off_dir` は現行 CLI で required であり、既存 config の `evaluation.run_dir` は評価入力ディレクトリとしては使われていない。

## 適用した仕様判断

- 最小互換変更として `--config` を追加する。
- `--config` 指定時は、同じ JSON を ON/OFF 両方の設定として扱う。
- 単一 config には `evaluation.label_review_sheet` と `evaluation.normal_ledger_sheet` の両方を置く。
- 既存の 2 config 形式は維持し、`--config` を使わない場合の contract は変えない。
- `--config` と 2 config 形式の混在は、優先順位を増やさず usage error とする。
- 入力ディレクトリや評価計算、出力契約は今回の仕様対象に含めない。

## 編集した成果物

- `.codex/spec-drafts/eval-score-csv-single-config.md`
- `.codex/open-issues/eval-score-csv-single-config.md`
- `.codex/work-notes/eval-score-csv-single-config.md`
- `docs/specs/overview.md`
- `docs/specs/requirements.md`
- `docs/specs/design.md`
- `docs/specs/decision_log.md`

## 既存成果物との関係

- `.codex/spec-drafts/add-parchange.md` は threshold 分布統計拡張の昇格済み草案であり、今回の単一 config 対応とは別スコープとして扱った。
- `.codex/spec-drafts/inference-fpr-tpr-selection.md` は推論から評価までの統合フローに関する未昇格草案であり、今回の `eval_score_csv` CLI config 指定方式とは直接競合しない。

## 未実施

- 本体コード編集
- テストファイル編集
- テスト実行

## test-designer へ渡す成果物

- 最新仕様草案: `.codex/spec-drafts/eval-score-csv-single-config.md`
- 未確定仕様: なし。`.codex/open-issues/eval-score-csv-single-config.md` は昇格時に解消済みとして削除した。
- 作業メモ: `.codex/work-notes/eval-score-csv-single-config.md`

## 実装許可待ちへの移行可否

- test-designer がテスト可能かつ昇格 OK と判断し、人間判断待ちの未確定仕様が残っていないため、2026-06-22 に承認済み仕様へ昇格した。
- 昇格先:
  - `docs/specs/overview.md`
  - `docs/specs/requirements.md`
  - `docs/specs/design.md`
  - `docs/specs/decision_log.md`
- test-designer の非ブロック指摘を反映し、存在しない config path の失敗条件を受け入れ条件として明示した。
- 承認済み仕様、テスト計画、テストケース定義、受け入れ条件対応表が揃ったため、`UNLOCK:IMPLEMENT` に基づく実装許可待ちへ移行可能。
