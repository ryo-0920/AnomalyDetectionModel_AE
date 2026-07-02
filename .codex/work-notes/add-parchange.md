# threshold 分布統計拡張 作業メモ

## 依頼

threshold/stat handling に、保存 artifact の追加分布統計として `p95`, `p99.5`, `p99.9`, `p99.99`, `p99.999` を追加する。

## 現行確認

- 承認済み仕様では、最終採用モデルに対して threshold を再算出し、artifacts に保存する方針がある。
- `docs/specs/**` は threshold の再算出と保存先を仕様化しているが、`threshold.json` 内の全 percentile キーまでは列挙していない。
- 本流通常版 `src/gofumi_ae/training/standard.py` は `threshold.json` に `p10`, `p50`, `p90`, `p99` を保存している。
- 本流 Nstep 版 `src/gofumi_ae/training/nstep.py` も同じ統計を保存し、追加で `tail_steps` を保存している。
- 本流推論 `src/gofumi_ae/inference/standard.py` と `src/gofumi_ae/inference/nstep.py` は `p10` から `p99` を読み込み、`p10` から `p99` の範囲で `y_conv_score` を正規化している。
- 旧 CLI 互換として `1_transformer/` 配下にも同等の通常版・Nstep 版がある。

## 仕様判断

- 追加する値は保存 artifact の分布統計であり、判定しきい値や正規化式を変える変更ではない。
- 既存 artifact の読み込みを壊さないため、新キーの欠落は推論エラーにしない。
- 分布が保存されていない既存 artifact に対し、高分位点を `mean/std` などから推定して「保存済み統計」として扱うことは避ける。
- 小数分位点のキーは `p99_5` のように `_` で表し、既存の `p10`/`p99` 形式に寄せる。

## 編集した成果物

- `.codex/spec-drafts/add-parchange.md`
- `.codex/open-issues/add-parchange.md`
- `.codex/work-notes/add-parchange.md`

## 未実施

- 本体コード編集
- テストファイル編集
- テスト実行
- test-designer 成果物の実行

## 昇格結果

- test-designer がテスト可能かつ昇格 OK と判断し、人間判断待ちの未確定事項が残っていないため、2026-06-22 に承認済み仕様へ昇格した。
- 昇格先:
  - `docs/specs/overview.md`
  - `docs/specs/requirements.md`
  - `docs/specs/design.md`
  - `docs/specs/decision_log.md`
- `.codex/open-issues/add-parchange.md` は解消済みのため削除対象とした。
