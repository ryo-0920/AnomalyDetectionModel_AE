# threshold 分布統計拡張 テスト計画

## 対象

spec-designer 最新出力 `.codex/spec-drafts/add-parchange.md` に基づき、`threshold.json` の分布統計として `p95`, `p99_5`, `p99_9`, `p99_99`, `p99_999` が追加され、既存の threshold 意味、temperature、score normalization、異常判定、CLI 契約が変わらないことを確認する。

対象範囲は本流 `src/gofumi_ae` の通常版・Nstep 版と、旧 CLI 互換の通常版・Nstep 版である。現在の旧 CLI 通常版は wrapper で本流 `gofumi_ae.cli.train` / `gofumi_ae.cli.score` へ委譲するため、旧 CLI 通常版の artifact writer / loader 互換は wrapper 経由の到達性と本流通常版の contract テストを組み合わせて確認する。

## 前提

- `UNLOCK:IMPLEMENT` は未提供のため、本体コード、`tests/**`、fixture、検証スクリプトは編集しない。
- spec-designer の最新出力は `.codex/spec-drafts/add-parchange.md`、未確定事項は `.codex/open-issues/add-parchange.md`、作業メモは `.codex/work-notes/add-parchange.md` とする。
- `.codex/open-issues/add-parchange.md` には人間判断待ちの未解決仕様項目はない。
- 既存テストは `unittest` ベースであり、`pytest` 設定と `pyproject.toml` は確認できない。後続の実行可能受け入れテストは、追加依存なしの `python -m unittest` を第一候補とする。
- Docker は不要。通常はホストのプロジェクトローカル Python 環境で検証する。

## テスト可能性判定

仕様はテスト可能であり、テスト設計側からは仕様昇格 OK と判断する。

- 追加キー名は `threshold.json` の JSON key として観測可能である。
- 期待値は同一 MAE 分布に対する `np.percentile` の 95, 99.5, 99.9, 99.99, 99.999 percentile として算出可能である。
- `threshold == p99_5` は `percentile=99.5` の代表ケースで観測可能である。
- 後方互換性は、追加キーがない `threshold.json` 相当の `thr_info` で inference context 構築が成功し、追加キーが合成されないことで確認可能である。
- 追加キーが既存 score normalization や異常判定へ影響しないことは、追加キーあり/なしの同一 `thr_info` から構築した context の `y_conv_threshold`, `y_pre_threshold`, `p10`, `p50`, `p90`, `p99`, `temperature`, `threshold` を比較して確認可能である。

## 受け入れ条件対応表

| ID | 受け入れ条件 | 検証方法 | 自動化 |
| --- | --- | --- | --- |
| AC-001 | 本流通常版の学習 artifact 作成後、`threshold.json` に既存キーを維持したまま追加キーが保存される。 | TC-01, TC-03 | 実装フェーズで自動化 |
| AC-002 | 本流 Nstep 版の学習 artifact 作成後、既存キーと `tail_steps` を維持したまま追加キーが保存される。 | TC-01, TC-03 | 実装フェーズで自動化 |
| AC-003 | 旧 CLI 互換の通常版および Nstep 版の学習 artifact 作成後も、本流と同じ追加キーが保存される。 | TC-02, TC-03 | 実装フェーズで自動化 |
| AC-004 | 追加キーの値が、同一 MAE 分布の `np.percentile` 値と一致する。 | TC-01 | 実装フェーズで自動化 |
| AC-005 | `p99`, `p99_5`, `p99_9`, `p99_99`, `p99_999` が数値誤差を除き単調非減少である。 | TC-01, TC-04 | 実装フェーズで自動化 |
| AC-006 | `percentile=99.5` では `threshold` と `p99_5` が一致する。 | TC-01 | 実装フェーズで自動化 |
| AC-007 | 追加キーがない既存 `threshold.json` を各推論 loader がエラーにせず読み込める。 | TC-05 | 実装フェーズで自動化 |
| AC-008 | 追加キーが存在する場合、推論 loader が `float` として context に保持する。 | TC-06 | 実装フェーズで自動化 |
| AC-009 | 追加キーの有無で `y_conv_score`, `y_conv_threshold`, `y_pre_threshold`, `is_anomaly` が変わらない。 | TC-07 | 実装フェーズで自動化 |
| AC-010 | CLI help、対話起動、保存先、既存 CSV 出力列、依存管理、実行環境要件が変わらない。 | TC-08 | 一部自動、一部手動確認 |

## 境界条件

- 非整数 percentile key: `p99.5`, `p99.9`, `p99.99`, `p99.999` は JSON key では `p99_5`, `p99_9`, `p99_99`, `p99_999` であること。
- 高分位点の補間: 代表 MAE 分布 `[0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0]` では `np.percentile` 既定方式の値と一致すること。
- 定数分布: すべての percentile 値が同じ値になり、単調非減少を満たすこと。
- 同値を含む分布: 単調非減少を満たし、キー欠落や型崩れがないこと。
- `percentile` が 99.5 以外の場合でも追加キーは固定の分布統計として保存されること。ただし `threshold == p99_5` の一致確認は `percentile=99.5` に限定する。
- 既存 artifact: 追加キーがない場合は読み込み成功し、追加高分位点を `mean/std` や `threshold` から合成しないこと。

## 実装フェーズの Red/Green 方針

1. `UNLOCK:IMPLEMENT` 後に `tests/test_add_parchange_acceptance.py` を追加する。
2. まず本体実装前に `python -m unittest tests.test_add_parchange_acceptance` を実行し、追加 percentile key の欠落または inference context 未保持で red になることを validator が確認する。
3. 実装後は同じコマンドで green を確認する。
4. 必要に応じて既存 smoke と組み合わせて、旧 CLI help と wrapper 経由の到達性を確認する。

## 検証コマンド案

最小 red/green 確認:

```sh
python -m unittest tests.test_add_parchange_acceptance
```

既存 smoke を含む範囲確認:

```sh
python -m unittest tests.test_add_parchange_acceptance tests.test_script_organization_smoke
```

pytest/JUnit XML が validator に必要になった場合の候補:

```sh
uv run pytest tests/test_add_parchange_acceptance.py --junit-xml=.codex/validation/artifacts/add-parchange/pytest.xml
```

ただし現時点で pytest 依存と設定は確認できないため、pytest/JUnit XML は追加依存または既存環境確認後に採用する。

## 対象外と未自動化理由

- 実データによるフル学習は、時間・GPU・データ配置に依存するため、最小受け入れテストでは直接実行しない。writer/loader の contract は MAE 分布を固定した関数単位テストと artifact JSON round-trip で検証する。
- CLI の対話起動そのもの、CSV 出力列全体、依存管理ファイル差分は AC-010 の回帰確認対象だが、この変更が CLI や出力列を変更しない非目標であるため、最小では help smoke、既存 smoke、差分レビューで確認する。
- Docker/CUDA 検証は不要。CUDA の有無によって追加 percentile key の JSON contract は変わらない。

## 未解決事項

テスト設計側から仕様へ差し戻すべき blocker はない。
