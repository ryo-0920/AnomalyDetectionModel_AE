# スクリプト整理 テスト計画

## 対象

仕様草案 `.codex/spec-drafts/script-organization.md` に基づき、スクリプト配置整理後も既存の公開 CLI、設定参照、出力契約が壊れていないことを確認する。

## 前提

- `UNLOCK:IMPLEMENT` 済みのため、最小の実行可能受け入れテストとして `tests/test_script_organization_smoke.py` を追加する。
- 現在 `pytest` は利用できないため、標準ライブラリ `unittest` で実行可能な smoke test を採用する。
- 本体コードは編集せず、red を先に固定する。
- `app/collect_intentional_accel.py`、`app/gofumi_accel_keyboard.py`、動画/Excel 補助スクリプトは実験用であり、本流の必須受け入れ保証対象外だが、`experiments/` 配下へ整理される期待はテストで固定する。

## 受け入れ条件対応表

| ID | 受け入れ条件 | 検証方法 | 自動化 |
| --- | --- | --- | --- |
| AC-01 | `python 1_transformer/train_transformer_autoencoder.py --help` が成功する。 | `tests.test_script_organization_smoke` の CLI help smoke | 実装フェーズで自動化 |
| AC-02 | `python 1_transformer/train_score_csv.py --help` が成功する。 | `tests.test_script_organization_smoke` の CLI help smoke | 実装フェーズで自動化 |
| AC-03 | `python 1_transformer/plot_timechart.py --help` が成功する。 | `tests.test_script_organization_smoke` の CLI help smoke | 実装フェーズで自動化 |
| AC-04 | `python 1_transformer/eval_score_csv.py --help` が成功する。 | `tests.test_script_organization_smoke` の CLI help smoke | 実装フェーズで自動化 |
| AC-05 | 旧 CLI パスから起動しても既定 config パスが変わらない。 | path constant check | 自動化候補 |
| AC-06 | `train_score_csv.py` の既存出力契約を維持する。 | 出力先/ファイル名契約の smoke または fixture 検証 | 実装後に範囲確定 |
| AC-07 | 新パッケージの主要モジュールが import できる。 | `tests.test_script_organization_smoke` の import smoke | 実装フェーズで自動化 |
| AC-08 | wrapper 方針と cache/output の Git 管理方針が記録されている。 | 仕様/README/decision log 確認 | 手動確認 |
| AC-09 | CARLA/動画/Excel 補助スクリプトが実験用であり、本流の必須受け入れテスト対象外であることが記録され、`experiments/` 配下へ整理される前提が固定されている。 | 仕様/decision log 確認 + `tests.test_script_organization_smoke` の配置期待チェック | 一部自動 |

## 最小検証コマンド案

```sh
python -m unittest tests.test_script_organization_smoke
```

## Red/Green 方針

1. 先に `tests/test_script_organization_smoke.py` を追加し、`unittest` で実行できる red を固定する。
2. 実装前は、少なくとも `gofumi_ae` import 不可、旧 CLI help 失敗、`experiments/` 未配置のいずれかで red になることを想定する。
3. 実装では旧 CLI を薄い wrapper にし、新パッケージ側の `main()` へ委譲する。
4. green 確認は `python -m unittest tests.test_script_organization_smoke` を最小コマンドとし、必要に応じて path constant check や非TTY確認を広げる。
5. 推論の実データ・学習済み artifacts・ネットワーク共有・CARLA は、最小 smoke から分離する。

## 対象外と未解決事項

- CARLA、動画変換、Excel 出力補助スクリプトは実験用であり、本流の必須受け入れテスト対象外とする。
- tracked `__pycache__`、`desktop.ini` を Git 管理から外した後の差分整理手順。
- 旧 CLI help が依存不足のままでは red の原因に依存環境差分も混ざるため、implementer/validator はパッケージ再編と必要依存の扱いを切り分けて確認する。
- 現時点で、人間判断待ちの未解決事項はない。
