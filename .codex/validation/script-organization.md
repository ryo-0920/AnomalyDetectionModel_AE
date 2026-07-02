# 検証報告: script-organization

## Summary
| 項目 | 値 |
| --- | --- |
| Overall | Focused Green / Release Unknown |
| PR Ready | No |
| Feature | `script-organization` |
| Branch | `feature/add-parchange` |
| Commit | `d556e8ab08bee9f9f41308dec533b2fb29f88a97` |
| Date | `2026-06-19` |
| 検証 JSON | `.codex/validation/script-organization.json` |

## Red/Green
| Phase | Command | Result | Evidence |
| --- | --- | --- | --- |
| Red | `python -m unittest tests.test_script_organization_smoke` | Expected failed | 既存 red 証跡を継承 |
| Green | `python -m unittest tests.test_script_organization_smoke` | Passed | import / legacy `--help` / `experiments/` 再配置 smoke |
| Manual | `python -c "import gofumi_ae, ..."` | Passed | `gofumi_ae` と主要 module import を確認 |
| Manual | `python 1_transformer/train_transformer_autoencoder.py --help` | Passed | 旧 CLI help |
| Manual | `python 1_transformer/train_score_csv.py --help` | Passed | 旧 CLI help |
| Manual | `python 1_transformer/eval_score_csv.py --help` | Passed | 旧 CLI help |
| Manual | `python 1_transformer/plot_timechart.py --help` | Passed | 旧 CLI help |
| Manual | `./.venv/bin/python -c "import gofumi_ae.training.standard as ts, ..."` | Passed | repo-root path 解決と legacy runtime import lookup を確認 |
| Manual | `git ls-files 'desktop.ini' '1_transformer/desktop.ini' '1_transformer/models/__pycache__/*'` | Passed | tracked match なし |
| Validation | `python .agents/skills/run-feature-validation/scripts/validate_report.py .codex/validation/script-organization.json` | Passed | 非 PR-ready 形式の検査 |

## Acceptance Coverage
| 受け入れ条件 | テストケース | 実行テストまたは確認 | 結果 | Evidence |
| --- | --- | --- | --- | --- |
| AC-01 | TC-02 | `python -m unittest tests.test_script_organization_smoke` | Passed | `train_transformer_autoencoder.py --help` |
| AC-02 | TC-02 | `python -m unittest tests.test_script_organization_smoke` | Passed | `train_score_csv.py --help` |
| AC-03 | TC-02 | `python -m unittest tests.test_script_organization_smoke` | Passed | `plot_timechart.py --help` |
| AC-04 | TC-02 | `python -m unittest tests.test_script_organization_smoke` | Passed | `eval_score_csv.py --help` |
| AC-05 | TC-03 | `./.venv/bin/python -c "import gofumi_ae.training.standard as ts, ..."` | Passed | `PROJECT_ROOT` と既定 config path が repo root 基準 |
| AC-06 | TC-03, TC-05 | end-to-end 推論出力契約の再確認 | Unverified | runtime flow は未実行 |
| AC-07 | TC-01 | `python -m unittest tests.test_script_organization_smoke` | Passed | `gofumi_ae` import / CLI module smoke |
| AC-08 | TC-07 | 仕様書と Git index 確認 | Passed | wrapper / experimental / tracked cache 方針あり |
| AC-09 | TC-06, TC-07 | `python -m unittest tests.test_script_organization_smoke` + 仕様確認 | Passed | `experiments/` 再配置と experimental 記録を確認 |

## Required Fixes
- なし。focused validation で確認した範囲に blocking issue は残っていない。

## Unverified Items
- `train_score_csv.py` の従来 `result/*_anomaly.csv` と TF 互換フォルダ出力の両立は未実行。
- 対話実行時の full runtime behavior は import/setup までしか確認していない。
- 非TTY 時の明示エラー動作は今回の green 範囲では未再確認。

## Code Review / Quality Notes
| 分類 | 観点 | 対象 | 内容 | 対応 |
| --- | --- | --- | --- | --- |
| follow-up | runtime coverage | `src/gofumi_ae/training/standard.py`, `src/gofumi_ae/inference/standard.py` | repo-root path 解決と legacy import lookup は確認できたが、実データでの end-to-end runtime は未確認 | 最小 fixture か既存 artifacts を使って AC-06 を追加検証する |
| follow-up | coverage gap | `tests/test_script_organization_smoke.py` | 現在の green は import / help / 配置 / path constant に寄っている | runtime fixture と non-TTY check を段階的に追加する |

## Human Decision Required
- なし。

## Commit Readiness
| 項目 | 値 |
| --- | --- |
| Commit Ready | No |
| Candidate Files | 実装差分一式と `.codex/validation/script-organization.*` |
| Exclude Files | `config/hyperparams_common.json`, `requirements.txt` |
| Notes | focused smoke は green。release-ready 判定には AC-06 と runtime coverage の追加確認が必要 |

## Notes
- `gofumi_ae` package shim、旧 CLI `--help`、`experiments/` 再配置、tracked cache/desktop.ini の Git 管理解除は現ワークスペースで確認済み。
- validator sub-agent は初回報告で runtime path 解決を赤判定したが、その後の follow-up implementer 修正と focused manual verification で AC-05 は解消した。
