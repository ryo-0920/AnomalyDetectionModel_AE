# 検証報告: <機能名>

## Summary
| 項目 | 値 |
| --- | --- |
| Overall | Green \| Red \| Blocked \| Unknown |
| PR Ready | Yes \| No |
| Feature | `<feature>` |
| Branch | `feature/<name>` |
| Commit | `<commit>` |
| Date | `<YYYY-MM-DD>` |
| 検証 JSON | `.codex/validation/<feature>.json` |

## Red/Green
| Phase | Command | Result | Evidence |
| --- | --- | --- | --- |
| Red | `<command>` | Expected failed | `.codex/validation/artifacts/<feature>/<red-log>` |
| Green | `<command>` | Passed | `.codex/validation/artifacts/<feature>/<report>` |

## Acceptance Coverage
| 受け入れ条件 | テストケース | 実行テストまたは確認 | 結果 | Evidence |
| --- | --- | --- | --- | --- |
| AC1 | TC1 | `<test-or-check>` | Passed | `.codex/validation/artifacts/<feature>/<path>` |

## Required Fixes
- なし。

## Unverified Items
- なし。

## Code Review / Quality Notes
| 分類 | 観点 | 対象 | 内容 | 対応 |
| --- | --- | --- | --- | --- |
| 必須修正 | 重複コード / 不要コード / 可読性 / 責務分離 / 命名 / 複雑性 | `<file-or-symbol>` | なし | - |
| 実装フェーズ内の推奨修正 | 重複コード / 不要コード / 可読性 / 責務分離 / 命名 / 複雑性 | `<file-or-symbol>` | なし | - |
| 別タスク候補 | 保守性 / リファクタリング余地 | `<file-or-symbol>` | なし | - |

## Human Decision Required
- なし。

## Commit Readiness
| 項目 | 値 |
| --- | --- |
| Commit Ready | Yes \| No |
| Candidate Files | `<path1>`, `<path2>` |
| Exclude Files | `.codex/validation/artifacts/<feature>/...`, `outputs/...` |
| Notes | `<commit 前に確認すべき事項>` |

## Notes
- <補足、環境差分、既知の制約>
