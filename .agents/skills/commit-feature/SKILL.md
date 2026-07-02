---
name: commit-feature
description: Use after validator has marked a feature PR-ready, when the main agent needs to create a local Conventional Commits commit for the validated feature changes without modifying implementation, validation, or specification artifacts.
---

# commit-feature

validator が実装、テスト、仕様対応、品質を確認し、PR 準備可能と判断した後、メインエージェントが検証済み差分を local commit する場合に使う。

## 目的
成果物の作成や修正は行わず、validator が OK とした差分を Git 境界で確定する。push、下書き PR 作成、ready for review、merge はこの skill では行わない。

## 入力
- validator が作成した `.codex/validation/<feature>.json`。
- validator が作成した `.codex/validation/<feature>.md`。
- validator が示した PR 準備可否、commit 可否、commit 対象候補、残リスク。
- `git status --short` と `git diff --stat`。
- 必要に応じて `git diff -- <path>`。

## 前提条件
- validator の総合判断が PR 準備可能である。
- `.codex/validation/<feature>.json` が存在する。
- `validate_report.py --require-pr-ready` が通る。
- `required_fixes`、`unverified_items`、`human_decision_required` が残っていない。
- commit 対象に、validator が未確認の実装差分、意図しない生成物、秘密情報、認証情報、巨大 artifact が混ざっていない。

## 手順
1. 現在のブランチが対象の `feature/*`、または人間が明示した作業ブランチであることを確認する。
2. `.codex/validation/<feature>.json` を確認し、`pr_ready=true`、`overall_status=green` 相当であることを確認する。
3. 次の検査を実行する。

```sh
python .agents/skills/run-feature-validation/scripts/validate_report.py .codex/validation/<feature>.json --require-pr-ready
```

4. `git status --short` と `git diff --stat` で差分全体を確認する。
5. validator が示した commit 対象候補と実際の差分を照合する。
6. 想定外の差分がある場合は commit せず、対象外ファイル、理由、必要な差し戻し先を整理して停止する。
7. commit 前に、秘密情報、認証情報、巨大 artifact、再生成可能なキャッシュ、`.codex/validation/artifacts/`、`outputs/` が staging 対象に含まれないことを確認する。
8. 必要なファイルだけを `git add <path>...` で stage する。`git add .` は使わない。
9. `git diff --cached --stat` と、必要に応じて `git diff --cached -- <path>` で staged 差分を確認する。
10. Conventional Commits 形式で commit する。
11. commit 後に `git status --short` を確認し、未コミット差分が残る場合は内容と扱いを報告する。
12. commit hash、commit message、stage したファイル、未コミット差分の有無をメインエージェントの最終報告に含める。

## commit message
- 形式は Conventional Commits とする。
- 既存運用がない場合は、機能追加なら `feat: ...`、修正なら `fix: ...`、文書や標準更新なら `docs: ...`、テストのみなら `test: ...`、内部整理なら `chore: ...` を使う。
- 1 commit に複数責務が混ざる場合は、意味のある単位に分割する。ただし validator の確認範囲を越える分割や再編集は行わない。

## 出力
- 実行した検査コマンドと結果。
- staged に含めたファイル。
- staged から除外したファイルと理由。
- commit hash。
- commit message。
- commit 後の `git status --short`。
- push または下書き PR 作成前に残る確認事項。

## ガードレール
- この skill では、仕様、テスト、実装、検証報告の内容を修正しない。
- validator が PR 準備可能と判断していない場合は commit しない。
- `validate_report.py --require-pr-ready` が失敗した場合は commit しない。
- validator が未確認の差分を commit しない。
- `git add .`、`git commit -a`、`git reset --hard`、`git clean`、履歴書き換えは使わない。
- push、下書き PR 作成、ready for review、レビュー依頼、merge は行わない。
- commit 後に push や下書き PR 作成へ進む場合は、メインエージェントが別途ユーザー承認を取る。
