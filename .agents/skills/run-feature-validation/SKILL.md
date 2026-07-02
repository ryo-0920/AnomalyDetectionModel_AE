---
name: run-feature-validation
description: Use after executable acceptance tests or implementation changes when validating red/green evidence, quality, PR readiness, and integration risk.
---

# run-feature-validation

実行可能受け入れテスト作成後、実装変更後、PR 作成前、または validator エージェントが red/green、品質、PR 準備可否、統合可否の根拠を必要とする場合に使う。

## 目的
最小の関連チェックから実行し、必要に応じて検証を広げる。仕様、テストケース定義、実行可能受け入れテスト、実装の対応を確認する。テスト結果だけで完了判断せず、変更差分と周辺コードを読み、品質、保守性、リファクタリング候補、残リスクを正直に報告する。

validator の総合判断は、文章だけで完了扱いにしない。`.codex/validation/<feature>.json` に機械可読の検証レポートを作成し、`scripts/validate_report.py` で検査する。

## 入力
- 変更されたファイル。
- 有効な仕様と受け入れ条件。
- テスト計画。
- テストケース定義。
- 実行可能受け入れテスト。
- 受け入れ条件対応表。
- 実装前後の red/green 結果。
- 変更差分と関連する周辺コード。
- README と CI の検証コマンド。
- プロジェクト設定ファイル。
- Docker を使う場合は image、tag、CUDA/GPU の有無、主要コマンド。

## sub-agent 運用
- メインエージェントは、検証・整理フェーズで `validator` を sub-agent として必ず起動する。
- sub-agent の `reasoning_effort` は、`AGENTS.md` のエージェント協働に定義された方針に従う。
- 編集を伴う validator は、実行環境が対応する場合 `worker` として起動し、担当成果物、`write_scope`、他者変更を戻さないことを明示する。
- `validator` は割り当てられた `.codex/validation/**`、検証 JSON、検証 Markdown、検証メモを直接作成または更新する。メインエージェントは、利用可能な `validator` の検証成果物を代行編集しない。
- 同一検証・整理フェーズ内の差し戻しでは、既に起動した sub-agent を再利用し、再確認のたびに新しい sub-agent を spawn しない。
- `validator` は red/green 証跡、受け入れ条件対応、未検証項目、コード品質指摘、PR 準備可否を確認する。
- 実装フェーズ中の検証では、`validator` は `test-designer` と `implementer` の成果物を順に確認する。テスト不備は `test-designer`、実装不備、品質上の必須修正、実装フェーズ内に直すべきリファクタリング候補は `implementer`、仕様未確定はメインエージェントへ差し戻す。
- 仕様意図とのずれや受け入れ条件の解釈が問題になる場合は、必要に応じて `spec-designer` または `test-designer` へ差し戻す。
- red/green 証跡、検証報告、受け入れ条件対応、品質確認が揃い、`必須修正` がなくなるまで PR 準備可能と判断しない。
- `validator` を起動できない場合、またはユーザー依頼に sub-agent 利用の明示がなく実行環境の上位ルールにより起動できない場合、検証を開始しない。
- 起動できない場合は、理由と次の依頼文をユーザーへ返して停止する: `multi-agent 標準運用で進めてください。検証・整理フェーズでは validator を使ってください。`

## 手順
1. 実行可能受け入れテストが仕様、受け入れ条件、テストケース定義に対応しているか確認する。
2. 実装前の red が妥当か確認する。確認できない場合は理由を記録する。
3. red が妥当でない場合は、失敗理由を整理して `test-designer` へ差し戻す。仕様未確定が原因の場合は実装を止める。
4. 変更された挙動に最も近い集中検証コマンドを特定する。
5. 集中検証コマンドを実行する。
6. 受け入れ条件対応表の各項目が、自動テスト、手動検証、または未自動化理由へ対応しているか確認する。
7. 通過した場合は、リスクと利用可能時間に応じて広めの検証を実行する。
8. 失敗した場合は、想定原因と影響範囲を特定できる程度に失敗内容を確認し、テスト不備なら `test-designer`、実装不備なら `implementer` へ差し戻す。
9. 実装、テスト、README、設定が有効な仕様と整合しているか確認する。
10. 変更差分と関連する周辺コードを読み、重複コード、不要コード、未使用コード、到達不能コード、過度な抽象化、責務混在、命名不整合、不要な複雑化、危険な暫定実装、デバッグコード残存、仕様外の汎用化、可読性低下を確認する。
11. コードが仕様を満たす範囲で最小であり、既存パターンと整合し、読みやすく、テストしやすい構造か確認する。
12. リファクタリング候補を `必須修正`、`実装フェーズ内の推奨修正`、`別タスク候補` に分類する。分類には、対象ファイル、問題、理由、推奨対応を含める。
13. `必須修正` または `実装フェーズ内の推奨修正` がある場合は、根拠と対象ファイルを整理して `implementer` へ差し戻す。
14. コマンド、結果、失敗内容、未検証範囲、品質指摘、コードレビュー指摘を `.codex/validation/<feature>.md` へ記録する。Markdown は `examples/validation_report_template.md` の形を基準にする。PR 上で artifact が参照できない場合でも判断できるよう、検証要約、失敗要約、受け入れ条件対応、未検証項目、必須修正の有無、コードレビュー指摘を本文に残す。必要に応じて実行日時、出力先、使用環境、主要パラメータ、Git 状態、Docker image、tag、CUDA/GPU の有無も記録する。長い生ログ、JUnit XML、テストランナーの生出力は `.codex/validation/artifacts/<feature>/` に保存してよい。
15. validator の総合判断を `.codex/validation/<feature>.json` に記録する。JSON は `examples/validation_report_template.json` の形に合わせる。
16. PR 準備可能と判断する前に、次の形式で検証レポートを検査する。

```sh
python .agents/skills/run-feature-validation/scripts/validate_report.py .codex/validation/<feature>.json --require-pr-ready
```

ローカルで artifact を作成した場合は、可能な限り成果物の存在も検査する。ただし `.codex/validation/artifacts/` は原則 Git 管理外であり、PR 上では `.codex/validation/<feature>.md` と `.codex/validation/<feature>.json` を正本として確認する。

```sh
python .agents/skills/run-feature-validation/scripts/validate_report.py .codex/validation/<feature>.json --require-pr-ready --check-artifacts
```

17. `.codex/validation/` や `.codex/validation/artifacts/<feature>/` が存在しない場合は、成果物を書き込む時点で親ディレクトリを作成する。`outputs/` はプロダクトやスクリプトの実行成果物置き場であり、Codex の検証ログやテスト結果は置かない。
18. 検証で `__pycache__/`、`.pytest_cache/`、`*.pyc`、JUnit XML、生ログなどの再生成可能なファイルが出た場合は、削除を繰り返すのではなく `.gitignore` の標準方針で除外する。作業に支障がない限り、これらを削除するためだけにユーザー確認を求めない。
19. 変更が PR 準備可能か、commit 可能か、commit 対象候補は何か、下書き PR 作成前または統合前に残る懸念があるかを明示する。
20. sub-agent が直接編集した検証ファイルを確認する。メインエージェントによる直接編集は、機械的統合、衝突解消、標準導入や標準更新そのもの、またはユーザーが明示した例外に限る。
21. PR 準備可能と判断できる場合でも、push や下書き PR 作成は実行せず、メインエージェントへ判断材料を返す。

## 補助スクリプト
- `scripts/run_tests.sh`: 可能な範囲でローカルテストコマンドを選ぶ。
- `scripts/collect_failures.sh`: コマンド出力を `.codex/validation/artifacts/<feature>/` に保存する。
- `scripts/validate_report.py`: `.codex/validation/<feature>.json` の形式、PR 準備可否、red/green 証跡、未検証項目、必須修正の有無を検査する。

## 機械可読レポート
`.codex/validation/<feature>.json` は次の判定項目を含める。

- `overall_status`: `green`、`red`、`blocked`、`unknown`。
- `pr_ready`: PR 準備可能かどうか。
- `spec_status`: `ok`、`unresolved`、`missing`。
- `acceptance_status`: `ok`、`unmapped`、`unverified`、`missing`。
- `test_status`: `green`、`red`、`not_run`、`partial`。
- `red_green_status`: `ok`、`not_applicable`、`not_checked`、`invalid`。
- `quality_status`: `ok`、`required_fixes`、`warnings`。
- `required_fixes`: PR 準備前に必須の修正。
- `code_review_findings`: コード品質、保守性、リファクタリング余地の指摘。各項目は `severity` として `required`、`in_scope_recommended`、`follow_up` のいずれかを持つ。
- `unverified_items`: 未検証の受け入れ条件や確認項目。
- `human_decision_required`: 人間判断が必要な事項。
- `commit`: commit 可否、commit 対象候補、commit から除外すべき差分。
- `commands`: red、green、validation、manual の検証証跡。
- `artifacts`: 人間向け報告、JUnit XML、ログなどの出力先。検証証跡は原則 `.codex/validation/artifacts/<feature>/` に置く。

`overall_status=green` または `pr_ready=true` と記録する場合、`validate_report.py` は次を必須条件として検査する。

- 仕様、受け入れ条件、テスト、品質の状態がすべて通過状態である。
- `required_fixes`、`unverified_items`、`human_decision_required` が空である。
- `code_review_findings` に `required` または `in_scope_recommended` の項目が残っていない。
- `red_green_status=ok` の場合、red の `expected_failed` と green の `passed` の両方のコマンド証跡がある。
- 失敗またはスキップされた検証コマンドが残っていない。

## 出力
- 実行したコマンド。
- sub-agent が直接編集した検証ファイル。
- 必要に応じた実行日時、出力先、使用環境、主要パラメータ、Git 状態。
- 成功または失敗。
- red/green の確認結果。
- 失敗要約。
- 未検証範囲。
- 受け入れ条件対応表との対応確認結果。
- 変更差分と周辺コードを読んだコードレビュー結果。
- 重複コード、不要コード、未使用コード、過度な抽象化、責務混在、命名不整合、可読性低下に関する指摘。
- 品質、保守性、リファクタリング候補。
- `必須修正`、`実装フェーズ内の推奨修正`、`別タスク候補` の分類。
- test-designer または implementer への差し戻し有無と理由。
- `.codex/validation/<feature>.json` の機械可読な総合判断。
- `validate_report.py` の検査結果。
- PR 準備可否の判断。
- commit 可否、commit 対象候補、commit 前に除外すべき差分。
- 下書き PR 作成前または統合前に残る懸念。
- 使用した sub-agent。

## ガードレール
- 補助スクリプトの推測より、文書化されたプロジェクトコマンドを優先する。
- 必須 sub-agent が未使用の場合、このワークフローを完了扱いにしない。
- 失敗またはスキップされたチェックを隠さない。
- validator は原則として本体コードを編集せず、必要な修正を implementer へ差し戻す。
- validator はリファクタリングを自分で実施しない。対象ファイル、理由、推奨対応を整理して implementer へ返す。
- 仕様達成に不要な大規模再設計や、今回の変更範囲と関係が薄い改善は、`別タスク候補` として分離する。
- 再生成可能なキャッシュや検証生ログを作業ごとに削除して状態を整えようとしない。必要な ignore が不足していれば `.gitignore` を更新し、tracked 化されている場合のみ別途判断する。
- validator は `git add`、`git commit`、feature ブランチの push、下書き PR 作成、ready for review 変更、PR の merge を行わない。
- 必須検証が不足している場合、未解決の仕様項目が残る場合、受け入れ条件対応表と検証結果が対応していない場合、`必須修正` または未対応の `実装フェーズ内の推奨修正` が残る場合、または暫定実装に人間判断が必要な場合は PR 準備可能または統合可能と扱わない。
- `.codex/validation/artifacts/` の生ログや JUnit XML だけに判断根拠を閉じ込めず、レビューに必要な要約は `.codex/validation/<feature>.md` に残す。
- `.codex/validation/<feature>.json` が存在しない場合、または `validate_report.py --require-pr-ready` が失敗する場合は PR 準備可能と扱わない。
