---
name: refine-spec
description: Use when clarifying requirements, drafting or updating specifications, resolving open specification questions, or preparing a feature for SDD/TDD test design.
---

# refine-spec

要求の仕様整理が必要な場合、新しい機能ブランチを準備する場合、または曖昧な要求によって実装が止まっている場合に使う。

## 目的
ユーザー依頼、現行挙動、暫定メモを、プロダクト判断を黙って補完せずにテスト設計可能な仕様へ整理する。
test-designer と共同で、仕様が観測可能な受け入れ条件、テスト計画、テストケース定義へ落とせることを確認する。

## 入力
- ユーザー依頼と現在のブランチ。
- 適用される `docs/specs/**` ファイル。
- 適用される `.codex/spec-drafts/**`、`.codex/open-issues/**`、`.codex/work-notes/**`、`.codex/implementation_notes.md` などの作業メモ。
- 現行挙動を示す既存コード、テスト、README、CI。

## sub-agent 運用
- メインエージェントは、仕様策定フェーズで `spec-designer` と `test-designer` を sub-agent として必ず起動する。
- sub-agent の `reasoning_effort` は、`AGENTS.md` のエージェント協働に定義された方針に従う。
- 編集を伴う sub-agent は、実行環境が対応する場合 `worker` として起動し、担当成果物、`write_scope`、他者変更を戻さないことを明示する。
- `spec-designer` と `test-designer` は割り当てられた仕様草案、open issue、work note、test plan、test case を直接作成または更新する。メインエージェントは、利用可能な担当 sub-agent の成果物を代行編集しない。
- 同一仕様策定フェーズ内の差し戻しでは、既に起動した sub-agent を再利用し、再確認のたびに新しい sub-agent を spawn しない。
- `spec-designer` を先に起動し、要求、制約、非目標、未確定仕様、仕様草案、受け入れ条件案を整理させる。
- メインエージェントは `spec-designer` の出力を待ち、最新の仕様草案、未確定仕様、非目標、受け入れ条件案を `test-designer` に渡す。
- `test-designer` は渡された最新仕様を前提に、テスト可能性、期待値、境界条件、受け入れ条件、テストケース化できない箇所を確認する。
- `test-designer` が仕様不備を指摘した場合、メインエージェントは `spec-designer` に差し戻し、仕様更新後に `test-designer` の再確認を行う。
- `test-designer` のレビューが NG の場合、メインエージェントは仕様を代行修正せず、指摘内容、対象ファイル、期待する再出力を `spec-designer` へ返す。
- `spec-designer` が `done`、`completed`、または終了済みの場合でも、実行環境が再開機能を提供するなら同じ agent id を再開して差し戻す。再開できない場合に限り、同じ責務の後任 sub-agent を起動し、前回成果物とレビュー指摘を引き継ぐ。
- `test-designer` が仕様昇格 OK と判断し、未解決の未確定仕様と人間判断待ちが残っていない場合、メインエージェントは `spec-designer` に承認済み仕様への昇格を指示する。メインエージェントは昇格作業を代行編集しない。
- `spec-designer` は承認済み仕様を `docs/specs/` へ直接反映し、確定済み判断を decision log へ移し、解消済み項目を open issue から削除する。昇格後に仕様策定フェーズを完了扱いにする。
- 人間判断が必要な事項が残る場合は、承認済み仕様へ昇格せず、open issue を現在判断が必要な項目だけに整頓してユーザー判断を待つ。
- この差し戻しループが収束するまで、仕様策定フェーズを完了扱いにしない。
- どちらかを起動できない場合、またはユーザー依頼に sub-agent 利用の明示がなく実行環境の上位ルールにより起動できない場合、仕様策定を開始しない。
- 起動できない場合は、理由と次の依頼文をユーザーへ返して停止する: `multi-agent 標準運用で進めてください。仕様策定フェーズでは spec-designer と test-designer を使ってください。`

## 手順
1. リポジトリが `新規プロジェクト` か `既存プロジェクト` かを判定する。
2. 最も近い適用対象の `docs/specs/` ファイルと `.codex/` メモを探す。
3. `spec-designer` に、確定済み要求、現行挙動、依頼された変更、実装判断に影響する `未確定仕様`、受け入れ条件案を整理させる。
4. `spec-designer` の出力を待ち、最新の仕様草案、未確定仕様、非目標、受け入れ条件案を `test-designer` に渡す。
5. `test-designer` に、テスト不能、期待値不明、境界条件不足、観測方法不明、受け入れ条件のずれを確認させる。
6. `test-designer` が仕様課題を指摘した場合は、論点を `spec-designer` に差し戻し、仕様更新後に再び `test-designer` へ渡す。メインエージェントはこの仕様修正を代行しない。
7. 差し戻しごとに、何を更新したか、どの指摘が解消したか、残る未確定仕様は何かを記録する。
8. `test-designer` が仕様昇格 OK と判断した場合、未解決の未確定仕様と人間判断待ちが残っていないことを確認する。
9. 昇格条件を満たす場合は、`spec-designer` に承認済み仕様への昇格を指示し、関連する `docs/specs/` ファイルを直接更新させる。
10. 昇格条件を満たさない場合は、未解決項目ごとに代表的な選択肢と各選択肢の影響を示し、承認前の草案を `.codex/spec-drafts/<feature>.md` に置く。
11. 人間が判断した事項は、決定内容、理由、却下した主な選択肢、日付または文脈を `docs/specs/decision_log.md` または `docs/specs/decisions/` に記録する。
12. `test-designer` OK かつ未解決事項なしで標準フロー昇格した判断も、必要に応じて decision log に記録する。
13. 現在判断が必要な未解決項目だけを `.codex/open-issues/<feature>.md` に残す。確定済みの判断、却下した選択肢、判断理由は open issue から削除し、decision log へ移す。
14. 暫定的な理由付けや実験メモは `.codex/work-notes/<feature>.md` に置く。
15. `.codex/spec-drafts/`、`.codex/open-issues/`、`.codex/work-notes/` が存在しない場合は、成果物を書き込む時点で親ディレクトリを作成する。
16. sub-agent が直接編集したファイルを確認する。メインエージェントによる直接編集は、機械的統合、衝突解消、標準導入や標準更新そのもの、またはユーザーが明示した例外に限る。
17. 承認済み仕様への昇格可否、昇格した `docs/specs/` ファイル、昇格しない場合の理由を明示する。
18. 現在のブランチが実装許可待ちへ移行可能かどうかを明示する。

## 出力
- 更新または草案化した仕様文。
- sub-agent が直接編集したファイル。
- 昇格した承認済み仕様、または昇格しない理由。
- 未解決論点がある場合はその明示。
- 更新した `.codex/spec-drafts/**`、`.codex/open-issues/**`、`.codex/work-notes/**`。
- decision log に移した確定済み判断。
- open issue に残した現在判断が必要な項目。
- 受け入れ条件。
- test-designer に渡した最新仕様の要約。
- test-designer からの指摘と、それに対する spec-designer の更新結果。
- test-designer と相互確認したテスト可能性の論点。
- 実装許可待ちへの移行可否:
  - 承認済み仕様へ昇格済みで実装許可待ちへ移行可能、または
  - 列挙した `未確定仕様` によりブロック。
- 使用した sub-agent。

## ガードレール
- このワークフローでは本体コードを編集しない。
- 必須 sub-agent が未使用の場合、このワークフローを完了扱いにしない。
- このワークフローではテスト計画の詳細化やテストファイル編集を行わない。
- 実験結果や仕様草案は、昇格条件を満たして `docs/specs/` に記録されるまで承認済み仕様として扱わない。
- 公開挙動、データ形式、セキュリティ、不可逆操作、外部契約に影響する既定値を勝手に選んで不確実性を隠さない。
