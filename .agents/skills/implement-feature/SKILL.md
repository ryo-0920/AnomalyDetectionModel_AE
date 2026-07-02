---
name: implement-feature
description: Use after UNLOCK:IMPLEMENT when implementing approved specifications against executable acceptance tests while preserving existing project conventions.
---

# implement-feature

`UNLOCK:IMPLEMENT` 後に、承認済み仕様、テストケース定義、受け入れ条件対応表、実行可能受け入れテストに基づいて本体コードを変更する場合に使う。

## 目的
仕様スコープに閉じた実装を行い、検証済みの実行可能受け入れテストを green にし、validator へ引き渡せる状態にする。

## 入力
- `UNLOCK:IMPLEMENT` を含む現在のユーザー依頼
- 有効な `docs/specs/**`
- テスト計画、テストケース定義、受け入れ条件対応表
- `test-designer` が作成し `validator` が確認した実行可能受け入れテスト
- `.codex/implementation_notes.md` を含む作業メモ
- 既存コード、依存定義、実行入口、テスト設定、CI、README

## sub-agent 運用
- `test-designer`、`validator`、`implementer` を必須とする。
- 起動順、`reasoning_effort`、再利用、差し戻し、代行編集禁止、起動不可時の停止条件は `AGENTS.md` のエージェント協働に従う。
- 実装前に `validator` が red を確認し、その後に `implementer` が本体コードを変更する。
- 実装後は `validator` が green、仕様対応、品質、検証報告、PR 準備可否を確認する。

## 手順
1. `UNLOCK:IMPLEMENT` が現在の対象ブランチに適用可能で、`未確定仕様` がなく、テスト計画、テストケース定義、受け入れ条件対応表が揃っていることを確認する。
2. `test-designer` に、最新仕様、テスト計画、テストケース定義、受け入れ条件対応表から実行可能受け入れテストと検証コマンドを作成または更新させる。
3. `validator` に、受け入れ条件、想定 red、検証コマンドを渡し、実装前 red の妥当性を確認させる。
4. red 不成立、仕様との矛盾、期待値不明があれば `test-designer` へ差し戻す。仕様の未確定事項が見つかった場合は実装を止める。
5. red が妥当と確認された後、既存コード、依存管理、実行方法、命名規約、公開インターフェイス、互換性要件、副作用範囲を確認する。
6. `implementer` に、検証済みテスト、変更範囲、既存規約、事前指摘を渡す。
7. 受け入れテストは原則変更せず、本体コードを変更して green にする。
8. 実装成立に必要な範囲で、本体コード、README、設定、依存定義、補助的な低レベルテストを更新する。
9. 実装後は `validator` に、変更差分、red/green 証跡、検証コマンド、README や設定変更、未検証範囲を渡す。
10. `validator` の `必須修正` または `実装フェーズ内の推奨修正` は担当 sub-agent へ差し戻し、収束するまで完了扱いにしない。
11. `git add`、`git commit`、push、下書き PR 作成、ready for review、merge は行わず、必要な判断材料をメインエージェントへ返す。

## 環境と依存関係
- `既存プロジェクト` では既存の依存管理、実行方法、環境構成を優先する。
- 依存変更、ロック更新、ランタイム変更は、許可されたフェーズでプロジェクト内に閉じる場合のみ行う。
- システム環境、プロジェクト外環境、共有環境、本番相当環境は変更しない。
- Docker は必要な場合のみ検討し、追加時は必要性と前提を記録する。

## 出力
- 実装した挙動
- 変更したファイル
- 実行可能受け入れテストを弱めていないこと
- red 確認結果、green 根拠、差し戻し対応結果
- README や設定の更新有無
- 互換性要件と副作用範囲への影響
- 暫定判断、未解決事項、未検証範囲
- 実装フェーズ完了可否

## ガードレール
- 仕様外の実装、受け入れテストの弱体化、不要な大規模リファクタリングを混ぜない。
- 必須 sub-agent が未使用なら完了扱いにしない。
- 必須検証が不足している場合、未解決の仕様項目が残る場合、高リスクな人間判断が必要な場合は PR 準備可能と扱わない。
