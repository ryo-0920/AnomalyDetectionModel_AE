---
name: generate-acceptance-tests
description: Use when a specification has acceptance criteria and needs a test plan, test case definitions, acceptance traceability, or executable acceptance tests after implementation is unlocked.
---

# generate-acceptance-tests

仕様に受け入れ条件があり、テスト計画、テストケース定義、受け入れ条件対応表、または実行可能受け入れテストが必要な場合に使う。

## 目的
要求を、`.codex/test-plans/`、`.codex/test-cases/`、`.codex/traceability/`、必要なら `tests/**` の成果物へ落とす。`UNLOCK:IMPLEMENT` 前は設計まで、後は実行可能受け入れテストまで扱う。

## 入力
- 有効な仕様と受け入れ条件
- 現在フェーズが仕様策定か実装フェーズか
- 既存テスト構成、テストフレームワーク、既存検証コマンド

## sub-agent 運用
- `test-designer` を必須とする。仕様解釈が揺れる場合は `spec-designer`、`UNLOCK:IMPLEMENT` 後の red 確認では `validator` を追加する。
- sub-agent の起動、`reasoning_effort`、再利用、差し戻し、代行編集禁止、起動不可時の停止条件は `AGENTS.md` のエージェント協働に従う。
- `validator` が red 不成立、テスト不備、仕様不整合を指摘した場合は `test-designer` へ差し戻し、再確認を受ける。
- `validator` が妥当な red を確認するまで `implementer` へ本体実装を渡さない。

## 手順
1. 有効な仕様と受け入れ条件を読み、入力、期待値、前提、判定方法を整理する。
2. `.codex/test-cases/<feature>.md` にテストケース定義を作成または更新する。
3. 有用な場合のみ、`.codex/test-cases/<feature>.csv`、`.yaml`、`.json` などのデータ駆動テスト用ケースを追加する。
4. 各受け入れ条件を、自動テスト、手動またはスモークチェック、意図的な未自動化のいずれかに対応付ける。
5. `.codex/test-plans/<feature>.md` にテスト計画と受け入れ条件対応表を作成または更新する。
6. 既存テストの置き場、命名規約、最小検証コマンドを特定する。
7. `UNLOCK:IMPLEMENT` 前は、明示許可がない限り `tests/**`、fixture、検証スクリプトを編集しない。
8. `UNLOCK:IMPLEMENT` 後または明示許可後は、最小かつ有用な実行可能受け入れテストを追加または更新する。
9. 受け入れテストは仕様から導き、実装都合で条件を弱めない。
10. 実装前 red の想定、失敗理由、検証コマンドを `validator` が確認できる形で残す。
11. どの条件が検証済みか、未検証か、実装へ進めるかを根拠付きで示す。

## 出力
- テスト計画
- テストケース定義
- 必要なデータ駆動テスト用ケース
- 受け入れ条件対応表
- 検証コマンド
- `UNLOCK:IMPLEMENT` 後は新規または更新した実行可能受け入れテスト
- red 確認結果または差し戻し対応結果
- 実装移行可否

## ガードレール
- 本体挙動を変更しない。
- 必須 sub-agent が未使用なら完了扱いにしない。
- `UNLOCK:IMPLEMENT` 前は、明示許可がない限りテストファイル編集、テスト実行、検証スクリプト実行を行わない。
- 壊れやすい実装依存テスト、大きな fixture、不要な生成データを増やさない。
