# プロジェクト標準導入チェックリスト

標準テンプレート導入後のリポジトリで、作業開始前に標準構成と運用前提を確認するときに使う。新規プロジェクトのテンプレート適用、初期コミット、`develop` 作成そのものは、標準リポジトリの `skills/bootstrap-team-repo` に従う。

## リポジトリ前提
- [ ] 標準導入済みリポジトリとして扱える状態か確認する。
- [ ] 既存運用の追加制約がある場合は記録する。
- [ ] 安定版ブランチを特定する。
- [ ] 統合ブランチを特定する。
- [ ] 既存ブランチ規約が標準運用を上書きするか確認する。
- [ ] remote の有無を確認する。
- [ ] 新しい feature ブランチを切る場合は `git fetch --prune origin` により更新有無を確認する。
- [ ] 安定版ブランチ、統合ブランチ、現在の feature ブランチが remote から遅れていないか確認する。

## 仕様基準
- [ ] `docs/specs/overview.md` を作成または確認する。
- [ ] `docs/specs/requirements.md` を作成または確認する。
- [ ] `docs/specs/design.md` を作成または確認する。
- [ ] `docs/specs/decision_log.md` または `docs/specs/decisions/` を作成または確認する。
- [ ] 承認済み仕様と暫定 `.codex/` メモを分離する。

## Codex 基準
- [ ] `AGENTS.md` を導入または更新する。
- [ ] `.codex/config.toml` を導入または更新する。
- [ ] `.codex/agents/` を導入または更新する。
- [ ] `.agents/skills/` を導入または更新する。
- [ ] 標準成果物の置き場を確認する。空ディレクトリ保持用の `.gitkeep` は標準では作成しない。
- [ ] `.codex/spec-drafts/`、`.codex/open-issues/`、`.codex/work-notes/`、`.codex/test-plans/`、`.codex/test-cases/`、`.codex/traceability/`、`.codex/validation/` は、成果物を書き込む時点で作成する。
- [ ] 暫定メモが必要な場合は `.codex/implementation_notes.md` を作成する。

## 開発基準
- [ ] 依存管理を確認する。
- [ ] セットアップコマンドを確認する。
- [ ] 実行コマンドを確認する。
- [ ] 集中検証コマンドを確認する。
- [ ] 広めの回帰検証コマンドを確認する。
- [ ] CI の挙動を確認する。

## ドキュメント
- [ ] `README.md` に現行セットアップ、使い方、テスト、エントリポイント、設定、仕様リンクを記載する。
- [ ] README が将来構想ではなく現行実装を説明していることを確認する。
- [ ] README に秘密情報、長い議論履歴、却下案を入れない。

## Git 整備
- [ ] `.gitignore` を作成または更新する。
- [ ] `.github/pull_request_template.md` を作成または既存テンプレートとの対応を確認する。
- [ ] 生成出力とキャッシュが除外されていることを確認する。
- [ ] 共有すべき `.codex/` テンプレートファイルが誤って除外されていないことを確認する。
- [ ] 既存運用がない場合、`main` が安定版ブランチ、`develop` が統合ブランチとして扱えることを確認する。
