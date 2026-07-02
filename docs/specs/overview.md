# Overview

## 1. 目的
本プロジェクトは、時系列CSVデータを用いて Transformer Autoencoder を学習し、ペダル踏み間違い検知のための異常判定モデルを生成する。

## 2. 現行スコープ
- 学習スクリプト: `1_transformer/train_transformer_autoencoder.py`
- 推論スクリプト: `1_transformer/train_score_csv.py`
- 可視化補助: `1_transformer/plot_timechart.py`
- モデル実装: `1_transformer/models/transformer_autoencoder.py`
- 共通設定: `config/hyperparams_common.json`
- 対話UI: `app/ui/`

## 2.1 スクリプト整理方針
- 本流の Python パッケージ名は `gofumi_ae` とする。
- 既存利用者向けに、旧 CLI パスと既存の公開起動方法は当面維持する。
- Nstep 版は派生系として分離維持せず、本流実装へ統合したうえで、実行時に通常版と切り替えられる前提とする。
- CARLA/動画/Excel補助スクリプトは実験用とし、本流の正式サポート対象には含めない。
- tracked `__pycache__/` と `desktop.ini` は成果物ではなく補助生成物として扱い、Git 管理から外す方針とする。

## 3. 学習実行の基本方針
- 学習開始時は常に対話起動とする。
- 対話で次を選択する。
  - optimizer: `AdamW` または `RAdam`
  - 学習データセット: 候補一覧から選択、または自由入力
  - 必要に応じて `tagged dataset (ledger + ver_tag001/2/3)` を選択し、台帳ベースで学習CSVを構成する
- `--no-dataset-prompt` は後方互換のため残すが、現行仕様では無効（常時対話）とする。

## 4. 推論・可視化実行の基本方針
- `train_score_csv.py` は常に対話起動とする。
- 対話で次を選択する。
  - artifactsディレクトリ: 候補一覧から選択、または自由入力
  - スコア対象（CSV/ディレクトリ/グロブ）: 候補一覧から選択、または自由入力
  - 必要に応じて `tagged dataset (ledger + ver_tag001/2/3)` を選択し、台帳ベースで推論CSVを構成する
- `train_score_csv.py` は既存の `result/*_anomaly.csv` を維持しつつ、TF互換の推論結果フォルダを追加出力する。
  - 出力先: `output/Valid_results/<timestamp>_transformer_ae_inference_csv`
  - 主な出力: `basic_info.csv`, `file_scores.csv`, `decision_times.csv`, `*_Inference_confusion_matrix.csv/.png`, `*_Inference_ROC.csv/.png`
- メタ情報CSV（`config/inference_ground_truth.csv`）は任意とする。
  - テンプレート例は `config/inference_ground_truth.example.csv` とする。
  - 存在する場合は `label`, `abnormal_start_time`, `collision_time` を優先利用する。
  - 存在しない場合はフォルダ名ルールで `label` を推定する（`normal`=0, `accident|abnormal|anomaly`=1）。
- 推論列は TF互換名を追加する。
  - `y_conv_score`, `y_conv(=is_anomaly)`, `y_pre_score`, `y_pre`
  - `y_pre` は擬似ヘッド（EWMA + 連続点判定）で算出する。
- `plot_timechart.py` は常に対話起動とする。
- 対話で可視化対象（CSV/ディレクトリ/グロブ）を候補一覧または自由入力で選択する。

## 5. データセット選択仕様
候補一覧は以下の配下から自動収集する。
- `datarecode_train`
- `datarecode_test`
- `02_20260213_dataset`

`tagged dataset` を選んだ場合は、設定ファイルを使って台帳フィルタを適用する。
- 学習時: `config/tagged_dataset_filter_train.json`
- 推論時: `config/tagged_dataset_filter_infer.json`

設定ファイルの `aq_to_ay_filters` で AQ〜AY を `enabled + mode(include/exclude) + values(複数値)` で定義し、
`combine=AND` で絞り込む。ファイル名突合は `TTDC提供ファイル名(タグ情報あり)` を基準に、拡張子を除いた stem 一致で行う。
CSV実体は `network_roots` 配下を再帰探索し、重複時は `network_roots` の順序で先勝ち採用する。

候補は `pattern`（既定 `*.csv`）でCSVが見つかるディレクトリ、またはCSVファイルを対象とする。

## 6. 学習・評価の運用方針（現行）
- 学習時にクロスバリデーションは実施しない。
- 単一の train/val 分割で学習し、その学習モデルを最終採用する。
- しきい値（threshold）は最終採用モデルに対して再算出する。
- `threshold.json` には既存の threshold/stat 値に加えて、高分位点統計 `p95`, `p99_5`, `p99_9`, `p99_95`, `p99_99`, `p99_999` を保存する。
- 再現性は strict deterministic モードをサポートする。
- `eval_score_csv.py` は ON/OFF 評価設定を 1 つの JSON で指定する `--config` をサポートする。
- 既存の `--config_on` / `--config_off` による ON/OFF 分割 config 指定も当面維持する。
- 評価入力ディレクトリは引き続き `--on_dir` / `--off_dir` で指定し、config 内の `evaluation.run_dir` をこの機能だけを理由に入力ディレクトリ既定値へ昇格しない。

## 7. 正式サポート範囲
- 正式サポート対象は、学習、推論、可視化、およびそれらを支える共通モジュールと設定に限定する。
- スクリプト整理後も、既存 CLI 互換、既存設定参照、既存出力契約は維持対象とする。
- 実験用スクリプトは本流の受け入れ保証対象に含めず、正式機能と切り分けて扱う。
