# Requirements

## 1. 機能要件

### R-001: 学習起動方式
- 学習スクリプトは常に対話モードで起動すること。
- 非TTY環境で起動した場合は、対話が必要である旨を明示して終了すること。

### R-002: optimizer 選択
- 学習開始時に `AdamW` / `RAdam` を対話で選択できること。
- 既定値は `hyperparams_common.json` の `optimizer` とすること。

### R-003: 学習データセット選択
- 学習開始時にデータセット候補を一覧表示すること。
- 一覧選択に加えて、自由入力（手入力パス）を受け付けること。
- 一覧候補は以下から探索すること。
  - `datarecode_train`
  - `datarecode_test`
  - `02_20260213_dataset`

### R-003f: タグ付けデータ選択（台帳連携）
- 学習データセット選択に `tagged dataset (ledger + ver_tag001/2/3)` を追加すること。
- `tagged dataset` 選択時は設定ファイルを読み込み、以下で学習対象CSVを決定すること。
  - 学習用設定: `config/tagged_dataset_filter_train.json`
  - 推論用設定: `config/tagged_dataset_filter_infer.json`
  - 設定項目: `enabled`, `ledger`, `network_roots`, `aq_to_ay_filters`, `combine`, `sampling`
- `aq_to_ay_filters` は AQ〜AY 列ごとに `mode`（`include` / `exclude`）と `values`（複数値）を持ち、`enabled=true` の列だけを適用すること。
- 複数列フィルタは `combine=AND` で評価すること。
- valid群番号の除外も AQ フィルタ設定で定義し、対話入力では受け付けないこと。
- CSV実体は `network_roots` 配下を再帰探索し、ファイル名突合は拡張子を除いた stem（大小文字無視）で行うこと。
- 同名CSVが複数箇所にある場合は、`network_roots` に定義された順序で先勝ち採用すること。
- 台帳にあるが実ファイルがないもの、実ファイルがあるが台帳にないものは除外し、件数を警告表示すること。
- `sampling.enabled=true` の場合、`tagged dataset` 選択時に `% (files)` を対話選択できること。
- `%` 候補は `10,20,...,100` とし、表示は `XX% (NN files)` とすること。
- 抽出件数は四捨五入で算出し、0件になる場合はエラー停止すること。
- サンプリング単位はファイル数とし、seed固定で再現可能な抽出とすること。
- `sampling.enabled=false` の場合、サンプリングは行わず100%利用すること。

### R-003a: 推論スクリプトの対話選択
- `train_score_csv.py` は常時対話起動とすること。
- artifactsディレクトリを候補一覧 + 自由入力で選択できること。
- スコア対象（CSV/ディレクトリ/グロブ）を候補一覧 + 自由入力で選択できること。
- スコア対象の選択に `tagged dataset (ledger + ver_tag001/2/3)` を含めること。

### R-003c: TF互換の推論結果フォルダ出力
- `train_score_csv.py` は、既存の `result/*_anomaly.csv` 出力を維持したまま、次を追加出力すること。
  - `output/Valid_results/<timestamp>_transformer_ae_inference_csv/basic_info.csv`
  - `output/Valid_results/<timestamp>_transformer_ae_inference_csv/file_scores.csv`
  - `output/Valid_results/<timestamp>_transformer_ae_inference_csv/decision_times.csv`
  - `output/Valid_results/<timestamp>_transformer_ae_inference_csv/*_Inference_confusion_matrix.csv/.png`
  - `output/Valid_results/<timestamp>_transformer_ae_inference_csv/*_Inference_ROC.csv/.png`

### R-003d: メタ情報CSV（任意）
- `config/inference_ground_truth.csv` は任意入力とすること。
- メタCSVが存在する場合、`label`, `abnormal_start_time`, `collision_time` を優先利用すること。
- メタCSVが存在しない場合、フォルダ名ルールで `label` を推定すること。
  - `normal` を含む: `label=0`
  - `accident` / `abnormal` / `anomaly` を含む: `label=1`
  - 両方含む、またはどちらも含まない場合: 評価用ラベル不明として扱うこと。

### R-003e: TF互換列と擬似 y_pre
- 推論CSVに次の列を追加すること。
  - `y_conv_score`, `y_conv`, `y_pre_score`, `y_pre`
- `y_conv` は `is_anomaly` と同義であること。
- `y_pre` は擬似ヘッドとして、`y_conv_score` のEWMAと連続点判定で算出すること。

### R-003b: 可視化スクリプトの対話選択
- `plot_timechart.py` は常時対話起動とすること。
- 可視化対象（CSV/ディレクトリ/グロブ）を候補一覧 + 自由入力で選択できること。

### R-004: ハイパーパラメータ読み込み
- `config/hyperparams_common.json` を既定読み込み対象とすること。
- JSONはBOM付きUTF-8も読み込めること。

### R-005: 学習/検証分割
- 学習時にクロスバリデーションは実行しないこと。
- 学習/検証は単一分割で実行すること。
- マルチCSV（セグメント）時は、セグメント分割で train/val を作成すること。

### R-006: 最終モデル採用
- 最終採用モデルは単一分割学習で得たモデルとすること。

### R-007: threshold 決定
- threshold は最終採用モデルに対して再算出すること。

### R-008: 再現性（strict deterministic）
- strict モード有効時、以下を適用すること。
  - `torch.backends.cudnn.deterministic = True`
  - `torch.backends.cudnn.benchmark = False`
  - `torch.use_deterministic_algorithms(True)`

### R-009: 勾配クリップ
- 勾配クリップ値は設定で制御可能とし、運用値は `grad_clip=1` を既定とすること。
- `grad_clip` 未指定時は `max_norm` をフォールバックとして使用すること。

## 2. 非機能要件
- 既存CLI引数との後方互換性を可能な限り維持すること。
- 変更は `feature/ae-update` のスコープ内に限定すること。

## 3. 受け入れ条件
- `python 1_transformer/train_transformer_autoencoder.py --help` でヘルプ表示できること。
- TTY実行時に optimizer と dataset の対話選択が表示されること。
- dataset 選択で「候補一覧」「manual input」が選べること。
- dataset 選択で `tagged dataset (ledger + ver_tag001/2/3)` が選べること。
- `tagged dataset` 選択時に `config/tagged_dataset_filter_train.json` を参照して AQ〜AY フィルタが適用されること。
- TTY実行時に `train_score_csv.py` で artifacts とスコア対象の対話選択が表示されること。
- `train_score_csv.py` のスコア対象で `tagged dataset` を選択した場合、`config/tagged_dataset_filter_infer.json` が適用されること。
- `train_score_csv.py` 実行後に、従来の `result/*_anomaly.csv` と TF互換推論結果フォルダの両方が生成されること。
- メタCSVなしでも推論は継続し、可能な範囲で評価出力が生成されること。
- TTY実行時に `plot_timechart.py` で可視化対象の対話選択が表示されること。
