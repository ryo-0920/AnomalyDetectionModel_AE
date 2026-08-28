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
  - 学習用設定: `config/tagged_dataset_train.json`
  - 推論用設定: `config/tagged_dataset_inference.json`
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

### R-010: パッケージ構成
- 本流の Python パッケージ名は `gofumi_ae` とすること。
- スクリプト整理後の本流実装は、学習、推論、可視化、および共通処理を `gofumi_ae` 配下へ集約できる構成を前提とすること。

### R-011: 旧 CLI 互換
- 既存の公開 CLI パスと主要な起動方法は当面維持すること。
- 旧 CLI 入口は互換 wrapper として残してよく、内部実装の整理後も既存利用者の起動導線を壊さないこと。

### R-012: 通常版と Nstep 版の統合
- Nstep 版は本流実装へ統合すること。
- 実行時に通常版と Nstep 版を切り替えられること。
- 通常版と Nstep 版の切替により、既存の学習・推論・評価フローの公開契約を不必要に変更しないこと。

### R-013: 実験用スクリプトの扱い
- CARLA/動画/Excel補助スクリプトは実験用として扱うこと。
- これらは本流の正式サポート対象および必須受け入れ保証対象に含めないこと。

### R-014: tracked 補助生成物の扱い
- tracked `__pycache__/` と `desktop.ini` は Git 管理から外す方針とすること。
- これらは本流機能の成果物、設定、公開インターフェイスとして扱わないこと。

### R-015: threshold 分布統計の保存
- 新規作成する `threshold.json` は、既存の threshold/stat フィールドを維持したまま、高分位点統計として `p95`, `p99_5`, `p99_9`, `p99_95`, `p99_99`, `p99_999` を保存すること。
- 追加高分位点は、既存の `p10`, `p50`, `p90`, `p99`, `threshold` と同じ MAE 分布から、既存 percentile 算出方式に揃えて算出すること。
- 通常版と Nstep 版の本流実装、および旧 CLI 互換入口から作成される artifact で同じ追加キーを保存すること。
- 追加高分位点の導入により、`threshold` の意味、`temperature` 算出、推論スコア正規化、異常判定、CLI 引数、対話 UI、既存出力契約を変更しないこと。
- 追加高分位点がない既存 artifact も、引き続き推論で読み込めること。

### R-016: eval_score_csv 単一 config 指定
- `eval_score_csv.py` は、ON/OFF 評価設定を 1 つの JSON で指定する `--config <path>` を受け付けること。
- `--config` に指定する JSON はトップレベルに `evaluation` セクションを持ち、`evaluation.label_review_sheet` と `evaluation.normal_ledger_sheet` の両方を含むこと。
- `--config` 指定時、ON 側のラベル区間生成には同一 JSON の `evaluation.label_review_sheet` を使い、OFF 側の正常台帳照合には同一 JSON の `evaluation.normal_ledger_sheet` を使うこと。
- `--config` 指定時、`evaluation.accel_column_name` は ON/OFF の両方へ同じ値を適用し、未指定時は `accelpedalangle` を使うこと。
- 既存の `--config_on <on_path> --config_off <off_path>` 形式を維持すること。
- `--config_on` / `--config_off` 形式では、ON config の `evaluation.label_review_sheet` と OFF config の `evaluation.normal_ledger_sheet` を現行どおり使うこと。
- `--config_on` / `--config_off` 形式では、`evaluation.accel_column_name` を ON/OFF それぞれの config から読み、未指定時はそれぞれ `accelpedalangle` を使うこと。
- `--config` と `--config_on` / `--config_off` を混在指定した場合は usage error とし、評価処理を開始しないこと。
- `--config` がなく、`--config_on` または `--config_off` の片方だけが指定された場合は usage error とし、評価処理を開始しないこと。
- 指定された config ファイルが存在しない場合は、対象 path を示して失敗すること。
- 必須の `evaluation` セクション、`evaluation.label_review_sheet`、`evaluation.normal_ledger_sheet` が不足する場合は、対象 path と不足キーを示して失敗すること。
- `--on_dir`, `--off_dir`, `--out_dir`, `--fpr_targets`, `--score_col`, `--verbose` の意味を変更しないこと。
- `--on_dir` と `--off_dir` は引き続き CLI 引数として指定し、`evaluation.run_dir` をこの変更だけを理由に入力ディレクトリの既定値として採用しないこと。
- per-file summary、混同行列、ROC、区間別集計、plot の算出ロジックと出力ファイル名を変更しないこと。
- 評価用設定は `config/tagged_dataset_inference.json` の `evaluation_on` と `evaluation_off` を正本とすること。

## 2. 非機能要件
- 既存CLI引数との後方互換性を可能な限り維持すること。
- 既存の公開 CLI パス、設定参照先、出力契約を可能な限り維持すること。
- スクリプト整理に伴う実験用スクリプトの再配置は、本流の正式サポート範囲を明確化する目的に限定すること。
- 変更は `feature/ae-update` のスコープ内に限定すること。

## 3. 受け入れ条件
- `python 1_transformer/train_transformer_autoencoder.py --help` でヘルプ表示できること。
- TTY実行時に optimizer と dataset の対話選択が表示されること。
- dataset 選択で「候補一覧」「manual input」が選べること。
- dataset 選択で `tagged dataset (ledger + ver_tag001/2/3)` が選べること。
- `tagged dataset` 選択時に `config/tagged_dataset_train.json` を参照して AQ〜AY フィルタが適用されること。
- TTY実行時に `train_score_csv.py` で artifacts とスコア対象の対話選択が表示されること。
- `train_score_csv.py` のスコア対象で `tagged dataset` を選択した場合、`config/tagged_dataset_inference.json` が適用されること。
- `train_score_csv.py` 実行後に、従来の `result/*_anomaly.csv` と TF互換推論結果フォルダの両方が生成されること。
- メタCSVなしでも推論は継続し、可能な範囲で評価出力が生成されること。
- TTY実行時に `plot_timechart.py` で可視化対象の対話選択が表示されること。
- 本流のパッケージ名が `gofumi_ae` として仕様化されていること。
- 旧 CLI 互換を当面維持する方針が仕様に記録されていること。
- Nstep 版が本流へ統合され、実行時切替対象であることが仕様に記録されていること。
- CARLA/動画/Excel補助スクリプトが実験用であり、本流の正式サポート対象外であることが仕様に記録されていること。
- tracked `__pycache__/` と `desktop.ini` を Git 管理から外す方針が仕様に記録されていること。
- 本流通常版の学習 artifact 作成後、`threshold.json` に既存キーを維持したまま `p95`, `p99_5`, `p99_9`, `p99_95`, `p99_99`, `p99_999` が保存されること。
- 本流 Nstep 版の学習 artifact 作成後、`threshold.json` に既存キーと `tail_steps` を維持したまま `p95`, `p99_5`, `p99_9`, `p99_95`, `p99_99`, `p99_999` が保存されること。
- 旧 CLI 互換の通常版および Nstep 版の学習 artifact 作成後も、本流と同じ追加キーが `threshold.json` に保存されること。
- 追加キーの値が、同一 MAE 分布に対する `np.percentile` の 95, 99.5, 99.9, 99.95, 99.99, 99.999 percentile と一致すること。
- 同一 MAE 分布から算出される `p99`, `p99_5`, `p99_9`, `p99_95`, `p99_99`, `p99_999` が、数値誤差を除き単調非減少であること。
- `percentile=99.5` で threshold を算出するケースでは、`threshold` と `p99_5` が同一 MAE 分布・同一 percentile 算出方式に基づく値として一致すること。
- 追加キーが存在しない既存 `threshold.json` を、本流通常版・本流 Nstep 版・旧 CLI 互換版の推論 loader がエラーにせず読み込めること。
- 追加キーが存在する `threshold.json` を推論 loader が読み込む場合、追加キーを `float` として threshold/stat context に保持すること。
- 追加キーの有無によって、既存の `threshold`, `p10`, `p50`, `p90`, `p99`, `temperature` に基づく `y_conv_score`, `y_conv_threshold`, `y_pre_threshold`, `is_anomaly` の結果が変わらないこと。
- 本変更により CLI help、対話起動、既存 artifact の保存先、既存 CSV 出力列、依存管理、実行環境要件が変更されないこと。
- `python 1_transformer/eval_score_csv.py --help` で `--config`, `--config_on`, `--config_off` が確認できること。
- `eval_score_csv.py --config <path>` だけを指定した場合、同一 JSON から `evaluation.label_review_sheet` と `evaluation.normal_ledger_sheet` が読み込まれること。
- `eval_score_csv.py --config_on <on_path> --config_off <off_path>` を指定した場合、ON config から `label_review_sheet`、OFF config から `normal_ledger_sheet` が現行どおり読み込まれること。
- `eval_score_csv.py` で `--config` と `--config_on` / `--config_off` を混在指定した場合、usage error になり、評価処理を開始しないこと。
- `eval_score_csv.py` で `--config` がなく `--config_on` または `--config_off` の片方だけを指定した場合、usage error になり、評価処理を開始しないこと。
- `eval_score_csv.py` の単一 config に `evaluation.label_review_sheet` または `evaluation.normal_ledger_sheet` が不足する場合、対象 path と不足キーを示して失敗すること。
- `eval_score_csv.py` で存在しない config path を指定した場合、対象 path を示して失敗すること。
- `--config` 単独指定では `config/tagged_dataset_inference.json` から `evaluation_on` と `evaluation_off` が読めること。
- `eval_score_csv.py` の単一 config 形式と 2 config 形式のどちらでも、`--on_dir`, `--off_dir`, `--out_dir`, `--fpr_targets`, `--score_col`, `--verbose` の意味と出力契約が変わらないこと。
