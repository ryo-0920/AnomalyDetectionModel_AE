# Design

## 1. 構成

### 1.1 主なディレクトリ
- `1_transformer/`: 学習・推論・モデル本体
- `app/ui/`: 対話UI層（選択メニュー、入力バリデーション）
- `config/`: ハイパーパラメータ設定
- `docs/specs/`: 承認済み仕様

### 1.3 スクリプト整理後の本流構成方針
- 本流パッケージは `gofumi_ae` とし、学習、推論、可視化、共通処理を集約できる構成を前提とする。
- 既存の `1_transformer/` 配下スクリプトは、整理後も旧 CLI 互換入口として残せる設計とする。
- Nstep 版は別系統として維持せず、本流実装の内部バリエーションとして統合する。
- CARLA/動画/Excel補助スクリプトは本流設計から切り離した実験用スクリプトとして扱う。
- tracked `__pycache__/` と `desktop.ini` は設計上の管理対象から除外し、生成補助物として扱う。

### 1.2 主要ファイル責務
- `1_transformer/train_transformer_autoencoder.py`
  - 学習エントリポイント
  - hparams読込
  - 対話UI呼び出し
  - データ準備、単一分割学習、threshold算出、成果物保存
- `1_transformer/train_score_csv.py`
  - 推論エントリポイント
  - 対話UIによる artifacts / 入力ターゲット選択
  - ストリーミングスコア算出とCSV出力（従来 `result/*_anomaly.csv`）
  - TF互換推論結果フォルダ出力（`basic_info.csv`, `file_scores.csv`, `decision_times.csv`, confusion matrix, ROC）
  - 任意メタCSV読込とフォルダ名ルールによるラベル補完
  - `y_pre` 擬似ヘッド（EWMA + 連続点）算出
- `1_transformer/plot_timechart.py`
  - 可視化エントリポイント
  - 対話UIによる入力ターゲット選択
  - タイムチャート画像出力
- `1_transformer/eval_score_csv.py`
  - 評価エントリポイント
  - 旧 CLI 互換 wrapper として本流評価実装へ委譲
  - ON/OFF の `*_anomaly.csv` から per-file summary、混同行列、ROC、区間別集計、plot を出力
  - 単一 config 形式 `--config` と既存 2 config 形式 `--config_on` / `--config_off` を受理
- `app/ui/prompts.py`
  - 整数入力、文字列入力、メニュー選択の共通処理
- `app/ui/interactive.py`
  - optimizer選択
  - データセット候補探索
  - データセット選択（候補一覧 + 自由入力）
  - artifacts選択（候補一覧 + 自由入力）
  - CSV/ディレクトリ/グロブ選択（候補一覧 + 自由入力）
- `app/tagged_dataset.py`
  - tagged dataset 用設定JSON読込
  - 台帳読み込み、AQ〜AYフィルタ評価
  - ネットワークCSV突合（stem一致）

### 1.4 構成整理時の責務分離
- 本流の正式サポート対象は、学習、推論、可視化、およびそれらを支える共通モジュールと設定に限定する。
- 旧 CLI 互換入口は、既存利用者向けの薄い wrapper とし、本流実装へ委譲する。
- 通常版と Nstep 版の差異は、CLI や設定から選択される実行モードとして吸収する。
- CARLA/動画/Excel補助スクリプトは、正式サポート境界の外側にある実験用ユーティリティとして分離する。

## 2. 実行フロー（学習）
1. 引数を受理し、hparamsをロードする。
2. 対話UIで optimizer を選択する。
3. 対話UIで dataset を選択する（候補一覧 or 手入力）。
4. 前処理後に学習を実行する。
5. 単一の train/val 分割で学習する。
6. 学習完了モデルで threshold を再算出し、artifactsに保存する。

## 3. 対話UI設計

### 3.1 optimizer 選択
- 入力値はメニュー番号で受け付ける。
- 既定値は hparams の `optimizer` を反映する。

### 3.2 dataset 選択
- 候補探索ルート:
  - `datarecode_train`
  - `datarecode_test`
  - `02_20260213_dataset`
- `pattern` に一致するCSVを持つディレクトリを候補化する。
- 最終メニューに `manual input` を追加し、自由入力を許可する。
- `tagged dataset (ledger + ver_tag001/2/3)` を追加し、台帳連携モードを起動できるようにする。
- 台帳連携モードでは以下を実行する。
  - 学習時は `config/tagged_dataset_train.json`、推論時は `config/tagged_dataset_inference.json` を読み込む。
  - `ledger`（path/sheet/header_row/file_name_column）設定で台帳Excel（市場走行一覧, ヘッダー3行目）を読み込む。
  - `aq_to_ay_filters`（AQ〜AY）を `enabled + mode + values` で評価する。
  - 複数列フィルタは `combine=AND` で評価する。
  - `TTDC提供ファイル名(タグ情報あり)` をキーに、`network_roots` を再帰探索した実CSVへ、拡張子除外のstem一致で突合する。
  - 同名ファイル重複時は `network_roots` の列挙順で先勝ち採用する。
  - `sampling.enabled=true` の場合、`10..100%` の固定候補を `XX% (NN files)` で対話提示し、選択率でファイル単位サンプリングする。
  - `sampling.enabled=false` の場合、サンプリングせず100%を利用する。

### 3.3 推論・可視化ターゲット選択
- `train_score_csv.py` は artifacts と入力ターゲットを対話で選択する。
- `train_score_csv.py` の入力ターゲットには `tagged dataset` を含める。
- `plot_timechart.py` は可視化対象を対話で選択する。
- いずれも候補一覧に加えて `manual input` を提供する。

### 3.4 推論評価情報
- `train_score_csv.py` はメタCSV（任意）を読み込む。
- メタCSVがない場合、CSVパスに基づくフォルダ名ルールで `label` を推定する。
- `label` 不明のCSVは推論結果CSVは出力し、混同行列/ROCの集計対象からは除外する。

### 3.5 eval_score_csv 評価 config 指定
- `eval_score_csv.py` は新しい任意引数 `--config <path>` を受理する。
- `--config` 指定時は、指定 JSON を ON 側設定と OFF 側設定の両方として扱う。
- 単一 config JSON はトップレベルに `evaluation` セクションを持ち、少なくとも `label_review_sheet` と `normal_ledger_sheet` の両方を含む。
- ON 側のラベル区間生成は `evaluation.label_review_sheet` を使い、OFF 側の正常台帳照合は `evaluation.normal_ledger_sheet` を使う。
- 単一 config の `evaluation.accel_column_name` は ON/OFF の両方に適用し、未指定時は `accelpedalangle` を使う。
- 既存の `--config_on <on_path> --config_off <off_path>` 形式は維持する。
- 2 config 形式では、ON config の `evaluation.label_review_sheet` と OFF config の `evaluation.normal_ledger_sheet` を使う。
- 2 config 形式の `evaluation.accel_column_name` は ON/OFF それぞれの config から読み、未指定時はそれぞれ `accelpedalangle` を使う。
- `--config` と `--config_on` / `--config_off` の混在指定、または `--config_on` / `--config_off` の片側だけの指定は usage error とし、評価処理を開始しない。
- 指定された config ファイルが存在しない場合は対象 path を示して失敗し、必須キーがない場合は対象 path と不足キーを示して失敗する。
- `--on_dir` と `--off_dir` は引き続き CLI 引数として指定する。`evaluation.run_dir` は、この変更だけを理由に入力ディレクトリの既定値として採用しない。
- `--out_dir`, `--fpr_targets`, `--score_col`, `--verbose` の意味、評価計算、出力ファイル名、依存管理、Docker/CUDA/OS 要件は変更しない。

## 4. 学習ロジック設計

### 4.1 学習/検証分割
- 学習時にクロスバリデーションは実行しない。
- マルチCSV（セグメント）時はセグメント分割で train/val を作成する。

### 4.2 最終モデル
- 単一分割学習で得たモデルを最終採用する。

### 4.3 threshold
- 最終採用モデルに対して再算出し、`threshold.json` に保存する。
- `threshold.json` には既存フィールドを維持したまま、次の追加高分位点統計を保存する。
  - `p95`: p95
  - `p99_5`: p99.5
  - `p99_9`: p99.9
  - `p99_95`: p99.95
  - `p99_99`: p99.99
  - `p99_999`: p99.999
- 小数分位点の JSON キーは、小数点を `_` に置換した `p99_5` 形式とする。
- 追加高分位点は、既存の `p10`, `p50`, `p90`, `p99`, `threshold` と同じ MAE 分布から `np.percentile` と同じ percentile 算出方式で算出する。
- 通常版は last-step MAE 分布を使い、Nstep 版は `tail_steps` を反映した tail MAE 分布を使う。
- `threshold` は引き続き `percentile` 設定値に基づく判定しきい値とし、追加高分位点の導入だけを理由に選択規則を変更しない。
- `temperature` は引き続き `p90` と `p50` から算出し、推論時の `y_conv_score` 正規化範囲は引き続き `p10` から `p99` とする。
- 推論 loader は、追加高分位点が存在する場合は `float` として threshold/stat context に保持し、存在しない既存 artifact もエラーにせず読み込む。
- 追加高分位点がない既存 artifact に対し、`mean`, `std`, `threshold` などから高分位点を推定して保存済み統計として扱わない。

### 4.4 再現性
- strict deterministic モードで cudnn / deterministic algorithms を強制する。

### 4.5 勾配クリップ
- `grad_clip` を優先して使用し、未指定時は `max_norm` を使用する。

### 4.6 推論スコア整合
- `y_conv_score` は現行 `anomaly` と同じ0〜1正規化スコアとする。
- `y_conv` は `is_anomaly` と同義とする。
- `y_pre_score` は `y_conv_score` のEWMAで算出する。
- `y_pre` は `y_pre_score >= thr_pre` の連続点条件で算出する。
- TF互換のファイル単位評価は `file_score_mode`（`tail` / `max`）で集約する。

## 5. 既知の制約
- 対話UIはTTY前提。
- `--no-dataset-prompt` は現行仕様では無効（常時対話）。
- 旧 CLI 互換は当面維持するため、整理後もしばらくは新旧入口が併存する。
- 実験用スクリプトは本流の必須受け入れ保証対象外であり、同じ品質保証境界には置かない。
- `eval_score_csv.py` の単一 config 対応は config 指定方式だけを変更対象とし、ON/OFF 評価入力ディレクトリの自動補完や評価ロジック変更は対象外とする。
