# Design

## 1. 構成

### 1.1 主なディレクトリ
- `1_transformer/`: 学習・推論・モデル本体
- `app/ui/`: 対話UI層（選択メニュー、入力バリデーション）
- `config/`: ハイパーパラメータ設定
- `docs/specs/`: 承認済み仕様

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
  - 学習時は `config/tagged_dataset_filter_train.json`、推論時は `config/tagged_dataset_filter_infer.json` を読み込む。
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

## 4. 学習ロジック設計

### 4.1 学習/検証分割
- 学習時にクロスバリデーションは実行しない。
- マルチCSV（セグメント）時はセグメント分割で train/val を作成する。

### 4.2 最終モデル
- 単一分割学習で得たモデルを最終採用する。

### 4.3 threshold
- 最終採用モデルに対して再算出し、`threshold.json` に保存する。

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
