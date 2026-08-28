# AE/scripts/analyze_accel_distribution.py 仕様書

このファイルは、AI に scripts/analyze_accel_distribution.py を生成・再生成させるための仕様書である。
ここに書かれている内容を真の仕様とし、未記載の内容は AI が勝手に補完してはならない。
曖昧な点がある場合は、必ず人間に確認すること。

---

## 0. 概要

- スクリプト名
  - scripts/analyze_accel_distribution.py
- 目的
  - 学習時に使用する train データの分布を集計する。
  - 推論結果に基づく inference データの分布を集計する。
- モード
  - train モード
    - 学習時に使う元データ CSV を対象にする。
    - train の対象ファイル選定は config/tagged_dataset_train.json を使う。
    - train には window 生成方法が異なる 2 モードを持つ。
  - inference モード
    - 推論結果 CSV を対象にする。
    - 元データ CSV と推論結果 CSV は別ファイルである。
    - Time 分布は inference モードでのみ集計する。
- モード切替
  - CLI フラグ --filtered-mode を inference モード切替に使う。
  - --filtered-mode なしが train モード、ありが inference モードとする。
- 設定ファイル
  - scripts/analyze_config.json を使う。
  - CLI が最優先、analyze_config.json は既定値として使う。

---

## 1. 既存コードから利用するもの

### 1.1 学習コード

src/gofumi_ae/training/nstep.py から以下の考え方・関数・設定を踏襲する。

- initialize_from_definition(definition_path: Optional[str] = None) -> Dict
- read_csv_lower(path: str) -> pd.DataFrame
- require_columns(df: pd.DataFrame, path: str) -> None
- preprocess_df_for_training(df: pd.DataFrame, context: str)
- extract_model_config_from_definition(...)
- FEATURES
- FEATURE_RULES
- CATEGORY_MAPS
- CATEGORICAL_FEATURES
- DEFAULT_UNKNOWN_ID

### 1.2 train 対象ファイル選定

train モードでは src/gofumi_ae/datasets/tagged_dataset.py の既存実装の考え方を使う。

- build_tagged_dataset_csvs_from_config(...)
- config/tagged_dataset_train.json

### 1.3 対応付けキー

元データ CSV と推論結果 CSV の対応付けキーは、src/gofumi_ae/datasets/tagged_dataset.py の _normalize_file_stem() と同じ正規化規則を使う。

- パスからファイル名のみを取得する。
- 拡張子を外す。
- 小文字化する。
- 先頭の 0 を削除する。
- 空になった場合は 0 とみなす。

---

## 2. 設定仕様

### 2.1 analyze_config.json

設定ファイルは scripts/analyze_config.json とする。

設定ファイルには、CLI で設定できる内容を既定値として持たせる。

最低限持たせる項目:

- feature
- pattern
- output_dir
- output_prefix_train
- output_prefix_inference
- use_training_preproc
- definition_path
- train_window_mode
- time_mode
- time_column
- train_filter_config_path
- inference_filter_column
- inference_filter_value

### 2.2 優先順位

- CLI 指定値
- scripts/analyze_config.json の値
- 実装内の最終フォールバック

---

## 3. 入力仕様

### 3.1 train モード

- --csv / -i は元データ CSV が格納されたフォルダを指す。
- 対象ファイルは、-i で指定したフォルダ配下から取得する。
- その中で config/tagged_dataset_train.json の条件に一致するファイルのみを使う。
- train スクリプトと同様の対象ファイル集合になることを優先する。

### 3.2 inference モード

- --csv / -i は推論結果 CSV が格納されたフォルダを指す。
- inference モードでは、-i で指定したフォルダ内の全 CSV を対象候補とする。
- 元データ CSV と推論結果 CSV は別ファイルである。
- 必要に応じて、対応付けキーを用いて元データと推論結果を結び付ける。

### 3.3 CSV 内容

- 元データ CSV は既存学習コードと同じ FEATURES 列を持つ前提。
- 推論結果 CSV は推論結果列を持つ前提。
- inference モードで使う条件列は推論結果 CSV 側の列を参照する。

---

## 4. 前処理仕様

前処理方式は --use-training-preproc で切り替える。

### 4.1 use_training_preproc=True

- read_csv_lower
- require_columns
- preprocess_df_for_training

を実行し、学習時と同じ前処理済み値を分布集計に使う。

### 4.2 use_training_preproc=False

- read_csv_lower
- require_columns

のみを実行する。

- 連続値は pd.to_numeric(..., errors="coerce") で扱う。
- NaN と vmin/vmax 範囲外値は分布集計から除外する。
- カテゴリ値は CATEGORY_MAPS に基づき UNKNOWN を含めて正規化する。

---

## 5. window 仕様

### 5.1 共通

- seq_len は definition.json から取得する。
- hparams には依存しない。
- N < seq_len の CSV は解析対象外とし、ログ出力する。

### 5.2 train モード1

- train モード1 は既存の train スクリプトの考え方と同じ window 生成にする。
- 学習時に実際に使用する window 群を、そのまま集計対象として扱う。
- frame_range_config の start_value より前方へ、seq_len * 0.1 だけ開始側を広げ得る。
- 終了側は end_value までとする。

### 5.3 train モード2

- train モード2 は、指定範囲の中に完全に収まる window だけを集計対象にする。
- 開始位置が start_value より前の window は対象外。
- 終了位置が end_value を超える window は対象外。
- train スクリプト側の mode2 と同じ考え方に揃える。
- window 数は次で決まる。
  - 範囲内フレーム数 - seq_len + 1

### 5.4 inference モードの time_mode

- inference モードでは time_mode を使う。
- first_window
  - start_value より seq_len * 0.1 前から end_value までを対象にする。
- frame_range
  - start_value から end_value の範囲内に完全に収まる window のみを対象にする。

---

## 6. 分布定義

### 6.1 連続値 FEATURE

- vmin / vmax は FEATURE_RULES[feature] を使う。
- bin 数は 10。
- 幅 w = (vmax - vmin) / 10。

集計する値:

1. window ごとの最大値分布
2. 全 window に含まれる全フレーム値分布

### 6.2 train モードの集計対象

- train モードでは学習対象 window の値を集計する。
- mode1 と mode2 の違いは、window 集合の作り方にのみある。
- 同一フレームが複数 window に含まれる場合は、その回数分カウントする。

### 6.3 inference モードの集計対象

- inference モードでは、推論結果 CSV の条件列と条件値に一致したデータを対象とする。
- 条件列と条件値は CLI または analyze_config.json から取得する。
- デフォルト値は持たせない。
- 一致方法は完全一致とする。
  - 数値として扱える場合は数値完全一致。
  - それ以外は trim と大文字小文字を吸収した文字列完全一致。

### 6.4 Time 分布

- Time 分布は inference モードでのみ出力する。
- 理由は、初検知タイミングが推論結果側にしかないため。
- Time 列は time_column で指定する。
- 負値を含む実数値のまま扱う。
- 同一フレームが複数 window に含まれる場合は、その回数分カウントする。

---

## 7. 出力仕様

### 7.1 出力ファイル名

- train
  - accel_dist_train_<feature>.xlsx
- inference
  - accel_dist_inference_<feature>.xlsx

### 7.2 シート名

train:

- SummaryTrain
- PerFileTrain

inference:

- SummaryInference
- PerFileInference
- SummaryInferenceTime
- PerFileInferenceTime

### 7.3 出力内容

- Summary 系は全ファイル集計。
- PerFile 系はファイル別集計。
- Time 系は inference モードのみ出力。

---

## 8. CLI 仕様

- --csv, -i
  - train では元データフォルダ。
  - inference では推論結果フォルダ。
- --csvdir, -d
  - 必要時のみ使う補助入力。
- --pattern
  - CSV 探索パターン。
- --feature
  - 集計対象 FEATURE。
- --output-dir
  - Excel 出力先。
- --use-training-preproc
  - 学習前処理再利用の有無。
- --filtered-mode
  - inference モード切替フラグ。
- --filter-column
  - inference モードで使う条件列名。
  - デフォルト値は持たせない。
- --filter-value
  - inference モードで使う条件値。
  - デフォルト値は持たせない。
- --time-column
  - inference モードの Time 列名。
- --time-mode
  - inference モードの範囲抽出方法。
  - first_window または frame_range。
- --train-window-mode
  - train モードの window 方式。
  - mode1 または mode2。

---

## 9. 未確定事項

以下は本仕様書更新時点で未確定のため、実装時に勝手に補完してはならない。

- 推論結果 CSV 側の初検知タイミング列名
- 推論結果 CSV 側の衝突タイミング列名
- 推論結果 CSV と元データ CSV の追加的な対応付け方法が必要かどうか

