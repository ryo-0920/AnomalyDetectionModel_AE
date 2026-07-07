# AE/scripts/analyze_accel_distribution.py 仕様書（AI生成用）
このファイルは、AI に `analyze_accel_distribution.py` を生成・再生成させるための仕様書である。
ここに書かれている内容は **唯一の真の仕様** とみなし、AI はこれを勝手に補完・変更してはならない。
※ 本仕様書に記載されていない振る舞い（例外時の詳細な挙動など）について、
AI は **自動的に補間・決定してはならない**。
曖昧な点がある場合は、必ず人間に仕様を問い合わせること。
スクリプト生成時にテスト等の元のスクリプトと関係ないファイルを生成した場合は必ず削除すること。

---
## 0. スクリプトの概要
- スクリプト名
  `AE/scripts/analyze_accel_distribution.py`
- 目的
  学習に使用する時系列 CSV 群に対して、指定した特徴量（FEATURE）の
  - 過去 `seq_len` フレーム分の **window** ごとの最大値の分布（連続値FEATUREのみ）
  - 全 window に含まれる **全フレーム値** の分布（連続値 / カテゴリFEATURE）
  を集計し、1つの Excel (`.xlsx`) ファイル（`Summary` / `PerFile` の2シート構成）に出力する。
  また、filtered モードでは、各 window 内で条件列が条件値に完全一致した行だけを対象にした
  - window ごとの最大値分布（連続値FEATUREのみ）
  - 全 window に含まれる条件一致フレーム値分布
  を追加集計し、同じ Excel に `SummaryFiltered` / `PerFileFiltered` シートとして出力する。
  さらに、異常発生のデータ集計モード（`--filtered-mode` 有効時）に対して、
  - **異常判定フレームの Time も分布で集計**し、`SummaryFilteredTime` / `PerFileFilteredTime` シートとして出力する。
  - **誤検知フレームにおける FEATURE（任意の連続FEATURE）の分布**も追加集計し、`SummaryFalseAccel` / `PerFileFalseAccel` シートとして出力する。
    - 誤検知フレームの定義は、`--filtered-mode` / `--filter-column` / `--filter-value` により指定された条件一致行とする。
    - 誤検知分布では、誤検知条件に一致した各フレーム行を **一度だけ**カウントし、window による重複は発生させない（max 系は誤検知では扱わない）。
- 特記事項
  - window の扱いは、既存の学習コード（`SequenceDatasetMasked` / `SequenceDatasetMaskedSegments`）と **完全に同一仕様** とする。
  - 既存の前処理関数群（`preprocess_df_for_training` など）を再利用できるようにする。
  - 前処理方式はフラグで切り替え可能（既存前処理 / 生値）。
  - filtered モードの条件は、現行仕様では **1 組の完全一致条件のみ** とする。
---
## 1. 既存環境・前提
### 1.1 既存コードから利用するもの
スクリプトは、既存の学習用コードから以下の関数・定数をインポートして利用する前提とする。
- 関数
  - `read_csv_lower(path: str) -> pd.DataFrame`
  - `require_columns(df: pd.DataFrame, path: str) -> None`
  - `preprocess_df_for_training(df: pd.DataFrame, context: str) -> Tuple[pd.DataFrame, np.ndarray, int]`
  - `list_csvs_in_dir(csv_dir: str, pattern: str = "*.csv", recursive: bool = False) -> List[str]`
- 定数 / 設定
  - `FEATURES: List[str]`
  - `FEATURE_RULES: Dict[str, Dict[str, Any]]`
  - `CATEGORY_MAPS: Dict[str, Dict[str, float]]`
  - `CATEGORICAL_FEATURES: List[str]`
  - `DEFAULT_UNKNOWN_ID: float`
AI は、これらがどのモジュールから提供されるかは **決めない**。
コード中では仮の import でよい（コメントで「実環境に合わせて修正」と記載）。
### 1.2 ハイパーパラメータ設定ファイル
- パス（デフォルト）
  `config/hyperparams_common.json`
- 内容のうち、本スクリプトが必ず参照するキー
  - `seq_len`: int
    window 長。学習コードと同じ値を使用する。
---
## 2. 入力仕様
### 2.1 CSVファイルの探索
CLI 引数をもとに、解析対象 CSV パス一覧を決定する。
- 引数
  - `--csv`, `-i`
    単一 CSV ファイル、または CSV を含むディレクトリのパス。
  - `--csvdir`, `-d`
    CSV ディレクトリのパス。指定されていれば `--csv` より優先。
  - `--pattern`
    ディレクトリ探索時の glob パターン（例: `"*.csv"`）。デフォルト `"*.csv"`。
- ロジック
  1. `csvdir` が非空なら、`csvdir` を探索対象ディレクトリとする。
  2. `csvdir` が空で `csv` がディレクトリなら、`csv` を探索対象ディレクトリとする。
  3. `csvdir` が空で `csv` がファイルなら、その1ファイルのみ解析対象とする。
  4. ディレクトリ探索時は `list_csvs_in_dir(source_dir, pattern, recursive=True)` を使ってパス一覧を取得する。
  5. パスは `os.path.normpath` で正規化し、重複を除去したうえでソートする。
  6. 有効な CSV が 1 件もなければエラー。
### 2.2 CSV の内容
- 各 CSV は、既存学習コードと同じ `FEATURES` 列を持っている前提。
- スクリプトでは各 CSV について:
  1. `read_csv_lower(path)` で読み込み（列名は小文字に揃えられる）。
  2. `require_columns(df, path)` で必要列の存在チェック。
     を **必ず** 行う。
---
## 3. 前処理仕様（2モード）
前処理方式は CLI フラグ `--use-training-preproc` で切り替える。
### 3.1 `--use-training-preproc=True`（既存前処理を再利用）
- 各 CSV について、以下を実行する:
  1. `read_csv_lower`
  2. `require_columns`
  3. `preprocess_df_for_training(df, context=os.path.basename(path))`
     - `drop_rows_with_missing_features` により、`FEATURES` に対して欠損行を削除。
     - `apply_categorical_mapping` によりカテゴリ列をコード化。
     - `clean_and_mask_by_rules` により、ルールに基づいて
       - センサ異常値・範囲外値を欠損扱い
       - 補間（linear / ffill）
       - vmin/vmax で clip
- 連続値FEATUREの解析には、**補間・clip 済みの値**を用いる。
- カテゴリFEATUREの解析には、**数値コード化された列**を用いる。
### 3.2 `--use-training-preproc=False`（前処理なし / 生値）
- 各 CSV について:
  1. `read_csv_lower`
  2. `require_columns`
  のみ実行し、その後はスクリプト内の処理で直接扱う。
- 連続値FEATURE:
  - `pd.to_numeric(df[feature], errors="coerce")` で数値変換。
  - NaN と、`FEATURE_RULES[feature]["vmin"]` 未満 / `["vmax"]` 超えの値は、**分布計算からスキップ**する。
  - 補間・clip は行わない。
- カテゴリFEATURE:
  - クラス集合は `CATEGORY_MAPS[feature]` に従う。
  - 列を文字列として読み、
    - `.astype(str).str.strip().str.upper()` で正規化。
    - この文字列を `CATEGORY_MAPS[feature]` のキーと照合し、一致すれば対応するコードに変換。
    - 一致しないものは `"UNKNOWN"` にまとめ、そのコードに変換（`CATEGORY_MAPS[feature]["UNKNOWN"]` または `DEFAULT_UNKNOWN_ID`）。
  - 分布計算時、NaN はスキップする。
---
## 4. window 仕様（最重要）
### 4.1 定義
- `seq_len` は `config/hyperparams_common.json` の `seq_len` を使用。
- 各 CSV の前処理後の行数を N とする。
- N < `seq_len` の CSV は、**解析対象外（スキップ）**とする。
- 対象フレームインデックス t:
  - `t = seq_len-1, seq_len, ..., N-1`
- 各 t に対する window:
  - `window_t` はインデックス `[t-seq_len+1, ..., t]` の **連続した seq_len 行**。
  - 過去のみを見る。
- window のスライド:
  - 開始インデックスは `0 .. N-seq_len`。
  - window は 1 フレーム刻みでスライドし、**強く重複する**。
  - これは学習コードの `SequenceDatasetMasked` / `SequenceDatasetMaskedSegments` と **完全に同一**。
### 4.2 N < seq_len の扱い
- 行数 N が `seq_len` 未満の CSV は、window が 1 つも構成できないため、**解析対象外**とし、その旨をログ出力する。
---
## 5. FEATURE 種別ごとの分布定義
解析対象の列は CLI 引数 `--feature` で指定。
**1 回の実行で 1 列のみ解析**する（単一列対応）。
FEATURE は以下の2種に分類される。
- 連続値FEATURE（`feature not in CATEGORICAL_FEATURES`）
- カテゴリFEATURE（`feature in CATEGORICALFEATURES`）
### 5.1 連続値FEATURE
#### 5.1.1 bin 定義
- vmin / vmax は `FEATURE_RULES[feature]["vmin"]`, `["vmax"]` を使用。
- bin 数は 10 個。
- 幅 `w = (vmax - vmin) / 10`。
- 区間（概念的定義）:
  - bin0: `[vmin, vmin+w]`
  - bin1: `(vmin+w, vvmin+2w]`
  - ...
  - bin9: `(vmin+9w, vmax]`
- 実装上は `np.digitize(value, edges, right=True)` を用いる想定。
  - edges: `[vmin+w, vmin+2w, ..., vmin+10w(=vmax)]`
  - value が NaN / vmin 未満 / vmax 超えの場合は **bin に割り当てずスキップ**。
#### 5.1.2 分布の対象
連続値FEATUREでは、次の 2 種類の値を集計する。
1. **windowごとの最大値分布**
   - 各 `window_t` の対象列値（長さ `seq_len`）の中から、有効値（NaN を除く）に対して最大値を計算する。
   - 有効値が1つもない window は、max 分布には乗せない（スキップ）。
   - 全 `window_t` について、この最大値を bin に割り当て、`max_count` を集計する。
2. **全フレーム値分布**
   - 各 `window_t` 内に含まれるすべてのフレームの対象列値を集計する。
   - 同一フレームが複数の window に現れる場合、その回数分カウントする（重複あり）。
   - NaN および vmin 未満 / vmax 超えはスキップ。
   - 全 window の全フレーム値を bin に割り当て、`all_count` を集計する。
#### 5.1.3 filtered モード（連続値FEATURE）
- filtered モードは CLI フラグ `--filtered-mode` で有効化する。
- 条件は次の 1 組を受け付ける。
  - `--filter-column`: 条件列名。デフォルトは `is_anomaly`。
  - `--filter-value`: 条件値。デフォルトは `"1"`。
- 一致方法は **完全一致** とする。
  - 列が数値として扱える場合は数値完全一致。
  - それ以外は `.astype(str).str.strip().str.upper()` による正規化後の文字列完全一致。
- window 自体は通常モードと同じく、**元データの連続行** で構成する。
- 各 window 内で、条件列が条件値に一致した行だけを filtered 集計対象とする。
- filtered モードの `max_count`:
  - 各 window 内の条件一致行に対応する対象列値だけを取り出す。
  - その部分に有効値が 1 つ以上ある場合、その最大値を 1 件として bin に割り当てる。
  - 条件一致行が 0 件、または条件一致行に有効値が 1 つもない window はスキップする。
- filtered モードの `all_count`:
  - 各 window 内の **学習対象となったフレーム** のうち、条件一致している行の対象列値だけを集計する。
  - 同一フレームが複数の window に含まれる場合、その回数分カウントする（重複あり）。
  - NaN および vmin 未満 / vmax 超えはスキップする。
- filtered モードの割合分母:
  - `max_ratio` / `max_ratio_total` の分母は、filtered 集計で採用された max 件数。
  - `all_ratio` / `all_ratio_total` の分母は、filtered 集計で採用された全フレーム件数。
#### 5.1.4 Time 分布（filtered モード / 異常判定フレーム）
- 異常発生のデータ集計モード（`--filtered-mode=True`）に対して、
  異常判定フレームの Time も分布で集計する。
- Time 列は CLI 引数 `--time-column` で指定する。
  - デフォルト値: `"time"`。
  - Time 列は **実数値（float 等）** として扱う。
  - 負の値も許容し、絶対値にはせず、**元の実数値のまま** 扱う。
- Time 分布の対象は、**filtered 条件一致フレームのみ** とする。
  - `--filtered-mode` が True のときのみ集計する。
  - 条件は `--filter-column` / `--filter-value` による完全一致（5.1.3 に準拠）とする。
  - これにより、「異常判定フレーム」の Time 分布を得る。
  - 同一フレームが複数の window に含まれる場合、**window に登場する回数ぶんカウントする**。
- Time 分布の bin は、**データから自動的に決める**。
  - `vmin`: filtered 対象フレームの Time の **最小値** を取得し、その値を **0.1 の位で切り捨てた値** とする。
    例: `-0.23` → `-0.3`、`0.27` → `0.2`。
  - `vmax`: filtered 対象フレームの Time の **最大値** をそのまま用いる。
  - bin 幅 `w` は **1.0 固定** とする。
  - bin 区間は `bin0: [vmin, vmin+w]`、`bin1: (vmin+w, vmin+2w]`、… のように 1.0 幅で拡張する。
  - 実装上は `n_bins = ceil((vmax - vmin) / w)` とし、`edges = [vmin+w, vmin+2w, ..., vmin+n_bins*w]` を用いる概念でよい。
  - Time 値が NaN である場合や範囲外となる場合は、**bin に割り当てずスキップ**する。
- `SummaryFilteredTime` / `PerFileFilteredTime` の集計は、
  - `time_count` / `time_count_total`: filtered 条件一致フレームの Time を各 bin に属する件数としてカウントしたもの。
  - `time_ratio` / `time_ratio_total`: 各分母は、filtered 条件一致フレームとして採用された Time 件数とする。
#### 5.1.5 誤検知分布（連続値FEATURE / filtered モード）
- 「誤検知フレーム」は、既存の `--filtered-mode` / `--filter-column` / `--filter-value` により
  条件列・条件値に完全一致した行として定義する。
  - ユーザーは、誤検知を表す列と値（例: `is_false_positive == 1`）を `--filter-column` / `--filter-value` に指定する。
- 対象 FEATURE:
  - 誤検知分布の対象 FEATURE は、現在 `--feature` で指定している FEATURE（任意の連続FEATURE）とする。
  - `accelpedalangle` 以外の連続FEATUREでも利用可能。
- 誤検知分布の対象は、**filtered 条件一致フレームのみ** とする。
  - `--filtered-mode` が True のときのみ誤検知分布を集計する。
  - 条件は 5.1.3 と同じく、`--filter-column` / `--filter-value` による完全一致。
- 誤検知分布における window の扱い:
  - 誤検知分布（FalseAccel）では、**window 構造は用いず**、DataFrame 全体のフレーム行単位で集計する。
  - 各 CSV について、`filter_column == filter_value` に一致した行を一度だけ抽出し、その行の FEATURE 値を分布の対象とする。
  - 同一フレームが複数の window に含まれる場合でも、誤検知分布では **フレーム行としては 1 回だけカウント**する（重複なし）。
- 誤検知分布の bin 定義:
  - bin の定義は、通常の連続値FEATURE分布と **同一仕様** とする（5.1.1 に準拠）。
    - vmin / vmax は `FEATURE_RULES[feature]["vmin"]`, `["vmax"]`。
    - bin 数は 10。
    - 幅 `w = (vmax - vmin) / 10`。
    - 区間・`np.digitize` の使用、NaN / 範囲外値のスキップも通常分布と同一。
- 誤検知分布で集計する値（max 系は誤検知では扱わない）:
  - **誤検知フレーム値分布（fp_all 系）**のみを扱う。
    - 各 CSV について、`filter_column == filter_value` に一致した全誤検知フレームの FEATURE 値を抽出する。
    - NaN および vmin 未満 / vmax 超えはスキップ。
    - 抽出された誤検知フレーム値を bin に割り当て、`fp_count` を集計する。
    - 同一フレームが複数の window に含まれる場合でも、誤検知分布ではそのフレーム行を 1 度だけカウントする。
- 割合の分母:
  - `fp_ratio_total` の分母は、全ファイルの誤検知条件に一致したフレーム行数（分布に採用された全有効フレーム件数）とする。
  - `fp_ratio` の分母は、当該ファイルの誤検知条件に一致したフレーム行数（分布に採用された全有効フレーム件数）とする。
### 5.2 カテゴリFEATURE
> カテゴリの場合は値の大小の関係はないため、
> 学習で使用された全フレームの解析のみ（window代表値は扱わない）
- カテゴリFEATUREでは、**window代表値（max系）を一切扱わない**。
- 学習で使用された全フレーム（前処理後の全行）に対して、クラスごとの出現回数を集計する。
#### 5.2.1 クラス集合
- クラス集合は `CATEGORY_MAPS[feature]` のキーとする。
  - 例: `"UNKNOWN"`, `"OFF"`, `"ON"` など。
- それぞれに対応する数値コードは `CATEGORY_MAPS[feature][クラス名]`。
- 分布の出力は **クラス名** 単位で行う（コード値ではない）。
#### 5.2.2 値のマッピング
- `--use-training-preproc=True` の場合:
  - `preprocess_df_for_training` により、対象列はすでに数値コードになっている前提。
  - `pd.to_numeric(df[feature], errors="coerce")` で float64 配列に変換して扱う。
  - コード → クラス名の逆引きは、`CATEGORY_MAPS[feature]` から構成する。
- `--use-training-preproc=False` の場合:
  - 列を文字列として読み、`.strip().upper()` で正規化。
  - 文字列が `CATEGORY_MAPS[feature]` のキーに存在すれば、そのコードに変換。
  - 存在しない場合は `"UNKNOWN"` クラスのコードに変換（`CATEGORY_MAPS[feature]["UNKNOWN"]` または `DEFAULT_UNKNOWN_ID`）。
  - 分布計算時、NaN はスキップする。
---
## 6. 出力仕様
### 6.1 出力ファイル
- 形式: Excel `.xlsx`
- 出力先: CLI `--output-dir` で指定（デフォルト: `AE/outputs`）。
- ファイル名: `<output-prefix>_<feature>.xlsx`
  - `--output-prefix` のデフォルト: `accel_dist`。
### 6.2 シート構成
- シート1: `Summary`
- シート2: `PerFile`
- シート3: `SummaryFiltered`（filtered モード時のみ）
- シート4: `PerFileFiltered`（filtered モード時のみ）
- シート5: `SummaryFilteredTime`（filtered モード時のみ）
- シート6: `PerFileFilteredTime`（filtered モード時のみ）
- シート7: `SummaryFalseAccel`（filtered モード時のみ）
- シート8: `PerFileFalseAccel`（filtered モード時のみ）
#### 6.2.1 Summary シート（連続値FEATURE）
列構成:
| 列名             | 説明                                                                                   |
|------------------|----------------------------------------------------------------------------------------|
| bin_min          | bin の下端値（float）                                                                 |
| bin_max          | bin の上端値（float）                                                                 |
| max_count_total  | 全ファイルの全 window における、その bin の max 値の件数                              |
| max_ratio_total  | `max_count_total / 全 max 値件数`（全 window数、スキップを除く）                      |
| all_count_total  | 全ファイルの全 window に含まれる全フレーム値のうち、その bin に入った件数（重複あり） |
| all_ratio_total  | `all_count_total / 全フレーム値件数`（スキップを除く）                                |
- 全 max 値件数 = 各ファイルの `n_valid_max` の総和。
- 全フレーム値件数 = 各ファイルの `n_valid_all` の総和。
#### 6.2.2 PerFile シート（連続値FEATURE）
列構成:
| 列名       | 説明                                                                               |
|------------|------------------------------------------------------------------------------------|
| file_name  | CSVファイル名（ベース名）                                                         |
| bin_min    | bin の下端値                                                                      |
| bin_max    | bin の上端値                                                                      |
| max_count  | 当該ファイル内の全 window における、その bin の max 値件数                        |
| max_ratio  | `max_count / n_valid_max`（当該ファイルで分布に採用された max 値件数）            |
| all_count  | 当該ファイル内の全 window における、その bin のフレーム値件数                     |
| all_ratio  | `all_count / n_valid_all`（当該ファイルで分布に採用された全フレーム値件数）       |
#### 6.2.3 Summary シート（カテゴリFEATURE）
列構成:
| 列名             | 説明                                            |
|------------------|-------------------------------------------------|
| feature          | FEATURE名                                       |
| class_value      | クラス名（CATEGORY_MAPS のキー）               |
| all_count_total  | 全ファイルにおける、そのクラスの出現件数       |
| all_ratio_total  | `all_count_total / 全フレーム値件数`（スキップを除く） |
- max 系列は一切出力しない。
#### 6.2.4 PerFile シート（カテゴリFEATURE）
列構成:
| 列名        | 説明                                                                                   |
|-------------|----------------------------------------------------------------------------------------|
| file_name   | CSVファイル名（ベース名）                                                             |
| feature     | FEATURE名                                                                             |
| class_value | クラス名                                                                              |
| all_count   | 当該ファイルにおける、そのクラスの出現件数                                            |
| all_ratio   | `all_count / 当該ファイルで分布に採用された全フレーム値件数`                          |
- max 系列は一切出力しない。
#### 6.2.5 SummaryFiltered シート（連続値FEATURE, filtered モード）
列構成:
| 列名             | 説明                                                         |
|------------------|--------------------------------------------------------------|
| feature          | FEATURE名                                                    |
| condition_column | 条件列名                                                     |
| condition_value  | 条件値                                                       |
| bin_min          | bin の下端値（float）                                       |
| bin_max          | bin の上端値（float）                                       |
| max_count_total  | 全ファイルの全 window における、条件一致部分の max 値件数   |
| max_ratio_total  | `max_count_total / filtered 集計で採用された全 max 件数`     |
| all_count_total  | 全ファイルの全 window における、条件一致フレーム値件数      |
| all_ratio_total  | `all_count_total / filtered 集計で採用された全フレーム件数` |
#### 6.2.6 PerFileFiltered シート（連続値FEATURE, filtered モード）
列構成:
| 列名             | 説明                                                           |
|------------------|----------------------------------------------------------------|
| file_name        | CSVファイル名（ベース名）                                     |
| feature          | FEATURE名                                                      |
| condition_column | 条件列名                                                       |
| condition_value  | 条件値                                                         |
| bin_min          | bin の下端値（float）                                         |
| bin_max          | bin の上端値（float）                                         |
| max_count        | 当該ファイル内の全 window における、条件一致部分の max 値件数 |
| max_ratio        | `max_count / 当該ファイルで採用された filtered max 件数`       |
| all_count        | 当該ファイル内の全 window における、条件一致フレーム値件数    |
| all_ratio        | `all_count / 当該ファイルで採用された filtered 全フレーム件数` |
- 現行仕様では `SummaryFiltered` / `PerFileFiltered` は **連続値FEATUREのみ対応** とする。
#### 6.2.7 SummaryFilteredTime シート（Time 分布, filtered モード）
- 異常発生のデータ集計モード（`--filtered-mode=True`）において、
  異常判定フレーム（filtered 条件一致フレーム）の Time を分布で集計する。
- Time 分布は、`--time-column` で指定された列を対象とする。
列構成:
| 列名             | 説明                                                         |
|------------------|--------------------------------------------------------------|
| feature          | FEATURE名（`--feature` 引数）                               |
| condition_column | 条件列名（`--filter-column`）                               |
| condition_value  | 条件値（`--filter-value`）                                  |
| time_bin_min     | Time bin の下端値（float, 実数値。単位は Time 列と同じ）   |
| time_bin_max     | Time bin の上端値（float, 実数値）                          |
| time_count_total | 全ファイルで、その bin に入った Time の件数                |
| time_ratio_total | `time_count_total / 全 Time 件数`（スキップを除く）        |
- `time_count_total` / `time_ratio_total` の対象は、**filtered 条件一致フレームのみ** である。
- 同一フレームが複数の window に含まれる場合、**window に登場する回数ぶんカウントする**。
#### 6.2.8 PerFileFilteredTime シート（Time 分布, filtered モード）
列構成:
| 列名             | 説明                                                         |
|------------------|--------------------------------------------------------------|
| file_name        | CSVファイル名（ベース名）                                   |
| feature          | FEATURE名（`--feature` 引数）                               |
| condition_column | 条件列名                                                     |
| condition_value  | 条件値                                                       |
| time_bin_min     | Time bin の下端値（float, 実数値）                          |
| time_bin_max     | Time bin の上端値（float, 実数値）                          |
| time_count       | 当該ファイルで、その bin に入った Time の件数              |
| time_ratio       | `time_count / 当該ファイルで分布に採用された Time 件数`     |
- `time_count` / `time_ratio` の対象は、**filtered 条件一致フレームのみ**。
- 同一フレームが複数の window に含まれる場合、**window に登場する回数ぶんカウントする**。
#### 6.2.9 SummaryFalseAccel シート（誤検知分布, 連続値FEATURE, filtered モード）
- 誤検知分布は、`--filtered-mode` / `--filter-column` / `--filter-value` で指定された条件に完全一致した行を
  「誤検知フレーム」とみなし、その FEATURE 分布を集計したもの。
- 誤検知分布では、誤検知条件に一致した各フレーム行を **一度だけ** カウントし、window による重複は発生させない。
- 誤検知分布では max 系列（fp_max_*）は扱わず、フレーム値分布（fp_all 系）のみを出力する。
列構成:
| 列名             | 説明                                                                                  |
|------------------|---------------------------------------------------------------------------------------|
| feature          | FEATURE名（`--feature` 引数）                                                        |
| condition_column | 誤検知条件列名（`--filter-column`）                                                  |
| condition_value  | 誤検知条件値（`--filter-value`）                                                     |
| bin_min          | bin の下端値（float）                                                                |
| bin_max          | bin の上端値（float）                                                                |
| fp_count_total   | 全ファイルにおける、誤検知条件に一致したフレーム値のうち、その bin に入った件数（重複なし） |
| fp_ratio_total   | `fp_count_total / 誤検知フレーム値分布で採用された全有効フレーム値件数`              |
- `fp_count_total` の分母となる「全有効フレーム値件数」は、
  全ファイルの誤検知条件に一致し、かつ vmin〜vmax 範囲内・非 NaN であったフレーム行数の総数とする。
#### 6.2.10 PerFileFalseAccel シート（誤検知分布, 連続値FEATURE, filtered モード）
列構成:
| 列名             | 説明                                                                                         |
|------------------|----------------------------------------------------------------------------------------------|
| file_name        | CSVファイル名（ベース名）                                                                    |
| feature          | FEATURE名                                                                                    |
| condition_column | 誤検知条件列名                                                                              |
| condition_value  | 誤検知条件値                                                                                |
| bin_min          | bin の下端値                                                                                |
| bin_max          | bin の上端値                                                                                |
| fp_count         | 当該ファイルにおける、誤検知条件に一致したフレーム値のうち、その bin に入った件数（重複なし） |
| fp_ratio         | `fp_count / 当該ファイルの誤検知フレーム値分布で採用された全有効フレーム値件数`              |
- `fp_count` の分母となる「当該ファイルの全有効フレーム値件数」は、
  当該ファイルの誤検知条件に一致し、かつ vmin〜vmax 範囲内・非 NaN であったフレーム行数の総数とする。
- 誤検知分布は、通常の連続値FEATURE分布と同じ bin 定義・ NaN／範囲外値の扱いを用いる（5.1.1 に準拠）。
---
## 7. CLI 仕様
`analyze_accel_distribution.py` の引数仕様:
- `--csv`, `-i` (str, required)
  単一 CSV ファイル、または CSV を含むディレクトリのパス。
- `--csvdir`, `-d` (str, default="")
  CSV ディレクトリのパス。指定時は `--csv` より優先。
- `--pattern` (str, default="*.csv")
  ディレクトリ探索時の glob パターン。
- `--hparams` (str, default="config/hyperparams_common.json")
  `seq_len` 等が定義されたハイパーパラメータ JSON のパス。
- `--feature` (str, default="accelpedalangle")
  解析対象 FEATURE 名（単一列）。
- `--output-dir` (str, default="AE/outputs")
  Excel 出力先ディレクトリ。
- `--output-prefix` (str, default="accel_dist")
  出力ファイル名のプレフィックス。
- `--use-training-preproc` (bool, flag)
  指定時 True:
  - 既存前処理 (`preprocess_df_for_training`) を利用する。
  指定しない場合 False:
  - `read_csv_lower` + `require_columns` のみ行い、生値を上述のルールで扱う。
- `--filtered-mode` (bool, flag)
  指定時 True:
  - 条件列・条件値に完全一致した行だけを対象にした filtered 集計を追加実行し、
    `SummaryFiltered` / `PerFileFiltered` シートを出力する。
  - さらに、異常判定フレーム（filtered 条件一致フレーム）の Time 分布を集計し、
    `SummaryFilteredTime` / `PerFileFilteredTime` シートを出力する。
  - また、誤検知フレーム（同じ filtered 条件一致フレーム）における FEATURE 分布を集計し、
    `SummaryFalseAccel` / `PerFileFalseAccel` シートを出力する。
- `--filter-column` (str, default="is_anomaly")
  filtered 集計で使用する条件列名。
  誤検知分布でも、この列・値の組み合わせが「誤検知条件」として使用される。
- `--filter-value` (str, default="1")
  filtered 集計で使用する条件値。
- `--time-column` (str, default="time")
  異常判定フレームの Time を表す列名。
  - Time 列は **実数値（float など）** として扱う。
  - 負の値も許容し、絶対値に変換したりはしない（**実数値そのもの** を使用する）。
---
## 8. ディレクトリ構成
プロジェクトルートから見て、構成は以下を前提とする。
```text
PROJECT_ROOT/
  AE/
    outputs/
      （解析結果の .xlsx を出力）
  scripts/
    analyze_accel_distribution.py
    analyze_accel_distribution.md
  config/
    hyperparams_common.json
    （その他設定ファイル）
  1_transformer/
  app/
  artifacts/
  datarecode_test/
  datarecode_train/
  docs/
  output/
  result/
  02_20260213_dataset/
  requirements-ae-core.txt
  requirements-ae-carla.txt

