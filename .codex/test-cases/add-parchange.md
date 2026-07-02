# threshold 分布統計拡張 テストケース定義

## 共通データ

代表 MAE 分布:

```text
[0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0]
```

`np.percentile` 既定方式での期待値:

| key | percentile | expected |
| --- | ---: | ---: |
| `p10` | 10 | 0.55 |
| `p50` | 50 | 12.0 |
| `p90` | 90 | 243.20000000000005 |
| `p95` | 95 | 371.1999999999998 |
| `p99` | 99 | 483.84000000000015 |
| `p99_5` | 99.5 | 497.9200000000001 |
| `p99_9` | 99.9 | 509.1840000000002 |
| `p99_99` | 99.99 | 511.71839999999975 |
| `p99_999` | 99.999 | 511.9718399999997 |

追加 percentile key:

- `p95`
- `p99_5`
- `p99_9`
- `p99_99`
- `p99_999`

既存 key:

- `threshold`
- `mean`
- `std`
- `p10`
- `p50`
- `p90`
- `p99`
- `percentile`
- `temperature`
- `n_samples`
- `score_policy`
- Nstep のみ `tail_steps`

## TC-01 本流 training threshold 統計 contract

- 目的: 本流通常版・Nstep 版の threshold 算出結果に追加 percentile key が含まれ、同一 MAE 分布の `np.percentile` 値と一致することを確認する。
- 対応: AC-001, AC-002, AC-004, AC-005, AC-006
- 対象候補:
  - `src/gofumi_ae/training/standard.py::compute_threshold_on_dataset`
  - `src/gofumi_ae/training/nstep.py::compute_threshold_on_dataset`
- 入力:
  - 代表 MAE 分布
  - `percentile=99.5`
  - Nstep は `tail_steps=3`
- 前提:
  - `collect_mae_distribution` を固定 MAE 分布へ差し替え、実モデル・実 DataLoader・実 CSV には依存しない。
- 期待値:
  - 既存 key が維持される。
  - 追加 key がすべて存在する。
  - 各追加 key の値が共通データの expected と `np.isclose` で一致する。
  - `p99 <= p99_5 <= p99_9 <= p99_99 <= p99_999`。
  - `threshold == p99_5`。
  - Nstep では `tail_steps == 3` が維持される。
- 判定方法:
  - `unittest` と `unittest.mock.patch` で対象 module の分布収集を差し替え、戻り値辞書を検査する。

## TC-02 旧 CLI 互換 training threshold 統計 contract

- 目的: 旧 CLI 互換の通常版・Nstep 版でも、本流と同じ追加 percentile key contract が成立することを確認する。
- 対応: AC-003, AC-004, AC-005, AC-006
- 対象候補:
  - `1_transformer/train_transformer_autoencoder.py`
  - `1_transformer/train_transformer_autoencoder_Nstep.py::compute_threshold_on_dataset`
- 入力:
  - 代表 MAE 分布
  - `percentile=99.5`
  - Nstep は `tail_steps=3`
- 前提:
  - 現行の旧 CLI 通常版は wrapper で本流 `gofumi_ae.cli.train` へ委譲するため、通常版は wrapper の到達性と本流通常版 TC-01 で contract を確認する。
  - 旧 CLI Nstep 版は `compute_threshold_on_dataset` を直接検査する。
- 期待値:
  - 旧 CLI 通常版 wrapper の `--help` で `--variant` が利用可能で、標準 variant が本流通常版へ到達できる。
  - 旧 CLI Nstep 版の戻り値に追加 key がすべて存在し、期待値と一致する。
  - 旧 CLI Nstep 版で `threshold == p99_5`、かつ高分位点が単調非減少である。
- 判定方法:
  - wrapper は subprocess help smoke または import 経由で到達性を確認する。
  - Nstep は `importlib.util.spec_from_file_location` で module をロードし、TC-01 と同じ分布差し替えで検査する。

## TC-03 threshold.json artifact round-trip

- 目的: training 側で作成した threshold stats が `threshold.json` に JSON 保存され、既存 key と追加 key が欠落しないことを確認する。
- 対応: AC-001, AC-002, AC-003
- 対象候補:
  - `save_artifacts` を持つ本流通常版・本流 Nstep 版・旧 CLI Nstep 版
  - 旧 CLI 通常版は wrapper 経由の本流通常版保存として扱う
- 入力:
  - TC-01/TC-02 で得た threshold stats
  - 一時ディレクトリ
  - `state_dict()` を持つ最小 dummy model
  - joblib 保存可能な dummy scaler
- 期待値:
  - `threshold.json` が作成される。
  - JSON の key set に既存 key と追加 key が含まれる。
  - 追加 key の JSON 値は `float` として読み戻せる。
  - Nstep では `tail_steps` も保存される。
- 判定方法:
  - 一時ディレクトリに保存し、`json.load` で読み戻して key と型を検査する。

## TC-04 高分位点境界分布

- 目的: 高分位点の単調性と補間値の扱いが、分布形状に依存して崩れないことを確認する。
- 対応: AC-004, AC-005
- 入力:
  - 定数分布 `[7.0, 7.0, 7.0, 7.0]`
  - 同値を含む分布 `[0.0, 0.0, 1.0, 1.0, 10.0, 10.0, 100.0]`
- 期待値:
  - 定数分布では `p95 == p99 == p99_5 == p99_9 == p99_99 == p99_999 == 7.0`。
  - 同値を含む分布でも `np.percentile` 期待値と一致し、単調非減少である。
- 判定方法:
  - TC-01 と同じ差し替え方式で分布別 subTest を実行する。

## TC-05 既存 artifact の後方互換 loader

- 目的: 追加 key がない既存 `threshold.json` 相当の `thr_info` を、推論 loader/context 構築がエラーにせず扱えることを確認する。
- 対応: AC-007
- 対象候補:
  - `src/gofumi_ae/inference/standard.py::build_inference_context`
  - `src/gofumi_ae/inference/nstep.py::build_inference_context`
  - `1_transformer/train_score_csv.py`
  - `1_transformer/train_score_csv_Nstep.py::build_inference_context`
- 入力:
  - `threshold`, `mean`, `std`, `p10`, `p50`, `p90`, `p99`, `temperature`, `score_policy` を含み、追加 key を含まない `thr_info`
  - 最小 `cfg`
  - dummy scaler
  - dummy model load
- 前提:
  - 実モデル file には依存せず、`CausalTransformerAutoencoder`, `torch.load`, `load_state_dict` を mock する。
- 期待値:
  - context 構築が例外を送出しない。
  - `threshold`, `p10`, `p50`, `p90`, `p99`, `temperature`, `y_conv_threshold`, `y_pre_threshold` が計算される。
  - 追加 high-percentile key は context に合成されない。
- 判定方法:
  - `unittest.mock.patch` で model load を無害化し、戻り値 context を検査する。

## TC-06 追加 key あり artifact の loader context 保持

- 目的: 追加 key が存在する `thr_info` を推論 loader/context 構築が `float` として保持することを確認する。
- 対応: AC-008
- 入力:
  - TC-05 の `thr_info` に追加 key と代表 MAE 分布の期待値を加えたもの
- 期待値:
  - context に `p95`, `p99_5`, `p99_9`, `p99_99`, `p99_999` が含まれる。
  - 各値は `float` であり、`thr_info` の値と一致する。
- 判定方法:
  - TC-05 と同じ mock context 構築で戻り値を検査する。

## TC-07 追加 key の非影響性

- 目的: 追加 key の有無が既存 score normalization、threshold context、異常判定基準へ影響しないことを確認する。
- 対応: AC-009
- 入力:
  - 追加 key なしの `thr_info`
  - 同じ既存値に追加 key だけを加えた `thr_info`
  - 代表 MAE 値 `[0.0, 0.55, 12.0, 243.2, 483.84, 512.0]`
- 期待値:
  - 両 context の `threshold`, `p10`, `p50`, `p90`, `p99`, `temperature`, `y_conv_threshold`, `y_pre_threshold` が一致する。
  - `y_conv_score = clip((mae - p10) / max(p99 - p10, 1e-6), 0, 1)` が両 context で一致する。
  - `is_anomaly = mae > threshold` が両 context で一致する。
- 判定方法:
  - TC-05 と同じ mock context 構築後、同じ MAE 値で score と判定を比較する。

## TC-08 CLI/API 非変更 smoke

- 目的: 本変更が CLI help、公開起動経路、依存定義、既存 smoke の範囲を変えないことを確認する。
- 対応: AC-010
- 入力:
  - `python 1_transformer/train_transformer_autoencoder.py --help`
  - `python 1_transformer/train_score_csv.py --help`
  - `python -m unittest tests.test_script_organization_smoke`
- 期待値:
  - CLI help が成功する。
  - 既存 smoke が維持される。
  - `requirements.txt` に本変更だけを理由とした追加依存がない。
  - `threshold.json` 以外の artifact format や CSV 出力列を変更していないことを差分レビューで確認できる。
- 判定方法:
  - 実装フェーズで自動 smoke と差分レビューを組み合わせる。

## 実装前 red 想定

`tests/test_add_parchange_acceptance.py` を追加した直後、本体実装前は以下のいずれかで red になる想定である。

- training threshold stats に `p95`, `p99_5`, `p99_9`, `p99_99`, `p99_999` が存在しない。
- inference context が追加 key を保持しない。
- 旧 CLI Nstep 版の threshold stats に追加 key が存在しない。

最小 red 確認コマンド:

```sh
python -m unittest tests.test_add_parchange_acceptance
```

## 対象外

- フル学習ジョブによる artifact 作成は、最小受け入れテストでは実施しない。
- GPU/CUDA 固有検証は、この JSON contract 変更の必須受け入れ条件に含めない。
