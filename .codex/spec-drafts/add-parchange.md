# threshold 分布統計拡張 仕様草案

## ステータス

- 2026-06-22 に承認済み仕様へ昇格済み。
- 昇格先は `docs/specs/overview.md`, `docs/specs/requirements.md`, `docs/specs/design.md`, `docs/specs/decision_log.md`。
- test-designer は本草案をテスト可能かつ昇格 OK と判断し、人間判断待ちの未確定事項は残っていない。

## 適用される承認済み仕様

- `docs/specs/overview.md`
  - 最終採用モデルに対して threshold を再算出する。
  - 既存 CLI 互換、既存設定参照、既存出力契約は維持対象。
- `docs/specs/requirements.md`
  - R-007: threshold は最終採用モデルに対して再算出する。
  - 非機能要件: 既存 CLI 引数、設定参照先、出力契約を可能な限り維持する。
- `docs/specs/design.md`
  - 学習完了モデルで threshold を再算出し、artifacts に保存する。
  - `threshold.json` を threshold 保存先とする。

## 現行挙動

- 本流の学習処理は `threshold.json` に次の threshold/stat 値を保存する。
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
- Nstep 版は上記に加えて `tail_steps` を保存する。
- 推論処理は既存 artifact の `threshold.json` を読み込み、少なくとも `threshold`, `mean`, `std`, `p10`, `p50`, `p90`, `p99`, `temperature` を使って正規化スコアと判定用しきい値を構築する。
- 旧 CLI 互換のため、`1_transformer/` 配下にも同等の学習・推論処理が残っている。

## 要求

threshold/stat handling に、保存 artifact の分布統計として次の分位点を追加する。

- p95
- p99.5
- p99.9
- p99.99
- p99.999

## 仕様案

### S-001: 保存される追加フィールド

新規作成される `threshold.json` は、既存フィールドを維持したうえで次のキーを追加する。

| 分位点 | `threshold.json` キー |
| --- | --- |
| p95 | `p95` |
| p99.5 | `p99_5` |
| p99.9 | `p99_9` |
| p99.99 | `p99_99` |
| p99.999 | `p99_999` |

小数分位点は JSON キー上では小数点を `_` に置換する。既存の `p10`, `p50`, `p90`, `p99` と同じ prefix 形式を維持し、ドット付きキーによる参照互換性リスクを避けるためである。

### S-002: 算出元と算出方法

- 追加フィールドは、既存の `p10`, `p50`, `p90`, `p99`, `threshold` と同じ MAE 分布から算出する。
- 通常版では現行 `compute_threshold_on_dataset` が収集する last-step MAE 分布を使う。
- Nstep 版では現行 `compute_threshold_on_dataset` が `tail_steps` を反映して収集する tail MAE 分布を使う。
- 分位点の算出方式は、現行 `np.percentile` による既存 percentile 算出と同じ方式に揃える。
- 各値は JSON に保存可能な Python `float` として保存する。

### S-003: 後方互換性

- 既存フィールド名、既存フィールド値の意味、`threshold.json` の保存先は変更しない。
- `threshold` は引き続き既存の `percentile` 設定値に基づく判定しきい値であり、追加分位点の導入だけを理由に選択規則を変えない。
- `temperature` は引き続き既存の `p90` と `p50` から算出する。追加分位点の導入だけを理由に `temperature` の式を変えない。
- 推論時の `y_conv_score` 正規化範囲は引き続き既存の `p10` から `p99` を使う。追加分位点の導入だけを理由に `p99_999` などへ変更しない。
- 追加フィールドがない既存 artifact も推論で読み込めること。追加フィールドの欠落を理由にエラーにしない。
- 追加フィールドがない既存 artifact に対して、分布が保存されていない高分位点を `mean/std` や `threshold` から推定して保存済み統計として扱わない。

### S-004: 対象範囲

本変更の対象範囲は次の artifact writer / loader 互換である。

- 本流通常版の学習 artifact writer
  - `src/gofumi_ae/training/standard.py`
- 本流 Nstep 版の学習 artifact writer
  - `src/gofumi_ae/training/nstep.py`
- 本流通常版の推論 stats loader / context 構築
  - `src/gofumi_ae/inference/standard.py`
- 本流 Nstep 版の推論 stats loader / context 構築
  - `src/gofumi_ae/inference/nstep.py`
- 旧 CLI 互換通常版の artifact writer / stats loader
  - `1_transformer/train_transformer_autoencoder.py`
  - `1_transformer/train_score_csv.py`
- 旧 CLI 互換 Nstep 版の artifact writer / stats loader
  - `1_transformer/train_transformer_autoencoder_Nstep.py`
  - `1_transformer/train_score_csv_Nstep.py`

### S-005: 推論 loader の扱い

- 推論 loader は、新しい追加フィールドが存在する場合に `float` として読み込めること。
- 推論 loader が threshold/stat context を返す場合、存在する追加フィールドを同じキー名で context に含めること。
- 追加フィールドの有無は、既存の異常判定、`y_conv_threshold`, `y_pre_threshold`, `y_conv_score`, `y_pre_score` の算出結果を変えないこと。

## 非目標

- CLI 引数、対話 UI、設定キーを追加または変更しない。
- `threshold` の percentile 選択規則を変えない。
- 推論の異常判定式、正規化式、EWMA、連続点判定を変えない。
- `threshold.json` 以外の artifact 形式を、この変更だけを理由に変更しない。
- README、依存定義、Docker/CUDA/OS 要件を変更しない。
- 本体コードやテストは本仕様草案フェーズでは編集しない。

## 受け入れ条件案

- AC-001: 本流通常版の学習 artifact 作成後、`threshold.json` に既存キーを維持したまま `p95`, `p99_5`, `p99_9`, `p99_99`, `p99_999` が保存される。
- AC-002: 本流 Nstep 版の学習 artifact 作成後、`threshold.json` に既存キーと `tail_steps` を維持したまま `p95`, `p99_5`, `p99_9`, `p99_99`, `p99_999` が保存される。
- AC-003: 旧 CLI 互換の通常版および Nstep 版の学習 artifact 作成後も、本流と同じ追加キーが `threshold.json` に保存される。
- AC-004: 追加キーの値は、同一 MAE 分布に対する `np.percentile` の 95, 99.5, 99.9, 99.99, 99.999 percentile と一致する。
- AC-005: 同一 MAE 分布から算出される `p99`, `p99_5`, `p99_9`, `p99_99`, `p99_999` は、数値誤差を除き単調非減少である。
- AC-006: `percentile=99.5` で threshold を算出する既定相当のケースでは、`threshold` と `p99_5` が同一 MAE 分布・同一 percentile 算出方式に基づく値として一致する。
- AC-007: 追加キーが存在しない既存 `threshold.json` を、本流通常版・本流 Nstep 版・旧 CLI 互換版の推論 loader がエラーにせず読み込める。
- AC-008: 追加キーが存在する `threshold.json` を推論 loader が読み込む場合、追加キーは `float` として threshold/stat context に保持される。
- AC-009: 追加キーの有無によって、既存の `threshold`, `p10`, `p50`, `p90`, `p99`, `temperature` に基づく `y_conv_score`, `y_conv_threshold`, `y_pre_threshold`, `is_anomaly` の結果が変わらない。
- AC-010: 本変更により CLI help、対話起動、既存 artifact の保存先、既存 CSV 出力列、依存管理、実行環境要件は変更されない。

## test-designer へ渡す確認観点

- 上記キー名でテストケース化できるか。
- 小数分位点のキー名を `p99_5` 形式にする仕様が、artifact contract として十分に観測可能か。
- 既存 artifact 欠落時の後方互換性を、推論 loader 単体または最小推論経路で検証できるか。
- 追加キーが判定挙動へ影響しないことを、既存 score normalization の比較で検証できるか。

## 昇格可否

- test-designer の昇格 OK 判定に基づき、2026-06-22 に `docs/specs/**` へ昇格済み。
- 人間判断待ちの未確定仕様はない。
- 承認済み仕様、テスト計画、テストケース定義、受け入れ条件対応表が揃ったため、`UNLOCK:IMPLEMENT` 待ちへ移行可能。
