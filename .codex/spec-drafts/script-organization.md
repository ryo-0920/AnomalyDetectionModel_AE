# スクリプト整理 仕様草案

## 目的

今のプロジェクトにある学習、推論、評価、可視化、補助スクリプトを、チーム標準に近い Python パッケージ構成へ整理する。新しいパッケージ名は `gofumi_ae` とし、既存利用者への影響を避けるため、旧 CLI パスは互換 wrapper として残す。

## 現行構成の要約

- `1_transformer/`: Transformer AE の学習、推論、評価、可視化 CLI とモデル実装が混在している。
- `1_transformer/models/`: モデル実装を格納している。
- `app/`: tagged dataset、対話 UI、CARLA/データ収集、動画変換、Excel 出力などの共通処理と補助スクリプトが混在している。
- `config/`: ハイパーパラメータ、tagged dataset、推論評価用の設定を格納している。

## 整理後の目標構成案

```text
src/gofumi_ae/
  cli/
    train.py
    score.py
    evaluate.py
    plot_timechart.py
  models/
  training/
  inference/
  evaluation/
  visualization/
  datasets/
  ui/
```

Nstep 版は本流へ統合し、通常版と Nstep 版をスクリプト実行時に使い分けられる構成にする。統合後は、新パッケージ側の CLI または設定切替で両モードを選択できることを前提とする。

CARLA/動画/Excel 補助スクリプトは正式機能ではなく実験用として扱い、次のように `experiments/` 配下へ整理する前提とする。

```text
experiments/
tools/
```

`tools/` は正式サポート対象の汎用補助機能を置く候補として残すが、CARLA/動画/Excel 補助スクリプトはここへ置かない。

## 互換 wrapper 方針

- 以下の旧 CLI パスは削除せず、薄い wrapper として残す。
  - `1_transformer/train_transformer_autoencoder.py`
  - `1_transformer/train_score_csv.py`
  - `1_transformer/plot_timechart.py`
  - `1_transformer/eval_score_csv.py`
- wrapper は新パッケージ側の `main()` に処理を委譲する。
- 旧パスから実行しても、既存の CLI 引数、`--help`、対話起動、既定 config パス、出力先、出力形式を維持する。
- `config/hyperparams_common.json` などの既定パスは、移動後もプロジェクトルート基準で解決する。
- 旧 CLI 互換は、将来ほかのモデルを追加する可能性を考慮し、当面維持する。
- Nstep 系の旧スクリプトが残る場合も、最終的には本流実装への互換入口として扱い、内部では通常版と Nstep 版の選択可能な共通実装へ委譲する。

## Git 管理方針

- `__pycache__/` と `desktop.ini` は成果物や補助ファイルとして扱い、Git 管理対象から外す方針とする。
- 既に tracked されている `__pycache__/` や `desktop.ini` は、実装フェーズで差分を確認できる形で Git 管理から外す。

## 非目標

- 学習、推論、評価、可視化ロジックの意味を変えない。
- CLI 引数、設定キー、出力 CSV の列名、出力フォルダ契約を変えない。
- tagged dataset のフィルタ仕様や sampling 仕様を変えない。
- 依存管理方式や実行環境を、この整理だけを理由に変更しない。
- CARLA/動画/Excel 補助スクリプトを正式サポート機能へ昇格しない。
- `docs/specs/`、本体コード、tests、README は、この草案作成時点では編集しない。

## 受け入れ条件案

- `python 1_transformer/train_transformer_autoencoder.py --help` が成功する。
- `python 1_transformer/train_score_csv.py --help` が成功する。
- `python 1_transformer/plot_timechart.py --help` が成功する。
- `python 1_transformer/eval_score_csv.py --help` が成功する。
- 旧 CLI パスから起動しても既定 config パスが変わらない。
- `train_score_csv.py` の既存出力契約を維持する。
- 新パッケージの主要モジュールが import できる。
- 通常版と Nstep 版をスクリプト実行時に切り替えられる。
- wrapper 方針と cache/output の Git 管理方針が記録されている。
- CARLA/動画/Excel 補助スクリプトが実験用として `experiments/` 配下へ整理される前提が記録されている。
