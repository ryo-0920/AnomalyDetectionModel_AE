import torch
import numpy as np
# 1) モデル読み込み（あなたのモデル読み込みコードに置き換えてください）
model = torch.load("artifacts/transformer_ae/model.pt")  # 学習済みモデルをロード
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
# 2) 検証データローダ（バッチで回す）
val_loader = ...  # DataLoader that yields batches: inputs shape (B, T, C) etc.
errors = []  # 各シーケンスの誤差を貯める
with torch.no_grad():
    for batch in val_loader:
        inputs = batch["inputs"].to(device)  # 実装に合わせて取得
        outputs = model(inputs)              # モデルの出力形状に合わせる
        # 例: 再構成誤差をシーケンス内で MSE 平均して 1 スカラーにする
        # ここでは per-sample mse: ( (outputs - inputs)**2 ).mean(dim=[1,2]) 等
        per_sample_mse = ((outputs - inputs) ** 2).mean(dim=[1,2])  # adjust dims as needed
        errors.append(per_sample_mse.cpu().numpy())
errors = np.concatenate(errors, axis=0)  # shape = (N_val,)
# 3) 99 パーセンタイルを計算
p = 99
threshold_99 = np.percentile(errors, p)
print(f"{p}th percentile threshold = {threshold_99:.6g}")
# 4) 閾値を保存
import json
with open("thresholds.json", "w", encoding="utf-8") as f:
    json.dump({"percentile": p, "threshold": float(threshold_99)}, f, ensure_ascii=False, indent=2)