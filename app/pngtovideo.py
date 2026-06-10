import cv2
import os
import glob
import re
import csv

def pngs_to_video(input_folder, output_file="output.mp4", fps=30):
    # フォルダ内の .png ファイルを取得（ソートして順序を揃える）
    images = sorted(glob.glob(os.path.join(input_folder, "*.png")))

    if not images:
        print("❌ 画像が見つかりません")
        return

    # 最初の画像からサイズを取得
    frame = cv2.imread(images[0])
    height, width, _ = frame.shape
    # テキスト（フレーム番号）描画用の設定
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, min(2.0, height / 480.0))
    text_color = (255, 255, 255)
    shadow_color = (0, 0, 0)
    thickness = 1
    shadow_thickness = 3
    org = (10, height - 20)

    # 入力フォルダ内のCSVから、強調表示するフレーム番号を収集
    # 条件: intent 列が "gofum" または "intentional_accel"
    highlight_frames = set()
    try:
        csv_path = os.path.join(input_folder, "intentional_accel.csv")
        if not os.path.isfile(csv_path):
            csv_candidates = sorted(glob.glob(os.path.join(input_folder, "*.csv")))
            csv_path = csv_candidates[0] if csv_candidates else None
        if csv_path and os.path.isfile(csv_path):
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # 小文字の列名で扱う
                    row_l = { (k or "").strip().lower(): (v.strip() if isinstance(v, str) else v) for k, v in row.items() }
                    intent = (row_l.get("intent") or "").lower()
                    frame_s = row_l.get("frame")
                    try:
                        frame_i = int(frame_s) if frame_s is not None else None
                    except Exception:
                        frame_i = None
                    if frame_i is None:
                        continue
                    if intent in ("gofumi", "intentional_accel"):
                        highlight_frames.add(frame_i)
    except Exception:
        highlight_frames = set()

    # 動画ライター作成
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # mp4 出力
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # 画像を1枚ずつ動画に追加
    for idx, img_path in enumerate(images):
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ 読み込めない画像: {img_path}")
            continue
        # ファイル名からフレーム番号を抽出（例: frame_000123.png）。見つからなければ連番を使用
        base = os.path.basename(img_path)
        m = re.search(r"(\d+)(?=\.png$)", base)
        frame_no = m.group(1) if m else str(idx)
        try:
            frame_no_int = int(frame_no)
        except Exception:
            frame_no_int = None
        text = f"frame {frame_no}"
        # 読みやすいように影→本体の順で描画
        cv2.putText(img, text, org, font, font_scale, shadow_color, shadow_thickness, cv2.LINE_AA)
        main_color = (0, 0, 255) if (frame_no_int is not None and frame_no_int in highlight_frames) else text_color
        cv2.putText(img, text, org, font, font_scale, main_color, thickness, cv2.LINE_AA)
        out.write(img)

    out.release()
    print(f"✅ 動画を書き出しました: {output_file}")


if __name__ == "__main__":
    # ここに対象フォルダを指定する
    pngs_to_video(r"C:\Users\user\Desktop\Work\carla\Gofumi\datarecode_test", "output.mp4", fps=30)
