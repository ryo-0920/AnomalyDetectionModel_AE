import csv
import glob
import os
import re

import cv2


def pngs_to_video(input_folder, output_file="output.mp4", fps=30):
    images = sorted(glob.glob(os.path.join(input_folder, "*.png")))

    if not images:
        print("No PNG files found.")
        return

    frame = cv2.imread(images[0])
    height, width, _ = frame.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, min(2.0, height / 480.0))
    text_color = (255, 255, 255)
    shadow_color = (0, 0, 0)
    thickness = 1
    shadow_thickness = 3
    org = (10, height - 20)

    highlight_frames = set()
    try:
        csv_path = os.path.join(input_folder, "intentional_accel.csv")
        if not os.path.isfile(csv_path):
            csv_candidates = sorted(glob.glob(os.path.join(input_folder, "*.csv")))
            csv_path = csv_candidates[0] if csv_candidates else None
        if csv_path and os.path.isfile(csv_path):
            with open(csv_path, "r", encoding="utf-8") as file_obj:
                reader = csv.DictReader(file_obj)
                for row in reader:
                    row_l = {
                        (key or "").strip().lower(): (
                            value.strip() if isinstance(value, str) else value
                        )
                        for key, value in row.items()
                    }
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

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    for idx, img_path in enumerate(images):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not load image: {img_path}")
            continue
        base = os.path.basename(img_path)
        match = re.search(r"(\d+)(?=\.png$)", base)
        frame_no = match.group(1) if match else str(idx)
        try:
            frame_no_int = int(frame_no)
        except Exception:
            frame_no_int = None
        text = f"frame {frame_no}"
        cv2.putText(
            img,
            text,
            org,
            font,
            font_scale,
            shadow_color,
            shadow_thickness,
            cv2.LINE_AA,
        )
        main_color = (
            (0, 0, 255)
            if frame_no_int is not None and frame_no_int in highlight_frames
            else text_color
        )
        cv2.putText(
            img,
            text,
            org,
            font,
            font_scale,
            main_color,
            thickness,
            cv2.LINE_AA,
        )
        out.write(img)

    out.release()
    print(f"Saved video: {output_file}")


if __name__ == "__main__":
    pngs_to_video(r"C:\Users\user\Desktop\Work\carla\Gofumi\datarecode_test", "output.mp4", fps=30)
