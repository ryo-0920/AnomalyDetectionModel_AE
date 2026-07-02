from pathlib import Path
import sys


def _ensure_repo_root_on_sys_path():
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


def pngs_to_video(input_folder, output_file="output.mp4", fps=30):
    _ensure_repo_root_on_sys_path()
    from experiments.media.pngtovideo import pngs_to_video as relocated_pngs_to_video

    return relocated_pngs_to_video(input_folder, output_file=output_file, fps=fps)


if __name__ == "__main__":
    pngs_to_video(r"C:\Users\user\Desktop\Work\carla\Gofumi\datarecode_test", "output.mp4", fps=30)
