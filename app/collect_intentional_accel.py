from pathlib import Path
import sys


def _ensure_repo_root_on_sys_path():
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


def main():
    _ensure_repo_root_on_sys_path()
    from experiments.carla.collect_intentional_accel import main as relocated_main

    relocated_main()

if __name__ == "__main__":
    main()
