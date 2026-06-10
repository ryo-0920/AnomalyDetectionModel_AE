import glob
import os
import sys
from typing import List, Optional, Sequence

from app.ui.prompts import choose_from_list, prompt_text

TTDC_ROOT = (
    "\\\\161.94.64.164"
    "\\\u5e02\u6280\u5831"
    "\\PMAR(\u6b21\u671f\u8aa4\u8e0f\u307f)"
    "\\\u4f9d\u983c\u6848\u4ef6"
    "\\Can300\u306b\u3088\u308b\u4e8b\u6545\u30c7\u30fc\u30bf\u89e3\u6790"
    "\\\u30bf\u30b0\u4ed8\u3051\u30c7\u30fc\u30bf"
    "\\\u691c\u8a0e\u30c7\u30fc\u30bftemp"
    "\\TTDC"
)
NETWORK_DATASET_DIRS = [
    os.path.join(TTDC_ROOT, "ver_tag001"),
    os.path.join(TTDC_ROOT, "ver_tag002"),
    os.path.join(TTDC_ROOT, "ver_tag003"),
]

TAGGED_DATASET_TOKEN = "__tagged_dataset__"
TAGGED_DATASET_LABEL = "tagged dataset (ledger + ver_tag001/2/3)"

_NETWORK_DATASET_DIRS_NORM = {os.path.normpath(p) for p in NETWORK_DATASET_DIRS}


def _require_tty() -> None:
    if sys.stdin is None or not sys.stdin.isatty():
        raise RuntimeError("Interactive mode requires a TTY terminal.")


def ensure_tty() -> None:
    _require_tty()


def prompt_optimizer(default_optimizer: str) -> str:
    _require_tty()
    default_opt = (default_optimizer or "adamw").strip().lower()
    if default_opt not in {"adamw", "radam"}:
        default_opt = "adamw"
    options = ["AdamW", "RAdam"]
    default_index = 1 if default_opt == "adamw" else 2
    selected = choose_from_list(options, "Select optimizer", default_index=default_index)
    return "adamw" if selected == 0 else "radam"


def _has_matching_csv(path: str, pattern: str, recursive: bool) -> bool:
    if recursive:
        for _ in glob.iglob(os.path.join(path, "**", pattern), recursive=True):
            return True
        return False
    return len(glob.glob(os.path.join(path, pattern))) > 0


def _is_network_dataset_path(path: str) -> bool:
    return os.path.normpath(path) in _NETWORK_DATASET_DIRS_NORM


def _is_dataset_candidate(path: str, pattern: str, recursive: bool = False) -> bool:
    if not path:
        return False
    if os.path.isfile(path):
        return path.lower().endswith(".csv")
    if os.path.isdir(path):
        return _has_matching_csv(path, pattern, recursive=recursive)
    return False


def _append_unique(candidates: List[str], seen: set, path: str) -> None:
    if not path:
        return
    normalized = os.path.normpath(path)
    if normalized in seen:
        return
    seen.add(normalized)
    candidates.append(normalized)


def prompt_path_with_manual(
    *,
    title: str,
    default_path: str,
    candidates: Sequence[str],
    manual_prompt: str,
    manual_label: str = "manual input",
    normalize: bool = True,
) -> str:
    _require_tty()
    all_candidates: List[str] = []
    seen = set()
    _append_unique(all_candidates, seen, default_path)
    for item in candidates:
        _append_unique(all_candidates, seen, item)

    if not all_candidates:
        selected = prompt_text(manual_prompt, default=default_path or "")
        return os.path.normpath(selected) if normalize else selected

    default_norm = os.path.normpath(default_path) if default_path else ""
    if default_norm in all_candidates:
        default_index = all_candidates.index(default_norm) + 1
    else:
        default_index = 1

    menu = [f"candidate: {path}" for path in all_candidates]
    menu.append(manual_label)
    chosen = choose_from_list(menu, title, default_index=min(default_index, len(menu)))

    if chosen == len(menu) - 1:
        selected = prompt_text(manual_prompt, default=default_norm)
    else:
        selected = all_candidates[chosen]

    return os.path.normpath(selected) if normalize else selected


def _discover_dataset_candidates(project_root: str, pattern: str) -> List[str]:
    candidates: List[str] = []
    seen = set()
    # Network paths are prioritized and scanned recursively.
    for net_root in NETWORK_DATASET_DIRS:
        if _is_dataset_candidate(net_root, pattern, recursive=True):
            _append_unique(candidates, seen, net_root)

    roots = [
        os.path.join(project_root, "datarecode_train"),
        os.path.join(project_root, "datarecode_test"),
        os.path.join(project_root, "02_20260213_dataset"),
    ]
    for root in roots:
        if not os.path.isdir(root):
            continue
        if _is_dataset_candidate(root, pattern):
            _append_unique(candidates, seen, root)
        try:
            for entry in sorted(os.listdir(root)):
                child = os.path.join(root, entry)
                if _is_dataset_candidate(child, pattern):
                    _append_unique(candidates, seen, child)
        except OSError:
            continue
    return candidates


def _discover_artifacts_candidates(project_root: str) -> List[str]:
    artifacts_root = os.path.join(project_root, "artifacts")
    if not os.path.isdir(artifacts_root):
        return []
    candidates: List[str] = []
    seen = set()
    for entry in sorted(os.listdir(artifacts_root)):
        child = os.path.join(artifacts_root, entry)
        if not os.path.isdir(child):
            continue
        required = [
            os.path.join(child, "config.json"),
            os.path.join(child, "threshold.json"),
            os.path.join(child, "scaler.pkl"),
            os.path.join(child, "model.pt"),
        ]
        if all(os.path.exists(p) for p in required):
            _append_unique(candidates, seen, child)
    return candidates


def _discover_result_targets(project_root: str) -> List[str]:
    candidates: List[str] = []
    seen = set()
    # Keep network candidates first (reachable + CSV exists).
    for net_root in NETWORK_DATASET_DIRS:
        if _is_dataset_candidate(net_root, "*.csv", recursive=True):
            _append_unique(candidates, seen, net_root)

    roots = [
        os.path.join(project_root, "result"),
        os.path.join(project_root, "datarecode_test"),
        os.path.join(project_root, "datarecode_train"),
    ]
    for root in roots:
        if os.path.exists(root):
            _append_unique(candidates, seen, root)
    return candidates


def prompt_training_dataset(
    default_path: str,
    project_root: str,
    pattern: str,
    extra_candidates: Optional[Sequence[str]] = None,
    include_tagged_option: bool = False,
) -> str:
    _require_tty()
    candidates: List[str] = []
    seen = set()
    # Keep network candidates at top.
    for net_root in NETWORK_DATASET_DIRS:
        _append_unique(candidates, seen, net_root)
    _append_unique(candidates, seen, default_path)
    if extra_candidates:
        for item in extra_candidates:
            _append_unique(candidates, seen, item)
    for discovered in _discover_dataset_candidates(project_root, pattern):
        _append_unique(candidates, seen, discovered)

    usable = []
    for c in candidates:
        recursive = _is_network_dataset_path(c)
        if _is_dataset_candidate(c, pattern, recursive=recursive):
            usable.append(c)
    default_norm = os.path.normpath(default_path) if default_path else ""
    if default_norm in usable:
        default_index = usable.index(default_norm) + 1
    else:
        default_index = 1

    menu = [f"candidate: {path}" for path in usable]
    if include_tagged_option:
        menu.append(TAGGED_DATASET_LABEL)
    menu.append("manual input")
    chosen = choose_from_list(menu, "Select training dataset", default_index=min(default_index, len(menu)))

    manual_index = len(menu) - 1
    tagged_index = (len(menu) - 2) if include_tagged_option else -1
    if include_tagged_option and chosen == tagged_index:
        return TAGGED_DATASET_TOKEN
    if chosen == manual_index:
        selected = prompt_text(
            f"Enter dataset path (folder or CSV) [default={default_norm}]:",
            default=default_norm,
        )
    else:
        selected = usable[chosen]

    selected = os.path.normpath(selected)
    if not os.path.exists(selected):
        print(f"[WARN] dataset path does not exist yet: {selected}")
    return selected


def prompt_artifacts_dir(default_path: str, project_root: str, extra_candidates: Optional[Sequence[str]] = None) -> str:
    candidates: List[str] = []
    if extra_candidates:
        candidates.extend([c for c in extra_candidates if c])
    candidates.extend(_discover_artifacts_candidates(project_root))
    selected = prompt_path_with_manual(
        title="Select artifacts directory",
        default_path=default_path,
        candidates=candidates,
        manual_prompt=f"Enter artifacts directory path [default={os.path.normpath(default_path)}]:",
        manual_label="manual input",
        normalize=True,
    )
    if not os.path.exists(selected):
        print(f"[WARN] artifacts path does not exist yet: {selected}")
    return selected


def prompt_csv_or_dir_or_glob(
    default_target: str,
    project_root: str,
    title: str,
    include_tagged_option: bool = False,
) -> str:
    candidates = _discover_result_targets(project_root)
    default_norm = os.path.normpath(default_target) if default_target else ""
    all_candidates: List[str] = []
    seen = set()
    for item in candidates:
        _append_unique(all_candidates, seen, item)
    if default_norm and default_norm not in seen:
        _append_unique(all_candidates, seen, default_norm)

    if default_norm in all_candidates:
        default_index = all_candidates.index(default_norm) + 1
    else:
        default_index = 1

    menu = [f"candidate: {path}" for path in all_candidates]
    if include_tagged_option:
        menu.append(TAGGED_DATASET_LABEL)
    menu.append("manual input (path or glob)")
    chosen = choose_from_list(menu, title, default_index=min(default_index, len(menu)))
    manual_index = len(menu) - 1
    tagged_index = (len(menu) - 2) if include_tagged_option else -1
    if include_tagged_option and chosen == tagged_index:
        return TAGGED_DATASET_TOKEN
    if chosen == manual_index:
        selected = prompt_text(
            f"Enter path or glob [default={default_target}]:",
            default=default_target,
        )
    else:
        selected = all_candidates[chosen]

    if not any(ch in selected for ch in "*?[]") and not os.path.exists(selected):
        print(f"[WARN] target path does not exist yet: {selected}")
    return selected
