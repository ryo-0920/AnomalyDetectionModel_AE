from typing import Sequence


def prompt_int(message: str, min_value: int = 1, max_value: int = None, default: int = None) -> int:
    while True:
        raw = input(f"{message} ").strip()
        if raw == "" and default is not None:
            value = default
        else:
            try:
                value = int(raw)
            except ValueError:
                print("Please enter an integer value.")
                continue
        if value < min_value:
            print(f"Please enter a value >= {min_value}.")
            continue
        if max_value is not None and value > max_value:
            print(f"Please enter a value <= {max_value}.")
            continue
        return value


def prompt_text(message: str, default: str = None) -> str:
    raw = input(f"{message} ").strip()
    if raw == "" and default is not None:
        return default
    return raw


def choose_from_list(options: Sequence[str], title: str, default_index: int = 1) -> int:
    if not options:
        raise ValueError("options must not be empty")
    print(f"--- {title} ---")
    for idx, option in enumerate(options, start=1):
        print(f"{idx}: {option}")
    selected = prompt_int(
        f"Select number [default={default_index}]:",
        min_value=1,
        max_value=len(options),
        default=default_index,
    )
    return selected - 1
