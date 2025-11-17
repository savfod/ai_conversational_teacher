import json
from pathlib import Path

import appdirs

APP_NAME = "conversa"
DEFAULT_DATA_DIR = Path(appdirs.user_data_dir(APP_NAME))
DEFAULT_MISTAKES_FILE = DEFAULT_DATA_DIR / "mistakes.jsonl"


def append_to_jsonl_file(data: dict, fpath: Path) -> None:
    """Append a dictionary as a JSON object to a JSONL file.
    Args:
        data: Dictionary to append as a JSON object.
        fpath: Path to the JSONL file.
    """
    fpath.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(fpath, "a") as f:
            json.dump(data, f)
            f.write("\n")
    except IOError as e:
        print(f"Error saving data to file {fpath}: {e}")
