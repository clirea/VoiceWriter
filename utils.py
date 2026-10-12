import json
import os
import tempfile
from typing import Any

from config import CONFIG_FILE


def load_config(path: str = CONFIG_FILE) -> dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError):
        # 壊れた設定で起動不能にしない。終了時に上書きされるので元ファイルは退避する
        _backup_broken_config(path)
        return {}
    if not isinstance(config, dict):
        _backup_broken_config(path)
        return {}
    return config


def save_config(config: dict[str, Any], path: str = CONFIG_FILE) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    fd, temp_path = tempfile.mkstemp(dir=directory, prefix=".voicewriter_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        # 書き込み途中で落ちても既存の設定を壊さない
        os.replace(temp_path, path)
    except BaseException:
        _remove_quietly(temp_path)
        raise


def mask_api_key(api_key: str) -> str:
    if len(api_key) <= 8:
        return "*" * len(api_key)
    return api_key[:4] + "*" * (len(api_key) - 8) + api_key[-4:]


def _backup_broken_config(path: str) -> None:
    try:
        backup_path = _reserve_backup_path(path)
    except OSError:
        return
    try:
        os.replace(path, backup_path)
    except OSError:
        _remove_quietly(backup_path)


def _reserve_backup_path(path: str) -> str:
    """過去の退避を上書きしない退避先（.broken、.broken.1、.broken.2 ...）を空ファイルとして確保する"""
    number = 0
    while True:
        candidate = f"{path}.broken" if number == 0 else f"{path}.broken.{number}"
        try:
            # 排他作成で名前を確保してから置き換えるので、既存の退避には書き込まない
            os.close(os.open(candidate, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            number += 1
            continue
        return candidate


def _remove_quietly(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass
