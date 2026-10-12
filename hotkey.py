import subprocess
import sys
from typing import Any, Callable

from config import VoiceWriterError


class InvalidShortcutError(VoiceWriterError):
    pass


class KeyboardBackendError(VoiceWriterError):
    pass


class KeyboardHotkey:
    def __init__(self, keyboard_module: Any) -> None:
        self.keyboard = keyboard_module

    def validate(self, shortcut_key: str) -> None:
        if not shortcut_key.strip():
            raise InvalidShortcutError("ショートカットキーが入力されていません。")
        try:
            self.keyboard.parse_hotkey(shortcut_key)
        except ValueError as e:
            raise InvalidShortcutError(f"ショートカットキー '{shortcut_key}' を解釈できません: {e}") from e

    def is_pressed(self, shortcut_key: str) -> bool:
        return bool(self.keyboard.is_pressed(shortcut_key))


def probe_keyboard_backend(
    platform: str = sys.platform,
    run: Callable[..., Any] = subprocess.run,
    executable: str = sys.executable,
) -> None:
    # Windows以外ではkeyboardのimportがプロセスごと異常終了する環境があるため、別プロセスで先に試す
    if platform == "win32":
        return
    try:
        result = run([executable, "-c", "import keyboard"], capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as e:
        raise KeyboardBackendError(f"キーボード監視ライブラリ keyboard を確認できません: {e}") from e
    if result.returncode != 0:
        raise KeyboardBackendError(
            "キーボード監視ライブラリ keyboard をこの環境で読み込めません"
            f"（終了コード {result.returncode}）。VoiceWriterはWindowsでの動作を前提としています。"
        )


def create_keyboard_hotkey() -> KeyboardHotkey:
    probe_keyboard_backend()
    import keyboard

    return KeyboardHotkey(keyboard)
