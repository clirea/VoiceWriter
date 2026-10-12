import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

from hotkey import (
    InvalidShortcutError,
    KeyboardBackendError,
    KeyboardHotkey,
    create_keyboard_hotkey,
    probe_keyboard_backend,
)


@pytest.fixture
def keyboard_module():
    return MagicMock()


@pytest.fixture
def hotkey(keyboard_module):
    return KeyboardHotkey(keyboard_module)


class TestValidate:
    def test正常系_validateメソッドが解釈できるショートカットを受け入れること(self, hotkey, keyboard_module):
        hotkey.validate("ctrl+space")

        keyboard_module.parse_hotkey.assert_called_once_with("ctrl+space")

    def test異常系_validateメソッドが空のショートカットでInvalidShortcutErrorを送出すること(self, hotkey):
        with pytest.raises(InvalidShortcutError, match="入力されていません"):
            hotkey.validate("  ")

    def test異常系_validateメソッドが解釈できないショートカットでInvalidShortcutErrorを送出すること(
        self, hotkey, keyboard_module
    ):
        keyboard_module.parse_hotkey.side_effect = ValueError("Key 'nosuchkey' is not mapped")

        with pytest.raises(InvalidShortcutError, match="ctrl\\+nosuchkey"):
            hotkey.validate("ctrl+nosuchkey")


class TestIsPressed:
    def test正常系_is_pressedメソッドが押下中にTrueを返すこと(self, hotkey, keyboard_module):
        keyboard_module.is_pressed.return_value = True

        assert hotkey.is_pressed("ctrl+space") is True

    def test正常系_is_pressedメソッドが離されているときFalseを返すこと(self, hotkey, keyboard_module):
        keyboard_module.is_pressed.return_value = False

        assert hotkey.is_pressed("ctrl+space") is False

    def test正常系_is_pressedメソッドが指定されたショートカットを問い合わせること(self, hotkey, keyboard_module):
        hotkey.is_pressed("f9")

        keyboard_module.is_pressed.assert_called_once_with("f9")

    def test異常系_is_pressedメソッドが権限不足のOSErrorをそのまま送出すること(self, hotkey, keyboard_module):
        keyboard_module.is_pressed.side_effect = OSError("Error 13 - Must be run as administrator")

        with pytest.raises(OSError):
            hotkey.is_pressed("ctrl+space")


class TestProbeKeyboardBackend:
    def test正常系_probe_keyboard_backendメソッドがWindowsでは別プロセスを起動しないこと(self):
        run = MagicMock()

        probe_keyboard_backend(platform="win32", run=run)

        run.assert_not_called()

    def test正常系_probe_keyboard_backendメソッドがWindows以外でimportを別プロセスで試すこと(self):
        run = MagicMock(return_value=MagicMock(returncode=0))

        probe_keyboard_backend(platform="darwin", run=run, executable="python")

        run.assert_called_once_with(["python", "-c", "import keyboard"], capture_output=True, timeout=30)

    def test異常系_probe_keyboard_backendメソッドがimportの異常終了でKeyboardBackendErrorを送出すること(self):
        run = MagicMock(return_value=MagicMock(returncode=-6))

        with pytest.raises(KeyboardBackendError, match="-6"):
            probe_keyboard_backend(platform="darwin", run=run)

    def test異常系_probe_keyboard_backendメソッドがタイムアウトでKeyboardBackendErrorを送出すること(self):
        run = MagicMock(side_effect=subprocess.TimeoutExpired("python", 30))

        with pytest.raises(KeyboardBackendError):
            probe_keyboard_backend(platform="linux", run=run)


class TestCreateKeyboardHotkey:
    def test正常系_create_keyboard_hotkeyメソッドがkeyboardモジュールを渡すこと(self):
        module = MagicMock()
        with patch.dict(sys.modules, {"keyboard": module}), patch("hotkey.probe_keyboard_backend"):
            assert create_keyboard_hotkey().keyboard is module

    def test異常系_create_keyboard_hotkeyメソッドが事前確認の失敗時にKeyboardBackendErrorを送出すること(self):
        with patch("hotkey.probe_keyboard_backend", side_effect=KeyboardBackendError("読み込めません")):
            with pytest.raises(KeyboardBackendError):
                create_keyboard_hotkey()
