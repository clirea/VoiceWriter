import sys
from unittest.mock import MagicMock, call, patch

import pytest

from text_output import ClipboardPaster, create_clipboard_paster


@pytest.fixture
def operations():
    # copyとhotkeyの呼び出し順を1つのモックで追う
    return MagicMock()


@pytest.fixture
def paster(operations):
    operations.get_active_window.return_value.title = "メモ帳"
    return ClipboardPaster(operations.copy, operations.hotkey, operations.get_active_window)


class TestPaste:
    def test正常系_pasteメソッドがコピーしてから貼り付けキーを送ること(self, paster, operations):
        paster.paste("こんにちは")

        assert operations.mock_calls[:2] == [call.copy("こんにちは"), call.hotkey("ctrl", "v")]

    def test正常系_pasteメソッドが貼り付け先のウィンドウタイトルを返すこと(self, paster):
        assert paster.paste("こんにちは") == "メモ帳"

    def test正常系_pasteメソッドが指定された修飾キーで貼り付けること(self, operations):
        paster = ClipboardPaster(operations.copy, operations.hotkey, paste_modifier="command")

        paster.paste("こんにちは")

        operations.hotkey.assert_called_once_with("command", "v")

    def test正常系_pasteメソッドがウィンドウ取得手段がない場合にNoneを返すこと(self, operations):
        paster = ClipboardPaster(operations.copy, operations.hotkey)

        assert paster.paste("こんにちは") is None

    def test正常系_pasteメソッドがアクティブウィンドウがない場合にNoneを返すこと(self, paster, operations):
        operations.get_active_window.return_value = None

        assert paster.paste("こんにちは") is None

    def test正常系_pasteメソッドがウィンドウ取得の失敗を貼り付け失敗として扱わないこと(self, paster, operations):
        operations.get_active_window.side_effect = RuntimeError("window error")

        assert paster.paste("こんにちは") is None

    def test異常系_pasteメソッドがクリップボードへコピーできない場合に例外を送出すること(self, paster, operations):
        operations.copy.side_effect = RuntimeError("clipboard error")

        with pytest.raises(RuntimeError):
            paster.paste("こんにちは")

    def test異常系_pasteメソッドがコピーに失敗した場合に貼り付けキーを送らないこと(self, paster, operations):
        operations.copy.side_effect = RuntimeError("clipboard error")

        with pytest.raises(RuntimeError):
            paster.paste("こんにちは")

        operations.hotkey.assert_not_called()


class TestCreateClipboardPaster:
    @pytest.fixture
    def modules(self):
        pyautogui = MagicMock()
        pyperclip = MagicMock()
        with patch.dict(sys.modules, {"pyautogui": pyautogui, "pyperclip": pyperclip}):
            yield MagicMock(pyautogui=pyautogui, pyperclip=pyperclip)

    def test正常系_create_clipboard_pasterメソッドがWindowsでCtrlを貼り付けキーにすること(self, modules):
        assert create_clipboard_paster("win32").paste_modifier == "ctrl"

    def test正常系_create_clipboard_pasterメソッドがmacOSでCommandを貼り付けキーにすること(self, modules):
        assert create_clipboard_paster("darwin").paste_modifier == "command"

    def test正常系_create_clipboard_pasterメソッドがpyperclipのコピーを使うこと(self, modules):
        assert create_clipboard_paster("win32").copy is modules.pyperclip.copy

    def test正常系_create_clipboard_pasterメソッドがpyautoguiのキー送信を使うこと(self, modules):
        assert create_clipboard_paster("win32").hotkey is modules.pyautogui.hotkey

    def test正常系_create_clipboard_pasterメソッドがgetActiveWindowのない環境でウィンドウ取得を無効にすること(
        self, modules
    ):
        del modules.pyautogui.getActiveWindow

        assert create_clipboard_paster("darwin").get_active_window is None

    def test異常系_create_clipboard_pasterメソッドがpyautogui未導入でImportErrorを送出すること(self):
        with patch.dict(sys.modules, {"pyautogui": None}):
            with pytest.raises(ImportError):
                create_clipboard_paster("win32")
