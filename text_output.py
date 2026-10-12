import sys
from typing import Any, Callable, Optional


class ClipboardPaster:
    def __init__(
        self,
        copy: Callable[[str], Any],
        hotkey: Callable[..., Any],
        get_active_window: Optional[Callable[[], Any]] = None,
        paste_modifier: str = "ctrl",
    ) -> None:
        self.copy = copy
        self.hotkey = hotkey
        self.get_active_window = get_active_window
        self.paste_modifier = paste_modifier

    def paste(self, text: str) -> Optional[str]:
        """クリップボード経由で貼り付け、貼り付け先のウィンドウタイトル（取得できなければNone）を返す"""
        self.copy(text)
        self.hotkey(self.paste_modifier, "v")
        return self._active_window_title()

    def _active_window_title(self) -> Optional[str]:
        if self.get_active_window is None:
            return None
        try:
            active_window = self.get_active_window()
        except Exception:
            # タイトル取得はログ用。貼り付け自体は済んでいるので失敗扱いにしない
            return None
        return getattr(active_window, "title", None) or None


def create_clipboard_paster(platform: str = sys.platform) -> ClipboardPaster:
    import pyautogui
    import pyperclip

    return ClipboardPaster(
        copy=pyperclip.copy,
        hotkey=pyautogui.hotkey,
        # getActiveWindowはpyautoguiがWindowsでのみ提供する
        get_active_window=getattr(pyautogui, "getActiveWindow", None),
        paste_modifier="command" if platform == "darwin" else "ctrl",
    )
