import queue
import tkinter as tk
from tkinter import messagebox
from typing import Any

from audio_processor import create_audio_processor
from gui import VoiceWriterGUI
from hotkey import create_keyboard_hotkey
from recognition import RecognitionSession
from text_output import create_clipboard_paster
from transcription_processor import TranscriptionProcessorFactory


def build_gui(root: tk.Misc) -> VoiceWriterGUI:
    events: "queue.Queue[Any]" = queue.Queue()
    hotkey = create_keyboard_hotkey()
    paster = create_clipboard_paster()
    audio_processor = create_audio_processor()
    session = RecognitionSession(audio_processor, hotkey, paster, emit=events.put)
    return VoiceWriterGUI(root, audio_processor, session, TranscriptionProcessorFactory(), events)


def main() -> None:
    root = tk.Tk()
    try:
        build_gui(root)
    except Exception as e:
        # 依存ライブラリや音声デバイスの初期化に失敗した理由を、コンソールなしの起動でも見えるようにする
        root.withdraw()
        messagebox.showerror("VoiceWriter", f"起動できませんでした。\n{type(e).__name__}: {e}")
        root.destroy()
        raise SystemExit(1) from e
    root.mainloop()


if __name__ == "__main__":
    main()
