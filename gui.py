import queue
import tkinter as tk
from tkinter import ttk
from typing import Any, Callable, Dict

from config import (
    DEFAULT_PROCESSING_MODE,
    DEFAULT_SHORTCUT_KEY,
    DEFAULT_WHISPER_MODEL,
    ProcessingMode,
    WhisperModel,
)
from recognition import LogEvent, RecognitionRequest, StoppedEvent, TranscribedEvent
from utils import load_config, mask_api_key, save_config

EVENT_POLL_INTERVAL_MS = 50
AUDIO_RELEASE_TIMEOUT_SECONDS = 2.0


class VoiceWriterGUI:
    def __init__(
        self,
        master: tk.Misc,
        audio_processor: Any,
        session: Any,
        processor_factory: Any,
        events: "queue.Queue[Any]",
        load_config: Callable[[], Dict[str, Any]] = load_config,
        save_config: Callable[[Dict[str, Any]], None] = save_config,
    ) -> None:
        self.master = master
        master.title("VoiceWriter")

        self.audio_processor = audio_processor
        self.session = session
        self.processor_factory = processor_factory
        self.events = events
        self.save_config = save_config
        self.is_recording = False
        self.is_closed = False

        # 設定の読み込み
        self.config = load_config()

        # GUI要素の初期化
        self.initialize_gui_elements()

        # ワーカースレッドからのイベントはメインスレッドで取り出して反映する
        self.poll_job = master.after(EVENT_POLL_INTERVAL_MS, self.poll_events)

        # ウィンドウが閉じられる時の処理を設定
        master.protocol("WM_DELETE_WINDOW", self.on_close)

    def initialize_gui_elements(self) -> None:
        # 処理モード選択
        ttk.Label(self.master, text="処理モード:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        mode_values = [mode.value for mode in ProcessingMode]
        self.processing_mode = tk.StringVar(
            value=self._choice(self.config.get("processingMode"), mode_values, DEFAULT_PROCESSING_MODE.value)
        )
        self.processing_mode_combo = ttk.Combobox(
            self.master, textvariable=self.processing_mode, values=mode_values, state="readonly"
        )
        self.processing_mode_combo.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        self.processing_mode_combo.bind("<<ComboboxSelected>>", self.on_processing_mode_change)

        # APIキー入力
        ttk.Label(self.master, text="OpenAIAPIキー:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.api_key = tk.StringVar(value=str(self.config.get("apiKey", "")))
        self.api_key_entry = ttk.Entry(self.master, textvariable=self.api_key, show="*")
        self.api_key_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=5)

        # Whisperモデル選択
        ttk.Label(self.master, text="Whisperモデル:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        model_values = [model.value for model in WhisperModel]
        self.whisper_model = tk.StringVar(
            value=self._choice(self.config.get("whisperModel"), model_values, DEFAULT_WHISPER_MODEL.value)
        )
        self.whisper_model_combo = ttk.Combobox(
            self.master, textvariable=self.whisper_model, values=model_values, state="readonly"
        )
        self.whisper_model_combo.grid(row=2, column=1, sticky="ew", padx=5, pady=5)

        # マイク選択
        ttk.Label(self.master, text="マイク:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        mic_values = self.audio_processor.get_input_devices()
        self.selected_mic = tk.StringVar(
            value=self._choice(self.config.get("selectedMic"), mic_values, mic_values[0] if mic_values else "")
        )
        self.mic_combo = ttk.Combobox(self.master, textvariable=self.selected_mic, values=mic_values, state="readonly")
        self.mic_combo.grid(row=3, column=1, sticky="ew", padx=5, pady=5)

        # ショートカットキー入力
        ttk.Label(self.master, text="ショートカットキー:").grid(row=4, column=0, sticky="w", padx=5, pady=5)
        self.shortcut_key = tk.StringVar(value=str(self.config.get("shortcutKey", DEFAULT_SHORTCUT_KEY)))
        self.shortcut_key_entry = ttk.Entry(self.master, textvariable=self.shortcut_key)
        self.shortcut_key_entry.grid(row=4, column=1, sticky="ew", padx=5, pady=5)

        # 認識開始/終了ボタン
        self.recognition_button = ttk.Button(self.master, text="認識開始", command=self.toggle_recognition)
        self.recognition_button.grid(row=5, column=0, columnspan=2, pady=10)

        # ログ表示エリア
        ttk.Label(self.master, text="ログ:").grid(row=6, column=0, columnspan=2, sticky="w", padx=5, pady=5)
        self.log_text = tk.Text(self.master, height=10, width=50, state="disabled")
        self.log_text.grid(row=7, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")

        # 最新の文字起こし結果表示エリア
        ttk.Label(self.master, text="最新の文字起こし結果:").grid(
            row=8, column=0, columnspan=2, sticky="w", padx=5, pady=5
        )
        self.latest_transcription_text = tk.Text(self.master, height=10, width=50)
        self.latest_transcription_text.grid(row=9, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")

        # ウィンドウサイズの調整
        self.master.columnconfigure(1, weight=1)
        self.master.rowconfigure(7, weight=1)
        self.master.rowconfigure(9, weight=1)

        self.set_settings_enabled(True)

    def on_close(self) -> None:
        if self.is_closed:
            return
        self.is_closed = True
        try:
            self.save_config(self.get_config_data())
        except OSError as e:
            # 設定を保存できなくても、マイクの解放と終了は続ける
            print(f"設定を保存できませんでした: {e}")
        self.master.after_cancel(self.poll_job)
        self.session.stop()
        # モデルのロード中など録音スレッドがすぐに終わらない場合は、PortAudioの解放をOSに任せて終了する
        if self.session.wait_audio_released(AUDIO_RELEASE_TIMEOUT_SECONDS):
            self.audio_processor.close()
        self.master.destroy()

    def get_config_data(self) -> Dict[str, Any]:
        return {
            "processingMode": self.processing_mode.get(),
            "apiKey": self.api_key.get(),
            "whisperModel": self.whisper_model.get(),
            "selectedMic": self.selected_mic.get(),
            "shortcutKey": self.shortcut_key.get(),
        }

    def on_processing_mode_change(self, event: Any = None) -> None:
        self.set_settings_enabled(True)

    def set_settings_enabled(self, enabled: bool) -> None:
        is_api = self.processing_mode.get() == ProcessingMode.API.value
        self.processing_mode_combo.config(state="readonly" if enabled else "disabled")
        self.api_key_entry.config(state="normal" if enabled and is_api else "disabled")
        self.whisper_model_combo.config(state="readonly" if enabled and not is_api else "disabled")
        self.mic_combo.config(state="readonly" if enabled else "disabled")
        self.shortcut_key_entry.config(state="normal" if enabled else "disabled")

    def toggle_recognition(self) -> None:
        if not self.is_recording:
            self.start_recognition()
        else:
            self.stop_recognition()

    def start_recognition(self) -> None:
        try:
            mode = ProcessingMode(self.processing_mode.get())
            processor = self.processor_factory.create(mode, self.api_key.get(), WhisperModel(self.whisper_model.get()))
            self.session.start(
                RecognitionRequest(
                    mic_name=self.selected_mic.get(),
                    shortcut_key=self.shortcut_key.get(),
                    processor=processor,
                )
            )
        except Exception as e:
            # 開始できなかった場合は設定を直せるよう、画面を開始前の状態のままにする
            self.log_message(f"認識を開始できません: {str(e) or type(e).__name__}")
            return

        self.is_recording = True
        self.recognition_button.config(text="認識終了")
        self.set_settings_enabled(False)
        self.log_message("認識を開始します...")
        if mode is ProcessingMode.API:
            api_key = self.processor_factory.resolve_api_key(self.api_key.get())
            self.log_message(f"APIキー: {mask_api_key(api_key)}")

    def stop_recognition(self) -> None:
        # 録音スレッドの終了はStoppedEventで受け取り、それまで再開始させない
        self.recognition_button.config(text="停止処理中...", state="disabled")
        self.log_message("認識を終了しています...")
        self.session.stop()

    def poll_events(self) -> None:
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            self.handle_event(event)
        self.poll_job = self.master.after(EVENT_POLL_INTERVAL_MS, self.poll_events)

    def handle_event(self, event: Any) -> None:
        if isinstance(event, LogEvent):
            self.log_message(event.message)
        elif isinstance(event, TranscribedEvent):
            self.show_transcription(event.text)
        elif isinstance(event, StoppedEvent):
            self.finish_recognition(event)

    def show_transcription(self, transcribed_text: str) -> None:
        self.log_message(f"文字起こし結果: {transcribed_text}")

        # 最新の文字起こし結果を表示
        self.latest_transcription_text.delete("1.0", tk.END)
        self.latest_transcription_text.insert(tk.END, transcribed_text)

    def finish_recognition(self, event: StoppedEvent) -> None:
        self.is_recording = False
        self.recognition_button.config(text="認識開始", state="normal")
        if event.error:
            self.log_message(f"エラーが発生したため認識を終了しました: {event.error}")
        else:
            self.log_message("認識を終了しました")

        # 設定項目を有効化
        self.set_settings_enabled(True)

    def log_message(self, message: str) -> None:
        self.log_text.config(state="normal")
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state="disabled")

    @staticmethod
    def _choice(value: Any, choices: list, default: str) -> str:
        return value if value in choices else default
