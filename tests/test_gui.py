import queue
import tkinter as tk
from unittest.mock import MagicMock

import pytest

from audio_processor import MicrophoneNotFoundError
from config import ProcessingMode, WhisperModel
from gui import VoiceWriterGUI
from recognition import LogEvent, RecognitionRequest, StoppedEvent, TranscribedEvent
from transcription_processor import MissingApiKeyError


@pytest.fixture(scope="module")
def root():
    try:
        root = tk.Tk()
    except tk.TclError as e:
        pytest.skip(f"Tkのウィンドウを作成できない環境です: {e}")
    root.withdraw()
    yield root
    root.destroy()


@pytest.fixture
def audio_processor():
    audio_processor = MagicMock()
    audio_processor.get_input_devices.return_value = ["Mic A", "Mic B"]
    return audio_processor


@pytest.fixture
def session():
    session = MagicMock()
    session.wait_audio_released.return_value = True
    return session


@pytest.fixture
def processor_factory():
    processor_factory = MagicMock()
    processor_factory.resolve_api_key.return_value = "sk-abcdefghijklmnop"
    return processor_factory


@pytest.fixture
def save_config():
    return MagicMock()


@pytest.fixture
def make_gui(root, audio_processor, session, processor_factory, save_config):
    windows = []

    def make(config=None):
        window = tk.Toplevel(root)
        window.withdraw()
        windows.append(window)
        return VoiceWriterGUI(
            window,
            audio_processor,
            session,
            processor_factory,
            queue.Queue(),
            load_config=lambda: dict(config or {}),
            save_config=save_config,
        )

    yield make
    for window in windows:
        if window.winfo_exists():
            window.destroy()


@pytest.fixture
def gui(make_gui):
    return make_gui()


@pytest.fixture
def running_gui(gui):
    gui.start_recognition()
    return gui


def log_lines(gui):
    return gui.log_text.get("1.0", tk.END).splitlines()[:-1] or []


def state(widget):
    return str(widget.cget("state"))


class TestInitializeGuiElements:
    def test正常系_initialize_gui_elementsメソッドが設定がない場合にCPUモードを選ぶこと(self, gui):
        assert gui.processing_mode.get() == "cpu"

    def test正常系_initialize_gui_elementsメソッドが設定がない場合にmediumモデルを選ぶこと(self, gui):
        assert gui.whisper_model.get() == "medium"

    def test正常系_initialize_gui_elementsメソッドが設定がない場合に既定のショートカットを入れること(self, gui):
        assert gui.shortcut_key.get() == "ctrl+space"

    def test正常系_initialize_gui_elementsメソッドが設定がない場合に先頭のマイクを選ぶこと(self, gui):
        assert gui.selected_mic.get() == "Mic A"

    def test正常系_initialize_gui_elementsメソッドがマイクがない場合に未選択にすること(self, make_gui, audio_processor):
        audio_processor.get_input_devices.return_value = []

        assert make_gui().selected_mic.get() == ""

    def test正常系_initialize_gui_elementsメソッドが保存済みのマイクを選ぶこと(self, make_gui):
        assert make_gui({"selectedMic": "Mic B"}).selected_mic.get() == "Mic B"

    def test正常系_initialize_gui_elementsメソッドが接続されていない保存済みマイクを先頭のマイクに置き換えること(
        self, make_gui
    ):
        assert make_gui({"selectedMic": "Old Mic"}).selected_mic.get() == "Mic A"

    def test正常系_initialize_gui_elementsメソッドが不正な処理モードの設定を既定値に置き換えること(self, make_gui):
        assert make_gui({"processingMode": "gpu"}).processing_mode.get() == "cpu"

    def test正常系_initialize_gui_elementsメソッドが不正なモデルの設定を既定値に置き換えること(self, make_gui):
        assert make_gui({"whisperModel": ["medium"]}).whisper_model.get() == "medium"

    def test正常系_initialize_gui_elementsメソッドが保存済みのAPIキーを復元すること(self, make_gui):
        assert make_gui({"apiKey": "sk-saved"}).api_key.get() == "sk-saved"

    def test正常系_initialize_gui_elementsメソッドがAPIキーを伏せ字で表示すること(self, gui):
        assert gui.api_key_entry.cget("show") == "*"

    def test正常系_initialize_gui_elementsメソッドがローカルモードでAPIキー欄を無効にすること(self, gui):
        assert state(gui.api_key_entry) == "disabled"

    def test正常系_initialize_gui_elementsメソッドがAPIモードでモデル選択を無効にすること(self, make_gui):
        assert state(make_gui({"processingMode": "api"}).whisper_model_combo) == "disabled"

    def test正常系_initialize_gui_elementsメソッドがturboを含むモデルを選択肢に出すこと(self, gui):
        assert list(gui.whisper_model_combo.cget("values")) == ["tiny", "base", "small", "medium", "large", "turbo"]


class TestOnProcessingModeChange:
    def test正常系_on_processing_mode_changeメソッドがAPIモードでAPIキー欄を有効にすること(self, gui):
        gui.processing_mode.set("api")

        gui.on_processing_mode_change()

        assert state(gui.api_key_entry) == "normal"

    def test正常系_on_processing_mode_changeメソッドがAPIモードでモデル選択を無効にすること(self, gui):
        gui.processing_mode.set("api")

        gui.on_processing_mode_change()

        assert state(gui.whisper_model_combo) == "disabled"

    def test正常系_on_processing_mode_changeメソッドがCUDAモードでモデル選択を選択専用にすること(self, gui):
        gui.processing_mode.set("cuda")

        gui.on_processing_mode_change()

        assert state(gui.whisper_model_combo) == "readonly"


class TestSetSettingsEnabled:
    @pytest.mark.parametrize(
        "widget_name", ["processing_mode_combo", "whisper_model_combo", "mic_combo", "shortcut_key_entry"]
    )
    def test正常系_set_settings_enabledメソッドが無効化で設定項目を操作不可にすること(self, gui, widget_name):
        gui.set_settings_enabled(False)

        assert state(getattr(gui, widget_name)) == "disabled"

    @pytest.mark.parametrize(
        ("widget_name", "expected"),
        [("processing_mode_combo", "readonly"), ("mic_combo", "readonly"), ("shortcut_key_entry", "normal")],
    )
    def test正常系_set_settings_enabledメソッドが有効化で設定項目を操作可能に戻すこと(self, gui, widget_name, expected):
        gui.set_settings_enabled(False)

        gui.set_settings_enabled(True)

        assert state(getattr(gui, widget_name)) == expected


class TestGetConfigData:
    def test正常系_get_config_dataメソッドが旧版と同じキーで画面の設定を返すこと(self, make_gui):
        config = {
            "processingMode": "api",
            "apiKey": "sk-saved",
            "whisperModel": "small",
            "selectedMic": "Mic B",
            "shortcutKey": "f9",
        }

        assert make_gui(config).get_config_data() == config


class TestStartRecognition:
    def test正常系_start_recognitionメソッドが画面の設定で文字起こし処理を作ること(self, make_gui, processor_factory):
        gui = make_gui({"processingMode": "cuda", "whisperModel": "small"})

        gui.start_recognition()

        processor_factory.create.assert_called_once_with(ProcessingMode.CUDA, "", WhisperModel.SMALL)

    def test正常系_start_recognitionメソッドが選択中のマイクとショートカットで認識を開始すること(
        self, make_gui, session, processor_factory
    ):
        gui = make_gui({"selectedMic": "Mic B", "shortcutKey": "f9"})

        gui.start_recognition()

        session.start.assert_called_once_with(RecognitionRequest("Mic B", "f9", processor_factory.create.return_value))

    def test正常系_start_recognitionメソッドがボタンを認識終了に変えること(self, running_gui):
        assert running_gui.recognition_button.cget("text") == "認識終了"

    def test正常系_start_recognitionメソッドが認識中に設定項目を無効にすること(self, running_gui):
        assert state(running_gui.mic_combo) == "disabled"

    def test正常系_start_recognitionメソッドが開始をログに出すこと(self, running_gui):
        assert log_lines(running_gui) == ["認識を開始します..."]

    def test正常系_start_recognitionメソッドがAPIモードでキーを伏せてログに出すこと(self, make_gui):
        gui = make_gui({"processingMode": "api"})

        gui.start_recognition()

        assert log_lines(gui)[-1] == "APIキー: sk-a***********mnop"

    def test異常系_start_recognitionメソッドがAPIキー未入力の例外をログに出すこと(self, gui, processor_factory):
        processor_factory.create.side_effect = MissingApiKeyError("APIキーが入力されていません")

        gui.start_recognition()

        assert log_lines(gui) == ["認識を開始できません: APIキーが入力されていません"]

    def test異常系_start_recognitionメソッドがAPIキー未入力の例外で認識を開始しないこと(
        self, gui, session, processor_factory
    ):
        processor_factory.create.side_effect = MissingApiKeyError("APIキーが入力されていません")

        gui.start_recognition()

        session.start.assert_not_called()

    def test異常系_start_recognitionメソッドがマイク未検出の例外でボタンを認識開始のままにすること(self, gui, session):
        session.start.side_effect = MicrophoneNotFoundError("マイクが選択されていません。")

        gui.start_recognition()

        assert gui.recognition_button.cget("text") == "認識開始"

    def test異常系_start_recognitionメソッドがマイク未検出の例外で設定項目を無効にしないこと(self, gui, session):
        session.start.side_effect = MicrophoneNotFoundError("マイクが選択されていません。")

        gui.start_recognition()

        assert state(gui.mic_combo) == "readonly"

    def test異常系_start_recognitionメソッドがマイク未検出の例外で認識中にならないこと(self, gui, session):
        session.start.side_effect = MicrophoneNotFoundError("マイクが選択されていません。")

        gui.start_recognition()

        assert gui.is_recording is False


class TestStopRecognition:
    def test正常系_stop_recognitionメソッドが認識セッションへ停止を依頼すること(self, running_gui, session):
        running_gui.stop_recognition()

        session.stop.assert_called_once_with()

    def test正常系_stop_recognitionメソッドが停止完了までボタンを無効にすること(self, running_gui):
        running_gui.stop_recognition()

        assert state(running_gui.recognition_button) == "disabled"

    def test正常系_stop_recognitionメソッドが停止完了まで設定項目を無効のままにすること(self, running_gui):
        running_gui.stop_recognition()

        assert state(running_gui.mic_combo) == "disabled"


class TestToggleRecognition:
    def test正常系_toggle_recognitionメソッドが停止中に認識を開始すること(self, gui, session):
        gui.toggle_recognition()

        session.start.assert_called_once()

    def test正常系_toggle_recognitionメソッドが認識中に停止を依頼すること(self, running_gui, session):
        running_gui.toggle_recognition()

        session.stop.assert_called_once_with()


class TestHandleEvent:
    def test正常系_handle_eventメソッドがログイベントをログ欄に出すこと(self, gui):
        gui.handle_event(LogEvent("録音中..."))

        assert log_lines(gui) == ["録音中..."]

    def test正常系_handle_eventメソッドが文字起こし結果を最新結果の欄に出すこと(self, gui):
        gui.handle_event(TranscribedEvent("こんにちは"))

        assert gui.latest_transcription_text.get("1.0", "end-1c") == "こんにちは"

    def test正常系_handle_eventメソッドが最新結果の欄を前回の結果から置き換えること(self, gui):
        gui.handle_event(TranscribedEvent("一回目"))

        gui.handle_event(TranscribedEvent("二回目"))

        assert gui.latest_transcription_text.get("1.0", "end-1c") == "二回目"

    def test正常系_handle_eventメソッドが文字起こし結果をログ欄に出すこと(self, gui):
        gui.handle_event(TranscribedEvent("こんにちは"))

        assert log_lines(gui) == ["文字起こし結果: こんにちは"]

    def test正常系_handle_eventメソッドが停止イベントでボタンを認識開始に戻すこと(self, running_gui):
        running_gui.stop_recognition()

        running_gui.handle_event(StoppedEvent())

        assert (running_gui.recognition_button.cget("text"), state(running_gui.recognition_button)) == (
            "認識開始",
            "normal",
        )

    def test正常系_handle_eventメソッドが停止イベントで設定項目を操作可能に戻すこと(self, running_gui):
        running_gui.handle_event(StoppedEvent())

        assert state(running_gui.mic_combo) == "readonly"

    def test正常系_handle_eventメソッドが停止イベントで再開始できる状態に戻すこと(self, running_gui):
        running_gui.handle_event(StoppedEvent())

        assert running_gui.is_recording is False

    def test正常系_handle_eventメソッドが正常な停止をログに出すこと(self, running_gui):
        running_gui.handle_event(StoppedEvent())

        assert log_lines(running_gui)[-1] == "認識を終了しました"

    def test異常系_handle_eventメソッドがエラーによる停止の理由をログに出すこと(self, running_gui):
        running_gui.handle_event(StoppedEvent("Invalid input device"))

        assert log_lines(running_gui)[-1] == "エラーが発生したため認識を終了しました: Invalid input device"

    def test異常系_handle_eventメソッドがエラーによる停止でも設定項目を操作可能に戻すこと(self, running_gui):
        running_gui.handle_event(StoppedEvent("Invalid input device"))

        assert state(running_gui.mic_combo) == "readonly"


class TestPollEvents:
    def test正常系_poll_eventsメソッドがキューのイベントを届いた順に反映すること(self, gui):
        gui.events.put(LogEvent("録音中..."))
        gui.events.put(LogEvent("録音終了"))

        gui.poll_events()

        assert log_lines(gui) == ["録音中...", "録音終了"]

    def test正常系_poll_eventsメソッドがキューを空にすること(self, gui):
        gui.events.put(LogEvent("録音中..."))

        gui.poll_events()

        assert gui.events.empty()

    def test正常系_poll_eventsメソッドが次回の確認を予約し直すこと(self, gui):
        previous_job = gui.poll_job

        gui.poll_events()

        assert gui.poll_job != previous_job


class TestLogMessage:
    def test正常系_log_messageメソッドがメッセージを追記すること(self, gui):
        gui.log_message("一行目")

        gui.log_message("二行目")

        assert log_lines(gui) == ["一行目", "二行目"]

    def test正常系_log_messageメソッドがログ欄を編集不可に戻すこと(self, gui):
        gui.log_message("一行目")

        assert state(gui.log_text) == "disabled"


class TestOnClose:
    def test正常系_on_closeメソッドが画面の設定を保存すること(self, gui, save_config):
        gui.on_close()

        save_config.assert_called_once_with(
            {
                "processingMode": "cpu",
                "apiKey": "",
                "whisperModel": "medium",
                "selectedMic": "Mic A",
                "shortcutKey": "ctrl+space",
            }
        )

    def test正常系_on_closeメソッドが認識セッションを停止すること(self, gui, session):
        gui.on_close()

        session.stop.assert_called_once_with()

    def test正常系_on_closeメソッドがマイクの解放後にPyAudioを終了すること(self, gui, audio_processor):
        gui.on_close()

        audio_processor.close.assert_called_once_with()

    def test正常系_on_closeメソッドがマイクを解放できていない間はPyAudioを終了しないこと(
        self, gui, session, audio_processor
    ):
        session.wait_audio_released.return_value = False

        gui.on_close()

        audio_processor.close.assert_not_called()

    def test正常系_on_closeメソッドがウィンドウを破棄すること(self, gui):
        gui.on_close()

        assert not gui.master.winfo_exists()

    def test正常系_on_closeメソッドが二重に呼ばれても一度だけ終了処理をすること(self, gui, session):
        gui.on_close()

        gui.on_close()

        session.stop.assert_called_once_with()

    def test異常系_on_closeメソッドが設定を保存できなくてもウィンドウを破棄すること(self, gui, save_config):
        save_config.side_effect = OSError("read-only")

        gui.on_close()

        assert not gui.master.winfo_exists()

    def test異常系_on_closeメソッドが設定を保存できなくてもPyAudioを終了すること(
        self, gui, save_config, audio_processor
    ):
        save_config.side_effect = OSError("read-only")

        gui.on_close()

        audio_processor.close.assert_called_once_with()
