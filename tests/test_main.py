from unittest.mock import MagicMock, patch

import pytest

import main


@pytest.fixture
def dependencies():
    names = [
        "create_keyboard_hotkey",
        "create_clipboard_paster",
        "create_audio_processor",
        "RecognitionSession",
        "TranscriptionProcessorFactory",
        "VoiceWriterGUI",
    ]
    patchers = [patch.object(main, name) for name in names]
    mocks = MagicMock()
    for name, patcher in zip(names, patchers, strict=True):
        setattr(mocks, name, patcher.start())
    yield mocks
    for patcher in patchers:
        patcher.stop()


class TestBuildGui:
    def test正常系_build_guiメソッドが組み立てたGUIを返すこと(self, dependencies):
        assert main.build_gui(MagicMock()) is dependencies.VoiceWriterGUI.return_value

    def test正常系_build_guiメソッドが同じ音声入力を認識セッションとGUIに渡すこと(self, dependencies):
        main.build_gui(MagicMock())

        assert (
            dependencies.RecognitionSession.call_args.args[0]
            is dependencies.VoiceWriterGUI.call_args.args[1]
            is dependencies.create_audio_processor.return_value
        )

    def test正常系_build_guiメソッドが認識セッションの通知先をGUIのイベントキューにすること(self, dependencies):
        main.build_gui(MagicMock())

        events = dependencies.VoiceWriterGUI.call_args.args[4]
        assert dependencies.RecognitionSession.call_args.kwargs["emit"] == events.put

    def test異常系_build_guiメソッドがキー監視を使えない場合にマイクを初期化せず例外を送出すること(self, dependencies):
        dependencies.create_keyboard_hotkey.side_effect = RuntimeError("keyboard unavailable")

        with pytest.raises(RuntimeError):
            main.build_gui(MagicMock())

        dependencies.create_audio_processor.assert_not_called()


class TestMain:
    @pytest.fixture
    def tk_root(self):
        with patch.object(main.tk, "Tk") as tk_class:
            yield tk_class.return_value

    @pytest.fixture
    def showerror(self):
        with patch.object(main.messagebox, "showerror") as showerror:
            yield showerror

    def test正常系_mainメソッドがGUIを組み立ててイベントループに入ること(self, tk_root, showerror):
        with patch.object(main, "build_gui"):
            main.main()

        tk_root.mainloop.assert_called_once_with()

    def test異常系_mainメソッドが起動失敗時に終了コード1でSystemExitを送出すること(self, tk_root, showerror):
        with patch.object(main, "build_gui", side_effect=OSError("No Default Input Device")):
            with pytest.raises(SystemExit) as exc_info:
                main.main()

        assert exc_info.value.code == 1

    def test異常系_mainメソッドが起動失敗の理由をダイアログで知らせること(self, tk_root, showerror):
        with patch.object(main, "build_gui", side_effect=OSError("No Default Input Device")):
            with pytest.raises(SystemExit):
                main.main()

        showerror.assert_called_once_with("VoiceWriter", "起動できませんでした。\nOSError: No Default Input Device")

    def test異常系_mainメソッドが起動失敗時にイベントループに入らないこと(self, tk_root, showerror):
        with patch.object(main, "build_gui", side_effect=OSError("No Default Input Device")):
            with pytest.raises(SystemExit):
                main.main()

        tk_root.mainloop.assert_not_called()
