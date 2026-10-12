import sys
from unittest.mock import MagicMock, patch

import pytest

from audio_processor import AudioProcessor, InputDevice, MicrophoneNotFoundError, create_audio_processor
from config import FRAME_LENGTH, SAMPLE_RATE, VoiceWriterError

PA_INT16 = 8


def make_pa(devices):
    pa = MagicMock()
    pa.get_device_count.return_value = len(devices)
    pa.get_device_info_by_index.side_effect = lambda i: {"name": devices[i][0], "maxInputChannels": devices[i][1]}
    return pa


@pytest.fixture
def pa():
    # 出力デバイスと同名の重複を含む、Windowsで典型的な並び
    return make_pa([("Speakers", 0), ("Mic A", 2), ("Mic A", 2), ("Headphones", 0), ("Mic B", 1)])


@pytest.fixture
def processor(pa):
    return AudioProcessor(pa, PA_INT16, preferred_encoding="utf-8")


class TestListInputDevices:
    def test正常系_list_input_devicesメソッドが入力デバイスを実デバイス番号つきで返すこと(self, processor):
        assert processor.list_input_devices() == [InputDevice(1, "Mic A"), InputDevice(4, "Mic B")]

    def test正常系_list_input_devicesメソッドがデバイスがない場合に空を返すこと(self):
        assert AudioProcessor(make_pa([]), PA_INT16).list_input_devices() == []

    def test正常系_list_input_devicesメソッドがロケールの文字コードで化けた名前を復元すること(self):
        garbled = "マイク (USB Audio)".encode("utf-8").decode("cp932")
        processor = AudioProcessor(make_pa([(garbled, 1)]), PA_INT16, preferred_encoding="cp932")

        assert processor.list_input_devices() == [InputDevice(0, "マイク (USB Audio)")]

    def test正常系_list_input_devicesメソッドが復元できない名前をそのまま返すこと(self):
        processor = AudioProcessor(make_pa([("マイク", 1)]), PA_INT16, preferred_encoding="ascii")

        assert processor.list_input_devices() == [InputDevice(0, "マイク")]

    def test正常系_list_input_devicesメソッドが未知の文字コード名でも名前をそのまま返すこと(self):
        processor = AudioProcessor(make_pa([("Mic", 1)]), PA_INT16, preferred_encoding="no-such-encoding")

        assert processor.list_input_devices() == [InputDevice(0, "Mic")]


class TestGetInputDevices:
    def test正常系_get_input_devicesメソッドが重複のない入力デバイス名を返すこと(self, processor):
        assert processor.get_input_devices() == ["Mic A", "Mic B"]


class TestFindDeviceIndex:
    def test正常系_find_device_indexメソッドが一覧の位置ではなく実デバイス番号を返すこと(self, processor):
        assert processor.find_device_index("Mic B") == 4

    def test正常系_find_device_indexメソッドが同名デバイスの先頭の番号を返すこと(self, processor):
        assert processor.find_device_index("Mic A") == 1

    def test異常系_find_device_indexメソッドが未選択でMicrophoneNotFoundErrorを送出すること(self, processor):
        with pytest.raises(MicrophoneNotFoundError, match="選択されていません"):
            processor.find_device_index("")

    def test異常系_find_device_indexメソッドが存在しないマイクでMicrophoneNotFoundErrorを送出すること(self, processor):
        with pytest.raises(MicrophoneNotFoundError, match="Mic C"):
            processor.find_device_index("Mic C")


class TestInitializeStream:
    def test正常系_initialize_streamメソッドが指定デバイスを16kHzモノラルで開くこと(self, processor, pa):
        processor.initialize_stream(4)

        pa.open.assert_called_once_with(
            rate=SAMPLE_RATE,
            channels=1,
            format=PA_INT16,
            input=True,
            input_device_index=4,
            frames_per_buffer=FRAME_LENGTH,
        )

    def test正常系_initialize_streamメソッドが開いたストリームを保持すること(self, processor, pa):
        processor.initialize_stream(4)

        assert processor.audio_stream is pa.open.return_value

    def test正常系_initialize_streamメソッドが開き直す前に古いストリームを閉じること(self, processor, pa):
        first_stream = MagicMock()
        pa.open.side_effect = [first_stream, MagicMock()]
        processor.initialize_stream(1)

        processor.initialize_stream(4)

        first_stream.close.assert_called_once_with()

    def test異常系_initialize_streamメソッドがデバイスを開けない場合にOSErrorを送出すること(self, processor, pa):
        pa.open.side_effect = OSError("Invalid input device")

        with pytest.raises(OSError):
            processor.initialize_stream(4)

    def test異常系_initialize_streamメソッドが失敗時にストリームを保持しないこと(self, processor, pa):
        pa.open.side_effect = OSError("Invalid input device")

        with pytest.raises(OSError):
            processor.initialize_stream(4)

        assert processor.audio_stream is None


class TestReadChunk:
    def test正常系_read_chunkメソッドが1フレーム分の音声を返すこと(self, processor, pa):
        pa.open.return_value.read.return_value = b"\x01\x00" * FRAME_LENGTH
        processor.initialize_stream(1)

        assert processor.read_chunk() == b"\x01\x00" * FRAME_LENGTH

    def test正常系_read_chunkメソッドがオーバーフローを例外にせず読むこと(self, processor, pa):
        processor.initialize_stream(1)

        processor.read_chunk()

        pa.open.return_value.read.assert_called_once_with(FRAME_LENGTH, exception_on_overflow=False)

    def test異常系_read_chunkメソッドがストリーム未初期化でVoiceWriterErrorを送出すること(self, processor):
        with pytest.raises(VoiceWriterError):
            processor.read_chunk()


class TestCloseStream:
    def test正常系_close_streamメソッドがストリームを閉じること(self, processor, pa):
        processor.initialize_stream(1)

        processor.close_stream()

        pa.open.return_value.close.assert_called_once_with()

    def test正常系_close_streamメソッドがストリームの参照を破棄すること(self, processor):
        processor.initialize_stream(1)

        processor.close_stream()

        assert processor.audio_stream is None

    def test正常系_close_streamメソッドが二重に呼ばれても一度だけ閉じること(self, processor, pa):
        processor.initialize_stream(1)
        processor.close_stream()

        processor.close_stream()

        pa.open.return_value.close.assert_called_once_with()

    def test正常系_close_streamメソッドがPyAudio自体は終了しないこと(self, processor, pa):
        processor.initialize_stream(1)

        processor.close_stream()

        pa.terminate.assert_not_called()


class TestClose:
    def test正常系_closeメソッドが開いているストリームを閉じること(self, processor, pa):
        processor.initialize_stream(1)

        processor.close()

        pa.open.return_value.close.assert_called_once_with()

    def test正常系_closeメソッドがPyAudioを終了すること(self, processor, pa):
        processor.close()

        pa.terminate.assert_called_once_with()

    def test正常系_closeメソッドが二重に呼ばれてもPyAudioを一度だけ終了すること(self, processor, pa):
        processor.close()

        processor.close()

        pa.terminate.assert_called_once_with()

    def test異常系_closeメソッドがストリームを閉じられなくてもPyAudioを終了すること(self, processor, pa):
        pa.open.return_value.close.side_effect = OSError("stream error")
        processor.initialize_stream(1)

        with pytest.raises(OSError):
            processor.close()

        pa.terminate.assert_called_once_with()


class TestCreateAudioProcessor:
    @pytest.fixture
    def pyaudio_module(self):
        module = MagicMock()
        module.paInt16 = PA_INT16
        with patch.dict(sys.modules, {"pyaudio": module}):
            yield module

    def test正常系_create_audio_processorメソッドがPyAudioを初期化して渡すこと(self, pyaudio_module):
        assert create_audio_processor().pa is pyaudio_module.PyAudio.return_value

    def test正常系_create_audio_processorメソッドが16bitのサンプル形式を設定すること(self, pyaudio_module):
        assert create_audio_processor().sample_format == PA_INT16

    def test異常系_create_audio_processorメソッドがPyAudio未導入でImportErrorを送出すること(self):
        with patch.dict(sys.modules, {"pyaudio": None}):
            with pytest.raises(ImportError):
                create_audio_processor()
