import io
import struct
import sys
import wave
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from config import API_MODEL, API_TIMEOUT_SECONDS, SAMPLE_RATE, ProcessingMode, WhisperModel
from transcription_processor import (
    APITranscriptionProcessor,
    LocalTranscriptionProcessor,
    MissingApiKeyError,
    ModelNotLoadedError,
    TranscriptionProcessor,
    TranscriptionProcessorFactory,
    _create_openai_client,
    _is_cuda_available,
    _load_whisper_model,
    encode_wav,
)

AUDIO = struct.pack("<4h", 0, 16384, -16384, -32768)


class TestTranscriptionProcessor:
    def test正常系_prepareメソッドが既定では通知せずに完了すること(self):
        notify = MagicMock()

        TranscriptionProcessor().prepare(notify)

        notify.assert_not_called()

    def test異常系_processメソッドが未実装でNotImplementedErrorを送出すること(self):
        with pytest.raises(NotImplementedError):
            TranscriptionProcessor().process(AUDIO)


class TestEncodeWav:
    @pytest.fixture
    def wave_file(self):
        with wave.open(io.BytesIO(encode_wav(AUDIO)), "rb") as f:
            yield f

    def test正常系_encode_wavメソッドがモノラルで書き出すこと(self, wave_file):
        assert wave_file.getnchannels() == 1

    def test正常系_encode_wavメソッドが16bitで書き出すこと(self, wave_file):
        assert wave_file.getsampwidth() == 2

    def test正常系_encode_wavメソッドが16kHzで書き出すこと(self, wave_file):
        assert wave_file.getframerate() == SAMPLE_RATE

    def test正常系_encode_wavメソッドが音声データを変えずに格納すること(self, wave_file):
        assert wave_file.readframes(wave_file.getnframes()) == AUDIO


class TestAPITranscriptionProcessor:
    @pytest.fixture
    def client(self):
        client = MagicMock()
        client.audio.transcriptions.create.return_value.text = "こんにちは"
        return client

    def test正常系_processメソッドがAPIの文字起こし結果を返すこと(self, client):
        assert APITranscriptionProcessor(client).process(AUDIO) == "こんにちは"

    def test正常系_processメソッドが既定のモデルを指定すること(self, client):
        APITranscriptionProcessor(client).process(AUDIO)

        assert client.audio.transcriptions.create.call_args.kwargs["model"] == API_MODEL

    def test正常系_processメソッドが音声をWAVとしてメモリから送ること(self, client):
        APITranscriptionProcessor(client).process(AUDIO)

        assert client.audio.transcriptions.create.call_args.kwargs["file"] == (
            "audio.wav",
            encode_wav(AUDIO),
            "audio/wav",
        )

    def test異常系_processメソッドがAPIの失敗をそのまま送出すること(self, client):
        client.audio.transcriptions.create.side_effect = RuntimeError("401 Unauthorized")

        with pytest.raises(RuntimeError, match="401"):
            APITranscriptionProcessor(client).process(AUDIO)

    def test異常系_processメソッドがAPIの失敗時に一時ファイルを作らないこと(self, client, tmp_path):
        client.audio.transcriptions.create.side_effect = RuntimeError("401 Unauthorized")

        with patch("tempfile.tempdir", str(tmp_path)), pytest.raises(RuntimeError):
            APITranscriptionProcessor(client).process(AUDIO)

        assert list(tmp_path.iterdir()) == []


class TestLocalTranscriptionProcessor:
    @pytest.fixture
    def model(self):
        model = MagicMock()
        model.transcribe.return_value = {"text": " こんにちは"}
        return model

    @pytest.fixture
    def load_whisper_model(self, model):
        return MagicMock(return_value=model)

    @pytest.fixture
    def notify(self):
        return MagicMock()

    def make(self, load_whisper_model, use_cuda=False, cuda_available=False):
        return LocalTranscriptionProcessor(
            use_cuda, WhisperModel.SMALL, load_whisper_model, MagicMock(return_value=cuda_available)
        )

    def test正常系_prepareメソッドがCPUモードでモデルをCPUに読み込むこと(self, load_whisper_model, notify):
        self.make(load_whisper_model).prepare(notify)

        load_whisper_model.assert_called_once_with("small", "cpu")

    def test正常系_prepareメソッドがCUDA利用可能な場合にモデルをCUDAに読み込むこと(self, load_whisper_model, notify):
        self.make(load_whisper_model, use_cuda=True, cuda_available=True).prepare(notify)

        load_whisper_model.assert_called_once_with("small", "cuda")

    def test正常系_prepareメソッドがCUDAを使えない場合にCPUへ切り替えること(self, load_whisper_model, notify):
        self.make(load_whisper_model, use_cuda=True, cuda_available=False).prepare(notify)

        load_whisper_model.assert_called_once_with("small", "cpu")

    def test正常系_prepareメソッドがCPUへの切り替えを通知すること(self, load_whisper_model, notify):
        self.make(load_whisper_model, use_cuda=True, cuda_available=False).prepare(notify)

        notify.assert_any_call("CUDAを利用できないため、CPUで処理します。")

    def test正常系_prepareメソッドがCPUモードでは切り替えを通知しないこと(self, load_whisper_model, notify):
        self.make(load_whisper_model).prepare(notify)

        assert [c.args[0] for c in notify.call_args_list] == [
            "モデル 'small' をロード中です。しばらくお待ちください...",
            "モデルのロードが完了しました。",
        ]

    def test正常系_prepareメソッドがロード済みのモデルを読み直さないこと(self, load_whisper_model, notify):
        processor = self.make(load_whisper_model)
        processor.prepare(notify)

        processor.prepare(notify)

        load_whisper_model.assert_called_once()

    def test異常系_prepareメソッドがモデルの読み込み失敗をそのまま送出すること(self, load_whisper_model, notify):
        load_whisper_model.side_effect = RuntimeError("download failed")

        with pytest.raises(RuntimeError, match="download failed"):
            self.make(load_whisper_model).prepare(notify)

    def test異常系_prepareメソッドが失敗後の再実行で読み込みをやり直すこと(self, load_whisper_model, model, notify):
        load_whisper_model.side_effect = [RuntimeError("download failed"), model]
        processor = self.make(load_whisper_model)
        with pytest.raises(RuntimeError):
            processor.prepare(notify)

        processor.prepare(notify)

        assert processor.model is model

    def test正常系_processメソッドがモデルの文字起こし結果を返すこと(self, load_whisper_model, notify):
        processor = self.make(load_whisper_model)
        processor.prepare(notify)

        assert processor.process(AUDIO) == " こんにちは"

    def test正常系_processメソッドが16bit音声をマイナス1から1のfloat32へ正規化すること(
        self, load_whisper_model, model, notify
    ):
        processor = self.make(load_whisper_model)
        processor.prepare(notify)

        processor.process(AUDIO)

        np.testing.assert_array_equal(
            model.transcribe.call_args.args[0], np.array([0.0, 0.5, -0.5, -1.0], dtype=np.float32)
        )

    def test正常系_processメソッドがfloat32の配列を渡すこと(self, load_whisper_model, model, notify):
        processor = self.make(load_whisper_model)
        processor.prepare(notify)

        processor.process(AUDIO)

        assert model.transcribe.call_args.args[0].dtype == np.float32

    def test正常系_processメソッドがCPUでは半精度を使わないこと(self, load_whisper_model, model, notify):
        processor = self.make(load_whisper_model)
        processor.prepare(notify)

        processor.process(AUDIO)

        assert model.transcribe.call_args.kwargs == {"fp16": False}

    def test正常系_processメソッドがCUDAでは半精度を使うこと(self, load_whisper_model, model, notify):
        processor = self.make(load_whisper_model, use_cuda=True, cuda_available=True)
        processor.prepare(notify)

        processor.process(AUDIO)

        assert model.transcribe.call_args.kwargs == {"fp16": True}

    def test異常系_processメソッドがモデル未ロードでModelNotLoadedErrorを送出すること(self, load_whisper_model):
        with pytest.raises(ModelNotLoadedError):
            self.make(load_whisper_model).process(AUDIO)


class TestTranscriptionProcessorFactory:
    @pytest.fixture
    def create_api_client(self):
        return MagicMock()

    @pytest.fixture
    def factory(self, create_api_client):
        return TranscriptionProcessorFactory(
            create_api_client=create_api_client,
            load_whisper_model=MagicMock(),
            is_cuda_available=MagicMock(return_value=False),
            environ={},
        )

    def test正常系_resolve_api_keyメソッドが入力されたキーを前後の空白なしで返すこと(self, factory):
        assert factory.resolve_api_key(" sk-entered ") == "sk-entered"

    def test正常系_resolve_api_keyメソッドが未入力のとき環境変数のキーを返すこと(self):
        factory = TranscriptionProcessorFactory(environ={"OPENAI_API_KEY": "sk-from-env"})

        assert factory.resolve_api_key("") == "sk-from-env"

    def test正常系_resolve_api_keyメソッドが入力されたキーを環境変数より優先すること(self):
        factory = TranscriptionProcessorFactory(environ={"OPENAI_API_KEY": "sk-from-env"})

        assert factory.resolve_api_key("sk-entered") == "sk-entered"

    def test異常系_resolve_api_keyメソッドがキーがない場合にMissingApiKeyErrorを送出すること(self, factory):
        with pytest.raises(MissingApiKeyError):
            factory.resolve_api_key("   ")

    def test正常系_createメソッドがAPIモードでAPI用の処理を返すこと(self, factory):
        processor = factory.create(ProcessingMode.API, "sk-entered", WhisperModel.MEDIUM)

        assert isinstance(processor, APITranscriptionProcessor)

    def test正常系_createメソッドがAPIモードで入力されたキーでクライアントを作ること(self, factory, create_api_client):
        factory.create(ProcessingMode.API, "sk-entered", WhisperModel.MEDIUM)

        create_api_client.assert_called_once_with("sk-entered")

    def test異常系_createメソッドがAPIモードでキーがない場合にMissingApiKeyErrorを送出すること(self, factory):
        with pytest.raises(MissingApiKeyError):
            factory.create(ProcessingMode.API, "", WhisperModel.MEDIUM)

    def test正常系_createメソッドがCPUモードでCUDAを使わないローカル処理を返すこと(self, factory):
        assert factory.create(ProcessingMode.CPU, "", WhisperModel.MEDIUM).use_cuda is False

    def test正常系_createメソッドがCUDAモードでCUDAを使うローカル処理を返すこと(self, factory):
        assert factory.create(ProcessingMode.CUDA, "", WhisperModel.MEDIUM).use_cuda is True

    def test正常系_createメソッドが選択されたモデルを設定すること(self, factory):
        assert factory.create(ProcessingMode.CPU, "", WhisperModel.TINY).model_type is WhisperModel.TINY

    def test正常系_createメソッドがローカルモードではAPIキーを要求しないこと(self, factory, create_api_client):
        factory.create(ProcessingMode.CPU, "", WhisperModel.MEDIUM)

        create_api_client.assert_not_called()

    def test正常系_createメソッドが同じ条件ではロード済みの処理を再利用すること(self, factory):
        first = factory.create(ProcessingMode.CPU, "", WhisperModel.MEDIUM)

        assert factory.create(ProcessingMode.CPU, "", WhisperModel.MEDIUM) is first

    def test正常系_createメソッドがモデルが変わると新しい処理を返すこと(self, factory):
        first = factory.create(ProcessingMode.CPU, "", WhisperModel.MEDIUM)

        assert factory.create(ProcessingMode.CPU, "", WhisperModel.SMALL) is not first

    def test正常系_createメソッドが処理モードが変わると新しい処理を返すこと(self, factory):
        first = factory.create(ProcessingMode.CPU, "", WhisperModel.MEDIUM)

        assert factory.create(ProcessingMode.CUDA, "", WhisperModel.MEDIUM) is not first


class TestDefaultDependencies:
    def test正常系__create_openai_clientメソッドがキーとタイムアウトを指定すること(self):
        openai = MagicMock()
        with patch.dict(sys.modules, {"openai": openai}):
            _create_openai_client("sk-entered")

        openai.OpenAI.assert_called_once_with(api_key="sk-entered", timeout=API_TIMEOUT_SECONDS)

    def test異常系__create_openai_clientメソッドがopenai未導入でImportErrorを送出すること(self):
        with patch.dict(sys.modules, {"openai": None}):
            with pytest.raises(ImportError):
                _create_openai_client("sk-entered")

    def test正常系__load_whisper_modelメソッドがモデル名とデバイスを指定して読み込むこと(self):
        whisper = MagicMock()
        with patch.dict(sys.modules, {"whisper": whisper}):
            _load_whisper_model("small", "cpu")

        whisper.load_model.assert_called_once_with("small", device="cpu")

    def test異常系__load_whisper_modelメソッドがwhisper未導入でImportErrorを送出すること(self):
        with patch.dict(sys.modules, {"whisper": None}):
            with pytest.raises(ImportError):
                _load_whisper_model("small", "cpu")

    def test正常系__is_cuda_availableメソッドがtorchの判定結果を返すこと(self):
        torch = MagicMock()
        torch.cuda.is_available.return_value = True
        with patch.dict(sys.modules, {"torch": torch}):
            assert _is_cuda_available() is True

    def test異常系__is_cuda_availableメソッドがtorch未導入でImportErrorを送出すること(self):
        with patch.dict(sys.modules, {"torch": None}):
            with pytest.raises(ImportError):
                _is_cuda_available()
