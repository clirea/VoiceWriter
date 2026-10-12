import io
import os
import threading
import wave
from typing import Any, Callable, Mapping, Optional

import numpy as np

from config import (
    API_KEY_ENV,
    API_MODEL,
    API_TIMEOUT_SECONDS,
    SAMPLE_RATE,
    SAMPLE_WIDTH_BYTES,
    ProcessingMode,
    VoiceWriterError,
    WhisperModel,
)

Notify = Callable[[str], None]


class MissingApiKeyError(VoiceWriterError):
    pass


class ModelNotLoadedError(VoiceWriterError):
    pass


class TranscriptionProcessor:
    def prepare(self, notify: Notify) -> None:
        """文字起こしを始められる状態にする。時間のかかる準備の経過はnotifyで知らせる"""

    def process(self, audio: bytes) -> str:
        """16kHz・モノラル・16bit PCMの音声を文字起こしする"""
        raise NotImplementedError("This method should be implemented by subclasses")


class APITranscriptionProcessor(TranscriptionProcessor):
    def __init__(self, client: Any, model: str = API_MODEL) -> None:
        self.client = client
        self.model = model

    def process(self, audio: bytes) -> str:
        # 一時ファイルを作らずメモリ上のWAVを送る（失敗時に音声ファイルを残さない）
        result = self.client.audio.transcriptions.create(
            model=self.model,
            file=("audio.wav", encode_wav(audio), "audio/wav"),
        )
        return result.text


class LocalTranscriptionProcessor(TranscriptionProcessor):
    def __init__(
        self,
        use_cuda: bool,
        model: WhisperModel,
        load_whisper_model: Callable[[str, str], Any],
        is_cuda_available: Callable[[], bool],
    ) -> None:
        self.use_cuda = use_cuda
        self.model_type = model
        self.model: Optional[Any] = None
        self.device: Optional[str] = None
        self._load_whisper_model = load_whisper_model
        self._is_cuda_available = is_cuda_available
        self._lock = threading.Lock()

    def prepare(self, notify: Notify) -> None:
        with self._lock:
            if self.model is not None:
                return
            device = "cpu"
            if self.use_cuda:
                if self._is_cuda_available():
                    device = "cuda"
                else:
                    notify("CUDAを利用できないため、CPUで処理します。")
            notify(f"モデル '{self.model_type.value}' をロード中です。しばらくお待ちください...")
            self.model = self._load_whisper_model(self.model_type.value, device)
            self.device = device
            notify("モデルのロードが完了しました。")

    def process(self, audio: bytes) -> str:
        with self._lock:
            if self.model is None:
                raise ModelNotLoadedError("モデルのロードが完了していません。")
            audio_array = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
            result = self.model.transcribe(audio_array, fp16=self.device == "cuda")
        return result["text"]


class TranscriptionProcessorFactory:
    def __init__(
        self,
        create_api_client: Optional[Callable[[str], Any]] = None,
        load_whisper_model: Optional[Callable[[str, str], Any]] = None,
        is_cuda_available: Optional[Callable[[], bool]] = None,
        environ: Optional[Mapping[str, str]] = None,
    ) -> None:
        self._create_api_client = create_api_client or _create_openai_client
        self._load_whisper_model = load_whisper_model or _load_whisper_model
        self._is_cuda_available = is_cuda_available or _is_cuda_available
        self._environ = os.environ if environ is None else environ
        self._local_processor: Optional[LocalTranscriptionProcessor] = None

    def resolve_api_key(self, entered_api_key: str) -> str:
        api_key = entered_api_key.strip() or self._environ.get(API_KEY_ENV, "").strip()
        if not api_key:
            raise MissingApiKeyError(f"APIキーが入力されていません（環境変数 {API_KEY_ENV} も未設定です）。")
        return api_key

    def create(self, mode: ProcessingMode, api_key: str, model: WhisperModel) -> TranscriptionProcessor:
        if mode is ProcessingMode.API:
            return APITranscriptionProcessor(self._create_api_client(self.resolve_api_key(api_key)))

        use_cuda = mode is ProcessingMode.CUDA
        cached = self._local_processor
        # 同じ条件での再開時にモデルを読み直さない
        if cached is not None and cached.use_cuda == use_cuda and cached.model_type is model:
            return cached
        self._local_processor = LocalTranscriptionProcessor(
            use_cuda, model, self._load_whisper_model, self._is_cuda_available
        )
        return self._local_processor


def encode_wav(audio: bytes) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wave_file:
        wave_file.setnchannels(1)
        wave_file.setsampwidth(SAMPLE_WIDTH_BYTES)
        wave_file.setframerate(SAMPLE_RATE)
        wave_file.writeframes(audio)
    return buffer.getvalue()


def _create_openai_client(api_key: str) -> Any:
    from openai import OpenAI

    return OpenAI(api_key=api_key, timeout=API_TIMEOUT_SECONDS)


def _load_whisper_model(model_name: str, device: str) -> Any:
    import whisper

    return whisper.load_model(model_name, device=device)


def _is_cuda_available() -> bool:
    import torch

    return torch.cuda.is_available()
