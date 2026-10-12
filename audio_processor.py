import locale
from dataclasses import dataclass
from typing import Any, List, Optional

from config import FRAME_LENGTH, SAMPLE_RATE, VoiceWriterError


class MicrophoneNotFoundError(VoiceWriterError):
    pass


@dataclass(frozen=True)
class InputDevice:
    index: int
    name: str


class AudioProcessor:
    def __init__(self, pa: Any, sample_format: int, preferred_encoding: Optional[str] = None) -> None:
        self.pa = pa
        self.sample_format = sample_format
        self.preferred_encoding = preferred_encoding or locale.getpreferredencoding()
        self.audio_stream: Optional[Any] = None
        self._terminated = False

    def list_input_devices(self) -> List[InputDevice]:
        devices: List[InputDevice] = []
        seen_names = set()
        for i in range(self.pa.get_device_count()):
            device_info = self.pa.get_device_info_by_index(i)
            if device_info["maxInputChannels"] <= 0:
                continue
            device_name = self._normalize_device_name(device_info["name"])
            if device_name in seen_names:  # 重複を削除（同名は先頭のデバイスを使う）
                continue
            seen_names.add(device_name)
            devices.append(InputDevice(index=i, name=device_name))
        return devices

    def get_input_devices(self) -> List[str]:
        return [device.name for device in self.list_input_devices()]

    def find_device_index(self, device_name: str) -> int:
        if not device_name:
            raise MicrophoneNotFoundError("マイクが選択されていません。")
        for device in self.list_input_devices():
            if device.name == device_name:
                return device.index
        raise MicrophoneNotFoundError(f"マイク '{device_name}' が見つかりません。接続を確認して選び直してください。")

    def initialize_stream(self, input_device_index: int) -> None:
        self.close_stream()
        self.audio_stream = self.pa.open(
            rate=SAMPLE_RATE,
            channels=1,
            format=self.sample_format,
            input=True,
            input_device_index=input_device_index,
            frames_per_buffer=FRAME_LENGTH,
        )

    def read_chunk(self) -> bytes:
        if self.audio_stream is None:
            raise VoiceWriterError("音声ストリームが初期化されていません。")
        return self.audio_stream.read(FRAME_LENGTH, exception_on_overflow=False)

    def close_stream(self) -> None:
        stream, self.audio_stream = self.audio_stream, None
        if stream is not None:
            stream.close()

    def close(self) -> None:
        if self._terminated:
            return
        self._terminated = True
        try:
            self.close_stream()
        finally:
            self.pa.terminate()

    def _normalize_device_name(self, name: str) -> str:
        # Windowsでロケールの文字コードとして読まれたUTF-8のデバイス名を復元する
        try:
            return name.encode(self.preferred_encoding).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError, LookupError):
            return name


def create_audio_processor() -> AudioProcessor:
    import pyaudio

    return AudioProcessor(pyaudio.PyAudio(), pyaudio.paInt16)
