from enum import Enum

SAMPLE_RATE = 16000
FRAME_LENGTH = 512
SAMPLE_WIDTH_BYTES = 2
CONFIG_FILE = "voicewriter_config.json"

# 文字起こしが録音に追いつかない場合のメモリ使用量を抑える上限
# 1件あたり約9.6MB（16kHz・16bitで5分）。APIの1ファイル上限（25MB）にも収まる
MAX_RECORDING_SECONDS = 300
MAX_PENDING_RECORDINGS = 4

API_MODEL = "whisper-1"
API_TIMEOUT_SECONDS = 120.0
API_KEY_ENV = "OPENAI_API_KEY"

DEFAULT_SHORTCUT_KEY = "ctrl+space"


class ProcessingMode(Enum):
    API = "api"
    CPU = "cpu"
    CUDA = "cuda"


class WhisperModel(Enum):
    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    TURBO = "turbo"


DEFAULT_PROCESSING_MODE = ProcessingMode.CPU
DEFAULT_WHISPER_MODEL = WhisperModel.MEDIUM


class VoiceWriterError(Exception):
    """利用者にそのまま表示できるメッセージを持つエラー"""
