import queue
import threading
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

from config import (
    FRAME_LENGTH,
    MAX_PENDING_RECORDINGS,
    MAX_RECORDING_SECONDS,
    SAMPLE_RATE,
    VoiceWriterError,
)
from transcription_processor import TranscriptionProcessor


@dataclass(frozen=True)
class RecognitionRequest:
    mic_name: str
    shortcut_key: str
    processor: TranscriptionProcessor


@dataclass(frozen=True)
class LogEvent:
    message: str


@dataclass(frozen=True)
class TranscribedEvent:
    text: str


@dataclass(frozen=True)
class StoppedEvent:
    error: Optional[str] = None


class RecognitionSession:
    """録音・文字起こし・貼り付けをGUIスレッドの外で実行し、経過をイベントとして通知する"""

    def __init__(
        self,
        audio_processor: Any,
        hotkey: Any,
        paster: Any,
        emit: Callable[[Any], None],
        max_recording_seconds: float = MAX_RECORDING_SECONDS,
        max_pending_recordings: int = MAX_PENDING_RECORDINGS,
    ) -> None:
        self.audio_processor = audio_processor
        self.hotkey = hotkey
        self.paster = paster
        self.emit = emit
        self.max_recording_seconds = max_recording_seconds
        self.max_recording_chunks = max(1, int(max_recording_seconds * SAMPLE_RATE / FRAME_LENGTH))
        self.max_pending_recordings = max(1, max_pending_recordings)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        # 停止と貼り付けを直列にし、stop()が戻った後の貼り付けを起こさない
        self._paste_lock = threading.Lock()
        self._audio_released = threading.Event()
        self._audio_released.set()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, request: RecognitionRequest) -> None:
        if self.is_running:
            raise VoiceWriterError("前回の認識がまだ終了していません。")
        self.hotkey.validate(request.shortcut_key)
        device_index = self.audio_processor.find_device_index(request.mic_name)

        # 停止フラグは回ごとに作り、前回の文字起こしが残っていても新しい回に影響させない
        self._stop_event = threading.Event()
        self._audio_released.clear()
        self._thread = threading.Thread(target=self._run, args=(request, device_index, self._stop_event), daemon=True)
        self._thread.start()

    def stop(self) -> None:
        # 待つのは実行中の貼り付け（クリップボードとキー送出）だけ。文字起こしの完了は待たない
        with self._paste_lock:
            self._stop_event.set()

    def wait_audio_released(self, timeout: Optional[float] = None) -> bool:
        return self._audio_released.wait(timeout)

    def join(self, timeout: Optional[float] = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def _run(self, request: RecognitionRequest, device_index: int, stop_event: threading.Event) -> None:
        error: Optional[str] = None
        recordings: "queue.Queue[Optional[bytes]]" = queue.Queue(maxsize=self.max_pending_recordings)
        worker = threading.Thread(
            target=self._transcription_loop, args=(request.processor, recordings, stop_event), daemon=True
        )
        try:
            request.processor.prepare(self._log)
            if not stop_event.is_set():
                self.audio_processor.initialize_stream(device_index)
                worker.start()
                self._log(f"'{request.shortcut_key}' を押している間、録音します。")
                self._capture_loop(request.shortcut_key, recordings, stop_event)
        except Exception as e:
            error = str(e) or type(e).__name__
        finally:
            stop_event.set()
            try:
                self.audio_processor.close_stream()
            except Exception as e:
                error = error or str(e) or type(e).__name__
            self._audio_released.set()
            if worker.is_alive():
                # 待ち行列が満杯でも終了の合図を入れられるよう、未着手の録音を先に破棄する
                self._discard_pending(recordings)
                recordings.put(None)
                # 実行中の文字起こしが同じモデルを使い終えるまで、次の開始を待たせる
                worker.join()
            self.emit(StoppedEvent(error))

    def _capture_loop(
        self, shortcut_key: str, recordings: "queue.Queue[Optional[bytes]]", stop_event: threading.Event
    ) -> None:
        chunks: List[bytes] = []
        recording = False
        # 上限で録音を受け付けなかった押下は、キーを離すまで録音しない
        refused = False
        while not stop_event.is_set():
            # 押していない間も読み捨て、押した瞬間に古い音声が混ざらないようにする
            chunk = self.audio_processor.read_chunk()
            if not self.hotkey.is_pressed(shortcut_key):
                if recording:
                    self._log("録音終了")
                    if chunks:
                        self._enqueue(recordings, chunks)
                chunks = []
                recording = False
                refused = False
                continue
            if refused:
                continue
            if not recording:
                if recordings.full():
                    # 話し終えてから破棄するのではなく、押した時点で録音できないことを知らせる
                    self._log(
                        f"文字起こし待ちの録音が上限（{self.max_pending_recordings}件）に達しているため、録音できません。"
                        "処理が進んでから押し直してください。"
                    )
                    refused = True
                    continue
                recording = True
                self._log("録音中...")
            chunks.append(chunk)
            if len(chunks) >= self.max_recording_chunks:
                self._log(
                    f"1回の録音が上限（{self.max_recording_seconds:g}秒）に達したため、"
                    "ここまでを区切って文字起こしし、録音を続けます。"
                )
                if not self._enqueue(recordings, chunks):
                    recording = False
                    refused = True
                chunks = []
        if chunks:
            self._log("認識を終了するため、録音中の音声を破棄しました。")

    def _enqueue(self, recordings: "queue.Queue[Optional[bytes]]", chunks: List[bytes]) -> bool:
        try:
            # 文字起こしが追いつかなくても、録音スレッドを待たせずメモリも増やし続けない
            recordings.put_nowait(b"".join(chunks))
        except queue.Full:
            self._log(
                f"文字起こし待ちの録音が上限（{self.max_pending_recordings}件）に達したため、今回の録音を破棄しました。"
                "処理が進んでから録音し直してください。"
            )
            return False
        self._log("文字起こしを開始します...")
        return True

    def _discard_pending(self, recordings: "queue.Queue[Optional[bytes]]") -> None:
        while True:
            try:
                recordings.get_nowait()
            except queue.Empty:
                return
            self._log("認識を終了したため、文字起こし前の音声を破棄しました。")

    def _transcription_loop(
        self,
        processor: TranscriptionProcessor,
        recordings: "queue.Queue[Optional[bytes]]",
        stop_event: threading.Event,
    ) -> None:
        while True:
            audio = recordings.get()
            if audio is None:
                return
            if stop_event.is_set():
                self._log("認識を終了したため、文字起こし前の音声を破棄しました。")
                continue
            try:
                self._transcribe_and_paste(processor, audio, stop_event)
            except Exception as e:
                self._log(f"エラーが発生しました: {str(e) or type(e).__name__}")

    def _transcribe_and_paste(
        self, processor: TranscriptionProcessor, audio: bytes, stop_event: threading.Event
    ) -> None:
        text = processor.process(audio)
        if stop_event.is_set():
            # 認識終了後に、利用者の意図しないウィンドウへ貼り付けない
            self._log("認識を終了したため、文字起こし結果を貼り付けませんでした。")
            return
        self.emit(TranscribedEvent(text))
        if not text.strip():
            self._log("音声を認識できなかったため、貼り付けませんでした。")
            return
        with self._paste_lock:
            # 通知の間に停止が完了している場合があるため、貼り付けの直前に確かめ直す
            stopped = stop_event.is_set()
            window_title = None if stopped else self.paster.paste(text)
        if stopped:
            self._log("認識を終了したため、文字起こし結果を貼り付けませんでした。")
            return
        if window_title:
            self._log(f"文字起こし結果を '{window_title}' に貼り付けました")
        else:
            self._log("文字起こし結果を貼り付けました")

    def _log(self, message: str) -> None:
        self.emit(LogEvent(message))
