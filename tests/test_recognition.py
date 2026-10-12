import threading
from unittest.mock import MagicMock, call

import pytest

from audio_processor import MicrophoneNotFoundError
from config import VoiceWriterError
from hotkey import InvalidShortcutError
from recognition import LogEvent, RecognitionRequest, RecognitionSession, StoppedEvent, TranscribedEvent

JOIN_TIMEOUT = 5
SHORTCUT = "ctrl+space"


class Harness:
    """音声機器・キーボード・貼り付けを差し替え、キーの押下列を台本として認識セッションを動かす"""

    def __init__(self, **session_options):
        self.events = []
        self.audio = MagicMock()
        self.audio.find_device_index.return_value = 4
        self.audio.read_chunk.side_effect = self._read_chunk
        self.hotkey = MagicMock()
        self.hotkey.is_pressed.side_effect = self._is_pressed
        self.paster = MagicMock()
        self.paster.paste.return_value = "メモ帳"
        self.processor = MagicMock()
        self.processor.process.return_value = "こんにちは"
        self.session = RecognitionSession(self.audio, self.hotkey, self.paster, self._emit, **session_options)
        self.pressed = []
        self.chunk_count = 0
        self.last_pressed = False
        self.pending_recordings = 0
        self.recordings_handled = threading.Condition()
        self.before_stop = self._wait_recordings_handled

    def _emit(self, event):
        self.events.append(event)
        # 1回の録音の処理は、貼り付け結果かエラーのログで終わる
        if isinstance(event, LogEvent) and ("貼り付け" in event.message or event.message.startswith("エラー")):
            with self.recordings_handled:
                self.pending_recordings -= 1
                self.recordings_handled.notify_all()

    def _read_chunk(self):
        self.chunk_count += 1
        return bytes([self.chunk_count])

    def _is_pressed(self, shortcut_key):
        if self.pressed:
            self.last_pressed = self.pressed.pop(0)
            return self.last_pressed
        # 台本を使い切ったら、録音済みの処理を待ってから、利用者が「認識終了」を押したものとして止める
        self.before_stop()
        self.session.stop()
        return self.last_pressed

    def _wait_recordings_handled(self):
        with self.recordings_handled:
            self.recordings_handled.wait_for(lambda: self.pending_recordings <= 0, JOIN_TIMEOUT)

    def start(self, pressed=(), expected_recordings=None):
        self.pressed = list(pressed)
        releases = zip(pressed, pressed[1:], strict=False)
        # 上限で録音が区切られる台本では、文字起こしへ渡る件数を明示する
        self.pending_recordings = (
            sum(1 for held, following in releases if held and not following)
            if expected_recordings is None
            else expected_recordings
        )
        self.session.start(RecognitionRequest("Mic B", SHORTCUT, self.processor))

    def run(self, pressed=(), expected_recordings=None):
        self.start(pressed, expected_recordings)
        self.session.join(JOIN_TIMEOUT)
        return self

    @property
    def logs(self):
        return [event.message for event in self.events if isinstance(event, LogEvent)]


@pytest.fixture
def harness():
    harness = Harness()
    yield harness
    harness.session.stop()
    harness.session.join(JOIN_TIMEOUT)


class TestStart:
    def test正常系_startメソッドが選択されたマイクの実デバイス番号でストリームを開くこと(self, harness):
        harness.run()

        harness.audio.initialize_stream.assert_called_once_with(4)

    def test正常系_startメソッドがモデルの準備を終えてからストリームを開くこと(self, harness):
        order = MagicMock()
        order.attach_mock(harness.processor.prepare, "prepare")
        order.attach_mock(harness.audio.initialize_stream, "initialize_stream")

        harness.run()

        assert [c[0] for c in order.mock_calls] == ["prepare", "initialize_stream"]

    def test正常系_startメソッドが準備の経過をログとして通知すること(self, harness):
        harness.processor.prepare.side_effect = lambda notify: notify("モデルのロードが完了しました。")

        harness.run()

        assert "モデルのロードが完了しました。" in harness.logs

    def test正常系_startメソッドが録音可能になったことをログで知らせること(self, harness):
        harness.run()

        assert "'ctrl+space' を押している間、録音します。" in harness.logs

    def test正常系_startメソッドが押下中の音声だけを文字起こしへ渡すこと(self, harness):
        harness.run([False, True, True, False])

        harness.processor.process.assert_called_once_with(bytes([2, 3]))

    def test正常系_startメソッドが押下のたびに別の録音として順番に文字起こしすること(self, harness):
        harness.run([True, False, True, True, False])

        assert [c.args[0] for c in harness.processor.process.call_args_list] == [bytes([1]), bytes([3, 4])]

    def test正常系_startメソッドがキーを押していない間は文字起こししないこと(self, harness):
        harness.run([False, False, False])

        harness.processor.process.assert_not_called()

    def test正常系_startメソッドが録音の開始と終了をログで知らせること(self, harness):
        harness.run([True, True, False])

        assert harness.logs[1:4] == ["録音中...", "録音終了", "文字起こしを開始します..."]

    def test異常系_startメソッドが実行中の再開始でVoiceWriterErrorを送出すること(self, harness):
        release = threading.Event()
        harness.processor.prepare.side_effect = lambda notify: release.wait(JOIN_TIMEOUT)
        harness.start()

        try:
            with pytest.raises(VoiceWriterError):
                harness.session.start(RecognitionRequest("Mic B", SHORTCUT, harness.processor))
        finally:
            release.set()

    def test異常系_startメソッドがマイク未検出でMicrophoneNotFoundErrorを送出すること(self, harness):
        harness.audio.find_device_index.side_effect = MicrophoneNotFoundError("見つかりません")

        with pytest.raises(MicrophoneNotFoundError):
            harness.start()

    def test異常系_startメソッドがマイク未検出のとき録音スレッドを起動しないこと(self, harness):
        harness.audio.find_device_index.side_effect = MicrophoneNotFoundError("見つかりません")

        with pytest.raises(MicrophoneNotFoundError):
            harness.start()

        assert harness.session.is_running is False

    def test異常系_startメソッドが不正なショートカットでInvalidShortcutErrorを送出すること(self, harness):
        harness.hotkey.validate.side_effect = InvalidShortcutError("解釈できません")

        with pytest.raises(InvalidShortcutError):
            harness.start()

    def test異常系_startメソッドが不正なショートカットのときマイクを開かないこと(self, harness):
        harness.hotkey.validate.side_effect = InvalidShortcutError("解釈できません")

        with pytest.raises(InvalidShortcutError):
            harness.start()

        harness.audio.initialize_stream.assert_not_called()


class TestTranscriptionResult:
    def test正常系_startメソッドが文字起こし結果を貼り付けること(self, harness):
        harness.run([True, False])

        harness.paster.paste.assert_called_once_with("こんにちは")

    def test正常系_startメソッドが文字起こし結果をイベントで通知すること(self, harness):
        harness.run([True, False])

        assert TranscribedEvent("こんにちは") in harness.events

    def test正常系_startメソッドが貼り付け先のウィンドウ名をログに出すこと(self, harness):
        harness.run([True, False])

        assert "文字起こし結果を 'メモ帳' に貼り付けました" in harness.logs

    def test正常系_startメソッドがウィンドウ名を取得できない場合も貼り付けをログに出すこと(self, harness):
        harness.paster.paste.return_value = None

        harness.run([True, False])

        assert "文字起こし結果を貼り付けました" in harness.logs

    def test正常系_startメソッドが空白だけの結果を貼り付けないこと(self, harness):
        harness.processor.process.return_value = "  "

        harness.run([True, False])

        harness.paster.paste.assert_not_called()

    def test正常系_startメソッドが空白だけの結果の理由をログに出すこと(self, harness):
        harness.processor.process.return_value = ""

        harness.run([True, False])

        assert "音声を認識できなかったため、貼り付けませんでした。" in harness.logs

    def test異常系_startメソッドが文字起こしの例外をログに出すこと(self, harness):
        harness.processor.process.side_effect = RuntimeError("401 Unauthorized")

        harness.run([True, False])

        assert "エラーが発生しました: 401 Unauthorized" in harness.logs

    def test異常系_startメソッドが文字起こしの例外の後も次の録音を処理すること(self, harness):
        harness.processor.process.side_effect = [RuntimeError("timeout"), "二回目"]

        harness.run([True, False, True, False])

        harness.paster.paste.assert_called_once_with("二回目")

    def test異常系_startメソッドが文字起こしの例外ではセッションをエラー終了にしないこと(self, harness):
        harness.processor.process.side_effect = RuntimeError("timeout")

        harness.run([True, False])

        assert harness.events[-1] == StoppedEvent(None)

    def test異常系_startメソッドが貼り付けの例外をログに出すこと(self, harness):
        harness.paster.paste.side_effect = RuntimeError("clipboard error")

        harness.run([True, False])

        assert "エラーが発生しました: clipboard error" in harness.logs


class TestStop:
    def test正常系_stopメソッドが録音スレッドを終了させること(self, harness):
        harness.run([False])

        assert harness.session.is_running is False

    def test正常系_stopメソッドがストリームを閉じること(self, harness):
        harness.run([False])

        harness.audio.close_stream.assert_called_once_with()

    def test正常系_stopメソッドが最後に正常終了を通知すること(self, harness):
        harness.run([False])

        assert harness.events[-1] == StoppedEvent(None)

    def test正常系_stopメソッドが録音途中の音声を文字起こしへ渡さないこと(self, harness):
        harness.run([True, True])

        harness.processor.process.assert_not_called()

    def test正常系_stopメソッドが録音途中の音声を破棄したことをログに出すこと(self, harness):
        harness.run([True, True])

        assert "認識を終了するため、録音中の音声を破棄しました。" in harness.logs

    def test正常系_stopメソッドが停止後に完了した文字起こしを貼り付けないこと(self, harness):
        def process(audio):
            harness.session.stop()
            return "遅れて届いた結果"

        harness.processor.process.side_effect = process

        harness.run([True, False])

        harness.paster.paste.assert_not_called()

    def test正常系_stopメソッドが実行中の文字起こしの完了を待ってから終了を通知すること(self, harness):
        in_process = threading.Event()
        stopped_before_finish = []

        def process(audio):
            in_process.set()
            # マイクの解放は、録音スレッドが停止処理に入ったことを示す
            harness.session.wait_audio_released(JOIN_TIMEOUT)
            stopped_before_finish.append(any(isinstance(event, StoppedEvent) for event in harness.events))
            return "結果"

        harness.processor.process.side_effect = process
        harness.before_stop = lambda: in_process.wait(JOIN_TIMEOUT)

        harness.run([True, False])

        assert stopped_before_finish == [False]

    def test正常系_stopメソッドが停止時に未着手の録音を破棄したことをログに出すこと(self, harness):
        # 1件目の文字起こしを停止まで引き延ばし、2件目を未着手のまま停止させる
        harness.processor.process.side_effect = lambda audio: harness.session.wait_audio_released(JOIN_TIMEOUT)
        harness.before_stop = lambda: None

        harness.run([True, False, True, False])

        assert "認識を終了したため、文字起こし前の音声を破棄しました。" in harness.logs

    def test正常系_stopメソッドがモデル準備中の停止でストリームを開かないこと(self, harness):
        harness.processor.prepare.side_effect = lambda notify: harness.session.stop()

        harness.run()

        harness.audio.initialize_stream.assert_not_called()

    def test正常系_stopメソッドが停止後に同じセッションを再開始できること(self, harness):
        harness.run([False])

        harness.run([True, False])

        harness.paster.paste.assert_called_once_with("こんにちは")

    def test正常系_stopメソッドが開始前に呼ばれても例外を出さないこと(self, harness):
        harness.session.stop()

        assert harness.session.is_running is False


class TestFailure:
    def test異常系_startメソッドがモデル準備の例外をエラー終了として通知すること(self, harness):
        harness.processor.prepare.side_effect = RuntimeError("download failed")

        harness.run()

        assert harness.events[-1] == StoppedEvent("download failed")

    def test異常系_startメソッドがストリームを開けない例外をエラー終了として通知すること(self, harness):
        harness.audio.initialize_stream.side_effect = OSError("Invalid input device")

        harness.run()

        assert harness.events[-1] == StoppedEvent("Invalid input device")

    def test異常系_startメソッドが録音中の読み取り例外をエラー終了として通知すること(self, harness):
        harness.audio.read_chunk.side_effect = OSError("Unanticipated host error")

        harness.run()

        assert harness.events[-1] == StoppedEvent("Unanticipated host error")

    def test異常系_startメソッドが読み取り例外の後にストリームを閉じること(self, harness):
        harness.audio.read_chunk.side_effect = OSError("Unanticipated host error")

        harness.run()

        harness.audio.close_stream.assert_called_once_with()

    def test異常系_startメソッドがキー監視の権限エラーをエラー終了として通知すること(self, harness):
        harness.hotkey.is_pressed.side_effect = OSError("Error 13 - Must be run as administrator")

        harness.run()

        assert harness.events[-1] == StoppedEvent("Error 13 - Must be run as administrator")

    def test異常系_startメソッドがメッセージのない例外を例外名で通知すること(self, harness):
        harness.processor.prepare.side_effect = MemoryError()

        harness.run()

        assert harness.events[-1] == StoppedEvent("MemoryError")

    def test異常系_startメソッドがストリームを閉じられない例外をエラー終了として通知すること(self, harness):
        harness.audio.close_stream.side_effect = OSError("close failed")

        harness.run([False])

        assert harness.events[-1] == StoppedEvent("close failed")

    def test異常系_startメソッドがエラー終了の後に再開始できること(self, harness):
        harness.audio.initialize_stream.side_effect = [OSError("Invalid input device"), None]
        harness.run()

        harness.run([True, False])

        harness.paster.paste.assert_called_once_with("こんにちは")


class TestWaitAudioReleased:
    def test正常系_wait_audio_releasedメソッドが開始前はTrueを返すこと(self, harness):
        assert harness.session.wait_audio_released(0) is True

    def test正常系_wait_audio_releasedメソッドが終了後にTrueを返すこと(self, harness):
        harness.run([False])

        assert harness.session.wait_audio_released(0) is True

    def test正常系_wait_audio_releasedメソッドがモデル準備中はFalseを返すこと(self, harness):
        release = threading.Event()
        harness.processor.prepare.side_effect = lambda notify: release.wait(JOIN_TIMEOUT)
        harness.start()

        try:
            assert harness.session.wait_audio_released(0) is False
        finally:
            release.set()

    def test正常系_wait_audio_releasedメソッドが文字起こしの完了を待たずにTrueを返すこと(self, harness):
        in_process = threading.Event()
        released_during_process = []

        def process(audio):
            in_process.set()
            released_during_process.append(harness.session.wait_audio_released(JOIN_TIMEOUT))
            return "結果"

        harness.processor.process.side_effect = process
        harness.before_stop = lambda: in_process.wait(JOIN_TIMEOUT)

        harness.run([True, False])

        assert released_during_process == [True]


class TestIsRunning:
    def test正常系_is_runningメソッドが開始前はFalseを返すこと(self, harness):
        assert harness.session.is_running is False

    def test正常系_is_runningメソッドが実行中はTrueを返すこと(self, harness):
        release = threading.Event()
        harness.processor.prepare.side_effect = lambda notify: release.wait(JOIN_TIMEOUT)
        harness.start()

        try:
            assert harness.session.is_running is True
        finally:
            release.set()


class TestJoin:
    def test正常系_joinメソッドが開始前に呼ばれても例外を出さないこと(self, harness):
        harness.session.join(0)

        assert harness.session.is_running is False


class TestStopPasteRace:
    def test正常系_stopメソッドが結果の通知後に完了しても貼り付けさせないこと(self, harness):
        transcribed = threading.Event()
        stop_done = threading.Event()
        emit = harness.session.emit

        def emit_and_wait_stop(event):
            emit(event)
            if isinstance(event, TranscribedEvent):
                # 停止の確認と貼り付けの間に、GUI側の停止が完了する状況を作る
                transcribed.set()
                stop_done.wait(JOIN_TIMEOUT)

        harness.session.emit = emit_and_wait_stop
        harness.before_stop = lambda: stop_done.wait(JOIN_TIMEOUT)
        harness.start([True, False])
        transcribed.wait(JOIN_TIMEOUT)

        harness.session.stop()
        stop_done.set()
        harness.session.join(JOIN_TIMEOUT)

        harness.paster.paste.assert_not_called()

    def test正常系_stopメソッドが実行中の貼り付けの完了を待ってから戻ること(self, harness):
        in_paste = threading.Event()
        stop_returned = threading.Event()
        paste_finished = []
        observed = []

        def paste(text):
            in_paste.set()
            # 停止が貼り付けを待たない実装なら、ここを抜ける前にstop()が戻る
            stop_returned.wait(0.2)
            paste_finished.append(True)
            return "メモ帳"

        harness.paster.paste.side_effect = paste
        harness.before_stop = lambda: stop_returned.wait(JOIN_TIMEOUT)
        harness.start([True, False])
        in_paste.wait(JOIN_TIMEOUT)

        harness.session.stop()
        observed.append(bool(paste_finished))
        stop_returned.set()
        harness.session.join(JOIN_TIMEOUT)

        assert observed == [True]

    def test正常系_stopメソッドが実行中の文字起こしの完了を待たずに戻ること(self, harness):
        in_process = threading.Event()
        release = threading.Event()
        harness.processor.process.side_effect = lambda audio: (in_process.set(), release.wait(JOIN_TIMEOUT), "結果")[2]
        harness.before_stop = lambda: release.wait(JOIN_TIMEOUT)
        harness.start([True, False])
        in_process.wait(JOIN_TIMEOUT)

        harness.session.stop()
        still_transcribing = not release.is_set() and harness.session.is_running
        release.set()
        harness.session.join(JOIN_TIMEOUT)

        assert still_transcribing is True


class BlockedTranscription:
    """1件目の文字起こしを止めたまま、後続の録音を待ち行列に溜める"""

    def __init__(self, harness, wait_before_call):
        self.harness = harness
        self.in_process = threading.Event()
        self.release = threading.Event()
        harness.processor.process.side_effect = self._process
        harness.hotkey.is_pressed.side_effect = self._is_pressed
        harness.before_stop = self.release.set
        self.wait_before_call = wait_before_call
        self.calls = 0

    def _process(self, audio):
        self.in_process.set()
        self.release.wait(JOIN_TIMEOUT)
        return "結果"

    def _is_pressed(self, shortcut_key):
        self.calls += 1
        if self.calls == self.wait_before_call:
            # 1件目が待ち行列から取り出されてから、次の録音を溜め始める
            self.in_process.wait(JOIN_TIMEOUT)
        return self.harness._is_pressed(shortcut_key)


class TestRecordingLimit:
    def test正常系_startメソッドが上限に達した録音を区切って文字起こしへ渡すこと(self, harness):
        harness.session.max_recording_chunks = 2

        harness.run([True, True, True, False], expected_recordings=2)

        assert harness.processor.process.call_args_list == [call(b"\x01\x02"), call(b"\x03")]

    def test正常系_startメソッドが録音を区切ったことをログで知らせること(self, harness):
        harness.session.max_recording_chunks = 2

        harness.run([True, True, True, False], expected_recordings=2)

        assert any(log.startswith("1回の録音が上限（300秒）に達したため") for log in harness.logs)

    def test正常系_startメソッドが上限ちょうどで離した録音を二重に渡さないこと(self, harness):
        harness.session.max_recording_chunks = 2

        harness.run([True, True, False])

        harness.processor.process.assert_called_once_with(b"\x01\x02")

    def test正常系_startメソッドが上限未満の録音を区切らないこと(self, harness):
        harness.session.max_recording_chunks = 3

        harness.run([True, True, False])

        harness.processor.process.assert_called_once_with(b"\x01\x02")

    def test正常系_initメソッドが録音時間の上限をチャンク数へ換算すること(self):
        session = RecognitionSession(MagicMock(), MagicMock(), MagicMock(), MagicMock(), max_recording_seconds=300)

        assert session.max_recording_chunks == 9375


class TestPendingLimit:
    @pytest.fixture
    def harness(self):
        harness = Harness(max_pending_recordings=1)
        yield harness
        harness.session.stop()
        harness.session.join(JOIN_TIMEOUT)

    def test異常系_startメソッドが待ち行列が満杯のとき録音できないことをログで知らせること(self, harness):
        BlockedTranscription(harness, wait_before_call=3)

        harness.run([True, False, True, False, True, False])

        assert any("上限（1件）に達しているため、録音できません。" in log for log in harness.logs)

    def test異常系_startメソッドが待ち行列が満杯のとき押下中の音声を文字起こしへ渡さないこと(self, harness):
        BlockedTranscription(harness, wait_before_call=3)

        harness.run([True, False, True, False, True, False])

        assert call(b"\x05") not in harness.processor.process.call_args_list

    def test異常系_startメソッドが待ち行列が満杯でも録音済みの音声を保持すること(self, harness):
        blocker = BlockedTranscription(harness, wait_before_call=3)
        harness.before_stop = lambda: (blocker.release.set(), harness._wait_recordings_handled())

        harness.run([True, False, True, False, True, False], expected_recordings=2)

        assert harness.processor.process.call_args_list == [call(b"\x01"), call(b"\x03")]

    def test異常系_startメソッドが区切った録音を溜められないとき破棄をログで知らせること(self, harness):
        harness.session.max_recording_chunks = 1
        BlockedTranscription(harness, wait_before_call=2)

        harness.run([True, True, True, True, False])

        assert any("上限（1件）に達したため、今回の録音を破棄しました。" in log for log in harness.logs)

    def test異常系_startメソッドが破棄を知らせた後は離すまで録音しないこと(self, harness):
        harness.session.max_recording_chunks = 1
        BlockedTranscription(harness, wait_before_call=2)

        harness.run([True, True, True, True, False])

        assert call(b"\x04") not in harness.processor.process.call_args_list

    def test異常系_startメソッドが溜められない録音で録音スレッドを止めないこと(self, harness):
        harness.session.max_recording_chunks = 1
        BlockedTranscription(harness, wait_before_call=2)

        harness.run([True, True, True, True, False])

        assert harness.session.is_running is False

    def test異常系_startメソッドが破棄の後に押し直した録音を受け付けること(self, harness):
        blocker = BlockedTranscription(harness, wait_before_call=3)
        drained = threading.Event()
        is_pressed = harness.hotkey.is_pressed.side_effect

        def process(audio):
            blocker.in_process.set()
            blocker.release.wait(JOIN_TIMEOUT)
            if audio == b"\x03":
                drained.set()
            return "結果"

        def press(shortcut_key):
            if blocker.calls == 6:
                # 録音を断られた後、処理が進んで待ち行列が空くのを待ってから押し直す
                blocker.release.set()
                drained.wait(JOIN_TIMEOUT)
            return is_pressed(shortcut_key)

        harness.processor.process.side_effect = process
        harness.hotkey.is_pressed.side_effect = press
        harness.before_stop = harness._wait_recordings_handled

        harness.run([True, False, True, False, True, False, True, False], expected_recordings=3)

        assert harness.processor.process.call_args_list[-1] == call(b"\x07")

    def test正常系_stopメソッドが待ち行列が満杯でも終了を通知すること(self, harness):
        blocker = BlockedTranscription(harness, wait_before_call=3)
        harness.before_stop = lambda: None
        harness.processor.process.side_effect = lambda audio: (
            blocker.in_process.set(),
            harness.session.wait_audio_released(JOIN_TIMEOUT),
            "結果",
        )[2]

        harness.run([True, False, True, False])

        assert harness.events[-1] == StoppedEvent(None)

    def test正常系_stopメソッドが待ち行列に残った録音を文字起こししないこと(self, harness):
        blocker = BlockedTranscription(harness, wait_before_call=3)
        harness.before_stop = lambda: None
        harness.processor.process.side_effect = lambda audio: (
            blocker.in_process.set(),
            harness.session.wait_audio_released(JOIN_TIMEOUT),
            "結果",
        )[2]

        harness.run([True, False, True, False])

        harness.processor.process.assert_called_once_with(b"\x01")
