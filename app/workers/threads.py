"""Background QThread workers for audio, transcription, and LLM tasks.

Each worker communicates with the GUI exclusively via Qt signals.
No direct GUI manipulation from worker threads.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
from PySide6.QtCore import QThread, Signal

from app.audio.capture import AudioCaptureError, MicrophoneRecorder
from app.config import AUDIO_OVERLAP_S, AUDIO_SAMPLE_RATE
from app.llm.client import OllamaClient, OllamaError
from app.transcription.engine import TranscriptionError, WhisperTranscriber

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════
# Model Loader Worker
# ════════════════════════════════════════════════════════════════════════

class ModelLoaderWorker(QThread):
    """Loads the Whisper model in the background so the GUI stays responsive."""

    finished = Signal(bool, str)  # success, message

    def __init__(self, transcriber: WhisperTranscriber, parent=None) -> None:
        super().__init__(parent)
        self._transcriber = transcriber

    def run(self) -> None:
        try:
            t0 = time.perf_counter()
            self._transcriber.load_model()
            elapsed = time.perf_counter() - t0
            self.finished.emit(True, f"Whisper model loaded in {elapsed:.1f}s")
        except TranscriptionError as exc:
            self.finished.emit(False, str(exc))
        except Exception as exc:
            self.finished.emit(False, f"Unexpected error loading model: {exc}")


# ════════════════════════════════════════════════════════════════════════
# Audio + Transcription Worker
# ════════════════════════════════════════════════════════════════════════

class AudioTranscriptionWorker(QThread):
    """Records audio and transcribes chunks, emitting text segments.

    Signals:
        text_segment(str)   – a new transcribed text segment
        error(str)          – a non-fatal error message
        status(str)         – informational status updates
        recording_started() – emitted once the stream is live
        recording_stopped() – emitted when recording ends
    """

    text_segment = Signal(str)
    error = Signal(str)
    status = Signal(str)
    recording_started = Signal()
    recording_stopped = Signal()

    def __init__(
        self,
        recorder: MicrophoneRecorder,
        transcriber: WhisperTranscriber,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._recorder = recorder
        self._transcriber = transcriber
        self._should_stop = False
        self._overlap_samples = int(AUDIO_SAMPLE_RATE * AUDIO_OVERLAP_S)
        self._prev_tail: Optional[np.ndarray] = None

    def request_stop(self) -> None:
        """Request a graceful stop from outside the thread."""
        self._should_stop = True

    def run(self) -> None:
        self._should_stop = False
        self._prev_tail = None

        try:
            self._recorder.start()
            self.recording_started.emit()
            self.status.emit("🎤 Recording…")
        except AudioCaptureError as exc:
            self.error.emit(str(exc))
            self.recording_stopped.emit()
            return

        try:
            while not self._should_stop:
                chunk = self._recorder.get_chunk(timeout=0.5)
                if chunk is not None:
                    self._process_chunk(chunk)
        except Exception as exc:
            self.error.emit(f"Recording error: {exc}")
        finally:
            self._recorder.stop()
            # Process any final chunk that was flushed
            final = self._recorder.get_chunk(timeout=0.2)
            if final is not None:
                self._process_chunk(final)
            self.recording_stopped.emit()
            self.status.emit("Recording stopped")

    def _process_chunk(self, chunk: np.ndarray) -> None:
        """Transcribe a single audio chunk and emit the result."""
        if chunk.size == 0:
            return

        # Prepend overlap from previous chunk to reduce boundary cuts
        if self._prev_tail is not None and self._prev_tail.size > 0:
            audio = np.concatenate([self._prev_tail, chunk])
        else:
            audio = chunk

        # Save tail for next overlap
        if chunk.size > self._overlap_samples:
            self._prev_tail = chunk[-self._overlap_samples:]
        else:
            self._prev_tail = chunk.copy()

        try:
            text = self._transcriber.transcribe(audio)
            if text:
                self.text_segment.emit(text)
        except TranscriptionError as exc:
            self.error.emit(f"Transcription: {exc}")


# ════════════════════════════════════════════════════════════════════════
# LLM Streaming Worker
# ════════════════════════════════════════════════════════════════════════

class LLMWorker(QThread):
    """Streams a response from Ollama for the given transcript.

    Signals:
        token(str)    – a single streamed token/chunk of text
        finished(str) – the complete response when streaming ends
        error(str)    – error message on failure
        started()     – emitted before the first token
    """

    token = Signal(str)
    finished = Signal(str)
    error = Signal(str)
    started = Signal()

    def __init__(
        self,
        client: OllamaClient,
        transcript: str,
        model: Optional[str] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._client = client
        self._transcript = transcript
        self._model = model
        self._cancelled = False

    def cancel(self) -> None:
        """Request cancellation of the generation."""
        self._cancelled = True

    def run(self) -> None:
        self._cancelled = False
        self.started.emit()
        full_response: list[str] = []

        try:
            t0 = time.perf_counter()
            first_token = True
            for chunk in self._client.stream_chat(
                self._transcript, model=self._model
            ):
                if self._cancelled:
                    logger.info("LLM generation cancelled by user")
                    break
                if first_token:
                    ttft = time.perf_counter() - t0
                    logger.info("Time to first token: %.2f s", ttft)
                    first_token = False
                full_response.append(chunk)
                self.token.emit(chunk)

            elapsed = time.perf_counter() - t0
            result = "".join(full_response)
            logger.info(
                "LLM generation complete: %d chars in %.1f s",
                len(result),
                elapsed,
            )
            self.finished.emit(result)

        except OllamaError as exc:
            self.error.emit(str(exc))
        except Exception as exc:
            self.error.emit(f"LLM error: {exc}")
