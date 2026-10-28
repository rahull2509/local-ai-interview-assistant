"""Local speech-to-text using faster-whisper.

Wraps the faster-whisper model for CPU int8 transcription of audio chunks.
Model loading is lazy — the model is loaded on first use to avoid blocking
application startup.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional

import numpy as np

from app.config import (
    AUDIO_SAMPLE_RATE,
    WHISPER_BEAM_SIZE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_DEVICE,
    WHISPER_LANGUAGE,
    WHISPER_MODEL_SIZE,
)

logger = logging.getLogger(__name__)


class TranscriptionError(Exception):
    """Raised when transcription fails."""


class WhisperTranscriber:
    """Lazily-loaded faster-whisper transcription engine.

    Thread-safe: a lock serialises access to the underlying model so it
    can be called from any worker thread.
    """

    def __init__(
        self,
        model_size: str = WHISPER_MODEL_SIZE,
        device: str = WHISPER_DEVICE,
        compute_type: str = WHISPER_COMPUTE_TYPE,
        language: str = WHISPER_LANGUAGE,
        beam_size: int = WHISPER_BEAM_SIZE,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.beam_size = beam_size

        self._model = None
        self._lock = threading.Lock()
        self._loaded = False

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ── Public API ──────────────────────────────────────────────────────

    def load_model(self) -> None:
        """Download (if needed) and load the Whisper model.

        This is a heavyweight operation that can take 10–60 s on first run.
        Call from a background thread.
        """
        if self._loaded:
            return

        with self._lock:
            if self._loaded:
                return
            try:
                # Disable Xet storage downloader — it can cause DLL issues
                # on some Windows configurations with Application Control
                os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
                os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

                from faster_whisper import WhisperModel  # lazy import

                logger.info(
                    "Loading faster-whisper model '%s' (device=%s, compute=%s) …",
                    self.model_size,
                    self.device,
                    self.compute_type,
                )
                self._model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                )
                self._loaded = True
                logger.info("Whisper model loaded successfully")
            except Exception as exc:
                raise TranscriptionError(
                    f"Failed to load Whisper model: {exc}"
                ) from exc

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe a numpy audio chunk (float32, 16 kHz mono) to text.

        Returns the concatenated text of all segments, or an empty string
        if nothing was recognised.
        """
        if not self._loaded or self._model is None:
            raise TranscriptionError("Whisper model is not loaded")

        if audio.size == 0:
            return ""

        # Ensure 1-D float32
        if audio.ndim > 1:
            audio = audio[:, 0]
        audio = audio.astype(np.float32)

        # Skip very short audio (< 0.3 s)
        if len(audio) < AUDIO_SAMPLE_RATE * 0.3:
            logger.debug("Audio chunk too short (%.2f s), skipping", len(audio) / AUDIO_SAMPLE_RATE)
            return ""

        with self._lock:
            try:
                segments, info = self._model.transcribe(
                    audio,
                    beam_size=self.beam_size,
                    language=self.language,
                    vad_filter=True,
                    vad_parameters=dict(
                        min_silence_duration_ms=500,
                        speech_pad_ms=200,
                    ),
                )
                parts: list[str] = []
                for seg in segments:
                    text = seg.text.strip()
                    if text:
                        parts.append(text)
                result = " ".join(parts)
                logger.debug("Transcribed %d samples → %r", len(audio), result[:80])
                return result
            except Exception as exc:
                logger.error("Transcription error: %s", exc)
                raise TranscriptionError(f"Transcription failed: {exc}") from exc
