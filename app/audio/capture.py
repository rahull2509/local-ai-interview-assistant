"""Microphone audio capture using sounddevice.

Captures 16 kHz mono float32 audio in a background stream and feeds
chunks into a queue for the transcription worker to consume.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Optional

import numpy as np
import sounddevice as sd

from app.config import (
    AUDIO_CHANNELS,
    AUDIO_CHUNK_DURATION_S,
    AUDIO_DTYPE,
    AUDIO_SAMPLE_RATE,
)

logger = logging.getLogger(__name__)


class AudioCaptureError(Exception):
    """Raised when microphone capture fails."""


def list_input_devices() -> list[dict]:
    """Return a list of available input (microphone) devices.

    Each dict has keys: index, name, max_input_channels, default_samplerate.
    """
    devices: list[dict] = []
    for info in sd.query_devices():
        if info["max_input_channels"] > 0:  # type: ignore[index]
            devices.append({
                "index": info["index"],  # type: ignore[index]
                "name": info["name"],  # type: ignore[index]
                "max_input_channels": info["max_input_channels"],  # type: ignore[index]
                "default_samplerate": info["default_samplerate"],  # type: ignore[index]
            })
    return devices


class MicrophoneRecorder:
    """Captures audio from the default microphone into a thread-safe queue.

    Usage::

        recorder = MicrophoneRecorder()
        recorder.start()
        while recorder.is_recording:
            chunk = recorder.get_chunk(timeout=1.0)
            if chunk is not None:
                process(chunk)
        recorder.stop()
    """

    def __init__(
        self,
        sample_rate: int = AUDIO_SAMPLE_RATE,
        channels: int = AUDIO_CHANNELS,
        chunk_duration_s: float = AUDIO_CHUNK_DURATION_S,
        device_index: Optional[int] = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration_s = chunk_duration_s
        self.device_index = device_index

        self._chunk_samples = int(sample_rate * chunk_duration_s)
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=50)
        self._buffer: list[np.ndarray] = []
        self._buffer_samples = 0
        self._stream: Optional[sd.InputStream] = None
        self._recording = False
        self._lock = threading.Lock()

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def is_recording(self) -> bool:
        return self._recording

    # ── Public API ──────────────────────────────────────────────────────

    def start(self) -> None:
        """Open the microphone stream and begin capturing audio."""
        if self._recording:
            logger.warning("Recording already in progress")
            return

        # Validate device
        try:
            if self.device_index is not None:
                info = sd.query_devices(self.device_index)
                if info["max_input_channels"] < 1:  # type: ignore[index]
                    raise AudioCaptureError(
                        f"Device {self.device_index} has no input channels"
                    )
        except sd.PortAudioError as exc:
            raise AudioCaptureError(f"Cannot access audio device: {exc}") from exc

        self._buffer.clear()
        self._buffer_samples = 0

        # Drain any stale data
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=AUDIO_DTYPE,
                device=self.device_index,
                blocksize=int(self.sample_rate * 0.5),  # 500 ms blocks
                callback=self._audio_callback,
            )
            self._stream.start()
            self._recording = True
            logger.info(
                "Microphone recording started (rate=%d, channels=%d)",
                self.sample_rate,
                self.channels,
            )
        except (sd.PortAudioError, Exception) as exc:
            raise AudioCaptureError(
                f"Failed to start microphone: {exc}"
            ) from exc

    def stop(self) -> None:
        """Stop recording and flush any remaining buffered audio."""
        if not self._recording:
            return

        self._recording = False

        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as exc:
                logger.warning("Error closing audio stream: %s", exc)
            self._stream = None

        # Flush remaining buffer as a final chunk
        with self._lock:
            if self._buffer:
                final = np.concatenate(self._buffer, axis=0)
                self._buffer.clear()
                self._buffer_samples = 0
                try:
                    self._audio_queue.put_nowait(final)
                except queue.Full:
                    logger.warning("Audio queue full, dropping final chunk")

        logger.info("Microphone recording stopped")

    def get_chunk(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Retrieve the next audio chunk from the queue.

        Returns None if no data is available within *timeout* seconds.
        The returned array is float32 mono with shape ``(samples,)``.
        """
        try:
            chunk = self._audio_queue.get(timeout=timeout)
            # Ensure 1-D
            if chunk.ndim > 1:
                chunk = chunk[:, 0]
            return chunk
        except queue.Empty:
            return None

    # ── Internal ────────────────────────────────────────────────────────

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        """sounddevice callback — runs in a separate C thread."""
        if status:
            logger.debug("Audio callback status: %s", status)

        with self._lock:
            self._buffer.append(indata.copy())
            self._buffer_samples += frames

            if self._buffer_samples >= self._chunk_samples:
                chunk = np.concatenate(self._buffer, axis=0)
                self._buffer.clear()
                self._buffer_samples = 0
                try:
                    self._audio_queue.put_nowait(chunk)
                except queue.Full:
                    logger.warning("Audio queue full, dropping chunk")
