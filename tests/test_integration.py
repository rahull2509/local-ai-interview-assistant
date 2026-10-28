"""Integration tests for audio capture and local whisper engine.

These tests verify that real numpy audio arrays can flow through the
transcription engine and microphone initialization succeeds on available hardware.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.audio.capture import MicrophoneRecorder, list_input_devices
from app.transcription.engine import WhisperTranscriber


def test_microphone_device_enumeration():
    """Verify system audio input devices can be queried."""
    devices = list_input_devices()
    assert isinstance(devices, list)
    # Most Windows developer machines have at least one input channel
    if devices:
        dev = devices[0]
        assert "name" in dev
        assert "index" in dev
        assert dev["max_input_channels"] > 0


def test_whisper_transcriber_load_and_silence():
    """Verify Whisper model loads and processes audio array without crashing."""
    transcriber = WhisperTranscriber()
    transcriber.load_model()
    assert transcriber.is_loaded is True

    # 1 second of silence at 16 kHz
    silence = np.zeros(16000, dtype=np.float32)
    result = transcriber.transcribe(silence)
    assert isinstance(result, str)
    assert result == ""  # VAD filter correctly silences empty noise
