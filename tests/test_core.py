"""Unit tests for transcript handling, LLM request construction,
response parsing, error handling, and UI initialization.

All external services (Ollama, microphone, Whisper) are mocked.
Tests require only pytest — no hardware, models, or running servers.
"""

from __future__ import annotations

import json
from typing import Generator
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

# ════════════════════════════════════════════════════════════════════════
# Transcript handling
# ════════════════════════════════════════════════════════════════════════


class TestTranscriptHandling:
    """Tests for WhisperTranscriber edge cases (mocked model)."""

    def _make_transcriber(self):
        from app.transcription.engine import WhisperTranscriber
        t = WhisperTranscriber()
        # Pretend model is loaded
        t._loaded = True
        t._model = MagicMock()
        return t

    def test_empty_audio_returns_empty_string(self):
        t = self._make_transcriber()
        result = t.transcribe(np.array([], dtype=np.float32))
        assert result == ""

    def test_short_audio_skipped(self):
        """Audio shorter than 0.3s should return empty."""
        t = self._make_transcriber()
        # 0.1s of silence at 16kHz
        audio = np.zeros(1600, dtype=np.float32)
        result = t.transcribe(audio)
        assert result == ""
        # Model should NOT have been called
        t._model.transcribe.assert_not_called()

    def test_transcribe_concatenates_segments(self):
        t = self._make_transcriber()

        # Mock segments
        seg1 = MagicMock()
        seg1.text = "  Hello world  "
        seg2 = MagicMock()
        seg2.text = "how are you"
        mock_info = MagicMock()
        t._model.transcribe.return_value = ([seg1, seg2], mock_info)

        audio = np.zeros(16000, dtype=np.float32)  # 1 second
        result = t.transcribe(audio)
        assert result == "Hello world how are you"

    def test_transcribe_not_loaded_raises(self):
        from app.transcription.engine import WhisperTranscriber, TranscriptionError
        t = WhisperTranscriber()
        with pytest.raises(TranscriptionError, match="not loaded"):
            t.transcribe(np.zeros(16000, dtype=np.float32))

    def test_multidimensional_audio_squeezed(self):
        """Stereo (2D) audio should be squeezed to mono."""
        t = self._make_transcriber()
        seg = MagicMock()
        seg.text = "test"
        t._model.transcribe.return_value = ([seg], MagicMock())

        audio_2d = np.zeros((16000, 2), dtype=np.float32)
        result = t.transcribe(audio_2d)
        assert result == "test"
        # Verify the model received 1D audio
        call_args = t._model.transcribe.call_args
        passed_audio = call_args[0][0]
        assert passed_audio.ndim == 1


# ════════════════════════════════════════════════════════════════════════
# LLM Request Construction & Response Parsing
# ════════════════════════════════════════════════════════════════════════


class TestOllamaClient:
    """Tests for OllamaClient without a running Ollama server."""

    def _make_client(self):
        from app.llm.client import OllamaClient
        return OllamaClient()

    def test_is_reachable_returns_false_on_connection_error(self):
        client = self._make_client()
        with patch("app.llm.client.requests.get", side_effect=ConnectionError):
            assert client.is_reachable() is False

    def test_list_models_returns_empty_on_error(self):
        client = self._make_client()
        with patch("app.llm.client.requests.get", side_effect=ConnectionError):
            assert client.list_models() == []

    def test_list_models_parses_response(self):
        client = self._make_client()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "models": [
                {"name": "qwen2.5:3b"},
                {"name": "llama3.2:1b"},
            ]
        }
        with patch("app.llm.client.requests.get", return_value=mock_resp):
            models = client.list_models()
            assert models == ["qwen2.5:3b", "llama3.2:1b"]

    def test_find_model_exact_match(self):
        client = self._make_client()
        with patch.object(client, "list_models", return_value=["qwen2.5:3b", "llama3.2:1b"]):
            assert client.find_model("qwen2.5:3b") == "qwen2.5:3b"

    def test_find_model_prefix_match(self):
        client = self._make_client()
        with patch.object(client, "list_models", return_value=["qwen2.5:3b-instruct-q4"]):
            result = client.find_model("qwen2.5:3b")
            assert result == "qwen2.5:3b-instruct-q4"

    def test_find_model_fallback_to_first(self):
        client = self._make_client()
        with patch.object(client, "list_models", return_value=["some-model:latest"]):
            result = client.find_model("nonexistent:1b")
            assert result == "some-model:latest"

    def test_find_model_none_when_empty(self):
        client = self._make_client()
        with patch.object(client, "list_models", return_value=[]):
            assert client.find_model() is None

    def test_stream_chat_empty_transcript_raises(self):
        from app.llm.client import OllamaError
        client = self._make_client()
        with pytest.raises(OllamaError, match="Empty transcript"):
            list(client.stream_chat(""))

    def test_stream_chat_connection_error(self):
        import requests as req
        from app.llm.client import OllamaError
        client = self._make_client()
        with patch("app.llm.client.requests.post", side_effect=req.ConnectionError):
            with pytest.raises(OllamaError, match="Cannot connect"):
                list(client.stream_chat("hello"))

    def test_stream_chat_parses_chunks(self):
        """Verify progressive token parsing from streamed JSON lines."""
        client = self._make_client()

        lines = [
            json.dumps({"message": {"content": "Hello"}, "done": False}),
            json.dumps({"message": {"content": " world"}, "done": False}),
            json.dumps({"message": {"content": "!"}, "done": True}),
        ]

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.iter_lines.return_value = iter(lines)
        mock_resp.close = MagicMock()

        with patch("app.llm.client.requests.post", return_value=mock_resp):
            tokens = list(client.stream_chat("test question"))
            assert tokens == ["Hello", " world", "!"]

    def test_stream_chat_truncates_long_transcript(self):
        """Transcript longer than MAX_TRANSCRIPT_CHARS gets truncated from the left."""
        from app.config import OLLAMA_MAX_TRANSCRIPT_CHARS
        client = self._make_client()

        long_text = "x" * (OLLAMA_MAX_TRANSCRIPT_CHARS + 500)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.iter_lines.return_value = iter([
            json.dumps({"message": {"content": "ok"}, "done": True})
        ])
        mock_resp.close = MagicMock()

        with patch("app.llm.client.requests.post", return_value=mock_resp) as mock_post:
            list(client.stream_chat(long_text))
            # Check the payload sent
            call_kwargs = mock_post.call_args
            payload = call_kwargs[1]["json"]  # keyword arg 'json'
            user_msg = payload["messages"][1]["content"]
            assert len(user_msg) == OLLAMA_MAX_TRANSCRIPT_CHARS


# ════════════════════════════════════════════════════════════════════════
# Audio Capture
# ════════════════════════════════════════════════════════════════════════


class TestMicrophoneRecorder:
    """Tests for MicrophoneRecorder without hardware access."""

    def test_not_recording_initially(self):
        from app.audio.capture import MicrophoneRecorder
        rec = MicrophoneRecorder()
        assert rec.is_recording is False

    def test_get_chunk_returns_none_on_empty_queue(self):
        from app.audio.capture import MicrophoneRecorder
        rec = MicrophoneRecorder()
        assert rec.get_chunk(timeout=0.1) is None

    def test_stop_when_not_recording_is_safe(self):
        from app.audio.capture import MicrophoneRecorder
        rec = MicrophoneRecorder()
        rec.stop()  # should not raise

    def test_list_input_devices_returns_list(self):
        from app.audio.capture import list_input_devices
        with patch("app.audio.capture.sd.query_devices") as mock_qd:
            mock_qd.return_value = [
                {"index": 0, "name": "Mic1", "max_input_channels": 2,
                 "default_samplerate": 44100.0},
                {"index": 1, "name": "Speaker", "max_input_channels": 0,
                 "default_samplerate": 44100.0},
            ]
            devices = list_input_devices()
            assert len(devices) == 1
            assert devices[0]["name"] == "Mic1"


# ════════════════════════════════════════════════════════════════════════
# UI Initialization (headless-safe)
# ════════════════════════════════════════════════════════════════════════


class TestUIInitialization:
    """Test that the UI can be constructed without crashing.

    Uses QApplication in offscreen mode so no display server is needed.
    """

    @pytest.fixture(autouse=True)
    def _qapp(self):
        """Create or reuse a QApplication for the test session."""
        import os
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        yield app

    def test_window_creates_without_error(self):
        """MainWindow should instantiate and have correct title."""
        # Mock out heavy init: model loading and Ollama checks
        with patch("app.ui.main_window.MainWindow._load_whisper_model"), \
             patch("app.ui.main_window.MainWindow._check_runtime_status"):
            from app.ui.main_window import MainWindow
            win = MainWindow()
            assert "Interview Assistant" in win.windowTitle()

    def test_buttons_exist(self):
        with patch("app.ui.main_window.MainWindow._load_whisper_model"), \
             patch("app.ui.main_window.MainWindow._check_runtime_status"):
            from app.ui.main_window import MainWindow
            win = MainWindow()
            assert win._start_btn is not None
            assert win._stop_btn is not None
            assert win._suggest_btn is not None
            assert win._clear_btn is not None

    def test_stop_button_disabled_initially(self):
        with patch("app.ui.main_window.MainWindow._load_whisper_model"), \
             patch("app.ui.main_window.MainWindow._check_runtime_status"):
            from app.ui.main_window import MainWindow
            win = MainWindow()
            assert win._stop_btn.isEnabled() is False

    def test_clear_empties_text(self):
        with patch("app.ui.main_window.MainWindow._load_whisper_model"), \
             patch("app.ui.main_window.MainWindow._check_runtime_status"):
            from app.ui.main_window import MainWindow
            win = MainWindow()
            win._transcript_edit.setPlainText("some text")
            win._answer_edit.setPlainText("some answer")
            win._on_clear()
            assert win._transcript_edit.toPlainText() == ""
            assert win._answer_edit.toPlainText() == ""

    def test_suggest_with_empty_transcript(self):
        """Suggest Answer with no text should show status message, not crash."""
        with patch("app.ui.main_window.MainWindow._load_whisper_model"), \
             patch("app.ui.main_window.MainWindow._check_runtime_status"):
            from app.ui.main_window import MainWindow
            win = MainWindow()
            win._on_suggest()  # Should not raise


# ════════════════════════════════════════════════════════════════════════
# Worker thread tests (mocked)
# ════════════════════════════════════════════════════════════════════════


class TestLLMWorker:
    """Test LLM worker cancellation and error handling."""

    @pytest.fixture(autouse=True)
    def _qapp(self):
        import os
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        yield app

    def test_cancel_sets_flag(self):
        from app.workers.threads import LLMWorker
        client = MagicMock()
        worker = LLMWorker(client, "test")
        worker.cancel()
        assert worker._cancelled is True

    def test_error_signal_on_ollama_error(self):
        from app.workers.threads import LLMWorker
        from app.llm.client import OllamaError

        client = MagicMock()
        client.stream_chat.side_effect = OllamaError("Server down")

        worker = LLMWorker(client, "test question")
        errors: list[str] = []
        worker.error.connect(errors.append)
        worker.run()  # run synchronously in test
        assert len(errors) == 1
        assert "Server down" in errors[0]
