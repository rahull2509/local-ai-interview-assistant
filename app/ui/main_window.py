"""Main window for the Local AI Interview Assistant.

Provides a professional, responsive PySide6 GUI with:
  • Model/runtime status indicators
  • Microphone start/stop
  • Live transcription text area
  • Suggest Answer with progressive LLM rendering
  • Clear/reset functionality
  • Non-blocking async operations for all heavy work
"""

from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from app import __app_name__, __version__
from app.audio.capture import MicrophoneRecorder, list_input_devices
from app.config import (
    APP_TITLE,
    WINDOW_MIN_HEIGHT,
    WINDOW_MIN_WIDTH,
)
from app.llm.client import OllamaClient
from app.transcription.engine import WhisperTranscriber
from app.workers.threads import (
    AudioTranscriptionWorker,
    LLMWorker,
    ModelLoaderWorker,
)

logger = logging.getLogger(__name__)

# ── Stylesheet ──────────────────────────────────────────────────────────

STYLESHEET = """
QMainWindow {
    background-color: #1e1e2e;
}
QWidget#centralWidget {
    background-color: #1e1e2e;
}
QLabel {
    color: #cdd6f4;
    font-size: 13px;
}
QLabel#titleLabel {
    color: #cba6f7;
    font-size: 22px;
    font-weight: bold;
}
QLabel#subtitleLabel {
    color: #a6adc8;
    font-size: 12px;
}
QLabel#statusIndicator {
    font-size: 12px;
    padding: 4px 10px;
    border-radius: 6px;
    background-color: #313244;
    color: #a6adc8;
}
QLabel#sectionLabel {
    color: #89b4fa;
    font-size: 14px;
    font-weight: bold;
    padding-top: 4px;
}
QPlainTextEdit {
    background-color: #181825;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 8px;
    padding: 10px;
    font-size: 13px;
    font-family: 'Segoe UI', 'Consolas', monospace;
    selection-background-color: #585b70;
}
QPlainTextEdit:focus {
    border-color: #89b4fa;
}
QPushButton {
    font-size: 13px;
    font-weight: 600;
    padding: 8px 20px;
    border-radius: 8px;
    border: none;
    color: #1e1e2e;
}
QPushButton#startBtn {
    background-color: #a6e3a1;
}
QPushButton#startBtn:hover {
    background-color: #94e2d5;
}
QPushButton#startBtn:disabled {
    background-color: #45475a;
    color: #6c7086;
}
QPushButton#stopBtn {
    background-color: #f38ba8;
}
QPushButton#stopBtn:hover {
    background-color: #eba0ac;
}
QPushButton#stopBtn:disabled {
    background-color: #45475a;
    color: #6c7086;
}
QPushButton#suggestBtn {
    background-color: #89b4fa;
}
QPushButton#suggestBtn:hover {
    background-color: #74c7ec;
}
QPushButton#suggestBtn:disabled {
    background-color: #45475a;
    color: #6c7086;
}
QPushButton#clearBtn {
    background-color: #f9e2af;
    color: #1e1e2e;
}
QPushButton#clearBtn:hover {
    background-color: #f5c2e7;
}
QPushButton#cancelLlmBtn {
    background-color: #fab387;
    color: #1e1e2e;
}
QPushButton#cancelLlmBtn:hover {
    background-color: #f9e2af;
}
QStatusBar {
    background-color: #181825;
    color: #a6adc8;
    font-size: 12px;
    border-top: 1px solid #313244;
}
"""


class MainWindow(QMainWindow):
    """Primary application window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)
        self.resize(1000, 780)

        # ── Core components ─────────────────────────────────────────────
        self._transcriber = WhisperTranscriber()
        self._recorder = MicrophoneRecorder()
        self._ollama = OllamaClient()
        self._active_model: Optional[str] = None

        # ── Worker references ───────────────────────────────────────────
        self._model_loader: Optional[ModelLoaderWorker] = None
        self._audio_worker: Optional[AudioTranscriptionWorker] = None
        self._llm_worker: Optional[LLMWorker] = None

        # ── Build UI ────────────────────────────────────────────────────
        self._build_ui()
        self.setStyleSheet(STYLESHEET)

        # ── Initial status checks ───────────────────────────────────────
        self._check_runtime_status()
        self._load_whisper_model()

    # ════════════════════════════════════════════════════════════════════
    # UI Construction
    # ════════════════════════════════════════════════════════════════════

    def _build_ui(self) -> None:
        central = QWidget()
        central.setObjectName("centralWidget")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(24, 20, 24, 12)
        root.setSpacing(10)

        # ── Header ──────────────────────────────────────────────────────
        title = QLabel(APP_TITLE)
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(title)

        subtitle = QLabel(
            "Local speech-to-text + LLM answer suggestions — fully offline"
        )
        subtitle.setObjectName("subtitleLabel")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(subtitle)

        # ── Status row ──────────────────────────────────────────────────
        status_row = QHBoxLayout()
        status_row.setSpacing(12)

        self._whisper_status = QLabel("⏳ Whisper: loading…")
        self._whisper_status.setObjectName("statusIndicator")
        status_row.addWidget(self._whisper_status)

        self._ollama_status = QLabel("⏳ Ollama: checking…")
        self._ollama_status.setObjectName("statusIndicator")
        status_row.addWidget(self._ollama_status)

        self._mic_status = QLabel("🎤 Mic: idle")
        self._mic_status.setObjectName("statusIndicator")
        status_row.addWidget(self._mic_status)

        status_row.addStretch()
        root.addLayout(status_row)

        # ── Transcription section ───────────────────────────────────────
        lbl_trans = QLabel("📝  Transcription")
        lbl_trans.setObjectName("sectionLabel")
        root.addWidget(lbl_trans)

        self._transcript_edit = QPlainTextEdit()
        self._transcript_edit.setPlaceholderText(
            "Transcribed speech will appear here. You can also edit the text before requesting an answer."
        )
        self._transcript_edit.setMinimumHeight(140)
        root.addWidget(self._transcript_edit, stretch=3)

        # ── Recording controls ──────────────────────────────────────────
        rec_row = QHBoxLayout()
        rec_row.setSpacing(12)

        self._start_btn = QPushButton("▶  Start Recording")
        self._start_btn.setObjectName("startBtn")
        self._start_btn.clicked.connect(self._on_start)
        rec_row.addWidget(self._start_btn)

        self._stop_btn = QPushButton("⏹  Stop Recording")
        self._stop_btn.setObjectName("stopBtn")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop)
        rec_row.addWidget(self._stop_btn)

        self._clear_btn = QPushButton("🗑  Clear All")
        self._clear_btn.setObjectName("clearBtn")
        self._clear_btn.clicked.connect(self._on_clear)
        rec_row.addWidget(self._clear_btn)

        rec_row.addStretch()
        root.addLayout(rec_row)

        # ── AI Answer section ───────────────────────────────────────────
        lbl_answer = QLabel("🤖  Suggested Answer")
        lbl_answer.setObjectName("sectionLabel")
        root.addWidget(lbl_answer)

        self._answer_edit = QPlainTextEdit()
        self._answer_edit.setPlaceholderText(
            "Click 'Suggest Answer' to get an AI-generated response to the transcribed question."
        )
        self._answer_edit.setReadOnly(True)
        self._answer_edit.setMinimumHeight(140)
        root.addWidget(self._answer_edit, stretch=3)

        # ── Answer controls ─────────────────────────────────────────────
        ans_row = QHBoxLayout()
        ans_row.setSpacing(12)

        self._suggest_btn = QPushButton("💡  Suggest Answer")
        self._suggest_btn.setObjectName("suggestBtn")
        self._suggest_btn.clicked.connect(self._on_suggest)
        ans_row.addWidget(self._suggest_btn)

        self._cancel_llm_btn = QPushButton("✖  Cancel Generation")
        self._cancel_llm_btn.setObjectName("cancelLlmBtn")
        self._cancel_llm_btn.setEnabled(False)
        self._cancel_llm_btn.clicked.connect(self._on_cancel_llm)
        ans_row.addWidget(self._cancel_llm_btn)

        ans_row.addStretch()
        root.addLayout(ans_row)

        # ── Status bar ──────────────────────────────────────────────────
        self._statusbar = QStatusBar()
        self.setStatusBar(self._statusbar)
        self._statusbar.showMessage(f"{__app_name__} v{__version__} — ready")

    # ════════════════════════════════════════════════════════════════════
    # Runtime status checks
    # ════════════════════════════════════════════════════════════════════

    def _check_runtime_status(self) -> None:
        """Check Ollama reachability and model availability."""
        # Microphone
        devices = list_input_devices()
        if devices:
            self._mic_status.setText(f"🎤 Mic: {len(devices)} device(s)")
            self._mic_status.setStyleSheet(
                "background-color: #313244; color: #a6e3a1;"
            )
        else:
            self._mic_status.setText("🎤 Mic: none found")
            self._mic_status.setStyleSheet(
                "background-color: #313244; color: #f38ba8;"
            )
            self._start_btn.setEnabled(False)

        # Ollama
        if self._ollama.is_reachable():
            model = self._ollama.find_model()
            if model:
                self._active_model = model
                self._ollama_status.setText(f"✅ Ollama: {model}")
                self._ollama_status.setStyleSheet(
                    "background-color: #313244; color: #a6e3a1;"
                )
            else:
                self._ollama_status.setText("⚠️ Ollama: no model")
                self._ollama_status.setStyleSheet(
                    "background-color: #313244; color: #f9e2af;"
                )
                self._statusbar.showMessage(
                    "Ollama is running but no model is installed. "
                    "Run: ollama pull qwen2.5:3b"
                )
        else:
            self._ollama_status.setText("❌ Ollama: offline")
            self._ollama_status.setStyleSheet(
                "background-color: #313244; color: #f38ba8;"
            )
            self._statusbar.showMessage(
                "Ollama not detected. Install from https://ollama.com and start it."
            )

    def _load_whisper_model(self) -> None:
        """Load the Whisper model in a background thread."""
        self._whisper_status.setText("⏳ Whisper: loading…")
        self._start_btn.setEnabled(False)

        self._model_loader = ModelLoaderWorker(self._transcriber)
        self._model_loader.finished.connect(self._on_model_loaded)
        self._model_loader.start()

    @Slot(bool, str)
    def _on_model_loaded(self, success: bool, message: str) -> None:
        if success:
            self._whisper_status.setText("✅ Whisper: ready")
            self._whisper_status.setStyleSheet(
                "background-color: #313244; color: #a6e3a1;"
            )
            # Enable start only if mic is available
            devices = list_input_devices()
            self._start_btn.setEnabled(len(devices) > 0)
        else:
            self._whisper_status.setText("❌ Whisper: failed")
            self._whisper_status.setStyleSheet(
                "background-color: #313244; color: #f38ba8;"
            )
        self._statusbar.showMessage(message, 8000)
        logger.info("Model load result: %s — %s", success, message)

    # ════════════════════════════════════════════════════════════════════
    # Recording controls
    # ════════════════════════════════════════════════════════════════════

    @Slot()
    def _on_start(self) -> None:
        """Start microphone recording and live transcription."""
        if not self._transcriber.is_loaded:
            self._statusbar.showMessage("Whisper model is not loaded yet")
            return

        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._mic_status.setText("🎤 Mic: recording…")
        self._mic_status.setStyleSheet(
            "background-color: #313244; color: #f38ba8;"
        )

        self._audio_worker = AudioTranscriptionWorker(
            self._recorder, self._transcriber
        )
        self._audio_worker.text_segment.connect(self._on_text_segment)
        self._audio_worker.error.connect(self._on_recording_error)
        self._audio_worker.status.connect(
            lambda msg: self._statusbar.showMessage(msg, 5000)
        )
        self._audio_worker.recording_stopped.connect(self._on_recording_finished)
        self._audio_worker.start()

    @Slot()
    def _on_stop(self) -> None:
        """Stop recording gracefully."""
        if self._audio_worker is not None:
            self._audio_worker.request_stop()
            self._stop_btn.setEnabled(False)
            self._statusbar.showMessage("Stopping recording…")

    @Slot(str)
    def _on_text_segment(self, text: str) -> None:
        """Append a transcribed segment to the text area."""
        current = self._transcript_edit.toPlainText()
        if current and not current.endswith(" "):
            self._transcript_edit.appendPlainText(" " + text)
        else:
            self._transcript_edit.appendPlainText(text)

        # Auto-scroll to bottom
        scrollbar = self._transcript_edit.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    @Slot(str)
    def _on_recording_error(self, msg: str) -> None:
        self._statusbar.showMessage(f"Error: {msg}", 10000)
        logger.error("Recording error: %s", msg)

    @Slot()
    def _on_recording_finished(self) -> None:
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._mic_status.setText("🎤 Mic: idle")
        self._mic_status.setStyleSheet(
            "background-color: #313244; color: #a6adc8;"
        )
        self._statusbar.showMessage("Recording stopped", 5000)

    # ════════════════════════════════════════════════════════════════════
    # LLM Answer
    # ════════════════════════════════════════════════════════════════════

    @Slot()
    def _on_suggest(self) -> None:
        """Send the transcript to the LLM and stream the response."""
        transcript = self._transcript_edit.toPlainText().strip()
        if not transcript:
            self._statusbar.showMessage("Nothing to send — record or type a question first")
            return

        if not self._ollama.is_reachable():
            self._statusbar.showMessage(
                "Ollama is not reachable. Start it and try again."
            )
            return

        if not self._active_model:
            # Re-check
            model = self._ollama.find_model()
            if model:
                self._active_model = model
            else:
                self._statusbar.showMessage(
                    "No LLM model found. Run: ollama pull qwen2.5:3b"
                )
                return

        # Clear previous answer
        self._answer_edit.setPlainText("")
        self._suggest_btn.setEnabled(False)
        self._cancel_llm_btn.setEnabled(True)

        self._llm_worker = LLMWorker(
            self._ollama, transcript, model=self._active_model
        )
        self._llm_worker.token.connect(self._on_llm_token)
        self._llm_worker.finished.connect(self._on_llm_finished)
        self._llm_worker.error.connect(self._on_llm_error)
        self._llm_worker.started.connect(
            lambda: self._statusbar.showMessage("Generating answer…")
        )
        self._llm_worker.start()

    @Slot()
    def _on_cancel_llm(self) -> None:
        if self._llm_worker is not None:
            self._llm_worker.cancel()
            self._cancel_llm_btn.setEnabled(False)
            self._statusbar.showMessage("Cancelling generation…")

    @Slot(str)
    def _on_llm_token(self, token: str) -> None:
        """Append a streamed token to the answer panel (progressive render)."""
        cursor = self._answer_edit.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(token)
        self._answer_edit.setTextCursor(cursor)
        # Auto-scroll
        scrollbar = self._answer_edit.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    @Slot(str)
    def _on_llm_finished(self, full_text: str) -> None:
        self._suggest_btn.setEnabled(True)
        self._cancel_llm_btn.setEnabled(False)
        self._statusbar.showMessage(
            f"Answer complete ({len(full_text)} chars)", 8000
        )

    @Slot(str)
    def _on_llm_error(self, msg: str) -> None:
        self._suggest_btn.setEnabled(True)
        self._cancel_llm_btn.setEnabled(False)
        self._answer_edit.setPlainText(f"[Error] {msg}")
        self._statusbar.showMessage(f"LLM error: {msg}", 10000)
        logger.error("LLM error: %s", msg)

    # ════════════════════════════════════════════════════════════════════
    # Clear / Reset
    # ════════════════════════════════════════════════════════════════════

    @Slot()
    def _on_clear(self) -> None:
        """Clear all transcription and answer text."""
        self._transcript_edit.setPlainText("")
        self._answer_edit.setPlainText("")
        self._statusbar.showMessage("Cleared", 3000)

    # ════════════════════════════════════════════════════════════════════
    # Cleanup
    # ════════════════════════════════════════════════════════════════════

    def closeEvent(self, event) -> None:
        """Clean up threads and audio streams on window close."""
        logger.info("Application closing — cleaning up…")

        # Stop recording
        if self._audio_worker is not None and self._audio_worker.isRunning():
            self._audio_worker.request_stop()
            self._audio_worker.wait(3000)

        # Cancel LLM
        if self._llm_worker is not None and self._llm_worker.isRunning():
            self._llm_worker.cancel()
            self._llm_worker.wait(3000)

        # Wait for model loader
        if self._model_loader is not None and self._model_loader.isRunning():
            self._model_loader.wait(5000)

        # Stop recorder just in case
        if self._recorder.is_recording:
            self._recorder.stop()

        event.accept()
