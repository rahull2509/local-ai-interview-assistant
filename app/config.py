"""Application configuration constants."""

# ── Window ──────────────────────────────────────────────────────────────
APP_TITLE = "Local AI Interview Assistant"
WINDOW_MIN_WIDTH = 900
WINDOW_MIN_HEIGHT = 700

# ── Audio ───────────────────────────────────────────────────────────────
AUDIO_SAMPLE_RATE = 16000          # 16 kHz mono, required by Whisper
AUDIO_CHANNELS = 1
AUDIO_DTYPE = "float32"
AUDIO_CHUNK_DURATION_S = 5         # seconds of audio per transcription chunk
AUDIO_OVERLAP_S = 0.5              # overlap between chunks to reduce boundary cuts

# ── Transcription (faster-whisper) ──────────────────────────────────────
WHISPER_MODEL_SIZE = "base"        # "tiny" if RAM/perf is an issue
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"
WHISPER_BEAM_SIZE = 3
WHISPER_LANGUAGE = "en"

# ── LLM (Ollama) ───────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_CHAT_ENDPOINT = "/api/chat"
OLLAMA_TAGS_ENDPOINT = "/api/tags"
OLLAMA_DEFAULT_MODEL = "qwen2.5:3b"  # preferred; detected at runtime
OLLAMA_TIMEOUT_S = 120
OLLAMA_MAX_TRANSCRIPT_CHARS = 4000   # bound context sent to LLM

OLLAMA_SYSTEM_PROMPT = (
    "You are a helpful interview preparation assistant. "
    "The user will speak a question aloud and you will see the transcription. "
    "Provide a clear, concise, and accurate answer to the question. "
    "If you are unsure or the question is ambiguous, say so honestly. "
    "Do not invent facts. Keep your answer focused and practical."
)
