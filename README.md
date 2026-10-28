# Local AI Interview Assistant

A Windows desktop application that captures microphone audio, transcribes speech locally using [faster-whisper](https://github.com/SYSTRAN/faster-whisper), and generates AI-powered answer suggestions using a locally running [Ollama](https://ollama.com) LLM — all completely offline after initial setup.

![Python 3.12](https://img.shields.io/badge/Python-3.12-blue)
![PySide6](https://img.shields.io/badge/GUI-PySide6-green)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow)

---

## Features

| Feature | Implementation |
|---|---|
| **Desktop GUI** | PySide6 with dark Catppuccin-inspired theme |
| **Microphone Capture** | Real-time via `sounddevice`, 16 kHz mono float32 |
| **Speech-to-Text** | `faster-whisper` (base model, CPU int8, VAD filtering) |
| **LLM Answer Suggestions** | Ollama local API with streaming progressive display |
| **Progressive Rendering** | Tokens streamed into UI as they are generated |
| **Thread Safety** | QThread workers with Qt signals — GUI never blocks |
| **Error Handling** | Graceful handling of missing mic, missing model, Ollama offline |
| **Fully Offline** | After initial model downloads, no internet required |

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                    MainWindow (UI)                    │
│  ┌────────┐  ┌──────────┐  ┌───────────┐  ┌───────┐ │
│  │ Status │  │Transcript│  │  Answer   │  │Buttons│ │
│  │Indicators│ │ TextArea │  │ TextArea  │  │       │ │
│  └────────┘  └──────────┘  └───────────┘  └───────┘ │
└──────────┬───────────┬──────────┬────────────────────┘
           │ Qt Signals│          │ Qt Signals
    ┌──────┴──────┐   ┌┴──────────┴──┐
    │ Audio+Trans │   │  LLM Worker  │
    │   Worker    │   │  (QThread)   │
    │  (QThread)  │   └──────┬───────┘
    └──────┬──────┘          │
           │                 │ HTTP streaming
    ┌──────┴──────┐   ┌──────┴───────┐
    │ Microphone  │   │   Ollama     │
    │ Recorder    │   │ Client       │
    │(sounddevice)│   │(requests)    │
    └──────┬──────┘   └──────┬───────┘
           │                 │
    ┌──────┴──────┐   ┌──────┴───────┐
    │  Whisper    │   │ Ollama Server│
    │ Transcriber │   │ (localhost   │
    │(faster-     │   │  :11434)     │
    │  whisper)   │   └──────────────┘
    └─────────────┘
```

**Key design decisions:**
- All heavy operations (model loading, audio capture, transcription, LLM inference) run in QThread workers
- GUI thread only handles Qt signals — never blocks
- Audio is captured in 5-second chunks with 0.5s overlap to reduce boundary artifacts
- VAD (Voice Activity Detection) filtering in Whisper reduces noise-induced false transcriptions
- Transcript context sent to LLM is bounded to 4000 characters
- Model loading is lazy — first use triggers download/load

---

## Prerequisites

| Requirement | Details |
|---|---|
| **OS** | Windows 10/11, 64-bit |
| **Python** | 3.10 – 3.12 |
| **RAM** | 16 GB recommended (8 GB minimum) |
| **GPU** | Not required — runs on CPU |
| **Microphone** | Any input device recognized by Windows |
| **Ollama** | Required for LLM answer suggestions |

---

## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd local-ai-interview-assistant
```

### 2. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Ollama

Download and install from: https://ollama.com/download

After installation, Ollama runs as a system service on `localhost:11434`.

### 5. Pull a local LLM model

```bash
ollama pull qwen2.5:3b
```

This downloads the **Qwen 2.5 3B Instruct** model (~2 GB). For faster but lower quality responses:

```bash
ollama pull qwen2.5:1.5b
```

### 6. Verify Ollama is running

```bash
# Check Ollama version
ollama --version

# List installed models
ollama list

# Test chat
ollama run qwen2.5:3b "Hello, how are you?"
```

---

## Running the Application

### Option A: Using run.bat (Windows)

```bash
run.bat
```

### Option B: Using Python directly

```bash
venv\Scripts\activate
python -m app.main
```

### First launch

On the first launch, the Whisper speech recognition model (`base`, ~140 MB) will be downloaded automatically from Hugging Face. This takes 30–60 seconds depending on your internet connection. Subsequent launches will use the cached model.

---

## Usage

1. **Check status indicators** at the top of the window:
   - ✅ Whisper: ready — speech recognition model loaded
   - ✅ Ollama: model-name — LLM connected and model detected
   - 🎤 Mic: N device(s) — microphone available

2. **Click "▶ Start Recording"** to begin capturing audio from your microphone

3. **Speak your question** — transcribed text appears in real time (with ~5 second latency per chunk)

4. **Click "⏹ Stop Recording"** when done speaking

5. **Edit the transcription** if needed (the text area is editable)

6. **Click "💡 Suggest Answer"** — the LLM generates an answer that streams progressively into the response panel

7. **Click "✖ Cancel Generation"** to stop the LLM mid-generation

8. **Click "🗑 Clear All"** to reset both panels

---

## Microphone Troubleshooting

| Issue | Solution |
|---|---|
| "Mic: none found" | Check Windows Sound Settings → Input devices |
| No sound captured | Set the correct default input device in Windows |
| Audio too quiet | Increase microphone volume in Windows Sound Settings |
| Bluetooth mic issues | Ensure the Bluetooth audio device is connected and set as default |

---

## Project Structure

```
local-ai-interview-assistant/
├── app/
│   ├── __init__.py           # Package metadata, version
│   ├── main.py               # Application entry point
│   ├── config.py             # All configuration constants
│   ├── audio/
│   │   ├── __init__.py
│   │   └── capture.py        # Microphone capture with sounddevice
│   ├── transcription/
│   │   ├── __init__.py
│   │   └── engine.py         # faster-whisper transcription engine
│   ├── llm/
│   │   ├── __init__.py
│   │   └── client.py         # Ollama HTTP client with streaming
│   ├── ui/
│   │   ├── __init__.py
│   │   └── main_window.py    # PySide6 main window
│   └── workers/
│       ├── __init__.py
│       └── threads.py        # QThread workers for async operations
├── tests/
│   ├── __init__.py
│   └── test_core.py          # 27 unit tests (all mocked)
├── requirements.txt
├── run.bat                    # Windows launcher script
├── .gitignore
└── README.md
```

---

## Testing

### Run all tests

```bash
venv\Scripts\activate
python -m pytest tests/ -v
```

### Test coverage

| Test Category | Count | What's Tested |
|---|---|---|
| Transcript handling | 5 | Empty audio, short audio, segment concatenation, unloaded model, 2D→1D squeeze |
| Ollama client | 8 | Reachability, model listing/parsing, model finding (exact/prefix/fallback/none), empty transcript, connection error, chunk parsing, transcript truncation |
| Audio capture | 4 | Initial state, empty queue, safe stop, device enumeration |
| UI initialization | 5 | Window creation, button existence, initial button states, clear functionality, empty-transcript suggest |
| Worker threads | 2 | Cancellation flag, error signal propagation |
| Integration tests | 2 | Real microphone device enumeration, Whisper base model load & silence transcription |
| **Total** | **29** | All 29 tests pass locally in ~1.6s |

---

## Offline Use

After completing the setup (installing dependencies, Whisper model download on first launch, and `ollama pull` for the LLM):

- **No internet is required** for any functionality
- Speech recognition runs locally via `faster-whisper` + `ctranslate2`
- LLM inference runs locally via Ollama
- No data is sent to any external service

---

## Performance Notes

**Latency characteristics (CPU-only, 16 GB RAM):**

| Operation | Typical Latency |
|---|---|
| Model loading (first launch) | 30–60s (download) + 5–10s (load) |
| Model loading (cached) | 3–8s |
| Audio chunk → transcription | 2–5s per 5s chunk |
| LLM time to first token | 2–10s (depends on model & CPU) |
| LLM full answer | 15–60s (depends on length & model) |

> **Note:** These are approximate values. Actual performance depends on your CPU, available RAM, background processes, and model size. The `tiny` Whisper model and `qwen2.5:1.5b` LLM offer faster performance at the cost of quality. Edit `app/config.py` to switch models.

---

## Known Limitations

1. **Transcription is chunk-based, not truly real-time.** Audio is buffered in ~5-second segments before transcription. This means there is inherent latency between speaking and seeing the text.

2. **Chunk boundary artifacts.** Despite the 0.5s overlap between chunks, some words at chunk boundaries may be duplicated or cut. This is a fundamental limitation of chunk-based processing.

3. **Ollama must be installed separately.** The application detects but cannot install Ollama.

4. **CPU-only inference is slow.** LLM responses take 15–60 seconds on CPU. A GPU significantly improves this but is not required.

5. **Single-language default.** Transcription defaults to English. Change `WHISPER_LANGUAGE` in `config.py` for other languages.

6. **No conversation history.** Each "Suggest Answer" sends only the current transcript, not a multi-turn history.

7. **`av` library pinned to <19.** The `av` 19.x series has a DLL loading issue on some Windows configurations with Application Control policies. Pinning to <19 resolves this.

---

## Configuration

Edit [`app/config.py`](app/config.py) to customize:

- `WHISPER_MODEL_SIZE` — `"tiny"`, `"base"`, `"small"`, `"medium"` (larger = better quality, more RAM, slower)
- `AUDIO_CHUNK_DURATION_S` — chunk size in seconds (shorter = lower latency, noisier)
- `OLLAMA_DEFAULT_MODEL` — preferred LLM model name
- `OLLAMA_SYSTEM_PROMPT` — customize the LLM's behavior
- `OLLAMA_MAX_TRANSCRIPT_CHARS` — maximum transcript context sent to the LLM

---

## License

MIT
