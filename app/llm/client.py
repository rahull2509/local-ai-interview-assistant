"""Ollama local LLM client.

Handles health checks, model discovery, and streaming chat completion
through Ollama's REST API at localhost:11434.
"""

from __future__ import annotations

import json
import logging
from typing import Generator, Optional

import requests

from app.config import (
    OLLAMA_BASE_URL,
    OLLAMA_CHAT_ENDPOINT,
    OLLAMA_DEFAULT_MODEL,
    OLLAMA_MAX_TRANSCRIPT_CHARS,
    OLLAMA_SYSTEM_PROMPT,
    OLLAMA_TAGS_ENDPOINT,
    OLLAMA_TIMEOUT_S,
)

logger = logging.getLogger(__name__)


class OllamaError(Exception):
    """Raised on Ollama communication failures."""


class OllamaClient:
    """Lightweight client for the Ollama local LLM HTTP API."""

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        default_model: str = OLLAMA_DEFAULT_MODEL,
        timeout: int = OLLAMA_TIMEOUT_S,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.timeout = timeout

    # ── Health & discovery ──────────────────────────────────────────────

    def is_reachable(self) -> bool:
        """Return True if the Ollama server responds."""
        try:
            resp = requests.get(self.base_url, timeout=5)
            return resp.status_code == 200
        except requests.ConnectionError:
            return False
        except Exception:
            return False

    def list_models(self) -> list[str]:
        """Return names of locally installed models, e.g. ['qwen2.5:3b']."""
        url = f"{self.base_url}{OLLAMA_TAGS_ENDPOINT}"
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            models = [m["name"] for m in data.get("models", [])]
            return models
        except Exception as exc:
            logger.error("Failed to list Ollama models: %s", exc)
            return []

    def find_model(self, preferred: Optional[str] = None) -> Optional[str]:
        """Find a suitable model, preferring *preferred* then the default.

        Returns the exact model name string or None.
        """
        models = self.list_models()
        if not models:
            return None

        target = preferred or self.default_model

        # Exact match
        if target in models:
            return target

        # Prefix match (e.g. 'qwen2.5:3b' matches 'qwen2.5:3b-instruct-…')
        for m in models:
            if m.startswith(target.split(":")[0]):
                return m

        # Any instruct model
        for m in models:
            if "instruct" in m.lower():
                return m

        # Fallback: first available
        return models[0]

    # ── Chat completion (streaming) ─────────────────────────────────────

    def stream_chat(
        self,
        transcript: str,
        model: Optional[str] = None,
        system_prompt: str = OLLAMA_SYSTEM_PROMPT,
    ) -> Generator[str, None, None]:
        """Stream LLM response tokens for the given *transcript*.

        Yields text chunks as they arrive from Ollama.
        Raises OllamaError on communication failures.
        """
        model = model or self.default_model
        transcript = transcript.strip()
        if not transcript:
            raise OllamaError("Empty transcript — nothing to send to the LLM")

        # Bound transcript length
        if len(transcript) > OLLAMA_MAX_TRANSCRIPT_CHARS:
            transcript = transcript[-OLLAMA_MAX_TRANSCRIPT_CHARS:]

        url = f"{self.base_url}{OLLAMA_CHAT_ENDPOINT}"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": transcript},
            ],
            "stream": True,
        }

        try:
            resp = requests.post(
                url,
                json=payload,
                stream=True,
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except requests.ConnectionError as exc:
            raise OllamaError(
                "Cannot connect to Ollama. Is it running? "
                f"(tried {self.base_url})"
            ) from exc
        except requests.HTTPError as exc:
            raise OllamaError(f"Ollama returned an error: {exc}") from exc
        except requests.Timeout as exc:
            raise OllamaError("Ollama request timed out") from exc

        try:
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    logger.debug("Skipping non-JSON line: %r", line[:100])
                    continue

                # Extract token text
                message = chunk.get("message", {})
                content = message.get("content", "")
                if content:
                    yield content

                if chunk.get("done", False):
                    return
        except requests.ConnectionError as exc:
            raise OllamaError(f"Connection lost during streaming: {exc}") from exc
        finally:
            resp.close()

    def generate_answer(
        self,
        transcript: str,
        model: Optional[str] = None,
    ) -> str:
        """Non-streaming convenience method. Returns the full response."""
        parts: list[str] = []
        for chunk in self.stream_chat(transcript, model=model):
            parts.append(chunk)
        return "".join(parts)
