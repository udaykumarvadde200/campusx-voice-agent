"""One-time model downloads and Ollama checks, so the first call does not hang."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from loguru import logger


def _ollama_root() -> str:
    base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    return base.removesuffix("/v1").rstrip("/")


def check_ollama() -> bool:
    """Return True if Ollama is running and the configured model is pulled."""
    from campusx_voice_agent.agent import DEFAULT_OLLAMA_MODEL

    model = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    try:
        with urllib.request.urlopen(f"{_ollama_root()}/api/tags", timeout=5) as resp:
            names = {m["name"] for m in json.load(resp).get("models", [])}
    except (urllib.error.URLError, OSError) as exc:
        logger.error("Cannot reach Ollama at {} ({}). Start the Ollama app first.", _ollama_root(), exc)
        return False
    wanted = model if ":" in model else f"{model}:latest"
    if wanted not in names:
        logger.error("Ollama model '{}' is not pulled. Run: ollama pull {}", model, model)
        return False
    logger.success("Ollama is running and '{}' is available", model)
    return True


def download_models() -> None:
    """Fetch the Whisper and Kokoro files now (a few hundred MB, one time)."""
    from campusx_voice_agent.agent import DEFAULT_WHISPER_MODEL

    logger.info("Downloading Whisper model (first run only)...")
    from faster_whisper import WhisperModel

    WhisperModel(os.getenv("WHISPER_MODEL", DEFAULT_WHISPER_MODEL), device="cpu", compute_type="int8")

    logger.info("Downloading Kokoro voice model (first run only)...")
    from pipecat.services.kokoro.tts import (
        KOKORO_CACHE_DIR,
        _ensure_model_files,
    )

    _ensure_model_files(KOKORO_CACHE_DIR / "kokoro-v1.0.onnx", KOKORO_CACHE_DIR / "voices-v1.0.bin")
    logger.success("Speech models are ready")
    check_ollama()
