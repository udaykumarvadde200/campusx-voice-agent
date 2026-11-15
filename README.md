# CampusX Voice Agent (fully local)

No API keys, no cloud credits.

| Stage | Tool |
|---|---|
| STT | faster-whisper `small` on CPU (int8) |
| LLM | Ollama `qwen2.5:7b` |
| TTS | Kokoro (kokoro-onnx) voice `af_heart` on CPU |
| VAD / transport | Silero VAD, WebRTC (browser at http://localhost:7860) |

## Setup

1. Install Ollama and pull the model: `ollama pull qwen2.5:7b`
2. Python 3.12 venv, then `pip install -r requirements.txt`
3. `python main.py download-models` (one-time, a few hundred MB)
4. `python main.py` and open http://localhost:7860

Other commands: `python main.py check` (is Ollama up and the model pulled?),
`python main.py bookings`, `python main.py moss-index` (optional, needs `pip install pipecat-moss`).

## Tuning

* Replies too slow: `ollama pull llama3.2:3b` and set `OLLAMA_MODEL=llama3.2:3b`.
* Misheard words: set `WHISPER_MODEL=medium` (slower).
* The first reply after idle is slow while Ollama loads the model.
