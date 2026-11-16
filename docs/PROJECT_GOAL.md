# PROJECT_GOAL: CampusX Local Voice Agent + Eval Suite

Paste this file at the start of every new Claude conversation about this project.

## What this is
A voice support and free-guidance-session booking agent for CampusX callers, built with Pipecat.
Callers ask about CampusX courses and schedules, or book a free 20-minute guidance session by voice.
It started as the OpenAI-based "Lecture 3 AI Voice Agent Workshop" code. The OpenAI key was unavailable and a
Hugging Face port failed with HTTP 402 (no credits), so it is now fully local and free to run.

Repo: https://github.com/udaykumarvadde200/campusx-voice-agent  (branch main)
Local path: C:\Users\uday2\Downloads\campusx_voice_agent_local\local_proj   (Windows, PowerShell 5.1, venv at .venv)

## Goal of the work
1. A voice agent that behaves reliably: answers only from verified CampusX knowledge, says so when it does not know,
   makes no promises (job, salary, discount, refund), and books sessions only after an explicit yes.
2. A rigorous, reproducible eval suite that measures every aspect (retrieval, answer quality, tools, booking,
   speech-to-text, latency, barge-in, robustness, security, stability).
3. Fix what the evals find and prove it with before/after numbers (paired comparison).
4. Intended as a portfolio project (assumption, edit if wrong).

## Stack (current, local, no keys)
- STT: faster-whisper "small", CPU, int8 (Pipecat WhisperSTTService)
- LLM: Ollama qwen2.5:7b (Pipecat OLLamaLLMService, OpenAI-compatible endpoint localhost:11434/v1). llama3.2:3b is also pulled.
- TTS: Kokoro (kokoro-onnx) voice af_heart, CPU
- VAD / turns: Silero VAD + SpeechTimeout user-turn stop (0.6 s), interruptions enabled
- Transport: Pipecat SmallWebRTC dev runner, browser at http://localhost:7860
- Tools: search_campusx_knowledge_base (local keyword search over data/knowledge_base/*.md),
  get_available_guidance_slots, book_free_guidance_session (SQLite data/guidance_bookings.db)
- Retrieval runs in code before every answer (grounding.py + knowledge_base.py); booking_gate.py and reply_guard.py enforce booking and speech rules
- Optional Moss semantic retrieval (needs its own keys; not used, not evaluated; replaces the local search when configured)
- Python 3.12 only. Pipecat pinned 1.6.0.
- Hardware: Lenovo LOQ, Core i7-13650HX, RTX 4050 6 GB VRAM. Whisper and Kokoro run on CPU so the GPU stays free for the LLM.

## Agent rules (system prompt in src/campusx_voice_agent/tools.py; since Section B the key ones are also enforced in code)
- Replies short and natural, no markdown (it is spoken aloud).
- Use the knowledge base for CampusX facts; never invent facts; if unverified say so and point to support@campusx.in.
- Never promise jobs, salaries, discounts, refunds, community links, account status, future launches, instructor availability.
- Booking: only if the caller explicitly asks; collect name, email, goal, time preference; use the availability tool first;
  offer at most 3 slots; read back name, email, goal and exact slot; call the booking tool only after an explicit yes.

## Non-goals
Paid APIs, cloud deployment, telephony, multi-language support (for now).
