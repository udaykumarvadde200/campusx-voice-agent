"""CampusX guidance-booking voice agent running fully locally (no cloud keys).

Pipeline: WebRTC mic -> faster-whisper STT (CPU) -> user aggregator -> (optional
Moss) -> Ollama LLM -> Kokoro TTS (CPU) -> WebRTC speaker -> assistant aggregator.
"""

from __future__ import annotations

import os

from pipecat.frames.frames import TTSSpeakFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.runner.types import RunnerArguments
from pipecat.services.kokoro.tts import KokoroTTSService
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.transports.base_transport import BaseTransport

from campusx_voice_agent.moss import make_moss_retrieval_processor
from campusx_voice_agent.pipeline import (
    MetricsPrinter,
    make_transport,
    make_user_turn_params,
    run_pipeline,
)
from campusx_voice_agent.tools import GUIDANCE_SYSTEM_PROMPT, GUIDANCE_TOOLS

DEFAULT_OLLAMA_MODEL = "qwen2.5:7b"
DEFAULT_WHISPER_MODEL = "small"
DEFAULT_KOKORO_VOICE = "af_heart"


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    # Whisper runs on CPU with int8 so the GPU's VRAM stays free for the Ollama model.
    stt = WhisperSTTService(
        device=os.getenv("WHISPER_DEVICE", "cpu"),
        compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
        settings=WhisperSTTService.Settings(
            model=os.getenv("WHISPER_MODEL", DEFAULT_WHISPER_MODEL),
        ),
    )
    llm = OLLamaLLMService(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        settings=OLLamaLLMService.Settings(
            model=os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
            system_instruction=GUIDANCE_SYSTEM_PROMPT,
            temperature=0.3,
            max_tokens=int(os.getenv("OLLAMA_MAX_TOKENS", "220")),
        ),
    )
    tts = KokoroTTSService(
        settings=KokoroTTSService.Settings(
            voice=os.getenv("KOKORO_VOICE", DEFAULT_KOKORO_VOICE),
        ),
    )

    @llm.event_handler("on_function_calls_started")
    async def on_function_calls_started(service, function_calls):
        await tts.queue_frame(
            TTSSpeakFrame("One moment while I check that.", append_to_context=False)
        )

    context = LLMContext(tools=GUIDANCE_TOOLS)
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=make_user_turn_params(interruptions=True),
    )
    moss_retrieval = await make_moss_retrieval_processor()

    processors = [transport.input(), stt, user_aggregator]
    if moss_retrieval is not None:
        processors.append(moss_retrieval)
    processors.extend(
        [llm, tts, transport.output(), assistant_aggregator, MetricsPrinter()]
    )

    await run_pipeline(transport, runner_args, processors, context=context, metrics=True)


async def bot(runner_args: RunnerArguments):
    """WebRTC bot entry point used by Pipecat's development runner."""
    transport = await make_transport(runner_args)
    await run_bot(transport, runner_args)
