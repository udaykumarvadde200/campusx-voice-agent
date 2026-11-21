from unittest.mock import patch

from pipecat.pipeline.pipeline import Pipeline
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.services.whisper.stt import WhisperSTTService

from campusx_voice_agent.tools import GUIDANCE_SYSTEM_PROMPT, GUIDANCE_TOOLS


def test_ollama_llm_points_at_local_server():
    llm = OLLamaLLMService(
        base_url="http://localhost:11434/v1",
        settings=OLLamaLLMService.Settings(
            model="qwen2.5:7b",
            system_instruction=GUIDANCE_SYSTEM_PROMPT,
            temperature=0.3,
            max_tokens=220,
        ),
    )
    params = llm.build_chat_completion_params({"messages": [], "tools": [], "tool_choice": "auto"})
    assert params["model"] == "qwen2.5:7b"
    assert params["max_tokens"] == 220
    assert "localhost:11434" in str(llm._client.base_url)


def test_whisper_settings_and_pipeline_builds():
    with patch.object(WhisperSTTService, "_load"):  # skip the model download in tests
        stt = WhisperSTTService(
            device="cpu",
            compute_type="int8",
            settings=WhisperSTTService.Settings(model="small"),
        )
    llm = OLLamaLLMService(settings=OLLamaLLMService.Settings(model="qwen2.5:7b"))
    Pipeline([stt, llm])
    assert [t.__name__ for t in GUIDANCE_TOOLS] == [
        "search_campusx_knowledge_base",
        "get_available_guidance_slots",
        "confirm_booking_details",
        "book_free_guidance_session",
    ]
