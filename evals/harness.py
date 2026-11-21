"""Runs the real agent brain (system prompt + tool schemas + tool code) outside the voice pipeline.

Text in, text out. STT/TTS/VAD are not involved here; they get their own evals later.
The prompt, the tool schemas (built by Pipecat's own adapter) and the tool implementations
are the same ones the live agent uses, so results transfer to the voice agent.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from types import SimpleNamespace

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipecat.adapters.services.open_ai_adapter import OpenAILLMAdapter  # noqa: E402
from pipecat.processors.aggregators.llm_context import LLMContext  # noqa: E402

from campusx_voice_agent.booking_flow import (
    allowed_tool_names,
    apply_booking_state,
    ever_booked,
    forced_readback,
    slots_were_offered,
)  # noqa: E402
from campusx_voice_agent.booking_gate import booking_confirmed_this_turn  # noqa: E402
from campusx_voice_agent.grounding import inject_kb_context  # noqa: E402
from campusx_voice_agent.reply_guard import ReplyGuard  # noqa: E402
from campusx_voice_agent.tools import ALLOWED_PHRASES, GUIDANCE_SYSTEM_PROMPT, GUIDANCE_TOOLS  # noqa: E402

ChatFn = Callable[[list[dict], list[dict]], Awaitable[dict]]


def tools_schema() -> list[dict]:
    """OpenAI-format tool schemas, produced by the same Pipecat adapter the live agent uses."""
    ctx = LLMContext(messages=[{"role": "user", "content": "hi"}], tools=GUIDANCE_TOOLS)
    params = OpenAILLMAdapter().get_llm_invocation_params(
        ctx, system_instruction=GUIDANCE_SYSTEM_PROMPT, convert_developer_to_user=False
    )
    return params["tools"]


def make_ollama_chat(
    base_url: str, model: str, temperature: float, max_tokens: int, seed: int | None = None
) -> ChatFn:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=base_url, api_key="ollama", timeout=240)

    async def chat(messages: list[dict], tools: list[dict]) -> dict:
        kwargs = {}
        if seed is not None:
            kwargs["seed"] = seed
        resp = await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        m = resp.choices[0].message
        calls = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in (m.tool_calls or [])
        ]
        return {"content": m.content, "tool_calls": calls}

    return chat


class ToolExecutor:
    """Calls the real tool functions with a stand-in for Pipecat's FunctionCallParams."""

    def __init__(self) -> None:
        self.registry = {fn.__name__: fn for fn in GUIDANCE_TOOLS}

    async def call(self, name: str, args: dict | None, messages: list[dict] | None = None):
        fn = self.registry.get(name)
        if fn is None:
            return None, f"unknown tool {name!r}"
        if args is None:
            return None, "arguments were not valid JSON"
        captured: dict = {}

        async def result_callback(result, *a, **k):
            captured["result"] = result

        try:
            # Like Pipecat's FunctionCallParams: the booking gate reads the real transcript from it.
            context = LLMContext(messages=list(messages or []))
            await fn(SimpleNamespace(result_callback=result_callback, context=context), **args)
        except TypeError as exc:  # wrong/missing argument names
            return None, f"bad arguments: {exc}"
        except Exception as exc:  # noqa: BLE001 - any tool crash is a finding
            return None, f"{type(exc).__name__}: {exc}"
        return captured.get("result"), None


CONFIRM_AGAIN_TEXT = "Yes, that's correct."


async def run_conversation(
    user_turns: list[str],
    chat: ChatFn,
    schema: list[dict],
    max_rounds: int = 4,
    *,
    grounding: bool = True,
    guard: bool = True,
    booking_state: bool = True,
    confirm_again: bool = False,
) -> list[dict]:
    """Run scripted caller turns through the brain exactly as the live pipeline wires it.

    ``grounding``: search the knowledge base in code before the model answers (live: KnowledgeContextProcessor).
    ``guard``: filter what the caller hears (live: ReplyGuardProcessor).
    ``booking_state``: hide the booking tool until read-back + yes (live: BookingStateProcessor).
    ``confirm_again``: a simulated caller who, after the scripted lines run out on a code-written read-back
    without a booking, answers \"Yes, that's correct.\" (at most twice). Opt in, and only for flows that
    should end in a booking; it measures end-to-end success, not first-try behaviour. The first three default to the live setup;
    turn them off only to measure what each one contributes.
    """
    executor = ToolExecutor()
    messages: list[dict] = [{"role": "system", "content": GUIDANCE_SYSTEM_PROMPT}]
    turns: list[dict] = []
    pending = [(text, False) for text in user_turns]
    simulated = 0
    while pending:
        user_text, is_simulated = pending.pop(0)
        messages.append({"role": "user", "content": user_text})
        retrieval = inject_kb_context(messages) if grounding else None
        offered = schema
        if booking_state:
            state = apply_booking_state(messages)
            names = allowed_tool_names([t["function"]["name"] for t in schema], state)
            offered = [t for t in schema if t["function"]["name"] in names]
        started = time.perf_counter()
        calls: list[dict] = []
        raw_text = ""
        rounds = 0
        for rounds in range(1, max_rounds + 1):
            msg = await chat(messages, offered)
            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                raw_text = (msg.get("content") or "").strip()
                break
            messages.append(
                {"role": "assistant", "content": msg.get("content") or "", "tool_calls": tool_calls}
            )
            for tc in tool_calls:
                name = tc["function"]["name"]
                raw = tc["function"]["arguments"]
                try:
                    args = json.loads(raw) if isinstance(raw, str) else dict(raw or {})
                    if not isinstance(args, dict):
                        args = None
                except (json.JSONDecodeError, TypeError, ValueError):
                    args = None
                result, error = await executor.call(name, args, messages)
                calls.append({"name": name, "args": args, "result": result, "error": error})
                payload = {"error": error} if error else result
                messages.append(
                    {"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(payload, default=str)}
                )

        interventions: list[str] = []
        text = raw_text
        if guard:
            booked = booking_confirmed_this_turn(messages)
            reply_guard = ReplyGuard(
                GUIDANCE_SYSTEM_PROMPT,
                allowed_phrases=ALLOWED_PHRASES,
                booked=lambda: booked,
                in_booking=lambda: slots_were_offered(messages),
            )
            text = reply_guard.clean(raw_text)
            interventions = list(reply_guard.interventions)
            readback = forced_readback(messages) if booking_state else None
            if readback:  # live: ReplyGuardProcessor speaks the code-written read-back instead
                text = readback
                interventions.append("forced_readback")
        # The context keeps what was SPOKEN, like the live assistant aggregator does.
        messages.append({"role": "assistant", "content": text})
        turns.append(
            {
                "user": user_text,
                "text": text,
                "raw_text": raw_text,
                "guard": interventions,
                "retrieval": retrieval,
                "tools_offered": [t["function"]["name"] for t in offered],
                "tool_calls": calls,
                "latency_s": round(time.perf_counter() - started, 3),
                "rounds": rounds,
                "hit_round_limit": rounds == max_rounds and not raw_text,
                "simulated_user": is_simulated,
            }
        )
        if (
            confirm_again
            and not pending
            and simulated < 2
            and "forced_readback" in interventions
            and not ever_booked(messages)
        ):
            pending.append((CONFIRM_AGAIN_TEXT, True))
            simulated += 1
    return turns


def new_isolated_db() -> str:
    """Point the booking tools at a throwaway SQLite file so evals never touch real bookings."""
    path = os.path.join(tempfile.mkdtemp(prefix="campusx_eval_"), "bookings.db")
    os.environ["GUIDANCE_DB_PATH"] = path
    return path


def read_bookings(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute("SELECT * FROM bookings")]
