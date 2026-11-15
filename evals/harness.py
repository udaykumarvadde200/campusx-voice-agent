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

from campusx_voice_agent.tools import GUIDANCE_SYSTEM_PROMPT, GUIDANCE_TOOLS  # noqa: E402

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

    async def call(self, name: str, args: dict | None):
        fn = self.registry.get(name)
        if fn is None:
            return None, f"unknown tool {name!r}"
        if args is None:
            return None, "arguments were not valid JSON"
        captured: dict = {}

        async def result_callback(result, *a, **k):
            captured["result"] = result

        try:
            await fn(SimpleNamespace(result_callback=result_callback), **args)
        except TypeError as exc:  # wrong/missing argument names
            return None, f"bad arguments: {exc}"
        except Exception as exc:  # noqa: BLE001 - any tool crash is a finding
            return None, f"{type(exc).__name__}: {exc}"
        return captured.get("result"), None


async def run_conversation(
    user_turns: list[str], chat: ChatFn, schema: list[dict], max_rounds: int = 4
) -> list[dict]:
    executor = ToolExecutor()
    messages: list[dict] = [{"role": "system", "content": GUIDANCE_SYSTEM_PROMPT}]
    turns: list[dict] = []
    for user_text in user_turns:
        messages.append({"role": "user", "content": user_text})
        started = time.perf_counter()
        calls: list[dict] = []
        text = ""
        rounds = 0
        for rounds in range(1, max_rounds + 1):
            msg = await chat(messages, schema)
            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                text = (msg.get("content") or "").strip()
                messages.append({"role": "assistant", "content": text})
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
                result, error = await executor.call(name, args)
                calls.append({"name": name, "args": args, "result": result, "error": error})
                payload = {"error": error} if error else result
                messages.append(
                    {"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(payload, default=str)}
                )
        else:
            messages.append({"role": "assistant", "content": text})
        turns.append(
            {
                "user": user_text,
                "text": text,
                "tool_calls": calls,
                "latency_s": round(time.perf_counter() - started, 3),
                "rounds": rounds,
                "hit_round_limit": rounds == max_rounds and not text,
            }
        )
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
