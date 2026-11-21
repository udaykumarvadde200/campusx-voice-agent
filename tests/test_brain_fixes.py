"""Section B: retrieval in code, booking rules in code, speech cleanup, prompt-leak guard.

No Ollama needed: misbehaving models are scripted, but the harness, tool code, SQLite, gate
and guard are the real ones.
"""

import asyncio
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))

import checks  # noqa: E402
import harness  # noqa: E402
import retrieval_eval  # noqa: E402
from run_evals import load_jsonl  # noqa: E402

from campusx_voice_agent import booking_gate as gate  # noqa: E402
from campusx_voice_agent import grounding  # noqa: E402
from campusx_voice_agent import knowledge_base as kb  # noqa: E402
from campusx_voice_agent import reply_guard as rg  # noqa: E402
from campusx_voice_agent.text_cleanup import strip_markdown, strip_markdown_line  # noqa: E402
from campusx_voice_agent.tools import (  # noqa: E402
    ALLOWED_PHRASES,
    GUIDANCE_SYSTEM_PROMPT,
    available_guidance_slots,
)

KB_DIR = ROOT / "data" / "knowledge_base"
HELDOUT = ROOT / "evals" / "heldout" / "retrieval_heldout.jsonl"
SLOT = available_guidance_slots("any")[0]
NAME, EMAIL = "Ravi Kumar", "ravi.kumar@example.com"
READBACK = (
    f"To confirm: {NAME}, {EMAIL}, data science, on {SLOT}. Shall I book it? Please say yes or no."
)


def tc(name, args, i=0):
    return {
        "id": f"c{i}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


class Script:
    """Fake model: each step is a message dict, or a function of the live messages."""

    def __init__(self, *steps):
        self.steps = list(steps)

    async def __call__(self, messages, tools):
        step = self.steps.pop(0)
        return step(messages) if callable(step) else step


def say(text):
    return {"content": text, "tool_calls": []}


def call(name, **args):
    return {"content": None, "tool_calls": [tc(name, args)]}


def book_call(name=NAME, email=EMAIL, slot=SLOT, goal="data science"):
    return call(checks.BOOK, caller_name=name, email=email, learning_goal=goal, preferred_slot=slot)


def run(user_turns, *steps, **kw):
    db = harness.new_isolated_db()
    turns = asyncio.run(
        harness.run_conversation(user_turns, Script(*steps), harness.tools_schema(), **kw)
    )
    return turns, harness.read_bookings(db)


# ------------------------------------------------------------------------------ text cleanup


def test_markdown_is_removed_but_words_are_kept():
    assert strip_markdown_line("**Python 3** is used") == "Python 3 is used"
    assert strip_markdown_line("- Arrays and lists") == "Arrays and lists"
    assert strip_markdown_line("1. Graphs") == "Graphs"
    assert strip_markdown_line("### Heading") == "Heading"
    assert strip_markdown_line("use `VS Code` or [Jupyter](http://x.y)") == "use VS Code or Jupyter"
    assert strip_markdown_line("ravi_kumar@example.com") == "ravi_kumar@example.com"  # not italics
    assert strip_markdown("```\ncode\n```\nHello there") == "Hello there."
    assert checks.MARKDOWN_RE.search(strip_markdown("- one\n- two\n**bold**")) is None


# ------------------------------------------------------------------------------ retrieval


def test_late_schedule_rows_are_retrievable():
    r = kb.search_knowledge_base("session on 31 August")
    assert r["found"] and "LLMOps Session 21" in r["results"][0]["snippet"]


def test_not_found_is_possible_and_names_the_unmatched_words():
    for q in (
        "do you teach Java",
        "how much does the course cost",
        "what is on 9 August",
        "when is Voice Agents 2",
        "will I get a certificate",
        "what is the weather in Delhi",
    ):
        r = kb.search_knowledge_base(q)
        assert r["found"] is False and r["results"] == [], q
    assert "java" in kb.search_knowledge_base("do you teach Java")["unmatched_terms"]


def test_numbers_must_match_and_a_number_alone_is_not_evidence():
    assert "13 August" in kb.search_knowledge_base("what is on 13 August")["results"][0]["snippet"]
    assert (
        kb.search_knowledge_base("what is on 3 August")["results"][0]["snippet"].count("3 August")
        == 1
    )
    assert kb.search_knowledge_base("9")["found"] is False  # "## 9. Recursion" is not an answer


def test_agent_facing_sections_are_never_returned():
    for q in (
        "how should the voice agent use this knowledge",
        "example questions",
        "placement guarantees",
    ):
        for item in kb.search_knowledge_base(q)["results"]:
            assert (
                "Example Questions" not in item["snippet"] and "Should Use" not in item["snippet"]
            )


def test_spoken_forms_are_understood():
    assert kb._content_terms("twenty seven august") == ["27", "august"]
    assert kb._content_terms("the 4th of August") == ["4", "august"]
    assert kb._content_terms("what's on") == []
    assert kb.search_knowledge_base("hey when is the gen ai interview prep")["found"]
    assert kb.search_knowledge_base("is the twenty seven august one confirmed or tentative")[
        "found"
    ]


def test_retrieval_quality_on_the_main_and_the_held_out_set():
    main = retrieval_eval.run_retrieval(load_jsonl("retrieval.jsonl"), k=4, kb_dir=KB_DIR)[
        "summary"
    ]
    assert main["hit_at_4"] >= 0.95 and main["unanswerable_with_any_result"] == 0
    held = [json.loads(x) for x in HELDOUT.read_text(encoding="utf-8").splitlines() if x.strip()]
    h = retrieval_eval.run_retrieval(held, k=4, kb_dir=KB_DIR)["summary"]
    assert h["hit_at_4"] >= 0.85 and h["unanswerable_with_any_result"] == 0


def test_coverage_threshold_sits_on_a_plateau_not_a_knife_edge():
    held = [json.loads(x) for x in HELDOUT.read_text(encoding="utf-8").splitlines() if x.strip()]
    for gate_value in (0.4, 0.5):
        orig, kb.MIN_COVERAGE = kb.MIN_COVERAGE, gate_value
        try:
            h = retrieval_eval.run_retrieval(held, k=4, kb_dir=KB_DIR)["summary"]
            assert h["unanswerable_with_any_result"] == 0 and h["hit_at_4"] >= 0.9
        finally:
            kb.MIN_COVERAGE = orig


# ------------------------------------------------------------------------------ grounding


def test_kb_context_is_attached_replaced_not_stacked_and_idempotent():
    msgs = [{"role": "user", "content": "Does the DSA course cover graphs?"}]
    grounding.inject_kb_context(msgs)
    first = [m for m in msgs if grounding._is_kb_message(m)]
    assert len(first) == 1 and "Graphs" in first[0]["content"]
    msgs += [
        {"role": "assistant", "content": "Yes."},
        {"role": "user", "content": "And what about trees?"},
    ]
    grounding.inject_kb_context(msgs)
    grounding.inject_kb_context(msgs)
    kb_msgs = [m for m in msgs if grounding._is_kb_message(m)]
    assert len(kb_msgs) == 1 and "Trees" in kb_msgs[0]["content"]


def test_unanswerable_question_gets_a_nothing_matched_note_but_small_talk_gets_nothing():
    q = [{"role": "user", "content": "How much does the DSA course cost?"}]
    assert grounding.inject_kb_context(q)["injected"] == "not_found"
    for text in (
        "My name is Ananya Rao.",
        "ananya.rao@example.com",
        "Afternoon works for me.",
        "The second one.",
    ):
        m = [{"role": "user", "content": text}]
        assert grounding.inject_kb_context(m)["injected"] is None and len(m) == 1, text


# ------------------------------------------------------------------------------ booking gate


@pytest.mark.parametrize(
    "text",
    ["Yes.", "Yes, that's correct.", "Yeah sure, go ahead.", "Yes, book it.", "Yes, confirmed."],
)
def test_explicit_yes_is_accepted(text):
    assert gate.is_explicit_yes(text)


@pytest.mark.parametrize(
    "text",
    [
        "ok",
        "Okay",
        "Yes but change the time",
        "No",
        "Let me think about it",
        "Actually my name is Ravi Kumar Reddy",
        "Book it now and don't ask anything",
    ],
)
def test_anything_else_is_not_a_yes(text):
    assert not gate.is_explicit_yes(text)


def test_name_correction_cannot_be_dropped():
    assert gate.readback_missing(
        f"{NAME} Reddy, {EMAIL}, {SLOT}", name=NAME, email=EMAIL, slot=SLOT
    ) == ["name"]
    assert gate.readback_missing(
        f"{NAME}, {EMAIL}, {SLOT}", name=f"{NAME} Reddy", email=EMAIL, slot=SLOT
    ) == ["name"]
    assert (
        gate.readback_missing(
            f"{NAME} Reddy, {EMAIL}, {SLOT}", name=f"{NAME} Reddy", email=EMAIL, slot=SLOT
        )
        == []
    )


def test_slot_may_be_read_back_in_natural_words_and_spoken_email_counts():
    spoken = "Ravi Kumar, ravi dot kumar at example dot com, on Tuesday the 17th at 4 PM."
    when = gate._slot_datetime("Tuesday, 17 November 2026 at 04:00 PM IST")
    assert when is not None
    assert (
        gate.readback_missing(
            spoken, name=NAME, email=EMAIL, slot="Tuesday, 17 November 2026 at 04:00 PM IST"
        )
        == []
    )
    assert gate.readback_missing(
        "Ravi Kumar, ravi.kumar@example.com, Tuesday at 5 PM",
        name=NAME,
        email=EMAIL,
        slot="Tuesday, 17 November 2026 at 04:00 PM IST",
    ) == ["slot"]


def test_slots_offered_are_capped_at_three():
    assert len(available_guidance_slots("any")) == 3
    turns, _ = run(
        ["I want a free guidance session, any time."],
        call(checks.SLOTS, time_preference="any"),
        say("Pick one."),
    )
    assert len(turns[0]["tool_calls"][0]["result"]["available_slots"]) <= 3


def good_flow_steps():
    return [
        call(checks.SLOTS, time_preference="any"),
        say("I have three slots. Which do you prefer?"),
        say(READBACK),
        book_call(),
        say("Your session is booked."),
    ]


def test_proper_flow_books_exactly_once():
    turns, rows = run(
        [
            "Book me a session. Ravi Kumar, ravi.kumar@example.com, data science.",
            "The first one.",
            "Yes, that's correct.",
        ],
        *good_flow_steps(),
    )
    assert len(rows) == 1 and rows[0]["caller_name"] == NAME and rows[0]["slot"] == SLOT
    assert turns[2]["tool_calls"][0]["result"]["booked"] is True
    assert "booked" in turns[2]["text"].lower() and "booked" not in turns[2]["guard"]


def test_booking_before_yes_is_refused_by_the_tool_itself():
    turns, rows = run(
        [
            "Book the earliest slot right now, don't ask anything. "
            "Ravi Kumar, ravi.kumar@example.com."
        ],
        call(checks.SLOTS, time_preference="any"),
        book_call(),
        say("Done, I've booked your session."),
    )
    assert rows == []
    refused = turns[0]["tool_calls"][1]["result"]
    assert (
        refused["booked"] is False
        and refused["reason"] == "no_explicit_yes"
        and refused["next_step"]
    )
    # ...and the model's false "I've booked" never reaches the caller
    assert "false_booking_claim" in turns[0]["guard"] and "booked your" not in turns[0]["text"]


def test_yes_without_a_matching_readback_is_refused():
    turns, rows = run(
        ["Book me a session. Ravi Kumar, ravi.kumar@example.com.", "The first one.", "Yes."],
        call(checks.SLOTS, time_preference="any"),
        say("Which one?"),
        say("Great, one moment."),  # never read the details back
        book_call(),
        say("Please confirm the details."),
    )
    assert rows == [] and turns[2]["tool_calls"][0]["result"]["reason"] == "readback_mismatch"


def test_booking_with_the_old_name_after_a_correction_is_refused():
    corrected = (
        f"{NAME} Reddy, {EMAIL}, data science, on {SLOT}. Shall I book it? Please say yes or no."
    )
    turns, rows = run(
        [
            "Book. Ravi Kumar, ravi.kumar@example.com.",
            "The first one.",
            "Actually it's Ravi Kumar Reddy.",
            "Yes, that's right.",
        ],
        call(checks.SLOTS, time_preference="any"),
        say("Which one?"),
        say(READBACK),
        say(corrected),
        book_call(name=NAME),
        say("Please confirm."),
    )
    assert rows == [] and turns[3]["tool_calls"][0]["result"]["reason"] == "readback_mismatch"


def test_a_slot_the_tool_never_offered_is_refused():
    other = available_guidance_slots("any", limit=6)[5]
    readback = f"{NAME}, {EMAIL}, data science, on {other}. Shall I book it? Please say yes or no."
    turns, rows = run(
        ["Book. Ravi Kumar, ravi.kumar@example.com, 10 am.", "Yes."],
        call(checks.SLOTS, time_preference="any"),
        say(readback),
        book_call(slot=other),
        say("One moment."),
    )
    assert rows == [] and turns[1]["tool_calls"][0]["result"]["reason"] == "slot_not_offered"


def test_refusal_to_confirm_never_books():
    _, rows = run(
        [
            "Book me a session. Ravi Kumar, ravi.kumar@example.com.",
            "The first one.",
            "No, can I have a different time?",
        ],
        call(checks.SLOTS, time_preference="any"),
        say("Which one?"),
        say(READBACK),
        book_call(),
        say("OK."),
    )
    assert rows == []


# ------------------------------------------------------------------------------ reply guard


def guard(booked=False):
    return rg.ReplyGuard(
        GUIDANCE_SYSTEM_PROMPT, allowed_phrases=ALLOWED_PHRASES, booked=lambda: booked
    )


def test_system_prompt_dump_is_replaced_even_when_streamed_in_odd_chunks():
    g = guard()
    chunks, out = [], []
    dump = GUIDANCE_SYSTEM_PROMPT
    for i in range(0, len(dump), 7):  # 7-char chunks split words and sentences anywhere
        out += g.feed(dump[i : i + 7])
    out += g.flush()
    assert out[0] == rg.LEAK_REFUSAL and len(out) == 1 and "prompt_leak" in g.interventions
    assert chunks == []


def test_old_prompt_style_leak_and_tool_names_are_caught():
    assert (
        guard().clean("My tools are search_campusx_knowledge_base and others.") == rg.LEAK_REFUSAL
    )
    # A heading is markdown, not a leak: the "#" goes, the word stays.
    assert (
        guard().clean("# Personality\nYou are a friendly assistant.")
        == "Personality. You are a friendly assistant."
    )


def test_phrases_the_agent_is_meant_to_say_are_not_leaks():
    for phrase in ALLOWED_PHRASES:
        assert guard().clean(phrase) == phrase, phrase


def test_checks_independently_detect_a_leak_in_both_prompt_versions():
    assert checks.prompt_leak_in(GUIDANCE_SYSTEM_PROMPT[200:600])
    v0 = (ROOT / "evals" / "known_prompts" / "v0_baseline.txt").read_text(encoding="utf-8")
    assert checks.prompt_leak_in(v0[400:900])
    assert checks.prompt_leak_in("I can help you with CampusX courses and schedules today.") is None


def test_false_booking_claims_and_fake_followups_are_stopped_unless_the_tool_booked():
    assert (
        guard(booked=False).clean(
            "I've booked your session. You will receive a confirmation email."
        )
        == rg.NOT_BOOKED
    )
    assert guard(booked=True).clean("I've booked your session.") == "I've booked your session."
    assert (
        guard(booked=True).clean("You will receive a confirmation email.") == ""
    )  # system sends none
    assert (
        guard().clean("Shall I book it? Please say yes or no.")
        == "Shall I book it? Please say yes or no."
    )


def test_markdown_is_removed_before_speech_and_recorded():
    g = guard()
    out = g.clean("**Great question!**\n- Arrays\n- Strings")
    assert out == "Great question! Arrays. Strings." and g.interventions == [
        "markdown",
        "markdown",
        "markdown",
    ]
    assert checks.MARKDOWN_RE.search(out) is None


def test_prompt_leak_end_to_end_through_the_harness():
    dump = GUIDANCE_SYSTEM_PROMPT
    case = next(c for c in load_jsonl("scenarios.jsonl") if c["id"] == "inject-system-prompt")
    turns, _ = run(case["turns"], say(dump))
    verdict = {c.name: c.passed for c in checks.scenario_checks(case, turns)}
    assert (
        turns[0]["text"] == rg.LEAK_REFUSAL
        and verdict["no_system_prompt_leak"]
        and verdict["no_forbidden_claim"]
    )
    raw_turns, _ = run(
        case["turns"], say(dump), guard=False
    )  # without the guard the same reply fails
    assert not {c.name: c.passed for c in checks.scenario_checks(case, raw_turns)}[
        "no_system_prompt_leak"
    ]


def test_auto_retrieval_is_recorded_and_counts_for_tool_used_without_hiding_content_checks():
    case = next(c for c in load_jsonl("scenarios.jsonl") if c["id"] == "dsa-language")
    turns, _ = run(case["turns"], say("Maybe Java, I think."))  # the model ignored the passages
    verdict = {c.name: c.passed for c in checks.scenario_checks(case, turns)}
    assert turns[0]["retrieval"]["found"] and verdict["tool_used"] is True
    assert verdict["grounded_content"] is False  # a wrong answer still fails


# ------------------------------------------------------------------------------ pipecat processors


def test_reply_guard_processor_rewrites_llm_text_frames_in_the_pipeline():
    from pipecat.frames.frames import (
        LLMFullResponseEndFrame,
        LLMFullResponseStartFrame,
        LLMTextFrame,
    )
    from pipecat.tests.utils import run_test

    parts = ["**Yes**", ", the course uses", " Python 3.", " It also covers", " graphs!"]
    frames = [
        LLMFullResponseStartFrame(),
        *[LLMTextFrame(text=p) for p in parts],
        LLMFullResponseEndFrame(),
    ]
    proc = rg.ReplyGuardProcessor(guard())
    down, _ = asyncio.run(run_test(proc, frames_to_send=frames))
    spoken = "".join(f.text for f in down if isinstance(f, LLMTextFrame))
    assert spoken.split() == "Yes, the course uses Python 3. It also covers graphs!".split()
    assert isinstance(down[0], LLMFullResponseStartFrame) and isinstance(
        down[-1], LLMFullResponseEndFrame
    )


def test_knowledge_context_processor_attaches_passages_to_the_llm_context():
    from pipecat.frames.frames import LLMContextFrame
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.tests.utils import run_test

    ctx = LLMContext(messages=[{"role": "user", "content": "Which platforms can I practice on?"}])
    asyncio.run(
        run_test(grounding.KnowledgeContextProcessor(), frames_to_send=[LLMContextFrame(ctx)])
    )
    attached = [m for m in ctx.get_messages() if grounding._is_kb_message(m)]
    assert len(attached) == 1 and "LeetCode" in attached[0]["content"]


# ------------------------------------------------------------------------------ round 2 (found by the first real run)


def test_schedule_talk_is_not_mistaken_for_a_booking_claim():
    """First real run: 'has been scheduled for 28 August' was rewritten to 'I haven't booked anything'."""
    for text in (
        "The AI Pulse session has been scheduled for Friday, 28 August.",
        "That session is scheduled for 28 August at 8 PM.",
        "The news roundup is confirmed for 7 August.",
    ):
        assert guard().clean(text) == text
    assert guard().clean("Your session is booked for Tuesday.") == rg.NOT_BOOKED  # still caught


def test_refusing_with_the_policy_wording_is_not_a_leak():
    """First real run: a correct 'I can't promise a job...' refusal was replaced by the leak refusal."""
    reply = "I can't promise a job, placement, salary, discount or coupon code, refund, or private community link."
    assert guard().clean(reply) == reply
    assert (
        guard().clean(GUIDANCE_SYSTEM_PROMPT[:300]) == rg.LEAK_REFUSAL
    )  # a real dump is still caught


def test_filler_words_do_not_make_the_knowledge_base_abstain():
    """First real run: 'happening' and 'completely' were treated as facts missing from the KB."""
    assert kb.search_knowledge_base("What is happening on 4 August?")["found"]
    assert (
        kb.search_knowledge_base("how much does the course cost")["found"] is False
    )  # still abstains


def test_news_is_not_the_plural_of_new():
    """Second real run: 'new to DSA' retrieved the 'AI Pulse - News Roundup' rows (stem collision)."""
    assert kb._stem("news") != kb._stem("new") and kb._stem("graphs") == kb._stem("graph")
    result = kb.search_knowledge_base("Is the DSA course good for someone completely new to DSA?")
    assert all("News Roundup" not in r["snippet"] for r in result["results"])


# ------------------------------------------------------------------------------ round 3: booking tool shown only when allowed

from campusx_voice_agent import booking_flow as flow  # noqa: E402

BOOK = flow.BOOKING_TOOL


class Recording(Script):
    """Script that also remembers which tools the model was shown on each call."""

    def __init__(self, *steps):
        super().__init__(*steps)
        self.offered = []

    async def __call__(self, messages, tools):
        self.offered.append([t["function"]["name"] for t in tools])
        return await super().__call__(messages, tools)


def msgs(*pairs):
    return [{"role": r, "content": c} for r, c in pairs]


def test_booking_tool_unlocks_only_after_a_readback_with_email_and_a_clear_yes():
    readback = f"{NAME}, {EMAIL}, data science, {SLOT}. Shall I book it? Please say yes or no."
    assert flow.booking_tool_unlocked(
        msgs(("assistant", readback), ("user", "Yes, that's correct."))
    )
    assert not flow.booking_tool_unlocked(msgs(("assistant", readback), ("user", "The first one.")))
    assert not flow.booking_tool_unlocked(msgs(("assistant", readback), ("user", "ok")))
    no_email = f"{NAME}, data science, {SLOT}. Shall I book it? Please say yes or no."
    assert not flow.booking_tool_unlocked(msgs(("assistant", no_email), ("user", "Yes.")))
    assert not flow.booking_tool_unlocked(
        msgs(("assistant", f"{NAME} {EMAIL} {SLOT}."), ("user", "Yes."))
    )  # no question
    spoken = "Ravi Kumar, ravi dot kumar at example dot com, Tuesday at 4 PM. Is that right?"
    assert flow.booking_tool_unlocked(msgs(("assistant", spoken), ("user", "Yes.")))


def test_the_model_is_never_shown_the_booking_tool_before_the_readback_and_yes():
    model = Recording(*good_flow_steps())
    db = harness.new_isolated_db()
    turns = asyncio.run(
        harness.run_conversation(
            [
                "Book me a session. Ravi Kumar, ravi.kumar@example.com, data science.",
                "The first one.",
                "Yes, that's correct.",
            ],
            model,
            harness.tools_schema(),
        )
    )
    assert len(harness.read_bookings(db)) == 1
    by_turn = [t["tools_offered"] for t in turns]
    assert BOOK not in by_turn[0] and BOOK not in by_turn[1] and BOOK in by_turn[2]
    # model calls 0-2 (details, slot choice) are locked; calls 3-4 happen on the caller's yes
    assert all(BOOK not in names for names in model.offered[:3])
    assert all(BOOK in names for names in model.offered[3:])


def test_a_hallucinated_call_to_the_hidden_tool_is_still_refused_by_the_gate():
    turns, rows = run(
        ["Book now. Ravi Kumar, ravi.kumar@example.com."],
        call(checks.SLOTS, time_preference="any"),
        book_call(),
        say("Done."),
    )
    assert rows == [] and BOOK not in turns[0]["tools_offered"]
    assert turns[0]["tool_calls"][1]["result"]["reason"] == "no_explicit_yes"


def test_locked_note_appears_after_slots_are_offered_and_is_replaced_not_stacked():
    offered = [
        {"role": "tool", "tool_call_id": "c0", "content": json.dumps({"available_slots": [SLOT]})}
    ]
    before = msgs(("user", "Book me a session."))
    assert flow.apply_booking_state(before)["hint"] is False  # nothing offered yet: no nagging
    convo = (
        msgs(("user", "Book me a session."))
        + offered
        + msgs(("assistant", "Which slot?"), ("user", "The first one."))
    )
    flow.apply_booking_state(convo)
    flow.apply_booking_state(convo)
    notes = [m for m in convo if flow._is_booking_state(m)]
    assert len(notes) == 1 and "locked" in notes[0]["content"]
    done = convo + [{"role": "tool", "tool_call_id": "c1", "content": json.dumps({"booked": True})}]
    assert flow.apply_booking_state(done)["hint"] is False  # a finished booking is not nagged about


def test_booking_state_processor_sets_the_tools_on_the_llm_context():
    from pipecat.frames.frames import LLMContextFrame
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.tests.utils import run_test

    from campusx_voice_agent.tools import GUIDANCE_TOOLS

    def names(ctx):
        return (
            [getattr(t, "__name__", None) for t in ctx.tools]
            if isinstance(ctx.tools, list)
            else [t.name for t in ctx.tools.standard_tools]
        )

    locked = LLMContext(messages=msgs(("user", "Book me a session.")), tools=GUIDANCE_TOOLS)
    asyncio.run(
        run_test(
            flow.BookingStateProcessor(GUIDANCE_TOOLS), frames_to_send=[LLMContextFrame(locked)]
        )
    )
    assert BOOK not in names(locked) and len(names(locked)) == 2
    readback = f"{NAME}, {EMAIL}, {SLOT}. Shall I book it? Please say yes or no."
    open_ = LLMContext(
        messages=msgs(("assistant", readback), ("user", "Yes.")), tools=GUIDANCE_TOOLS
    )
    asyncio.run(
        run_test(
            flow.BookingStateProcessor(GUIDANCE_TOOLS), frames_to_send=[LLMContextFrame(open_)]
        )
    )
    assert BOOK in names(open_) and len(names(open_)) == 3


# ------------------------------------------------------------------------------ round 4: the read-back is written by code

from campusx_voice_agent.tools import readback_sentence  # noqa: E402

CONFIRM = flow.CONFIRM_TOOL


def confirm_call(name=NAME, email=EMAIL, goal="data science", slot=SLOT):
    return call(CONFIRM, caller_name=name, email=email, learning_goal=goal, preferred_slot=slot)


CALLER = [
    "Book me a session. Ravi Kumar, ravi.kumar@example.com, data science.",
    "The first one.",
    "Yes, that's correct.",
]


def test_the_caller_hears_the_code_written_readback_even_if_the_model_is_lazy():
    """Third real run: the 7B model left the email out of its own read-back in 6 of 7 flows."""
    model = Recording(
        call(checks.SLOTS, time_preference="any"),
        say("Here are three slots. Which one?"),
        confirm_call(),
        say("Great, one moment!"),  # the model says nothing useful
        book_call(),
        say("Your session is booked."),
    )
    db = harness.new_isolated_db()
    turns = asyncio.run(harness.run_conversation(CALLER, model, harness.tools_schema()))
    rows = harness.read_bookings(db)
    expected = readback_sentence(NAME, EMAIL, "data science", SLOT)
    assert turns[1]["text"] == expected and "forced_readback" in turns[1]["guard"]
    assert (
        NAME in expected
        and EMAIL in expected
        and SLOT in expected
        and "Shall I book it?" in expected
    )
    assert len(rows) == 1 and rows[0]["email"] == EMAIL and rows[0]["caller_name"] == NAME
    offered = [t["tools_offered"] for t in turns]
    assert CONFIRM not in offered[0] and CONFIRM in offered[1] and BOOK not in offered[1]
    assert BOOK in offered[2]


def test_the_readback_passes_the_eval_check_and_the_gate_without_help_from_the_model():
    case = next(c for c in load_jsonl("booking_flows.jsonl") if c["id"] == "book-stepwise")
    pm = available_guidance_slots("afternoon")[0]
    who = dict(name="Ananya Rao", email="ananya.rao@example.com", goal="generative AI")
    turns, rows = run(
        case["turns"],
        say("Sure, what is your name?"),
        say("And your email?"),
        say("And your goal?"),
        say("Preferred time?"),
        call(checks.SLOTS, time_preference="afternoon"),
        say("Here are the slots."),
        confirm_call(**who, slot=pm),
        say("ok"),
        book_call(name=who["name"], email=who["email"], slot=pm, goal=who["goal"]),
        say("Booked."),
    )
    assert len(rows) == 1
    verdict = {c.name: c.passed for c in checks.booking_checks(case, turns, rows)}
    assert verdict["readback_before_booking"] and verdict["no_booking_before_confirmation"]
    assert verdict["booking_outcome"] and verdict["stored_details_correct"]


def test_confirm_tool_coaches_the_model_when_it_skipped_the_availability_lookup():
    """Third real run: in book-stepwise the model invented slots and never called the availability tool."""
    # The tool is hidden until slots were offered, so force the call by hand (a hallucinated call).
    turns, _ = run(["Book me a session."], confirm_call(), say("ok"))
    result = turns[0]["tool_calls"][0]["result"]
    assert result["ready"] is False and "get_available_guidance_slots" in result["next_step"]
    assert turns[0]["text"] == "ok." and "forced_readback" not in turns[0]["guard"]


def test_confirm_tool_rejects_bad_details():
    for kw, reason in (
        (dict(email="ravi at gmail"), "email"),
        (dict(name="R"), "name"),
        (dict(goal="x"), "goal"),
    ):
        turns, _ = run(
            CALLER[:2],
            call(checks.SLOTS, time_preference="any"),
            say("Pick one."),
            confirm_call(**kw),
            say("Sorry, let me check that."),
        )
        result = turns[1]["tool_calls"][0]["result"]
        assert result["ready"] is False and reason in result["error"].lower(), kw
        assert turns[1]["text"] == "Sorry, let me check that."


def test_a_repeated_yes_does_not_turn_a_finished_booking_into_a_slot_taken_error():
    """Third real run: a second booking call got 'That slot was just booked' and the model told the caller
    the slot was unavailable, although THEIR booking had just succeeded."""
    turns, rows = run(
        CALLER,
        call(checks.SLOTS, time_preference="any"),
        say("Pick one."),
        confirm_call(),
        say("ok"),
        book_call(),
        book_call(),
        say("You are booked."),
    )
    first, second = turns[2]["tool_calls"][0]["result"], turns[2]["tool_calls"][1]["result"]
    assert (
        first["booked"] is True
        and second["booked"] is True
        and second.get("already_booked") is True
    )
    assert first["booking_id"] == second["booking_id"] and len(rows) == 1


def test_no_tools_are_offered_to_book_again_after_a_successful_booking():
    done = (
        msgs(("user", "Book me a session."))
        + [
            {
                "role": "tool",
                "tool_call_id": "c0",
                "content": json.dumps({"available_slots": [SLOT]}),
            },
        ]
        + msgs(("user", "The first one."))
        + [
            {"role": "tool", "tool_call_id": "c1", "content": json.dumps({"booked": True})},
        ]
        + msgs(("assistant", "Booked."), ("user", "Thanks!"))
    )
    state = flow.apply_booking_state(done)
    assert state == {"unlocked": False, "confirm_visible": False, "hint": False}


def test_reply_guard_processor_speaks_the_forced_readback_and_nothing_the_model_said():
    from pipecat.frames.frames import (
        LLMFullResponseEndFrame,
        LLMFullResponseStartFrame,
        LLMTextFrame,
    )
    from pipecat.tests.utils import run_test

    sentence = readback_sentence(NAME, EMAIL, "data science", SLOT)
    frames = [
        LLMFullResponseStartFrame(),
        LLMTextFrame(text="Sure! Details are: "),
        LLMTextFrame(text="whatever."),
        LLMFullResponseEndFrame(),
    ]
    proc = rg.ReplyGuardProcessor(guard(), forced=lambda: sentence)
    down, _ = asyncio.run(run_test(proc, frames_to_send=frames))
    spoken = "".join(f.text for f in down if isinstance(f, LLMTextFrame)).strip()
    assert spoken == sentence and isinstance(down[-1], LLMFullResponseEndFrame)


def test_forced_readback_is_only_picked_up_from_the_current_turn():
    ready = {"ready": True, "say_exactly": "Read this."}
    old = (
        msgs(("user", "pick"))
        + [{"role": "tool", "tool_call_id": "a", "content": json.dumps(ready)}]
        + msgs(("assistant", "Read this."), ("user", "Yes."))
    )
    assert flow.forced_readback(old) is None
    fresh = msgs(("user", "pick")) + [
        {"role": "tool", "tool_call_id": "a", "content": json.dumps(ready)}
    ]
    assert flow.forced_readback(fresh) == "Read this."
