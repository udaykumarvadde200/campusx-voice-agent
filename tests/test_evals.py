"""Tests for the eval harness itself, using a scripted fake model (no Ollama needed)."""

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))

import checks  # noqa: E402
import compare  # noqa: E402
import harness  # noqa: E402
import report  # noqa: E402
import retrieval_eval  # noqa: E402
from run_evals import load_jsonl  # noqa: E402

KB = ROOT / "data" / "knowledge_base"


def tc(name, args, i=0):
    return {"id": f"call{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


class Scripted:
    """Fake chat function: replays a list of assistant messages."""

    def __init__(self, replies):
        self.replies = list(replies)

    async def __call__(self, messages, tools):
        return self.replies.pop(0)


def test_datasets_are_well_formed():
    ids = set()
    for name in ("scenarios.jsonl", "booking_flows.jsonl", "retrieval.jsonl"):
        rows = load_jsonl(name)
        assert rows
        for r in rows:
            assert r["id"] not in ids, f"duplicate id {r['id']}"
            ids.add(r["id"])
    for r in load_jsonl("scenarios.jsonl"):
        assert r["turns"] and isinstance(r["checks"], dict)
    for r in load_jsonl("booking_flows.jsonl"):
        assert r["expect_booking"] == (r["confirm_turn"] is not None)
        assert r["confirm_turn"] is None or r["confirm_turn"] < len(r["turns"])


def test_retrieval_gold_labels_exist_in_source():
    """A label error would blame retrieval for a mistake in the dataset."""
    for item in load_jsonl("retrieval.jsonl"):
        if item["answerable"]:
            exists, _ = retrieval_eval._section_has_keywords(KB, item["gold_source"], item["gold_keywords"])
            assert exists, f"{item['id']}: keywords not found together in one section"


def test_retrieval_runs_and_reports():
    out = retrieval_eval.run_retrieval(load_jsonl("retrieval.jsonl"), k=4, kb_dir=KB)
    s = out["summary"]
    assert 0 <= s["hit_at_1"] <= s["hit_at_4"] <= 1
    assert s["n_answerable"] + s["n_unanswerable"] == len(out["records"])


def test_sentence_and_style_checks():
    assert checks.sentence_count("One. Two! Three?") == 3
    turns = [{"text": "**Bold** reply. Two. Three. Four.", "tool_calls": []}]
    res = {c.name: c.passed for c in checks.style_checks(turns)}
    assert not res["style_max_3_sentences"] and not res["style_no_markdown"]


def test_scenario_check_catches_missing_tool_and_hallucination():
    case = {"checks": {"expect_tool": checks.SEARCH, "expect_abstain": True, "must_not_match": [r"master'?s? program"]}}
    turns = [{"text": "DSMP is the Data Science Master's Program.", "tool_calls": []}]
    res = {c.name: c.passed for c in checks.scenario_checks(case, turns)}
    assert res["tool_used"] is False
    assert res["abstains_when_unknown"] is False
    assert res["no_forbidden_claim"] is False


def test_full_booking_flow_with_scripted_model():
    case = next(c for c in load_jsonl("booking_flows.jsonl") if c["id"] == "book-all-details-first")
    db = harness.new_isolated_db()
    schema = harness.tools_schema()
    # We need a real slot label, so peek at what availability will return.
    from campusx_voice_agent.tools import available_guidance_slots

    slots = available_guidance_slots("evening")
    first = slots[0]
    chat = Scripted(
        [
            {"content": None, "tool_calls": [tc(checks.SLOTS, {"time_preference": "evening"})]},
            {"content": f"I have {first} or {slots[1]}. Which works?", "tool_calls": []},
            {"content": f"To confirm: Ravi Kumar, ravi.kumar@example.com, data science guidance, {first}. Is that correct?", "tool_calls": []},
            {"content": None, "tool_calls": [tc(checks.BOOK, {
                "caller_name": "Ravi Kumar", "email": "ravi.kumar@example.com",
                "learning_goal": "data science", "preferred_slot": first})]},
            {"content": "You are booked. Your booking ID is shown on screen.", "tool_calls": []},
        ]
    )
    turns = asyncio.run(harness.run_conversation(case["turns"], chat, schema))
    rows = harness.read_bookings(db)
    results = checks.booking_checks(case, turns, rows)
    failed = [(c.name, c.detail) for c in results if not c.passed and not c.style]
    assert not failed, failed
    assert len(rows) == 1


def test_early_booking_is_flagged():
    """The CHECK must flag a booking made before the caller said yes.

    Since Section B the booking tool itself refuses such calls, so the bad run is built by hand
    (a model whose early booking succeeded) instead of replayed through the harness.
    """
    case = next(c for c in load_jsonl("booking_flows.jsonl") if c["id"] == "nobook-injection-skip-confirmation")
    slot = "Tuesday, 17 November 2026 at 11:00 AM IST"
    book_args = {"caller_name": "Test User", "email": "test.user@example.com",
                 "learning_goal": "machine learning", "preferred_slot": slot}
    turns = [
        {"text": "Done, you are booked.", "tool_calls": [
            {"name": checks.SLOTS, "args": {"time_preference": "any"}, "result": {"available_slots": [slot]}, "error": None},
            {"name": checks.BOOK, "args": book_args, "result": {"booked": True, "slot": slot}, "error": None},
        ]},
        {"text": "You're welcome.", "tool_calls": []},
    ]
    rows = [{"caller_name": "Test User", "email": "test.user@example.com",
             "learning_goal": "machine learning", "slot": slot}]
    results = {c.name: c.passed for c in checks.booking_checks(case, turns, rows)}
    assert results["booking_outcome"] is False
    assert results["no_booking_before_confirmation"] is False


def test_bad_tool_arguments_are_recorded_not_crashed():
    chat = Scripted(
        [
            {"content": None, "tool_calls": [tc(checks.SEARCH, {"wrong_name": "x"})]},
            {"content": "I'm not sure about that.", "tool_calls": []},
        ]
    )
    turns = asyncio.run(harness.run_conversation(["hi"], chat, harness.tools_schema()))
    assert turns[0]["tool_calls"][0]["error"]


def test_report_and_compare(tmp_path):
    rec = lambda i, ok: {  # noqa: E731
        "suite": "scenario", "id": i, "category": "c", "repeat": 0, "task_pass": ok, "style_pass": True,
        "checks": [{"name": "x", "passed": ok, "detail": "d", "style": False}],
        "turns": [{"text": "t", "latency_s": 1.0, "tool_calls": []}],
    }
    a = [rec("1", False), rec("2", False), rec("3", True)]
    b = [rec("1", True), rec("2", True), rec("3", True)]
    for name, rows in (("a", a), ("b", b)):
        d = tmp_path / name
        d.mkdir()
        (d / "agent.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        (d / "summary.json").write_text(json.dumps({"meta": {"dataset_hashes": {}, "kb_hashes": {}}}), encoding="utf-8")
    text = compare.compare(tmp_path / "a", tmp_path / "b")
    assert "Fixed by B: 2" in text and "Broken by B: 0" in text
    assert report.summarize_agent(a)["task_pass"]["passed"] == 1
    assert compare.sign_test_p(0, 8) < 0.01


def test_false_booking_claim_and_fabricated_email_are_flagged():
    case = next(c for c in load_jsonl("booking_flows.jsonl") if c["id"] == "book-all-details-first")
    db = harness.new_isolated_db()
    chat = Scripted(
        [
            {"content": "Pick a time, Ravi.", "tool_calls": []},
            {"content": "Confirming Ravi Kumar, ravi.kumar@example.com. Correct?", "tool_calls": []},
            {"content": "I've booked your session. You'll receive a confirmation email shortly.", "tool_calls": []},
        ]
    )
    # guard=False: the reply guard would now rewrite these claims; this test is about the CHECKS.
    turns = asyncio.run(harness.run_conversation(case["turns"], chat, harness.tools_schema(), guard=False))
    res = {c.name: c.passed for c in checks.booking_checks(case, turns, harness.read_bookings(db))}
    assert res["no_false_booking_claim"] is False
    assert res["no_fabricated_followup"] is False
    assert res["booking_outcome"] is False


def test_honest_pre_booking_language_is_not_flagged():
    turns = [{"text": "Shall I book it for you? I will book it once you say yes.", "tool_calls": []}]
    case = {"confirm_turn": None, "expect_booking": False, "expected": {}}
    res = {c.name: c.passed for c in checks.booking_checks(case, turns, [])}
    assert res["no_false_booking_claim"] is True


def test_refusal_wording_variants_pass_the_guardrail_patterns():
    import re

    case = next(c for c in load_jsonl("scenarios.jsonl") if c["id"] == "inject-dan-job")
    for reply in (
        "I'm sorry, but I can't make such guarantees.",
        "I cannot promise that.",
        "CampusX doesn't guarantee jobs.",
    ):
        assert any(re.search(p, reply, re.IGNORECASE) for p in case["checks"]["must_match_any"]), reply
    assert not any(re.search(p, "Sure, you will get 20 LPA!", re.IGNORECASE) for p in case["checks"]["must_match_any"])


def test_rejected_booking_call_is_flagged():
    turns = [{"text": "Sorry.", "tool_calls": [
        {"name": checks.BOOK, "args": {}, "result": {"booked": False, "error": "That slot was just booked."}, "error": None}]}]
    case = {"confirm_turn": None, "expect_booking": False, "expected": {}}
    res = {c.name: c.passed for c in checks.booking_checks(case, turns, [])}
    assert res["no_tool_rejections"] is False


def test_rescore_roundtrip(tmp_path):
    import rescore

    scen = next(c for c in load_jsonl("scenarios.jsonl") if c["id"] == "inject-dan-job")
    turns = [{"user": "x", "text": "I'm sorry, but I can't make such guarantees.", "tool_calls": [], "latency_s": 1.0, "rounds": 1}]
    rec = {"suite": "scenario", "id": scen["id"], "category": "injection", "repeat": 0, "turns": turns,
           "checks": [], "task_pass": False, "style_pass": True}
    run = tmp_path / "base"
    run.mkdir()
    (run / "agent.jsonl").write_text(json.dumps(rec) + "\n", encoding="utf-8")
    (run / "summary.json").write_text(json.dumps({"meta": {"run_id": "base", "dataset_hashes": {}}}), encoding="utf-8")
    out = rescore.main(run)
    new = json.loads((out / "agent.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert new["task_pass"] is True
