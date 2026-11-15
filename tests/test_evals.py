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
    case = next(c for c in load_jsonl("booking_flows.jsonl") if c["id"] == "nobook-injection-skip-confirmation")
    db = harness.new_isolated_db()
    from campusx_voice_agent.tools import available_guidance_slots

    slot = available_guidance_slots("any")[0]
    chat = Scripted(
        [
            {"content": None, "tool_calls": [tc(checks.SLOTS, {"time_preference": "any"})]},
            {"content": None, "tool_calls": [tc(checks.BOOK, {
                "caller_name": "Test User", "email": "test.user@example.com",
                "learning_goal": "machine learning", "preferred_slot": slot})]},
            {"content": "Done, you are booked.", "tool_calls": []},
            {"content": "You're welcome.", "tool_calls": []},
        ]
    )
    turns = asyncio.run(harness.run_conversation(case["turns"], chat, harness.tools_schema()))
    results = {c.name: c.passed for c in checks.booking_checks(case, turns, harness.read_bookings(db))}
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
