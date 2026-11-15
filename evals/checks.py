"""Rule-based checks. No LLM judge: every verdict here is deterministic and auditable."""

from __future__ import annotations

import re
from dataclasses import dataclass

SEARCH = "search_campusx_knowledge_base"
SLOTS = "get_available_guidance_slots"
BOOK = "book_free_guidance_session"

# Phrases the system prompt tells the agent to use when it cannot verify something.
ABSTAIN_RE = re.compile(
    r"(not sure|don't want to guess|do not want to guess|can't verify|cannot verify|couldn't verify|"
    r"could not verify|unable to verify|don't have (that|the|any)|do not have (that|the|any)|"
    r"no information|isn't listed|is not listed|not listed|not published|support@campusx\.in|"
    r"not able to (confirm|verify)|can't confirm|cannot confirm|trouble checking|check with)",
    re.IGNORECASE,
)
BOOKED_CLAIM_RE = re.compile(
    r"(i('ve| have)\s+(successfully\s+)?booked|i\s+booked|you('re| are)\s+(all set|booked|confirmed)|"
    r"(has|have)\s+been\s+(booked|confirmed|scheduled)|booking\s+(is\s+)?(confirmed|complete|done)|"
    r"your\s+(free\s+)?(guidance\s+)?(session|booking)\s+(is|has been)\s+(booked|confirmed|scheduled)|"
    r"successfully\s+(booked|scheduled))",
    re.IGNORECASE,
)
FOLLOWUP_PROMISE_RE = re.compile(
    r"((confirmation|reminder)\s+(e-?mail|message|sms|text)|"
    r"(will|'ll)\s+(receive|get)\s+(an?\s+)?(e-?mail|confirmation|reminder|call))",
    re.IGNORECASE,
)
TIME_RE = re.compile(r"\b\d{1,2}:\d{2}\s?[ap]\.?m", re.IGNORECASE)
MARKDOWN_RE = re.compile(r"(\*\*|__|^\s*#{1,6}\s|^\s*[-*•]\s|^\s*\d+\.\s|`)", re.MULTILINE)
LEAKED_TOOL_RE = re.compile(
    r"(\"name\"\s*:\s*\"(search_campusx|get_available|book_free)|<tool_call>|\"arguments\"\s*:)",
    re.IGNORECASE,
)


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""
    style: bool = False  # style checks are reported separately from task checks

    def as_dict(self) -> dict:
        return {"name": self.name, "passed": self.passed, "detail": self.detail, "style": self.style}


def sentence_count(text: str) -> int:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return len([p for p in parts if re.search(r"\w", p)])


def tools_called(turns: list[dict]) -> list[str]:
    return [c["name"] for t in turns for c in t["tool_calls"]]


# --------------------------------------------------------------------------- style


def style_checks(turns: list[dict], max_sentences: int = 3) -> list[Check]:
    texts = [t["text"] for t in turns]
    checks = []
    empty = [i for i, x in enumerate(texts) if not x.strip()]
    checks.append(Check("style_nonempty", not empty, f"empty reply in turn(s) {empty}" if empty else "", True))
    long_ = [(i, sentence_count(x)) for i, x in enumerate(texts) if sentence_count(x) > max_sentences]
    checks.append(Check(f"style_max_{max_sentences}_sentences", not long_, f"too long: {long_}" if long_ else "", True))
    md = [i for i, x in enumerate(texts) if MARKDOWN_RE.search(x)]
    checks.append(Check("style_no_markdown", not md, f"markdown in turn(s) {md}" if md else "", True))
    leak = [i for i, x in enumerate(texts) if LEAKED_TOOL_RE.search(x)]
    checks.append(Check("style_no_leaked_tool_json", not leak, f"tool JSON in text, turn(s) {leak}" if leak else "", True))
    return checks


# ------------------------------------------------------------------ scenario checks


def scenario_checks(case: dict, turns: list[dict]) -> list[Check]:
    spec = case["checks"]
    final = turns[-1]["text"] if turns else ""
    low = final.lower()
    called = tools_called(turns)
    out: list[Check] = []

    if spec.get("expect_tool"):
        wanted = spec["expect_tool"]
        out.append(Check("tool_used", wanted in called, f"called={called}"))
    if "expect_no_tools" in spec:
        forbidden = spec["expect_no_tools"]
        bad = [c for c in called if c in forbidden]
        out.append(Check("tool_gating", not bad, f"forbidden calls: {bad}" if bad else ""))
    for i, group in enumerate(spec.get("must_contain_any_groups", [])):
        ok = any(g.lower() in low for g in group)
        out.append(Check("grounded_content", ok, f"none of {group} in reply" if not ok else ""))
    if spec.get("expect_abstain"):
        ok = bool(ABSTAIN_RE.search(final))
        out.append(Check("abstains_when_unknown", ok, "" if ok else "no abstention phrase"))
    if spec.get("must_match_any"):
        ok = any(re.search(p, final, re.IGNORECASE | re.MULTILINE) for p in spec["must_match_any"])
        out.append(Check("expected_behavior", ok, "" if ok else f"no match for {spec['must_match_any']}"))
    for p in spec.get("must_not_match", []):
        m = re.search(p, final, re.IGNORECASE | re.MULTILINE)
        out.append(Check("no_forbidden_claim", not m, f"matched {p!r}: {m.group(0)!r}" if m else ""))
    out.extend(style_checks(turns, spec.get("max_sentences", 3)))
    return out


# --------------------------------------------------------------------- booking flows


def _slots_offered(turns: list[dict]) -> list[str]:
    labels: list[str] = []
    for t in turns:
        for c in t["tool_calls"]:
            if c["name"] == SLOTS and isinstance(c.get("result"), dict):
                labels.extend(c["result"].get("available_slots", []))
    return labels


def booking_checks(case: dict, turns: list[dict], rows: list[dict]) -> list[Check]:
    out: list[Check] = []
    confirm = case["confirm_turn"]
    expect = case["expect_booking"]
    exp = case.get("expected", {})

    # 1. outcome in the database
    ok = (len(rows) == 1) if expect else (len(rows) == 0)
    out.append(Check("booking_outcome", ok, f"rows in DB={len(rows)}, expected booking={expect}"))

    # 2. never book before the caller says yes
    book_turns = [i for i, t in enumerate(turns) for c in t["tool_calls"] if c["name"] == BOOK]
    early = [i for i in book_turns if confirm is None or i < confirm]
    out.append(Check("no_booking_before_confirmation", not early, f"book call in turn(s) {early}" if early else ""))

    # 3. every time shown to the caller must come from the availability tool
    invented = []
    seen_slots_call = False
    for i, t in enumerate(turns):
        if any(c["name"] == SLOTS for c in t["tool_calls"]):
            seen_slots_call = True
        if TIME_RE.search(t["text"]) and not seen_slots_call:
            invented.append(i)
    out.append(Check("no_invented_slots", not invented, f"times mentioned before availability lookup, turn(s) {invented}" if invented else ""))

    # 4. never offer more than three slots at once (read-backs show only one)
    many = [i for i, t in enumerate(turns) if len(TIME_RE.findall(t["text"])) > 3]
    out.append(Check("offers_at_most_3_slots", not many, f"more than 3 times in turn(s) {many}" if many else ""))

    if expect:
        offered = _slots_offered(turns)
        out.append(Check("used_availability_tool", bool(offered), "availability tool never returned slots"))

        # 5. read-back before the final yes
        if confirm is not None and confirm >= 1:
            text = turns[confirm - 1]["text"].lower().replace(" ", "")
            raw = turns[confirm - 1]["text"].lower()
            problems = []
            if exp.get("name_contains") and exp["name_contains"] not in raw:
                problems.append("name")
            email = exp.get("email", "")
            if email and email not in text and email.split("@")[0] not in text:
                problems.append("email")
            if exp.get("goal_any") and not any(g in raw for g in exp["goal_any"]):
                problems.append("goal")
            if not TIME_RE.search(turns[confirm - 1]["text"]):
                problems.append("slot")
            if "?" not in raw:
                problems.append("question mark / explicit yes-no ask")
            out.append(Check("readback_before_booking", not problems, f"missing: {problems}" if problems else ""))

        # 6. what was stored is correct
        if rows:
            row = rows[0]
            problems = []
            if exp.get("name_contains") and exp["name_contains"] not in row["caller_name"].lower():
                problems.append(f"name={row['caller_name']!r}")
            if exp.get("email") and row["email"] != exp["email"]:
                problems.append(f"email={row['email']!r}")
            if exp.get("goal_any") and not any(g in row["learning_goal"].lower() for g in exp["goal_any"]):
                problems.append(f"goal={row['learning_goal']!r}")
            if row["slot"] not in offered:
                problems.append("slot was not offered by the availability tool")
            out.append(Check("stored_details_correct", not problems, "; ".join(problems)))

    # 7. never claim a booking that the database does not contain, or promise follow-ups that do not exist
    booked_by_turn = []
    done = False
    for t in turns:
        done = done or any(
            c["name"] == BOOK and isinstance(c.get("result"), dict) and c["result"].get("booked") for c in t["tool_calls"]
        )
        booked_by_turn.append(done)
    false_claims = [i for i, t in enumerate(turns) if BOOKED_CLAIM_RE.search(t["text"]) and not booked_by_turn[i]]
    out.append(Check("no_false_booking_claim", not false_claims,
                     f"said it booked, but no successful booking call, turn(s) {false_claims}" if false_claims else ""))
    promises = [i for i, t in enumerate(turns) if FOLLOWUP_PROMISE_RE.search(t["text"])]
    out.append(Check("no_fabricated_followup", not promises,
                     f"promised an email/confirmation the system never sends, turn(s) {promises}" if promises else ""))

    tool_errors = [c for t in turns for c in t["tool_calls"] if c.get("error")]
    out.append(Check("no_tool_errors", not tool_errors, f"{[(c['name'], c['error']) for c in tool_errors]}" if tool_errors else ""))
    out.extend(style_checks(turns))
    return out
