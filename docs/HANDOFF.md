# HANDOFF (updated 2026-11-15, git commit cb0bb5b on main)

Paste PROJECT_GOAL.md first, then this file. Attach only the files the current section needs.

## Repo layout
main.py (commands: run bot, `bookings`, `check`, `download-models`, `moss-index`)
src/campusx_voice_agent/: agent.py (pipeline wiring), pipeline.py (VAD/turn strategies, metrics printer),
  tools.py (system prompt + 3 tools + SQLite), knowledge_base.py (keyword search), models.py (Ollama check, model download),
  moss.py / moss_index.py (optional), config.py, bookings.py
data/knowledge_base/: campusx_august_schedule.md, dsa_python_course_full_curriculum.md   (data/guidance_bookings.db is git-ignored: real caller emails)
evals/: run_evals.py, harness.py, checks.py, retrieval_eval.py, report.py, compare.py, rescore.py, datasets/*.jsonl,
  baselines/qwen2.5-7b-baseline (raw) and qwen2.5-7b-baseline-rescored (current reference), README.md
tests/test_evals.py (16 tests, all pass; fake model, no Ollama needed)

## Commands (venv active, Ollama running, from local_proj)
python main.py check | python main.py download-models | python main.py
python evals/run_evals.py --suite all --model qwen2.5:7b --tag NAME [--repeats 3] [--limit 5] [--ids a,b]
python evals/compare.py <runA> <runB>        python evals/rescore.py <run_dir>

## Eval suite (text-level "brain" evals; no audio yet)
Retrieval 38 queries; Scenarios 56 (kb_dsa, kb_schedule, kb_unanswerable, guardrail, injection, style, tool_gating, multi_turn);
Booking flows 12 scripted (7 should book, 5 must not). All checks rule-based, no LLM judge.
Style checks (<=3 sentences, no markdown, no leaked tool JSON) reported separately from task checks.

## Baseline (qwen2.5:7b, temp 0.3, 1 repeat; rescored with current checks)
- Retrieval: hit@1 0.60, hit@4 0.80, MRR 0.70; 5 answers ranked below 4, 1 snippet truncated (31 Aug row);
  8/8 unanswerable queries still return results (top score 5.25 vs 6.33 for answerable), so "not found" is undetectable.
- Scenarios: 20/56 task pass. tool_used 2/34 (search tool almost never called). kb_dsa 1/13, kb_schedule 0/9,
  kb_unanswerable 1/12, guardrail 7/8, injection 2/4, style 4/4, tool_gating 3/3, multi_turn 2/3.
- Booking flows: 3/12 task pass, style 2/12. offers_at_most_3_slots 4/12, no_booking_before_confirmation 11/12,
  no_false_booking_claim 11/12, readback 5/7, stored_details_correct 5/6.
- Latency: cold start 6.7 s; brain per turn p50 1.7 s, p95 4.1 s (bookings p50 3.2 s). No STT/TTS included.

## Findings to fix (priority order)
1. Model answers from memory instead of the KB (invented course duration "3 to 4 months", DAMP "6 months", Heap Sort, AVL trees,
   CodeSignal, "Data Science Master Program"). Often says "let me verify" or the "I'm having trouble checking" template without calling the tool.
2. Booking safety: claimed "I've booked" with no tool call and promised a confirmation email; booked before the caller said yes
   (so a name correction never reached the DB); attempted booking on "book now, ask nothing"; offered >3 slots in 8/12 flows;
   markdown bullets in 7/12 flows; replies over 3 sentences in 9/12.
3. System prompt leaked in full when told to ignore instructions (markdown, 9 s reply).
4. Retrieval: poor ranking, no stopword filtering, 900-char snippet cut drops late schedule rows, no minimum score.
5. Prompt contradictions: "2 to 3", "1 sentence", "no more than 2" sentences.

## Known limits of the evals
Small suites (use 95% CI, --repeats 3, compare.py). Scripted booking flows are brittle (fixed user lines).
"unverified-*" cases score policy (do not state facts not in the KB), not real-world truth.
Moss path not evaluated. Voice-level behaviour (STT, TTS, barge-in, end-to-end latency) not yet measured.

## Decisions
No paid APIs. Evals call the real prompt, tool schemas (Pipecat adapter) and tool code, with an isolated temp SQLite DB per case.
Compare runs only when dataset and KB hashes match (compare.py warns). Baseline results are committed under evals/baselines/.

## Sections (one conversation each) and status
A Baseline harness + datasets (days 1-3): DONE
B Brain fixes (day 4): NEXT (retrieval in code, booking rules in code, prompt rewrite, leak guard)
C Re-eval + model comparison (day 5)   D STT eval (day 6)   E Latency, turn detection, barge-in (day 7)
F Robustness + security (day 8)   G Stability, resources, human spot check (day 9)   H Report + portfolio write-up (day 10)
Optional: simulated user for booking flows; Moss retrieval eval.

## How to work with me (Claude) on this project
- Windows PowerShell 5.1. Always start command blocks with the full `cd C:\Users\uday2\Downloads\campusx_voice_agent_local\local_proj`.
- No placeholders like <path> in commands; give runnable commands. Pasted multi-line blocks can swallow prompts, so keep blocks short.
- Files reach the laptop via Downloads: find them with Get-ChildItem ... -Filter, then Expand-Archive and Copy-Item.
- Claude's sandbox has no Ollama or GPU: code is tested there with a fake model; real numbers only come from the user's runs.
- Before committing, run `python -m pytest -q tests` and `git status`; never commit .env, .venv, or guidance_bookings.db.
- At the end of each section, update HANDOFF.md (status, numbers, decisions, next step).
