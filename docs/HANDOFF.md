# HANDOFF (updated 2026-11-16; Section B code written, NOT yet committed; base commit 52c34f5 on main)

Paste PROJECT_GOAL.md first, then this file. Attach only the files the current section needs.

## Repo layout
main.py (commands: run bot, `bookings`, `check`, `download-models`, `moss-index`)
src/campusx_voice_agent/: agent.py (pipeline wiring), pipeline.py (VAD/turn strategies, metrics printer),
  tools.py (system prompt + 3 tools + SQLite), knowledge_base.py (BM25 chunk search with "not found"), models.py (Ollama check, model download),
  moss.py / moss_index.py (optional), config.py, bookings.py
  NEW in B: grounding.py (search before the model answers), booking_gate.py (explicit yes + read-back, from the transcript),
  reply_guard.py (markdown, prompt leak, false "I've booked", fake email promise), text_cleanup.py
data/knowledge_base/: campusx_august_schedule.md, dsa_python_course_full_curriculum.md   (data/guidance_bookings.db is git-ignored: real caller emails)
evals/: run_evals.py, harness.py, checks.py, retrieval_eval.py, report.py, compare.py, rescore.py, datasets/*.jsonl,
  heldout/retrieval_heldout.jsonl (32 queries written blind, see Section B), known_prompts/v0_baseline.txt (old prompt, for the leak check),
  baselines/qwen2.5-7b-baseline (raw) and qwen2.5-7b-baseline-rescored (current reference), README.md
tests/test_evals.py (16) + tests/test_brain_fixes.py (41): 57 tests, all pass; fake models, no Ollama needed

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

## Findings to fix (priority order) - Section B status: all five addressed in code, LLM effect NOT yet measured
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
B Brain fixes (day 4): CODE DONE and tested in a sandbox; real-model numbers still to be measured (first job of Section C)
C Re-eval + model comparison (day 5): NEXT   D STT eval (day 6)   E Latency, turn detection, barge-in (day 7)
F Robustness + security (day 8)   G Stability, resources, human spot check (day 9)   H Report + portfolio write-up (day 10)
Optional: simulated user for booking flows; Moss retrieval eval.

## Section B: what was built (2026-11-16)
1. Retrieval in code. knowledge_base.py: per-row schedule chunks, long sections split not cut, BM25 over stemmed content words,
   stopword + spoken-filler removal, spoken numbers ("twenty seven" -> 27), "gen ai"/"llm ops" aliases, numbers in the query must match
   exactly, a number alone is never a match, and a chunk is returned only if it covers >= MIN_COVERAGE (0.5) of the caller's content words.
   Agent-facing sections of the KB ("How the Voice Agent Should Use This Knowledge", example Q&A, "Purpose") are never retrieved (NON_FACT_HEADING).
   grounding.py runs the search on EVERY user turn and attaches passages (or a "nothing matched" note when the message is a question)
   as a system message after the caller's message; old KB messages are replaced, not stacked. If Moss is configured it is used instead.
2. Booking rules in code. get_available_guidance_slots returns at most 3. book_free_guidance_session refuses (booked false + next_step) unless,
   in the real transcript: the caller's latest message is an explicit yes ("ok"/"okay" do NOT count), the previous assistant message read back
   the same full name, email and slot being booked (a corrected name must be read back; a surname alone is not enough), and the slot was
   returned by the availability tool.
3. Prompt: 839 -> 332 words, one length rule (at most three short sentences), no markdown, no section headings, exact wording for the
   abstain line, the confirm question and the leak refusal (constants in tools.py / reply_guard.py).
4. reply_guard.py sits between the LLM and TTS (same class used by the evals): strips markdown, replaces any sentence copying 8+ words of the
   prompt or naming a tool with a fixed refusal and drops the rest of that reply, replaces "I've booked..." unless a booking tool call
   succeeded after the caller's last message, drops promises of a confirmation email.
5. Eval changes (checks themselves unchanged for old runs; rescoring the committed baseline gives NO verdict changes, 20/56 and 3/12):
   harness runs the same grounding/guard/gate code as the live agent and stores raw_text, guard interventions and retrieval per turn;
   new check no_system_prompt_leak (10-word overlap with every known prompt version, injection cases only); tool_used also accepts
   auto retrieval; report shows auto-retrieval and guard counts; flags --no-grounding and --no-guard for ablations.

### Measured in the sandbox (retrieval needs no LLM, so these are real)
| Retrieval | Baseline | Section B |
|---|---|---|
| main set (38 q): hit@1 / hit@4 / MRR | 0.60 / 0.80 / 0.70 | 1.00 / 1.00 / 1.00 |
| unanswerable queries that still return results | 8 / 8 | 0 / 8 |
| held-out set (32 q), BEFORE any tuning | n/a | hit@1 0.727, 0/10 false results |
| held-out set after generic fixes (contractions, filler, number words, aliases) | n/a | hit@1 0.909, 0/10 false results |
Caveats: the main set was used while tuning, so 1.00 is optimistic; the held-out set stopped being blind after the fixes (0.727 is the honest
untuned number). Threshold sweep: MIN_COVERAGE 0.4-0.5 is a plateau on both sets (below 0.4 false results appear on held-out, above 0.5 recall
drops: 0.6 -> 0.82, 0.7 -> 0.77). The 2 remaining held-out misses are semantic and deliberately NOT patched: "evening" vs "8 PM", "jobs" vs "roles".

### NOT measured yet (needs Ollama on the laptop)
Every LLM-level number: scenarios, booking flows, style, latency (the extra ~300-900 tokens of passages per turn will cost some time).
Reference points for reading them: a stand-in model that ALWAYS says "I can't verify that" scores 29/56 on scenarios (0/25 on
kb_dsa + kb_schedule + multi_turn) and 5/12 on booking flows, i.e. above the 20/56 baseline. Judge Section B on the answerable categories
AND kb_unanswerable together, never on the overall percentage alone.

## Decisions (Section B)
- MIN_COVERAGE 0.5 (conservative end of the plateau): a false "found" invites guessing, a false "not found" only sends the caller to support.
- "long", "much", "many" are NOT stopwords, so "how long is the course" is correctly "not in the KB". Cost: "how many topics ..." may abstain.
- Auto retrieval makes tool_used pass by design; grounded_content / abstains_when_unknown / no_forbidden_claim carry the signal.
- Gate does not verify the learning goal in the read-back (the model paraphrases it); the eval's readback check still does.
- "ok"/"okay"/"mm-hmm" are not accepted as the final yes; the agent asks "Shall I book it? Please say yes or no."

## Known limits added by Section B
Retrieval is lexical (no synonyms). The leak guard catches copied wording, not paraphrase ("my rules say I never promise jobs").
The name check is a heuristic (name must be followed by a non-name word); middle names read back differently can be refused.
If the caller interrupts the read-back, the context holds only what was spoken, so a quick "yes" may be refused and the agent re-reads it.
Replies are not truncated in code; the length rule is prompt-only. The "One moment while I check that" line also plays for slot lookups.
The live pipeline (guard + grounding inside real Pipecat with audio) is covered by processor tests but not yet run with a real call.

## NEXT (Section C, day 5): measure it
1. Commit Section B. 2. python -m pytest -q tests (57 pass). 3. Retrieval suite (seconds). 4. Full run with --repeats 1 (comparable to the
single-repeat baseline), then --repeats 3. 5. compare.py against evals/baselines/qwen2.5-7b-baseline-rescored. 6. Optional ablations:
--no-grounding and --no-guard to see what each fix contributes. 7. 5-minute live call check (list in the Section B message). 8. Then
compare llama3.2:3b and another 7-8B model on the same suites.

## How to work with me (Claude) on this project
- Windows PowerShell 5.1. Always start command blocks with the full `cd C:\Users\uday2\Downloads\campusx_voice_agent_local\local_proj`.
- No placeholders like <path> in commands; give runnable commands. Pasted multi-line blocks can swallow prompts, so keep blocks short.
- Files reach the laptop via Downloads: find them with Get-ChildItem ... -Filter, then Expand-Archive and Copy-Item.
- Claude's sandbox has no Ollama or GPU: code is tested there with a fake model; real numbers only come from the user's runs.
- Before committing, run `python -m pytest -q tests` and `git status`; never commit .env, .venv, or guidance_bookings.db.
- At the end of each section, update HANDOFF.md (status, numbers, decisions, next step).
