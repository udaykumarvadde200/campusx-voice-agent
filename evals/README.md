# Evals for the CampusX voice agent (baseline days 1-3; Section B added day 4)

These evals test the agent's *brain*: the real system prompt, the real tool schemas
(built by Pipecat's own adapter) and the real tool code, driven by text. STT, TTS, VAD,
barge-in and end-to-end voice latency are **not** covered here; they come in later days.

## Run it (PowerShell, venv active, Ollama running)

```powershell
pip install pytest
python -m pytest -q tests/test_evals.py              # tests of the harness itself, no Ollama needed

python evals/run_evals.py --suite retrieval --tag baseline            # seconds, no LLM
python evals/run_evals.py --suite all --model qwen2.5:7b --tag baseline
python evals/run_evals.py --suite all --model qwen2.5:7b --repeats 3 --tag baseline-x3
```

Useful flags: `--suite {all,retrieval,scenarios,booking}`, `--ids dsa-graphs,guardrail`
(case ids or category names), `--limit 5` for a quick smoke test, `--repeats N` (LLMs are
noisy; use 3 or more for numbers you quote), `--seed 7`, `--temperature 0` for determinism.
Defaults match the live agent (temperature 0.3, max_tokens 220).

Each run writes `evals/results/<timestamp>_<model>_<tag>/` with `summary.md` (read this first),
`summary.json`, `agent.jsonl` and `booking.jsonl` (every reply and tool call), and
`retrieval.json`. Compare two runs with paired statistics:

```powershell
python evals/compare.py evals/results/<before> evals/results/<after>
```

## What is measured

| Suite | Cases | Checks (all rule-based, no LLM judge) |
|---|---|---|
| Retrieval | 38 queries (30 answerable, 8 not in the KB) | hit@1, hit@4, MRR, and a diagnosis for every miss: ranked below k, snippet truncated, or no token overlap. Also whether unanswerable queries can be told apart by score. |
| Scenarios | 56 (DSA course, schedule, not-in-KB, guardrails, injection, style, tool gating, multi-turn) | tool used when needed, grounded facts present, abstains when unknown, no forbidden claims (prices, durations, guarantees, coupon codes, links, leaked prompt), booking tools not misused |
| Booking flows | 12 scripted conversations (7 should book, 5 must not) | database outcome, no booking before the caller says yes, read-back of name/email/goal/slot before booking, slots come from the availability tool, at most 3 offered, stored values correct, no tool errors |
| Style (every reply) | all of the above | at most 3 sentences, no markdown, no leaked tool-call JSON, non-empty |

Task pass and style pass are reported separately. Cold-start time (first request, includes
the model load) is recorded as `warmup_s`.

## How to read the numbers honestly

* Small suites. A 95% Wilson interval is printed next to each pass rate; with 12 flows a
  difference of a few cases is noise. Use `--repeats` and `compare.py`.
* Rule-based checks are strict and sometimes unfair to a correct answer phrased unexpectedly.
  When a failure looks wrong, read the reply in `agent.jsonl`, then fix the check, not the model.
* Booking flows use fixed user lines, so a model that asks things in a different order can fail
  for reasons that are not its fault. A simulated user is a planned improvement.
* "Unverified claim" cases (`unverified-*`) score the policy "never state CampusX facts that are
  not in the knowledge base", even when the model's guess happens to be true in the real world.
* The knowledge base here is two files. Retrieval numbers describe this KB, not CampusX in general.
* Moss retrieval is not evaluated yet; only the local keyword search path is.

## Files

`datasets/*.jsonl` test cases | `checks.py` all pass/fail logic | `harness.py` runs conversations
| `retrieval_eval.py` | `report.py` | `run_evals.py` CLI | `compare.py` paired comparison.
Add cases by appending JSON lines; `tests/test_evals.py` validates the dataset shape and that
every retrieval label really exists in the source files.

## Roadmap

Day 4 fix what the baseline finds (forced retrieval, brevity), day 5 re-run and compare plus model
comparisons, day 6 STT, day 7 latency and barge-in, day 8 robustness and injection, day 9 long
sessions and resources, day 10 report.


## Section B notes

- The harness runs the SAME grounding, booking gate and reply guard as the live agent, so the numbers describe what a caller
  hears. Each turn stores `raw_text` (model) and `text` (spoken), plus `guard` and `retrieval`. `summary.md` shows how many
  turns the guard changed; a pass that needed the guard is a safety-net pass, not a model improvement.
- Ablations: `--no-grounding` (no search before the answer) and `--no-guard` (speak raw model text). The booking gate is part of
  the tool and has no switch.
- `evals/heldout/retrieval_heldout.jsonl` (32 queries) was written blind before tuning retrieval. It is not in `datasets/`, so it does
  not change the dataset hashes that `compare.py` checks. Score it with a small script or `retrieval_eval.run_retrieval`.
- `checks.no_system_prompt_leak` compares replies with every prompt in `evals/known_prompts/` plus the live prompt. Keep old prompts
  there when you rewrite the prompt, otherwise old runs stop being scorable.
- Floor to beat: a model that always answers "I can't verify that" scores 29/56 on scenarios (0/25 on the answerable categories) and
  5/12 on booking flows. Read category and check tables, not the overall percentage.
- `tool_used` is satisfied by auto retrieval; it no longer measures whether the model chose to search.
