# External Review Summaries

This directory holds Claude-authored summaries and takeaways for the six
external-review files dropped under `docs/external_review/{gemini,chatgpt}/`
by Gemini and ChatGPT. They are inputs to a forthcoming v1 release roadmap
and are NOT themselves the roadmap.

## Source corpus

| File | Lines | Role |
|---|---:|---|
| `gemini/gemini_report_v1.md` | 244 | Gemini's first research+roadmap report |
| `gemini/gemini_report_v2.md` | 246 | Gemini's second pass with sharper macro/empirical framing |
| `chatgpt/chatgpt_report_v1.md` | 149 | ChatGPT's first attempt — generic, since superseded |
| `chatgpt/leadforge_report_v1_critique.md` | 678 | Critique of ChatGPT v1 — methodology rebuke |
| `chatgpt/leadforge_second_attempt_guidance.md` | 1167 | Guidance attached to the v2 retry |
| `chatgpt/chatgpt_report_v2.md` | 781 | ChatGPT's evidence-grounded second attempt — the strongest single source |

Total: 3265 lines, ~240 KB.

## Summary files (per-source)

- `gemini_v1_summary.md`
- `gemini_v2_summary.md`
- `chatgpt_v1_summary.md` (brief — historical)
- `chatgpt_v1_critique_summary.md`
- `chatgpt_guidance_summary.md`
- `chatgpt_v2_summary.md` (the substantive review)

## Synthesis files (across sources)

- `cross_source_takeaways.md` — themes consolidated across all six sources, agreement vs. divergence vs. unique-to-source
- `key_findings.md` — action-prioritized synthesis: critical → high → medium → low/defer; this is the input list a roadmap would consume

## What's NOT here yet

- A consolidated roadmap. That comes after a "process and recommendations" pass on every key finding (accept / accept-with-different-approach / reject / out-of-scope-and-open-issue / defer) with sign-off from the user.

## How to read this corpus

1. Start with `key_findings.md` — the action-ranked list.
2. Read `cross_source_takeaways.md` for the agreement/divergence map.
3. Drill into `chatgpt_v2_summary.md` for the most actionable single source (relational-leakage blocker, gap matrix, milestone roadmap).
4. Dip into the per-source summaries when you want to know which reviewer said what, especially for items unique to one source.
5. Original review files remain untouched in `gemini/` and `chatgpt/`.
