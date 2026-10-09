# Open-source contribution candidates (vLLM / SGLang)

## Purpose

Setting up vLLM and SGLang for the cross-engine study (ADR-025,
`docs/xengine/SPEC.md`) will hit friction: a flag the docs don't explain, an
error message that points at the wrong cause, a crash that reproduces. Each of
those is a lead for an upstream contribution — a docs PR, a clearer error, or a
bug report with a repro. This file captures them **at the moment they happen**,
while the exact command, version, and output are still on screen.

Why it matters: friction that is not written down immediately gets worked
around and forgotten, and a vague memory ("the SGLang flags were confusing") is
not something an upstream maintainer can act on. A repro with a pinned version
is.

## Rules

- **Entries come only from friction actually encountered** while running this
  study. No speculative entries, no issues found by reading other people's bug
  reports, no "this might be confusing".
- Every entry names the **exact engine version** (pip version and, if built
  from source, the commit) and an **evidence path** — a log, terminal capture,
  or artifact committed under `results/xengine/` or `bench/xengine/logs/`.
- **Search upstream before proposing anything.** Record what was searched
  (issue tracker query, docs page) and what was found. A duplicate of an open
  issue becomes a comment on that issue, not a new one.
- An entry is a *candidate*. Nothing here is filed upstream without a fresh
  repro on the pinned version.

## Entry format

Copy this block for each entry. Number entries sequentially (`OSS-001`, …).

```markdown
### OSS-NNN — <one-line title>

- **Engine + version:** vLLM x.y.z / SGLang x.y.z (pip) — commit <sha> if from source
- **Category:** doc gap | confusing error | reproducible bug
- **Found while:** <what the study was doing, e.g. "launching the vllm-eager arm for W1">
- **Repro steps:**
  1. <exact command, with flags and env vars>
  2. ...
- **Expected:** <what the docs or a reasonable user would expect>
- **Actual:** <what happened; paste the key lines of the error verbatim>
- **Evidence:** <path to committed log/artifact>
- **Upstream search done?** yes/no — <queries run, links to related issues/PRs, or "none found">
- **Proposed fix:** <docs wording, error-message change, or patch sketch>
- **Status:** candidate | repro confirmed on pinned version | filed (<link>) | merged (<link>) | dropped (<reason>)
```

Category definitions:

- **doc gap** — the behavior is correct, but the documentation is missing,
  wrong, or out of date for this version.
- **confusing error** — the engine fails correctly, but the message points at
  the wrong cause or omits what the user must change.
- **reproducible bug** — the engine does the wrong thing, and the steps above
  make it happen every time on the stated version.

## Entries

*None yet.* Entries are added only from friction actually encountered during
the study.
