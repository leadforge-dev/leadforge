# AGENTS.md — leadforge

Agent-specific conventions layered on top of CLAUDE.md.

---

## PR Review Comment Workflow

When addressing PR review comments (Copilot, human reviewers, or otherwise):

1. Triage each comment — recommend one of: resolve as irrelevant, accept and implement, open a separate issue and resolve as out-of-scope, accept a different solution, or resolve as already treated.
2. After the user confirms decisions, implement accepted changes and push **all changes in a single commit** to the PR branch.
3. **After the commit lands, resolve the corresponding GitHub review threads** using the GraphQL API:

```bash
gh api graphql -f query='mutation { resolveReviewThread(input: {threadId: "PRRT_..."}) { thread { isResolved } } }'
```

Resolve every addressed thread — whether the action was "implement", "already treated", or "irrelevant/out-of-scope". Unresolved threads indicate open work; resolved threads mean the discussion is closed.

Do **not** leave threads unresolved after the commit is pushed.

---

## Branch & PR Conventions

See CLAUDE.md for the full mandatory branch/PR workflow (branch → commit → update `.agent-plan.md` → open PR).
