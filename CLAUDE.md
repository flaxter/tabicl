# CLAUDE.md — tabicl fork

This repository is a **fork** of `soda-inria/tabicl` maintained by Seth Flaxman at `flaxter/tabicl`. It is the TabICL side of the "Learning to Explain with a Generative Process" project (see companion repo `flaxter/learning-to-explain`).

## CRITICAL: PR targeting

**All pull requests from this repo MUST target `flaxter/tabicl`, NEVER `soda-inria/tabicl`.**

`soda-inria/tabicl` is the upstream package maintainer. We are not contributing to them. GitHub's default PR target on a fork is the upstream (via its fork-relationship metadata), which has already caused one embarrassing mis-targeted PR. Always pass `--repo flaxter/tabicl` to `gh pr create`, or change the "base repository" dropdown in the GitHub web UI before confirming.

For solo work by Claude agents, prefer skipping PRs entirely on branches you own end-to-end:

```
# 1. Fetch the feature branch explicitly — `git pull origin main` only
#    updates main, so origin/claude/<branch> would otherwise be stale or
#    absent locally and the merge below would silently fail with
#    "not something we can merge".
git fetch origin claude/<branch-name> main

# 2. Fast-forward main. If this aborts with "Not possible to fast-forward",
#    main has advanced since the branch was cut — rebase the feature
#    branch onto origin/main first (git rebase origin/main on the feature
#    branch, then force-push), then retry.
git checkout main
git merge --ff-only origin/main
git merge --ff-only origin/claude/<branch-name>

# 3. Push, and ONLY delete the remote branch after the push succeeds.
#    Chain with `&&` so a failed merge or push short-circuits the delete.
git push origin main && \
  git push origin --delete claude/<branch-name>
```

**Footgun:** never run `git push origin --delete <branch>` as an
unconditional final step of a merge recipe. If an earlier command failed
(unknown ref, non-ff, push rejected), the delete still runs and erases
the only remote copy of unmerged commits. Always gate cleanup on the
push succeeding (`&&`) or run it as a separate, verified step.

## Tooling

Use `uv` for all Python: `uv pip install -e .`, `uv run pytest`, `uv venv`. Do not use bare `pip`, `conda`, or `python -m venv`.

## Test commands

```
uv run pytest tests/test_heads.py -v                      # Phase 2 head modules
uv run pytest tests/test_return_column_embeddings.py -v   # Phase 1 trunk flag
```

## Companion repo

Paper, plan, notes, experiments: `../learning-to-explain/` (at `github.com/flaxter/learning-to-explain`). The phase roadmap (Phase 0 → 7) lives in `../learning-to-explain/notes/PLAN.md`; phase-specific decisions in `../learning-to-explain/notes/PHASE*.md`.
