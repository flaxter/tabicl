# CLAUDE.md — tabicl fork

This repository is a **fork** of `soda-inria/tabicl` maintained by Seth Flaxman at `flaxter/tabicl`. It is the TabICL side of the "Learning to Explain with a Generative Process" project (see companion repo `flaxter/learning-to-explain`).

## CRITICAL: PR targeting

**All pull requests from this repo MUST target `flaxter/tabicl`, NEVER `soda-inria/tabicl`.**

`soda-inria/tabicl` is the upstream package maintainer. We are not contributing to them. GitHub's default PR target on a fork is the upstream (via its fork-relationship metadata), which has already caused one embarrassing mis-targeted PR. Always pass `--repo flaxter/tabicl` to `gh pr create`, or change the "base repository" dropdown in the GitHub web UI before confirming.

## Branching workflow

`main` is the only live branch. The post-pivot refactor (conditional predictive value heads, new training loss, eval rewrite) was merged on 2026-04-29.

- **Agent-driven fixes** (reviewer findings on l2e clusters that touch tabicl files — e.g., `src/tabicl/prior/labels.py`, `src/tabicl/model/heads.py`, `src/tabicl/train/*`): branch off `main` as `fix/<slug>`, commit, push, open a PR to `main`, ask Seth to merge.
- **No more long-lived feature branches.** No `refactor/...`, no `claude/<phase>`. Work is small and lands on `main` quickly.

**Footgun:** never run `git push origin --delete <branch>` as an unconditional step. If an earlier merge/push failed, the delete still runs and erases the only remote copy of unmerged commits. Always gate cleanup on the push succeeding (`&&`) — or rely on GitHub's auto-delete-source-branch-on-merge setting (currently enabled on this repo).

## Upstream sync deferred

As of 2026-04-29, `flaxter/tabicl:main` is 5 commits behind `soda-inria/tabicl:main`. None are critical bug fixes. Notable: PR #84 ("Clarify public vs. private API") renamed `train/{run,optim,train_config}.py` → underscore-prefixed; this will conflict noisily with our refactor's edits to those files when we eventually re-sync. Defer upstream sync until post-NeurIPS submission.

## Tooling

Use `uv` for all Python: `uv pip install -e .`, `uv run pytest`, `uv venv`. Do not use bare `pip`, `conda`, or `python -m venv`.

## Test commands

```
uv run pytest tests/test_heads.py -v                      # Phase 2 head modules
uv run pytest tests/test_return_column_embeddings.py -v   # Phase 1 trunk flag
```

## Companion repo

Paper, plan, notes, experiments: `../learning-to-explain/` (at `github.com/flaxter/learning-to-explain`). The phase roadmap (Phase 0 → 7) lives in `../learning-to-explain/notes/PLAN.md`; phase-specific decisions in `../learning-to-explain/notes/PHASE*.md`.
