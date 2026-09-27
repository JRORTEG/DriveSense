# DriveSense — Project Instructions

## Git workflow

- Always create a new branch off `main` before starting work on a new `Plan.md` task. Never commit directly on `main`.
- Branch naming follows the existing pattern: `task-N-short-description` (e.g. `task-5-lead-vehicle-accel`, `task-6-ego-stationary`).
- Do not merge a task's PR into `main` yourself, even if asked with a short "merge"/"merge it" — leave every task PR open on its own branch until the user gives an explicit, deliberate instruction to merge that specific PR (not just "next step" or moving on to the next task). Each task's own branch is the source of truth; `main` only gets a task's code when the user has clearly decided it's ready.
- If a PR was merged into `main` and the user asks to undo it, use `git revert -m 1 <merge-commit>` (not a history rewrite / force-push) so the branch's own commits and PR history stay intact.
- Do not `git commit` automatically as part of finishing a task — implement and test the change, then stop and ask before committing. This applies on task branches too, not just `main`; committing to `main` needs even more caution (see above: never direct, never without an explicit ask).
