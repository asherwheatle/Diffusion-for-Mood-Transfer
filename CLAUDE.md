# Version Control Instructions

After **every** edit you (Claude) make to this codebase — creating, modifying, or deleting any file — commit and push the change immediately, without waiting for the user to ask. Do this at the end of the turn in which the edit was made, or immediately after each logical edit if a turn contains several unrelated changes.

Run, in order:

```
git add .
git commit -m "<detailed commit message>"
git push origin <current branch name>
```

Where:
- `<current branch name>` is the output of `git branch --show-current` (do not hardcode `main` or `master`).
- `<detailed commit message>` explains *why* the change was made, not just what changed, following the repository's existing commit message style (see `git log` for examples).
- End every commit message with:
  ```
  Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
  ```

## Notes

- Run `git status` before `git add .` and review what would be staged. Do not stage files that look like secrets, credentials, or large binaries that don't belong in version control.
- If `git push` fails (e.g. the remote has diverged), stop and surface the error instead of force-pushing.
- This standing instruction authorizes routine `add`/`commit`/`push` on the current branch after ordinary edits. It does not authorize destructive or history-rewriting operations (`push --force`, `reset --hard`, `rebase`, deleting branches, etc.) — those still require explicit user confirmation each time.
