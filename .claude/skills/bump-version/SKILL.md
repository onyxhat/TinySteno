---
name: bump-version
description: >-
  Bump the TinySteno version (patch by default) and cut a local release commit
  and tag. Use when landing changes on main, cutting a release, or when asked to
  bump, iterate, or roll the version. Mirrors what the version-bump GitHub
  Actions workflow does automatically on push to main.
---

# Bumping the version

`scripts/bump_version.py` is the single source of truth: it rewrites the version
in `pyproject.toml` and `tinysteno/__init__.py` together and prepends a dated
section to `CHANGELOG.md` from the commit subjects since the last tag. This skill
runs that script and wraps it in a release commit and tag. It never pushes.

The same script runs in CI (`.github/workflows/version-bump.yml`) on every push
to `main`, so normally you do **not** need this skill — reach for it only when
bumping by hand, e.g. a manual release or a `--set` version jump.

## Checklist

1. **Preconditions.** Confirm the current branch is `main` and `git status` is
   clean. If either fails, stop and tell the user — do not bump on a feature
   branch or over uncommitted work.
2. **Run the bump.** `uv run python scripts/bump_version.py` for a patch bump, or
   `uv run python scripts/bump_version.py --set X.Y.Z` for an explicit version.
   The new version is the last line on stdout; capture it as `NEW`.
3. **Handle the no-op case.** If the script prints `nothing to do` to stderr (the
   HEAD commit is already a bump, or `--set` matched the current version), stop
   here. Nothing was changed; there is nothing to commit.
4. **Review.** Show the user `git diff` for `pyproject.toml`,
   `tinysteno/__init__.py`, and `CHANGELOG.md`. Fix the changelog wording if a
   commit subject reads badly.
5. **Test.** `uv run pytest` and `uv run pylint $(git ls-files '*.py')` — the
   version change must not break either.
6. **Commit.** Stage exactly those three files, then
   `git commit -m "chore: bump version to $NEW [skip ci]"`. The `[skip ci]`
   marker and the `chore: bump version` prefix are what stop the CI workflow from
   bumping again on top of this commit — keep both.
7. **Tag.** `git tag "v$NEW"`.
8. **Report.** Tell the user the new version and that they can publish it with
   `git push --follow-tags`. Do not push for them.
