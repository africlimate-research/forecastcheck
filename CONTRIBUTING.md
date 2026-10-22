# Contributing to ForecastCheck

Thank you for helping. This page explains how we work together, so that everyone builds on
the same, current code. Please read it before your first branch.

Questions, or not sure where to start? Open an issue, or write to
[research@africlimate.ai](mailto:research@africlimate.ai).

## How we work: one `main`, short branches

`main` is the only long-lived branch. It always passes the tests, and every piece of work
starts from it and goes back into it, usually within a few days:

```
main ── feat/ghana       ── PR ──► main
main ── feat/era5-truth  ── PR ──► main
main ── fix/kenya-season ── PR ──► main
```

- `main` stays up to date.
- Everyone starts from the same, current code.
- Conflicts stay small.
- Each branch is deleted after merging (GitHub does this automatically).

**Countries are files, not branches.** A country is a `deployments/{name}.toml` file plus its
admin boundaries and settings, merged into `main` like any other change. Never keep a
long-lived branch per country: it drifts from `main`, and fixes stop reaching it.

### Branch names

Start the name with what the change is, then a few words:

| Prefix  | For                         | Example                      |
| ------- | --------------------------- | ---------------------------- |
| `feat/` | a new feature or country    | `feat/ghana`                 |
| `fix/`  | a bug fix                   | `fix/kenya-season-months`    |
| `docs/` | documentation only          | `docs/setup-on-windows`      |

Other prefixes are fine when none of these fit (`test/`, `ci/`, `chore/`).

### Step by step

```bash
# 1. Start from the latest main
git checkout main
git pull

# 2. Create your branch
git checkout -b feat/my-change

# 3. Work, commit, and run the tests
python -m pytest tests/

# 4. Push and open a pull request into main
git push -u origin feat/my-change
gh pr create --base main        # or open it on GitHub

# 5. After the merge, go back to main and delete your local branch
git checkout main
git pull
git branch -d feat/my-change
```

If `main` moves while your branch is open, bring it in with `git pull origin main` (or the
**Update branch** button on the pull request) and re-run the tests.

Working from outside the team? Fork the repository, push the branch to your fork, and open
the pull request from there — everything else is the same.

## Pull requests

- **Small and focused**: one change per pull request. Two unrelated fixes are two PRs.
- **Describe the why**: what problem it solves, and how you checked it (tests, a run, a
  screenshot of the dashboard).
- **Tests must pass.** CI runs the full suite on every pull request. A red check is fixed
  before merging.
- **One review** from a maintainer before merging.
- **Squash and merge**: `main` gets one clear commit per pull request.
- Open a **draft** pull request early if you want feedback before it is finished.

## Setting up

See the README's *Installation*. In short, with conda (Linux, macOS, DGX Spark):

```bash
conda env create -f environment.yml
conda activate forecastcheck
python -m pytest tests/          # confirms the installation
```

The tests need no data: tests that read a country's data skip when `data/` is absent.

## Code conventions

- **License header.** Every `.py` file starts with:

  ```python
  # Authors: The africlimate AI team
  # SPDX-License-Identifier: MIT
  ```

- **Absolute imports only** — `from verification import diagnostics`, never `from . import …`.
- **Never commit data or credentials.** `data/` is git-ignored; keep it that way. Keys are
  referenced by their path (`GOOGLE_APPLICATION_CREDENTIALS`), never copied into the
  repository. See [SECURITY.md](SECURITY.md).
- **Keep the change tested.** A bug fix comes with a test that would have caught it; a new
  feature comes with tests of its behaviour.

## Reporting a problem

Open an issue with the *Bug report* template: what you did, what you expected, what happened
instead, the command, the error message, and your country.

## Code of conduct

Everyone taking part follows the [code of conduct](CODE_OF_CONDUCT.md).
