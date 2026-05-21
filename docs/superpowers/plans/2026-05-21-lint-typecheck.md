# Lint + Typecheck Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add ruff (lint + format) and mypy (strict on the package) with CI enforcement via a parallel `lint` job.

**Architecture:** Configuration-only addition to `pyproject.toml` (no new files in the package), plus a new CI job. Source-code fixes are produced by `ruff --fix` / `ruff format` and a hand pass on whatever strict mypy surfaces.

**Tech Stack:** ruff, mypy + pydantic plugin, GitHub Actions, uv.

**Spec reference:** `docs/superpowers/specs/2026-05-21-lint-typecheck-design.md`

---

## File Structure

**Modified:**
- `pyproject.toml` — adds ruff and mypy dev deps, `[tool.ruff]` blocks, `[tool.mypy]` blocks.
- `.github/workflows/ci.yml` — adds `lint` job, updates `publish.needs`.
- `README.md` — appends two lint/typecheck command lines to the Development section.
- Source files under `kagent_a2a_proxy/` — any formatting changes from `ruff format`, autofixes from `ruff check --fix`, and hand fixes for mypy strict.

**Not modified:** test files (`test_*.py`, `conftest.py`) are excluded from mypy via override and only get `ruff format` whitespace changes (no rule violations targeted at them).

---

## Task 1: Add ruff + mypy config to pyproject.toml

Pure configuration. Adds two dev dependencies, three TOML blocks, and one override section for mypy. After this task `uv sync --all-groups` must succeed; nothing else needs to pass yet (we'll auto-fix and hand-fix violations in the next two tasks).

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add ruff and mypy to dev deps**

Edit `pyproject.toml`. Replace the existing `[dependency-groups]` block:

```toml
[dependency-groups]
dev = [
    "httpx>=0.28.1",
    "pytest>=9.0.3",
    "pytest-asyncio>=1.3.0",
    "respx>=0.23.1",
]
```

with:

```toml
[dependency-groups]
dev = [
    "httpx>=0.28.1",
    "pytest>=9.0.3",
    "pytest-asyncio>=1.3.0",
    "respx>=0.23.1",
    "ruff>=0.7",
    "mypy>=1.13",
]
```

- [ ] **Step 2: Append ruff and mypy config blocks**

Append the following to the end of `pyproject.toml`:

```toml

[tool.ruff]
target-version = "py312"
line-length = 88

[tool.ruff.lint]
select = ["E", "W", "F", "I", "B", "UP", "SIM", "RUF"]

[tool.ruff.lint.per-file-ignores]
"test_*.py" = ["E501"]
"conftest.py" = ["E501"]

[tool.mypy]
python_version = "3.12"
plugins = ["pydantic.mypy"]
warn_unused_ignores = true
check_untyped_defs = true

[[tool.mypy.overrides]]
module = "kagent_a2a_proxy.*"
strict = true

[[tool.mypy.overrides]]
module = ["test_*", "conftest"]
ignore_errors = true

[[tool.mypy.overrides]]
module = ["fastmcp.*", "respx.*"]
ignore_missing_imports = true
```

- [ ] **Step 3: Sync deps**

Run: `uv sync --all-groups`
Expected: ruff and mypy install successfully; `uv.lock` is updated.

- [ ] **Step 4: Confirm pytest still green**

Run: `uv run pytest -q`
Expected: 33 passed.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build: add ruff + mypy dev deps and config"
```

---

## Task 2: Apply ruff auto-fix and ruff format

Lets ruff fix everything mechanical first. After this task `uv run ruff check .` and `uv run ruff format --check .` both exit 0.

**Files:**
- Modify: any source files under `kagent_a2a_proxy/`, plus possibly tests, that ruff touches.

- [ ] **Step 1: Run ruff check with auto-fix**

Run: `uv run ruff check --fix .`
Expected: outputs the count of issues fixed (or "All checks passed!"). Some unfixable issues may remain — that's handled in Step 3.

- [ ] **Step 2: Run ruff format**

Run: `uv run ruff format .`
Expected: outputs the count of files reformatted (or "X files left unchanged").

- [ ] **Step 3: Handle any remaining ruff check failures**

Run: `uv run ruff check .`
Expected: exit code 0. If non-zero, the output lists remaining violations.

For each remaining violation:
- Read the rule code (e.g. `B008`, `SIM117`) and the message.
- If it's a legitimate code smell, fix it.
- If the rule conflicts with the project's intentional style (rare with this rule set), add a narrowly-scoped `# noqa: <code>` with a one-line comment explaining why, or add a per-file ignore in `pyproject.toml` if the rule is wrong for the whole file.

Re-run `uv run ruff check .` until it exits 0. If you find yourself adding more than two `# noqa` comments or thinking the rule set is wrong, STOP and report back as DONE_WITH_CONCERNS so the controller can decide whether to revisit the rule selection.

- [ ] **Step 4: Confirm format check passes**

Run: `uv run ruff format --check .`
Expected: exit 0 (every file already formatted by Step 2).

- [ ] **Step 5: Confirm pytest still green**

Run: `uv run pytest -q`
Expected: 33 passed. Auto-fixes should never change behavior; if pytest fails, something went wrong — STOP and report back.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "style: apply ruff auto-fix and ruff format"
```

If the working tree is clean (ruff had nothing to fix), skip the commit and report DONE with that observation.

---

## Task 3: Fix mypy strict violations on the package

Runs strict mypy on `kagent_a2a_proxy/` and fixes every violation. Tests are excluded by the override block from Task 1.

**Files:**
- Modify: any module under `kagent_a2a_proxy/` that mypy flags.

- [ ] **Step 1: Run mypy and capture output**

Run: `uv run mypy kagent_a2a_proxy`
Expected: a list of violations grouped by file, or "Success: no issues found".

If the run is already clean, skip to Step 4.

- [ ] **Step 2: Fix violations**

For each error reported, fix it in place. Common categories you'll likely see and how to handle each:

- **`Missing return statement`** / **`Incompatible return value type`** — tighten the return annotation, or add the missing branch.
- **`Function is missing a type annotation`** — add the annotation. Strict mypy requires every function to be fully annotated. Look at neighbors for the right type names (`AsyncIterator`, `dict[str, Any]`, etc.).
- **`Item "None" of "X | None" has no attribute "Y"`** — narrow with `if x is not None:` before access.
- **`Argument N to "f" has incompatible type`** — usually `str` vs `AnyHttpUrl` mismatch around `settings.kagent_base_url`. Cast at the boundary (`str(settings.kagent_base_url).rstrip("/")`, which is already used).
- **`Untyped decorator makes function "f" untyped`** — affects `@mcp.tool(...)` in `mcp_server.py`. If unavoidable, add `# type: ignore[misc]` on the decorator line with a comment explaining `fastmcp` doesn't ship type stubs for the decorator factory.
- **`Need type annotation for "X"`** — give the local an explicit type (`parts: list[str] = []`).

Do NOT relax the strict override. The goal is a clean strict pass on the package.

If you hit a single error category that requires more than three `# type: ignore` comments or that suggests a real type problem in the production code, STOP and report DONE_WITH_CONCERNS describing what you found.

- [ ] **Step 3: Re-run mypy until clean**

Run: `uv run mypy kagent_a2a_proxy`
Expected (final): `Success: no issues found in N source files`.

- [ ] **Step 4: Confirm pytest still green**

Run: `uv run pytest -q`
Expected: 33 passed.

- [ ] **Step 5: Confirm ruff still clean**

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: both exit 0. If any of your mypy fixes broke formatting, run `uv run ruff format .` again before committing.

- [ ] **Step 6: Commit**

```bash
git add kagent_a2a_proxy
git commit -m "fix(types): satisfy mypy strict on kagent_a2a_proxy"
```

If the working tree is clean (mypy had nothing to flag), skip the commit and report DONE with that observation.

---

## Task 4: Add lint job to CI workflow + README update

Adds the parallel `lint` job and updates `publish.needs`. Then appends two convenience commands to the README's Development section.

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `README.md`

- [ ] **Step 1: Add the `lint` job to ci.yml**

Edit `.github/workflows/ci.yml`. Locate the `jobs:` section. The current file has two top-level jobs under `jobs:`: `test` and `publish`. Insert a new `lint` job between them.

The full updated `jobs:` section should read:

```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python: ['3.12', '3.13', '3.14']
    steps:
      - uses: actions/checkout@v4

      - name: Set up uv
        uses: astral-sh/setup-uv@v3
        with:
          python-version: ${{ matrix.python }}

      - name: Install dependencies
        run: uv sync --all-groups

      - name: Run tests
        run: uv run pytest -q

  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up uv
        uses: astral-sh/setup-uv@v3
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: uv sync --all-groups

      - name: Ruff check
        run: uv run ruff check .

      - name: Ruff format check
        run: uv run ruff format --check .

      - name: Mypy
        run: uv run mypy kagent_a2a_proxy

  publish:
    needs: [test, lint]
    if: github.event_name != 'pull_request'
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to GHCR
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract image metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ghcr.io/${{ github.repository }}
          tags: |
            type=raw,value=latest,enable={{is_default_branch}}
            type=sha,prefix=sha-,enable={{is_default_branch}}
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            type=semver,pattern={{major}}

      - name: Build and push
        uses: docker/build-push-action@v6
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

The only changes compared to the current file are:

1. New `lint:` job inserted between `test:` and `publish:`.
2. `publish.needs:` changes from `test` to `[test, lint]`.

Everything else is unchanged.

- [ ] **Step 2: Validate the YAML**

Run: `uv run python -c "import yaml,sys; yaml.safe_load(open('.github/workflows/ci.yml')); print('ok')"`
Expected: `ok`.

If pyyaml isn't available, fall back to: `python -c "import yaml,sys; yaml.safe_load(open('.github/workflows/ci.yml')); print('ok')"`. If neither yaml is installed, skip validation — GitHub will validate on push.

- [ ] **Step 3: Update README Development section**

Edit `README.md`. Find the Development section. It currently reads:

````markdown
## Development

```bash
uv sync --all-groups
uv run pytest -q
```
````

Replace that code block with:

````markdown
## Development

```bash
uv sync --all-groups
uv run pytest -q                                       # tests
uv run ruff check . && uv run ruff format --check .    # lint
uv run mypy kagent_a2a_proxy                           # typecheck
```
````

- [ ] **Step 4: Run the full local pipeline once more**

Run, in sequence (any failure means stop and fix before continuing):

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run mypy kagent_a2a_proxy
```

Expected: each command exits 0. pytest still reports 33 passed.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/ci.yml README.md
git commit -m "ci: add lint job (ruff + mypy) and gate publish on it"
```

---

## Task 5: Push

The user already authorized pushing in the previous session.

- [ ] **Step 1: Confirm everything is committed**

Run: `git status --short`
Expected: empty.

- [ ] **Step 2: Push**

Run: `git push`
Expected: push succeeds. The new `lint` job will appear on the next workflow run; verify on GitHub that both `test` and `lint` succeed and `publish` runs only after both.
