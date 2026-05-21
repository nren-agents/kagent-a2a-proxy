# Add ruff + mypy with CI integration

**Date:** 2026-05-21
**Status:** Approved (design phase)

## Goal

Add linting (ruff), formatting (ruff format — black-compatible, replaces a
separate black step), and static type checking (mypy) to the project, with
sane defaults. Wire them into CI as a separate `lint` job that runs in
parallel with the existing `test` matrix; gate `publish` on both jobs.

## In scope

- Add `ruff` and `mypy` to `[dependency-groups].dev` in `pyproject.toml`.
- Add `[tool.ruff]`, `[tool.ruff.lint]`, `[tool.ruff.lint.per-file-ignores]`,
  and `[tool.mypy]` (+ overrides) blocks to `pyproject.toml`.
- Add a `lint` job in `.github/workflows/ci.yml` running on Python 3.12 only.
  Update `publish.needs` to `[test, lint]`.
- Fix any violations surfaced by the new config so CI lands green from the
  first push. Auto-fix what `ruff` can; fix mypy issues by hand.
- Add a short Development-section note in `README.md` showing the lint /
  typecheck commands.

## Out of scope

- Pre-commit hooks. (User said they should be CI-enforced.)
- Coverage tooling, security scanners, dependency-update bots.
- Black as a separate formatter — explicitly replaced by `ruff format`.

## Components

### `pyproject.toml`

**Dev deps**:

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

**Ruff config**:

```toml
[tool.ruff]
target-version = "py312"
line-length = 88

[tool.ruff.lint]
select = ["E", "W", "F", "I", "B", "UP", "SIM", "RUF"]

[tool.ruff.lint.per-file-ignores]
"test_*.py" = ["E501"]
"conftest.py" = ["E501"]
```

Rationale: pycodestyle (E/W), pyflakes (F), isort (I), bugbear (B),
pyupgrade (UP), simplify (SIM), ruff-specific (RUF) — the standard starter
set. No project-specific carve-outs at first; add per-file-ignores only if a
broad rule trips on something we deliberately keep. Line length 88 matches
the existing code style.

**Mypy config**:

```toml
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

Rationale: pydantic mypy plugin makes BaseModel / BaseSettings type-check
cleanly. Strict on the package; tests are excluded (they use respx /
fastmcp patterns that are awkward to type strictly). `fastmcp` and `respx`
get the missing-imports escape so we don't fail on absent stubs.

### `.github/workflows/ci.yml`

Add a third top-level job, parallel with `test`:

```yaml
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
        with:
          python-version: '3.12'
      - run: uv sync --all-groups
      - run: uv run ruff check .
      - run: uv run ruff format --check .
      - run: uv run mypy kagent_a2a_proxy
```

Update `publish.needs` from `test` to `[test, lint]`.

PRs and pushes both run `test` + `lint` in parallel; `publish` waits on both
and is still gated on `if: github.event_name != 'pull_request'`.

### `README.md`

Append two lines to the Development section, after `uv run pytest -q`:

```
uv run ruff check . && uv run ruff format --check .   # lint
uv run mypy kagent_a2a_proxy                          # typecheck
```

### Source-file fixes

After enabling the config, anything that fails `ruff check`,
`ruff format --check`, or `mypy` gets fixed in the same change:

- `ruff check --fix .` + `ruff format .` handle most issues mechanically.
- Remaining mypy failures fixed by hand (likely narrow type-narrowing or
  explicit `# type: ignore[code]` on irreducible spots — e.g. fastmcp's
  dynamic `@mcp.tool` decorator).

We commit all fixes together with the config so the CI history doesn't
start red.

## Risks

- **Mypy strict surfaces a lot.** The current code is well-typed but hasn't
  been validated against strict mypy. The fix pass may touch every module.
  Acceptable: violations are an audit, not a redesign.
- **Pydantic `AnyHttpUrl` typing.** `Settings.kagent_base_url` is `AnyHttpUrl`
  but used as a string after `str(...).rstrip("/")`. Mypy should be fine
  with this, but worth confirming.
- **Ruff `RUF` rules are pre-release-ish.** They evolve. If a future ruff
  version flips a rule and breaks CI, we'll deal with it then — pin the
  minor in dev deps to slow the churn.
- **`fastmcp` dynamic decorators.** `@mcp.tool` registration creates a
  closure that mypy may flag. If unavoidable, a single `# type: ignore` is
  acceptable; restructuring the decorator for typability is out of scope.

## Verification

Done means:

- `uv run ruff check .` exits 0.
- `uv run ruff format --check .` exits 0.
- `uv run mypy kagent_a2a_proxy` exits 0.
- `uv run pytest -q` still 33 passed.
- CI workflow runs both `test` and `lint` jobs on the next push, both
  succeed, and `publish` runs on main.
