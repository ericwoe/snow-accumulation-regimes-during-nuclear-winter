# CLAUDE.md

This file defines the rules and conventions for this project. All rules here are mandatory.

You obsessively plan, audit and implement using the totality of your skills and elegance. Think holistically through this from first-principles. Be methodical, genius, high effort - high touch. Execute elegantly, impeccably, thoroughly, production-grade. Don’t waste time optimizing something that shouldn’t exist, ask yourself the hard questions and audit any conclusions. Elegant interventions that break nothing. Robust and scalable.

---

## CRITICAL WORKFLOW REQUIREMENTS

**THESE RULES ARE NON-NEGOTIABLE.**

1. **Work modularly.** Complete one module at a time. After each module, report what you built and wait for confirmation before proceeding.

2. **Iterate and fix errors yourself.** Do not rely on the user to report errors back to you. Run the code, observe the output, and fix problems before presenting results.

3. **Be explicit about unknowns.** If you're uncertain about something, say so. Don't guess.

4. **Use `python3` and `pip3`.** Never use `python` or `pip` directly.

This project does not use an automated test suite or linting/pre-commit
checks (a deliberate choice, not an oversight) — see the README's "Status"
section. Don't add tests, Ruff config, or pre-commit hooks unless the user
asks for them.

---

## Environment & Dependency Management

- **Always use [UV](https://docs.astral.sh/uv/)** for all package and environment management. Never use conda, mamba, or bare pip.
- Every repository must have its own virtual environment managed by UV (lives in `.venv/`, gitignored).
- Common commands:
  - `uv sync` — install all dependencies from `pyproject.toml`
  - `uv run python script.py` — run a script inside the project environment
  - `uv add <package>` — add a new dependency

---

## Python Style Guide

Follow [PEP 8](https://peps.python.org/pep-0008/). Key rules:

### Docstrings

Every function needs a docstring in this exact format:

```python
def count_line(f, line):
    """
    Counts the number of times a line occurs. Case-sensitive.

    Arguments:
        f (file): the file to scan
        line (str): the line to count

    Returns:
        int: the number of times the line occurs.
    """
```

### Naming Conventions

| Thing | Style | Example |
|---|---|---|
| Variables, functions, modules | snake_case | `my_variable`, `my_module.py` |
| Classes | CamelCase | `MyClass` |
| Jupyter notebooks | Camel Case With Spaces | `Analyze Results.ipynb` |

### Code Quality Rules

- Write decoupled code: each function does exactly one thing and is as short as possible.
- Delete dead code. Never comment out unused code — use git history to recover it.
- No commented-out code blocks.
- Use `print()` for any logging output. Do not use the `logging` module.

### Code Organization

- All reusable logic goes in `.py` files under `src/`.
- Jupyter notebooks (in `notebooks/`) are for final visualizations and explanations only — not for implementing logic.

---

## Plotting Style

This project currently plots with the ALLFED matplotlib style sheet
(loaded from a URL in `main.py`), not the template's
`style/scientific.mplstyle`. Keep using the ALLFED style for consistency
with existing figures unless the user asks to switch.

Published maps (`src/postprocessing/snow_analysis.py`) use cartopy's
Robinson projection, not the template's Winkel Tripel convention. Don't
change the map projection unless the user asks for it.
