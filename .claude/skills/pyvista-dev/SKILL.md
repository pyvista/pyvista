---
name: pyvista-dev
description: Develop for PyVista the way the project works. Load before writing code for this repository.
---

# Developing for PyVista

Two documents already carry most of the rules, and this skill does not repeat them.
Read them, then come back here.

- `CONTRIBUTING.rst` is normative. Sections you will need:

  | Topic                                   | Section                                                    |
  | --------------------------------------- | ---------------------------------------------------------- |
  | Coding style, keyword-only arguments    | `Coding Style`                                             |
  | Standard library import rules           | `Import Conventions`                                       |
  | numpydoc rules and the sample docstring | `Docstrings`                                               |
  | Three-step deprecation policy           | `Deprecating Features or other Backwards-Breaking Changes` |
  | Image regression testing                | `Notes Regarding Image Regression Testing`                 |
  | Branch prefixes                         | `Branch Naming Conventions`                                |
  | `make` targets and tox environments     | `Quick Development Commands`                               |

- `context7.json` at the repository root holds the project's agent-facing API rules:
  wrap rather than subclass, prefer PyVista over raw VTK, filters return new datasets,
  the image-regression discipline. It is schema-validated by `pre-commit`, so it stays
  current. Treat it as the source for how to _use_ the API; this skill covers how to
  _change_ it.

What follows is only what neither document states.

## The workflow

| Stage    | Skill                      | Purpose                               |
| -------- | -------------------------- | ------------------------------------- |
| Build    | **pyvista-dev** (this one) | Conventions, sizing, local gates      |
| Critique | **pyvista-review**         | Adversarial review before a PR exists |
| Ship     | **pyvista-pr**             | Title and body in the project's style |

Run the review stage in a **subagent**. A reader who has to derive intent from the diff
alone finds what the author cannot. Fix what it finds before opening anything.

## What the project is for

PyVista is a Pythonic interface to VTK. A feature that makes users think in VTK terms
has failed even when it works.

`CONTRIBUTING.rst` names one value that is easy to skim past and that often decides
reviews: the project wants "good code, concise accurate documentation, and avoiding
unneeded code churn". Churn is called out by name. Prefer the smaller diff.

## Before writing code

Two assumptions cause most of the rework here, and both are one `grep` from certainty:

- **that the name is free.** `cell_connectivity` exists because `connectivity` collides
  with `DataSetFilters.connectivity`. Search the class, its bases, and the filter mixins
  before choosing a public name.
- **that the constant does not already exist.** Version checks, capability probes,
  `Literal` aliases and test guards are the single most requested reuse in review.
  Version-dependent behavior in particular must use the shared constant rather than a
  locally re-derived comparison, because the local copy is what goes stale.

Where a design question has two defensible answers that lead to different code, ask
rather than picking one silently and building on it.

## Size

Merged pull requests here are small: the median adds tens of lines, not hundreds. One
self-contained change per pull request, and refactors go in a separate one from features
and bug fixes. If you find an unrelated bug while working, note it and open a second pull
request rather than folding the fix in.

Large changes are sometimes correct, and a few percent of merged pull requests are large.
The recurring legitimate shapes are a cohesive new subsystem whose halves do not stand
alone, a mechanical sweep that must land atomically or the tree is inconsistent, a
deprecation that has to move every internal call site at once, and a feature whose tests
dwarf it. A change split into pieces that cannot be reviewed or merged independently is
worse than one large change.

When a change has to be large, spend the effort on the reviewer:

- Say in one line why it must land atomically.
- Keep mechanical commits separate from semantic ones. A wide rename plus a small
  behavior change is reviewable as two commits and unreadable as one.
- Name the two or three files carrying the real change and say the rest is mechanical.
- Never fold an opportunistic refactor into an already-large change.

Regenerate the size distribution with:

```bash
gh pr list -R pyvista/pyvista --state merged --limit 400 \
  --json additions,author --jq '.[] | select(.author.is_bot | not) | .additions'
```

Exclude bots, as above. Dependency bumps are a seventh of merged pull requests and pull
every percentile down.

## Conventions that are machine-enforced

`ruff` and `numpydoc` settle most style questions, so do not spend review budget on
them. The custom `pre-commit` hooks are easier to trip, and they produce a red job rather
than a comment:

| Hook                           | Rejects                                                         | Use instead                                |
| ------------------------------ | --------------------------------------------------------------- | ------------------------------------------ |
| `no-bare-import-pyvista`       | a bare `import pyvista`                                         | `import pyvista as pv`                     |
| `no-forbidden-plotter-names`   | `plotter`, `p`, `plot`, `plt`, `pltr` as the `Plotter` variable | `pl`                                       |
| `namespace-stdlib-imports`     | `import pathlib` and friends                                    | `from pathlib import Path`                 |
| `no-lint-suppression-comments` | `noqa` and `ruff:` directives under `examples/`                 | fix it, or `per-file-ignores` in pyproject |
| `no-import-error-skip`         | `importorskip`, `except ImportError`, `suppress(ImportError)`   | make the dependency available              |
| `warn_external`                | `warnings.warn` inside `pyvista/`                               | `warn_external`                            |

Scope matters when a snippet fails one of these: `namespace-stdlib-imports` and
`no-forbidden-plotter-names` run on Python, reStructuredText and Markdown;
`no-bare-import-pyvista` runs on reStructuredText and Markdown only. A documentation
snippet can fail a hook that the equivalent Python file would pass.

Two more that `ruff` enforces and reviewers still notice: error messages are assigned to
a variable before being raised (`EM`), and boolean arguments are keyword-only
(`FBT001`/`FBT002`). Making an existing signature keyword-only goes through
`_deprecate_positional_args`, never a hard break. The counterpart of
`namespace-stdlib-imports` is ruff's `banned-from` list (`ICN003`) in `pyproject.toml`,
which forbids the opposite direction; `Import Conventions` explains why both exist.

## Tests

The heaviest review axis by a wide margin. `CONTRIBUTING.rst` and `context7.json` cover
what to write. This is what gets sent back:

- **Prove the test fails without the fix.** Revert the change, watch it go red, restore.
  It is the only reliable way to catch a test that passes for an unrelated reason.
- **Watch for incidental passes.** A uniform mesh takes a different code path than a
  mixed one, and a fixture can make an assertion true regardless of the change.
- **Test the negative case.** If a feature guards against something, make that something
  happen in a test.
- **Assert enough.** Round-trip equality rather than a single attribute.
- Parametrize rather than branching inside the test body, and do not leave a case
  commented out in a `parametrize` list.
- `filterwarnings` in `pyproject.toml` begins with `error`, so any warning that is not
  explicitly ignored fails the suite. A new deprecation therefore has to migrate every
  internal call site in the same change.

## Local gates

Continuous integration runs on `pull_request`, so a red pull request costs the whole
matrix. `Quick Development Commands` lists the `make` targets, each mirroring a job. At
minimum run `make lint`, the test module you touched, and `make doctest` if you edited a
docstring example.

`tests/conftest.py` and the doctest tox environment already set off-screen rendering, so
the `make` targets are safe. Only a bare `pytest --doctest-modules` outside tox needs
`PYVISTA_OFF_SCREEN=true`; without it the examples open render windows and take over the
display.

## Approaching a red job

- Read the log and find the assertion before changing anything. Do not infer the cause
  from the job name.
- Check whether the failure also occurs on `main` before attributing it to the branch. A
  second worktree at `origin/main` settles this in a minute.
- Never re-run a job as the fix. A flaky test is a defect to be diagnosed.
- A version-specific failure needs the shared version constant, not a local workaround.
