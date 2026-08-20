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
  the image-regression discipline. It is schema-validated by `pre-commit`. Treat it as the source for how to _use_ the API; this skill covers how to
  _change_ it.

## The workflow

Build here, wrap with **pyvista-vtk**, test with **pyvista-testing**, critique with
**pyvista-review**, ship with **pyvista-pr**. `AGENTS.md` at the repository root routes to
all five and carries the rules that apply before any of them.

Run the critique stage in a **subagent** and fix what it finds before opening anything.

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

`warn_external` is the one that behaves differently: it is a `libcst` codemod, so it
rewrites the call for you and fails the run because the file changed. Stage its rewrite
rather than reverting it.

Scope matters when a snippet fails one of these: `namespace-stdlib-imports` and
`no-forbidden-plotter-names` run on Python, reStructuredText and Markdown;
`no-bare-import-pyvista` runs on reStructuredText and Markdown only;
`no-lint-suppression-comments` runs under `examples/` and `no-import-error-skip` under
`tests/`. A documentation snippet can fail a hook that the equivalent Python file would
pass.

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
  explicitly ignored fails the suite.

## Local gates

`Quick Development Commands` lists the `make` targets, each mirroring a CI job. Run
`make lint`, `make docstyle`, `make doctest`, and the test module you touched, and treat a
change as unfinished until they pass.

`make doctest` is the one that gets skipped for looking unrelated: it runs every docstring
example in the package rather than the ones in the diff, so anything that changes
import-time behavior or a plotting default can fail it without a docstring edit anywhere.
Moving `pv.BUILDING_GALLERY` out of `pyvista/ext/plot_directive.py`'s module scope did
exactly that -- collecting that module was what set the flag, and one example's
anti-aliasing warning is silenced only while a gallery is being built.

`tests/conftest.py` and the doctest tox environment already set off-screen rendering, so
the `make` targets are safe. Only a bare `pytest --doctest-modules` outside tox needs
`PYVISTA_OFF_SCREEN=true`; without it the examples open render windows and take over the
display.

**Never push a commit to find out whether something passes.** `AGENTS.md` opens with that
rule and `CONTRIBUTING.rst` states it as `Continuous Integration Etiquette`. It binds you
more tightly than a human contributor, because pushing is cheaper for you than it is for
the project. Keep the pull request in draft while you iterate.

## Approaching a red job

`Continuous Integration Etiquette` covers the basics: read the log and find the assertion,
reproduce locally, and diagnose a flaky test rather than re-running the job. Beyond that:

- Check whether the failure also occurs on `main` before attributing it to the branch. A
  second worktree at `origin/main` settles this in a minute.
- A version-specific failure needs the shared version constant, not a local workaround.
- An image regression failure is settled from the job's `failed_test_images-*` artifact,
  not by pushing a regenerated baseline to see whether it sticks. See **pyvista-testing**.
