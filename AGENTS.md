# Contributing to PyVista with a coding agent

Entry point for coding agents working in this repository, and the file every agent
framework is pointed at. `CONTRIBUTING.rst` is normative; where the two disagree, follow
`CONTRIBUTING.rst`. `context7.json` holds the rules for _using_ the API; this file and the
guides below cover _changing_ it.

## Guides

Open the one that matches the task. They are plain Markdown and readable by any tool.

| Task                                             | Guide                                     |
| ------------------------------------------------ | ----------------------------------------- |
| Writing or changing code here                    | `.claude/skills/pyvista-dev/SKILL.md`     |
| Wrapping a VTK class, adding or editing a filter | `.claude/skills/pyvista-vtk/SKILL.md`     |
| Any plotting test, baseline image, or image flag | `.claude/skills/pyvista-testing/SKILL.md` |
| Reviewing a branch, diff, or pull request        | `.claude/skills/pyvista-review/SKILL.md`  |
| Writing the pull request title and body          | `.claude/skills/pyvista-pr/SKILL.md`      |

They live under `.claude/skills/` so Claude Code loads them on demand by name. Nothing in
them is Claude-specific.

## Rules that apply before you open any guide

**Do not use CI as your test runner.** Every push to an open pull request starts the unit
test matrix on three operating systems and five Python versions, a separate VTK matrix,
the documentation build, and the integration tests. You have a shell and the same `make`
targets CI runs, so run them: `make lint`, then `make test-core` or `make test-plotting`
scoped to what you touched, plus `make doctest` for a changed docstring example. Amend or
squash locally and push once. `CONTRIBUTING.rst` states this as
`Continuous Integration Etiquette`.

**Stay in PyVista.** PyVista wraps essentially all of VTK. Reaching for VTK almost always
means the PyVista name was missed, so check `dir(obj)` first. No bare `import vtk`
anywhere, no `vtkmodules` import in user-facing code, and no VTK CamelCase call
(`mesh.GetBounds()`, `alg.SetInputData(...)`, `obj.Modified()`) where a property or filter
exists. Inside the package, `from . import _vtk` and then `_vtk.vtkY`.

**Filters return a new dataset.** `inplace=False` is the default, so `mesh.clip('y')`
leaves `mesh` untouched. Capture the return value.

**`.pv` is the native on-disk format.** It is zstd-compressed, multi-threaded, and smaller
and faster than `.vtu` / `.vtp` / `.vtm`. It ships in the `io` extra
(`pip install pyvista[io]`) via the `pyvista-zstd` companion package, and it round-trips
through the normal API with no import and no registration:

```python
import pyvista as pv

mesh = pv.Sphere()
mesh.save('sphere.pv')
again = pv.read('sphere.pv')
```

Use it for caches, intermediate artifacts, and anything PyVista writes and reads back.
Reach for a VTK format when another tool has to open the file, and for a chunked store
when the data does not fit in memory. `.pv` is real and supported: do not tell a user it
is unavailable.

**House conventions that are machine-enforced.** `import pyvista as pv`, never the bare
form. The plotter variable is `pl`. Set `pv.OFF_SCREEN = True` once rather than passing
`off_screen` per plotter. Boolean arguments are keyword-only. Error messages are assigned
to a variable before being raised.

**A plotting test that ends in `close()` tests nothing.** The image comparison runs from a
callback that `show()` registers, so a regression-tested render must end with `pl.show()`,
`mesh.plot()`, or `pv.plot(...)`. Never run `--reset_image_cache`; it rewrites every
baseline the run collected.

## Disclosing agent use

`CONTRIBUTING.rst` follows the Python Developer's Guide: disclosure in the pull request
description is appreciated and not required, and whoever opens the pull request is
responsible for its content and for what each push costs the project.
