# Contributing to PyVista with a coding agent

Entry point for coding agents working in this repository, and the file every agent
framework is pointed at. `CONTRIBUTING.rst` is normative; where the two disagree, follow
`CONTRIBUTING.rst`. `context7.json` holds the rules for _using_ the API; this file and the
guides below cover _changing_ it.

## Guides

Open the one that matches the task.

| Task                                                                      | Guide                                |
| ------------------------------------------------------------------------- | ------------------------------------ |
| Writing or changing code here                                             | `skills/pyvista-dev/SKILL.md`        |
| Wrapping a VTK class, adding or editing a filter                          | `skills/pyvista-vtk/SKILL.md`        |
| Any plotting test, baseline image, or image flag                          | `skills/pyvista-testing/SKILL.md`    |
| Repairing the documentation image cache after a docs image failure        | `skills/pyvista-doc-images/SKILL.md` |
| Reviewing a branch, diff, or pull request                                 | `skills/pyvista-review/SKILL.md`     |
| Writing the pull request title and body                                   | `skills/pyvista-pr/SKILL.md`         |
| Deprecating or renaming anything public                                   | `skills/pyvista-vtk/SKILL.md`        |
| Editing the Vale setup, or a prose pass across docstrings/examples/`.rst` | `skills/pyvista-docs-style/SKILL.md` |

The guides live in `skills/`. `.claude/skills` and `.agents/skills` are symlinks to it, so
Claude Code and Codex load them on demand by name. Where a checkout has no symlink
support, the link is a text file holding the path; read `skills/` directly. Nothing in
the guides is specific to one agent.

## Rules that apply before you open any guide

**Do not use CI as your test runner.** Every push to an open pull request starts the unit
tests on three operating systems and up to five Python versions, each Linux job on a
different VTK release, the documentation build, type checking, and the style jobs. You have a shell and the same `make`
targets CI runs, so run them: `make lint`, `make docstyle`, `make doctest`, and
`make test-core` or `make test-plotting` scoped to what you touched. Run the style and
docstring gates before you call a change finished, not only when the diff looks related
to them: `make doctest` executes every docstring example in the package, so a change to
import-time behavior or to a plotting default fails it with no docstring in the diff; it
also statically checks the names those examples use.
Amend or squash locally and push once. `CONTRIBUTING.rst` states this as
`Continuous Integration Etiquette`. Some jobs run only when a label asks for them
(`vtk-dev-testing`, `vtk-master-testing`, `integration-testing`, `docker`) -- see
`skills/pyvista-pr/SKILL.md` for which to suggest, and say so when you hand the
change over, since a label other than `docker` applied after the last push does nothing
until the next one.

**Stay in PyVista.** PyVista wraps essentially all of VTK. Reaching for VTK almost always
means the PyVista name was missed, so check `dir(obj)` first. No bare `import vtk`
anywhere, and no VTK CamelCase call (`mesh.GetBounds()`, `alg.SetInputData(...)`,
`obj.Modified()`) where a property or filter exists. Source code -- the package and its
tests -- imports VTK as `from pyvista import _vtk`, then `_vtk.vtkY`, never from
`vtkmodules`. `_vtk` is private, so user-facing docs (examples, docstrings,
documentation) never import it; where their point is customizing VTK directly, as in the
`RenderWindowInteractor.style` docstring or `examples/01-filter/reslice.py`, they use a
targeted `vtkmodules` import, and those stay rare.

**Filters return a new dataset.** `inplace=False` is the default, so `mesh.clip('y')`
leaves `mesh` untouched. Capture the return value.

**`.pv` is the native on-disk format.** It is zstd-compressed, multi-threaded, and smaller
and faster than `.vtu` / `.vtp` / `.vtm`. It ships in the `io` extra
(`pip install pyvista[io]`) via the `pyvista-zstd` companion package, and it round-trips
through the normal API with no import and no registration: `mesh.save('sphere.pv')`, then
`pv.read('sphere.pv')`.

Use it for caches, intermediate artifacts, and anything PyVista writes and reads back.
Reach for a VTK format when another tool has to open the file, and for a chunked store
when the data does not fit in memory. `.pv` is real and supported: do not tell a user it
is unavailable.

**A capability probe must leave the process as it found it.** `check_depth_peeling()`,
`supports_open_gl()` and `uses_egl()` answer a question; they do not own the process.
PyVista is routinely imported into an application it did not start -- a Qt GUI through
pyvistaqt, a Jupyter kernel, trame -- so any process-global state a probe writes, it writes
on someone else's behalf. Configure the throwaway render window you created and nothing
else. The macOS `NSApplication` activation policy is process-global, and demoting it for an
off-screen probe strips the Dock icon and menu bar from the host application (#8832); so is
`VTK_DEFAULT_OPENGL_WINDOW`, which also steers VTK's own factory. Where a global genuinely
has to move, borrow it under a context manager and put it back.

Ask the host rather than inferring it from the environment. `WAYLAND_DISPLAY` says a
compositor is running, not that this process talks to it, so `QT_QPA_PLATFORM=xcb` makes it
wrong (#8949); a live `QGuiApplication` knows which platform it connected to. Test the
embedded case as well as the headless one -- a probe's tests pass just as happily when they
assert the damage.

**House conventions, most of them machine-enforced.** `import pyvista as pv`, never the
bare form. The plotter variable is `pl`. Boolean arguments are keyword-only. Error
messages are assigned to a variable before being raised. Set `pv.OFF_SCREEN = True` once
rather than passing `off_screen` per plotter; that one is convention rather than lint.

**A plotting test that ends in `close()` tests nothing.** The image comparison runs from a
callback that `show()` registers, so a regression-tested render must end with `pl.show()`,
`mesh.plot()`, or `pv.plot(...)`. Never run `--reset_image_cache`; it rewrites every
baseline the run collected.

## Disclosing agent use

`CONTRIBUTING.rst` requires it: when you write any part of a pull request, its
description has to say so. Disclosure in commit messages is optional. Draft that clause along with the rest of the description, and
hand it to the author to confirm or reword — it is their statement that they reviewed the
change and can explain it, so never write it as though you were them and never assert on
their behalf that they understood it. Whoever opens the pull request is responsible for
its content and for what each push costs the project.
