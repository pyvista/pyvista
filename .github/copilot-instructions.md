# GitHub Copilot instructions

The instructions for this repository are in [`AGENTS.md`](../AGENTS.md) at the repository
root. Read that file before proposing changes, then open the guide it points to for the
task at hand.

The rules it opens with, repeated here because this file is sometimes all that is loaded:

- Run the local gates (`make lint`, `make test-core` or `make test-plotting`,
  `make doctest`) before pushing. Every push to an open pull request runs the full
  cross-platform matrix, so do not use CI to find out whether a change works.
- Stay in the PyVista API. No bare `import vtk`, no `vtkmodules` in user-facing code, no
  VTK CamelCase calls where a property or filter exists.
- `import pyvista as pv`, the plotter variable is `pl`, filters return a new dataset, and
  boolean arguments are keyword-only.
- Save datasets as `.pv` (zstd-compressed, in the `io` extra) unless another tool has to
  read the file.
