# Claude Code instructions

@AGENTS.md

Task-specific guides live in `skills/`, linked from `.claude/skills`. Load `pyvista-dev`
before writing code, `pyvista-vtk` before wrapping a VTK class or touching a filter,
`pyvista-testing` before touching a plotting test or a baseline image,
`pyvista-doc-images` after a docs image failure, `pyvista-docs-style` before a prose pass
or a Vale change, `pyvista-review` before reviewing, and `pyvista-pr` before writing the
pull request.
