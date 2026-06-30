# Change Log

- 2026-06-30: Added missing `Plotter` constructor kwargs to the class docstring so the generated API docs now include `border_width`, `groups`, `row_weights`, `col_weights`, `splitting_position`, `title`, and `point_smoothing` (issue #8781).
- 2026-06-30: Corrected the new docstring entries — reordered the parameters to match the `__init__` signature (numpydoc `PR03`), fixed types (`groups` is a `list`, weights are `sequence[float]`), and rewrote the descriptions to be accurate (mirroring the `Renderers` docstring, with a `:ref:` to `multi_window_example`). Verified all params documented in order with valid types/formatting.
