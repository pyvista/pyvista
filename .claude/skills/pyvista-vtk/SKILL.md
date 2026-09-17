---
name: pyvista-vtk
description: Wrap VTK the way this project wraps it. Load before adding or editing a filter, wrapping a VTK class, or writing anything that touches the VTK API.
---

# Wrapping VTK

PyVista exists so that users never write VTK. Every VTK call that escapes into user-facing
code bypasses the validation, breaks the snake_case surface, and couples us to an API that
moves between VTK releases. `context7.json` states the rules for consuming the API; this
skill is how the wrapper itself is written.

## Where VTK is allowed

Four places, and nowhere else:

1. Inside a filter body, driven through `_update_alg` and `_get_output`.
2. Inside `pyvista/_vtk.py` and the `_vtk` re-exports, which exist to lazily pull names out
   of `vtkmodules`.
3. Inside low-level helpers such as `pv.wrap` and the validation utilities.
4. Inside the wrapper machinery itself, where a PyVista class subclasses a VTK class.

Anywhere else, the call is a defect rather than a shortcut. Check `dir(obj)` before
concluding a wrapper is missing; if it genuinely is, add the property or filter first and
let the consumer land after it.

`import vtk` and `import vtkmodules` are both banned by ruff (`banned-api` in
`pyproject.toml`), which points you at `pyvista._vtk` instead. `examples/*` is exempt from
that rule (`TID251` in its per-file ignores) and is covered by the custom pre-commit hooks
instead. Inside the package the
sanctioned form is `from . import _vtk`, then `_vtk.vtkThreshold()`.

The wrapper classes also gate the VTK surface at runtime: `_NoNewAttrMixin`
(`pyvista/core/utilities/misc.py`) refuses unknown attributes, and `DisableVtkSnakeCase`
and `VTKObjectWrapperCheckSnakeCase` (`pyvista/core/_vtk_utilities.py`) keep VTK's
generated snake_case API from resolving quietly. A leak surfaces fast rather than silently
working.

| VTK call                                | Use instead                           |
| --------------------------------------- | ------------------------------------- |
| `mesh.GetBounds()` / `GetCenter()`      | `mesh.bounds` / `mesh.center`         |
| `mesh.GetNumberOfPoints()`              | `mesh.n_points`                       |
| `mesh.GetPoints()` / `SetPoints()`      | `mesh.points`, a live NumPy view      |
| `mesh.GetPointData()` / `GetCellData()` | `mesh.point_data` / `mesh.cell_data`  |
| `mesh.GetCell(i)`                       | `mesh.get_cell(i)`                    |
| `actor.GetMapper()` / `GetProperty()`   | `actor.mapper` / `actor.prop`         |
| `alg.Update()`                          | `_update_alg(alg, progress_bar=...)`  |
| `alg.GetOutput()` plus a manual wrap    | `_get_output(alg)`                    |
| `obj.Modified()`                        | nothing. PyVista handles invalidation |

To sweep a file, `rg '\.(Get|Set)[A-Z]\w*\('` finds the static leaks. It cannot see an
access built with `getattr`, so read the diff as well.

## The filter pattern

Every filter in `pyvista/core/filters/` has the same shape. Copy a neighbour rather than
inventing a variant.

```python
def threshold(
    self,
    value: float | VectorLike[float] | None = None,
    scalars: str | None = None,
    *,
    invert: bool = False,
    inplace: bool = False,
    progress_bar: bool = False,
) -> DataSet:
    """Apply a threshold filter.

    Parameters
    ----------
    value : float | VectorLike[float], optional
        Threshold value or ``(min, max)`` range. ``None`` uses the data range.

    invert : bool, default: False
        Invert the threshold.

    inplace : bool, default: False
        Update this dataset in place. When ``False``, return a new dataset.

    progress_bar : bool, default: False
        Display a progress bar.

    Returns
    -------
    pyvista.DataSet
        Thresholded dataset.

    Examples
    --------
    >>> import pyvista as pv
    >>> mesh = pv.Wavelet()
    >>> result = mesh.threshold(value=100)
    >>> result.n_cells < mesh.n_cells
    True

    """
    if scalars is None:
        scalars = self.active_scalars_name
    _validation.check_string(scalars, name='scalars')

    alg = _vtk.vtkThreshold()
    alg.SetInputDataObject(self)
    alg.SetInvert(invert)

    _update_alg(alg, progress_bar=progress_bar, message='Thresholding')
    output = _get_output(alg)

    if not inplace:
        return output
    self.copy_from(output, deep=False)
    return self
```

Points that reviewers raise when they are missing:

- **`_update_alg` and `_get_output`** (`pyvista/core/filters/__init__.py`) are the only
  pipeline driver. They handle the progress bar, VTK errors, and wrapping the output into
  the right PyVista subclass with its metadata intact.
- **Validate at the boundary.** The `pyvista-validation` package, imported as `_validation`, has the checks already:
  `check_string`, `check_contains`, `check_range`, `check_subdtype`, `validate_array`,
  `validate_array3`, `validate_arrayNx3`, `validate_axes`, `validate_transform4x4`,
  `validate_number`, and more. Internal helpers can trust their inputs; public entry points
  cannot.
- **`inplace=False` is the default**, and the in-place branch is
  `self.copy_from(output, deep=False)`.
- **Booleans are keyword-only** and positional arguments are limited to one or two. Write
  the final signature on a new API. `@_deprecate_positional_args`
  (`pyvista/_deprecate_positional_args.py`) is for tightening a signature that already
  shipped with positional callers, never for greenfield code.
- **Return a PyVista type.** `pv.wrap` promotes any VTK dataset, NumPy point array,
  `trimesh`, or `meshio` object zero-copy. A public API never returns a raw VTK object.

## Wrapping a new VTK class

Dataset classes combine a PyVista base, a filter mixin, and the VTK class:

```python
class PolyData(_PointSet, PolyDataFilters, _vtk.vtkPolyData):
    """Wrap :vtk:`vtkPolyData`."""
```

Module docstring is one line and uses the `:vtk:` role. `@abstract_class` marks a base
that must not be instantiated. Mixins that use `_NoNewAttrMixin` declare `__slots__ = ()`.
Map each VTK getter and setter you need onto a snake_case property with validation in the
setter, document it with numpydoc plus an `Examples` block, register the type with
`pv.wrap`, and add tests. Do not expose a CamelCase method as public API.

New top-level subpackages with heavy imports go in the lazy `__getattr__` list in
`pyvista/__init__.py` rather than being imported eagerly.

## Version gating and deprecation

Gate on `pv.vtk_version_info`, never on a parsed version string, so the branch shows up
when a VTK version is dropped:

```python
if pv.vtk_version_info >= (9, 5):
    ...
```

A shared constant already exists for most capability checks; reuse it rather than
re-deriving one locally (see **pyvista-dev**).

Deprecation applies to a shipped public contract only. `CONTRIBUTING.rst` sets the
lifecycle: warn with `PyVistaDeprecationWarning` through `warn_external`, then raise
`DeprecationError`, then remove, seeking at least three minor versions of overlap and
recording the trail in a comment (`# deprecated 0.47.0, convert to error in 0.50.0, remove
0.51.0`). Add the `.. deprecated::` directive to the docstring and a test that asserts the
warning. Because `filterwarnings` starts with `error`, a new deprecation has to migrate
every internal call site in the same change.
