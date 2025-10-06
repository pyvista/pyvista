"""PyVista's command-line interface."""

from __future__ import annotations

from ast import literal_eval
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Annotated
from typing import Literal
from typing import get_type_hints
import warnings

from cyclopts import App
from cyclopts import Parameter
from cyclopts import Token

import pyvista
from pyvista import Report
from pyvista.core.errors import PyVistaDeprecationWarning

if TYPE_CHECKING:
    from collections.abc import Sequence

# from pyvista.plotting._typing import ColorLike, Color

# Assign annotations to be able to use the Report class using
# cyclopts when __future__ annotations are enabled. See https://github.com/BrianPugh/cyclopts/issues/570
Report.__init__.__annotations__ = get_type_hints(Report.__init__)


# help format needs to be `plaintext` due to unsupported `versionadded` sphinx directive in rich.
# Change to `rst` when cyclopts v4 is out. See https://github.com/BrianPugh/cyclopts/issues/568
app = App(
    name=pyvista.__name__,
    help_format='plaintext',
    version=f'{pyvista.__name__} {pyvista.__version__}',
    help_on_error=True,
)


@app.command
@wraps(Report)
def report(*args, **kwargs):  # noqa: ANN201, D103
    return Report(*args, **kwargs)


def _validator_window_size(type_: type, value: list[int] | None) -> None:  # noqa: ARG001
    if value is not None and len(value) != 2:
        msg = 'Window size must be a list of two integers.'
        raise ValueError(msg)


def _validator_files(type_: type, value: list[str] | None) -> None:  # noqa: ARG001
    if value is None:
        return

    # Test file exists
    if not all((files := {v: Path(v).exists() for v in value}).values()):
        missing = [k for k, v in files.items() if not v]
        msg = f'File(s) not found: {missing}'
        raise ValueError(msg)

    # Test file can be read by pyvista
    def readable(file: str) -> bool:
        try:
            pyvista.read(file)
        except Exception:  # noqa: BLE001
            return False
        else:
            return True

    if not all((files := {v: readable(v) for v in value}).values()):
        not_readable = [k for k, v in files.items() if not v]
        msg = f'File(s) not readable by pyvista: {not_readable}'
        raise ValueError(msg)


def _kwargs_converter(type_, tokens: Sequence[Token]):  # noqa: ANN001, ANN202, ARG001
    for token in tokens:
        # Check hyphen in keyword value
        if (key := token.keys[0]) is not None and '-' in key:
            msg = f'cannot use hyphen `-`, try with --{key.replace("-", "_")}={token.value}'
            raise ValueError(msg)

        # Coerce using literal_eval with fallback to str value
        try:
            return literal_eval(token.value)
        except (ValueError, SyntaxError):
            return token.value
    return None


_HELP_KWARGS = """\
Additional keyword arguments passed to ``Plotter.add_mesh``.
See documentation for more details https://docs.pyvista.org/api/plotting/_autosummary/pyvista.plotter.add_mesh#pyvista.Plotter.add_mesh

Note that contrary to other arguments, hyphens CANNOT not be used (ie. use ``--show_edges=True`` instead of ``--show-edges=True``).

"""  # noqa: E501


@app.command
def _plot(
    files: Annotated[
        list[str] | None,
        Parameter(
            consume_multiple=True,
            help='File(s) to plot. Must be readable with ``pv.read``. If nothing is provided, show an empty window.',  # noqa: E501
            validator=_validator_files,
        ),
    ] = None,
    *,
    off_screen: bool | None = None,
    full_screen: bool | None = None,
    screenshot: str | None = None,
    interactive: bool = True,
    window_size: Annotated[
        list[int] | None, Parameter(consume_multiple=True, validator=_validator_window_size)
    ] = None,
    show_bounds: bool = False,
    show_axes: bool | None = None,
    notebook: bool | None = None,
    background: str | None = None,
    text: str = '',
    eye_dome_lighting: bool = False,
    volume: bool = False,
    parallel_projection: bool = False,
    return_cpos: bool = False,
    anti_aliasing: Literal['ssaa', 'msaa', 'fxaa'] | None = None,
    zoom: float | str | None = None,
    border: bool = False,
    border_color: str = 'k',
    border_width: float = 2.0,
    ssao: bool = False,
    **kwargs: Annotated[
        dict,
        Parameter(help=_HELP_KWARGS, converter=_kwargs_converter),
    ],
) -> None:
    pyvista.plot(
        var_item=files or [],
        off_screen=off_screen,
        full_screen=full_screen,
        screenshot=screenshot,
        interactive=interactive,
        window_size=window_size,
        show_bounds=show_bounds,
        show_axes=show_axes,
        notebook=notebook,
        background=background,
        text=text,
        eye_dome_lighting=eye_dome_lighting,
        volume=volume,
        parallel_projection=parallel_projection,
        return_cpos=return_cpos,
        anti_aliasing=anti_aliasing,
        zoom=zoom,
        border=border,
        border_color=border_color,
        border_width=border_width,
        ssao=ssao,
        **kwargs,
    )


_plot.__doc__ = pyvista.plot.__doc__


def main(argv: list[str] | str | None = None) -> None:
    """PyVista Command-Line Interface entry point."""
    # Ignore warnings emitted because arguments are passed positionally by the
    # inspect module. See https://docs.python.org/3/library/inspect.html#inspect.BoundArguments.kwargs
    # and https://github.com/BrianPugh/cyclopts/issues/567
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=PyVistaDeprecationWarning)
        result = app(tokens=argv)

    if result is not None:
        print(result)  # noqa: T201


if __name__ == '__main__':
    main()
