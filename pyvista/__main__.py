"""PyVista's command-line interface."""

from __future__ import annotations

from functools import wraps
from typing import Annotated
from typing import Literal
from typing import get_type_hints
import warnings

from cyclopts import App
from cyclopts import Parameter

import pyvista
from pyvista import Report
from pyvista.core.errors import PyVistaDeprecationWarning

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


@app.command
def _plot(
    files: Annotated[list[str], Parameter(consume_multiple=True)],
    *,
    off_screen: bool | None = None,
    full_screen: bool | None = None,
    screenshot: str | None = None,
    interactive: bool = True,
    window_size: list[int] | None = None,
    show_bounds: bool = False,
    show_axes: bool | None = None,
    notebook: bool | None = None,
    background: str | None = None,
    text: str = '',
    eye_dome_lighting: bool = False,
    volume: bool = False,
    parallel_projection: bool = False,
    return_cpos: bool = False,
    anti_aliasing: Literal['ssaa', 'msaa', 'fxaa'] | bool | None = None,
    zoom: float | str | None = None,
    border: bool = False,
    border_color: str = 'k',
    border_width: float = 2.0,
    ssao: bool = False,
    **kwargs,
) -> None:
    pyvista.plot(
        var_item=files,
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
