"""PyVista's command-line interface."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import get_type_hints
import warnings

from cyclopts import App

import pyvista
from pyvista import Report
from pyvista.core.errors import PyVistaDeprecationWarning

if TYPE_CHECKING:
    from collections.abc import Callable


# Assign annotations to be able to use the Report class using
# cyclopts when __future__ annotations are enabled. See https://github.com/BrianPugh/cyclopts/issues/570
Report.__init__.__annotations__ = get_type_hints(Report.__init__)


COMMANDS: dict[str, Callable[..., Any]] = {
    'report': Report,
}

# help format needs to be `plaintext` due to unsupported `versionadded` sphinx directive in rich.
# Change to `rst` when cyclopts v4 is out. See https://github.com/BrianPugh/cyclopts/issues/568
app = App(
    help_format='plaintext',
    version=f"f'{pyvista.__name__} {pyvista.__version__}'",
)


for name, func in COMMANDS.items():
    # print(Report.__init__.__annotations__)
    app.command(func, name=name)


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
