"""Run a build from a JSON configuration: ``python -m pyvista._build '<json>'``."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from rich.console import Console
from rich.markup import escape

from .phases import BuildConfig
from .phases import Builder
from .phases import BuildError

_PATH_FIELDS = ('build_dir', 'source', 'cache_dir')


def main(argv: list[str]) -> int:
    """Build from the JSON configuration in ``argv[0]``.

    Parameters
    ----------
    argv : list[str]
        Command-line arguments after the module name.

    Returns
    -------
    int
        Exit status.

    """
    data = json.loads(argv[0])
    for key in _PATH_FIELDS:
        if data.get(key) is not None:
            data[key] = Path(data[key])
    try:
        Builder(BuildConfig(**data)).build()
    except BuildError as err:
        Console(stderr=True).print(f'[red]{escape(str(err))}[/red]')
        return 1
    return 0


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
