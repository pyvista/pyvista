from __future__ import annotations

import itertools
import pathlib
import re
import sys
import warnings


def warn_external(message: str, category: type[Warning] | None = None) -> None:
    """`warnings.warn` wrapper that sets *stacklevel* to "outside PyVista".

    Taken and modified from Matplotlib
    https://github.com/matplotlib/matplotlib/blob/db83efff4d7d3849f8bffbd1f6cdfc43d74c9aea/lib/matplotlib/_api/__init__.py#L395

    """
    kwargs = {}
    if sys.version_info[:2] >= (3, 12):
        # Go to Python's `site-packages` or `pyvista` from an editable install.
        basedir = pathlib.Path(__file__).parents[1]
        kwargs['skip_file_prefixes'] = (str(basedir / 'pyvista'),)
    else:
        frame = sys._getframe()
        for stacklevel in itertools.count(1):
            if frame is None:
                # when called in embedded context may hit frame is None
                kwargs['stacklevel'] = stacklevel
                break
            if not re.match(
                r'\Apyvista(\Z|\.(?!tests\.))',
                # Work around sphinx-gallery not setting __name__.
                frame.f_globals.get('__name__', ''),
            ):
                kwargs['stacklevel'] = stacklevel
                break
            frame = frame.f_back
        # preemptively break reference cycle between locals and the frame
        del frame
    warnings.warn(message, category, **kwargs)
