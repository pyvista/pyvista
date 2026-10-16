"""Render ``pyvista-plot`` snippets in a separate process during parallel Sphinx builds.

Sphinx runs parallel builds with forked worker processes. On macOS a forked process that
never calls ``exec`` cannot use Cocoa, Metal or libdispatch, and VTK needs all three to
render, so rendering inside such a worker aborts the build. Each worker therefore starts one
long-lived interpreter with :mod:`subprocess` and sends it every snippet it has to render as
a JSON line on stdin. The render process executes the snippet, writes the images, and replies
with the image names, the warnings to log and the sphinx-autocodelink records. Every parallel
build uses it, on every platform; serial builds render in the Sphinx process itself.
"""

from __future__ import annotations

import atexit
import contextlib
import importlib
import json
import multiprocessing
import os
import subprocess
import sys
import traceback
from typing import TYPE_CHECKING
from typing import Any
from typing import cast
import warnings

import pyvista as pv
from pyvista import _vtk
from pyvista.core.utilities.observers import set_error_output_file
from pyvista.plotting.themes import Theme

if TYPE_CHECKING:
    from typing import IO

    from docutils.parsers.rst.states import RSTState
    from sphinx.environment import BuildEnvironment


def in_forked_worker() -> bool:
    """Return whether this is a forked Sphinx worker."""
    return multiprocessing.parent_process() is not None


def _class_name(cls: type) -> str:
    """Return ``module:qualname`` for ``cls``."""
    return f'{cls.__module__}:{cls.__qualname__}'


def _class_from_name(name: str) -> type:
    """Return the class ``module:qualname`` names."""
    module, _, qualname = name.partition(':')
    return getattr(importlib.import_module(module), qualname)


def _error_output_file() -> str | None:
    """Return the file VTK errors are written to, if one is set."""
    window = _vtk.vtkOutputWindow.GetInstance()
    if isinstance(window, _vtk.vtkFileOutputWindow):
        return window.GetFileName()
    return None


class RenderProcess:
    """A spawned interpreter that renders the snippets of one Sphinx worker."""

    def __init__(self) -> None:
        """Start the interpreter and send it this process's theme, paths and warning filters."""
        self._proc = subprocess.Popen(
            [sys.executable, '-c', 'from pyvista.ext._plot_subprocess import main; main()'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            env={**os.environ, 'PYTHONPATH': os.pathsep.join(p for p in sys.path if p)},
        )
        self._stdin = cast('IO[str]', self._proc.stdin)
        self._stdout = cast('IO[str]', self._proc.stdout)
        theme = pv.global_theme.to_dict()
        theme['trame'].pop('jupyter_extension_available')
        self._send(
            {
                'warning_filters': [
                    [
                        action,
                        getattr(message, 'pattern', message),
                        _class_name(cast('type', category)),
                        getattr(module, 'pattern', module),
                        lineno,
                    ]
                    for action, message, category, module, lineno in warnings.filters
                ],
                'theme': theme,
                'plot_directive_theme': pv.PLOT_DIRECTIVE_THEME,
                'building_gallery': pv.BUILDING_GALLERY,
                'figure_path': pv.FIGURE_PATH,
                'error_output_file': _error_output_file(),
            }
        )
        self._recv()

    def _send(self, message: dict[str, Any]) -> None:
        """Write one JSON message to the render process."""
        try:
            self._stdin.write(json.dumps(message) + '\n')
            self._stdin.flush()
        except OSError:
            self._raise_exited()

    def _recv(self) -> dict[str, Any]:
        """Read one JSON message from the render process."""
        line = self._stdout.readline()
        if not line:
            self._raise_exited()
        return json.loads(line)

    def _raise_exited(self) -> None:
        """Raise for a render process that exited; its traceback is on stderr."""
        msg = (
            f'The pyvista-plot render process exited with code {self._proc.wait()}; '
            'its error output precedes this message'
        )
        raise RuntimeError(msg)

    def close(self) -> None:
        """End the render process and release its pipes."""
        with contextlib.suppress(OSError):
            self._stdin.close()
        self._proc.wait()
        self._stdout.close()

    def run(
        self, job: dict[str, Any], *, want_records: bool
    ) -> tuple[list[tuple[str, list[str]]], list[str], list[dict[str, Any]] | None]:
        """Run ``job``, the keyword arguments of ``_execute_pieces``, in the render process."""
        self._send({'job': job, 'want_records': want_records})
        reply = self._recv()
        if 'error' in reply:
            raise RuntimeError(reply['error'])
        results = [(code, basenames) for code, basenames in reply['results']]
        return results, reply['warnings'], reply['records']


_render_process: RenderProcess | None = None


def get_render_process() -> RenderProcess:
    """Return this process's render process, starting one when none is running."""
    global _render_process  # noqa: PLW0603
    if _render_process is None or _render_process._proc.poll() is not None:
        _render_process = RenderProcess()
        atexit.register(_render_process.close)
    return _render_process


def store_records(
    env: BuildEnvironment, records: list[dict[str, Any]], state: RSTState | None
) -> None:
    """Store the records a render process computed, as ``record_namespace`` would have."""
    from sphinx_autocodelink import DEFAULT_DOCSTRING_EXAMPLE_CATEGORY  # noqa: PLC0415
    from sphinx_autocodelink import _from_jsonable  # noqa: PLC0415
    from sphinx_autocodelink import _store_records  # noqa: PLC0415
    from sphinx_autocodelink import is_inside_autodoc_desc  # noqa: PLC0415

    category = ''
    if state is not None and is_inside_autodoc_desc(state):
        category = DEFAULT_DOCSTRING_EXAMPLE_CATEGORY
    _store_records(env, env.docname, [_from_jsonable(e) for e in records], category=category)


def main() -> None:  # pragma: no cover
    """Serve render jobs over stdin/stdout, one JSON message per line."""
    from pyvista.ext.plot_directive import PlotError  # noqa: PLC0415
    from pyvista.ext.plot_directive import _execute_pieces  # noqa: PLC0415

    # keep the original stdout for the protocol and send fd 1 to stderr
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), 'w')
    sys.stdout.flush()
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())

    def reply(message: dict[str, Any]) -> None:
        """Write one JSON message to the worker."""
        protocol.write(json.dumps(message) + '\n')
        protocol.flush()

    init = json.loads(sys.stdin.readline())
    warnings.resetwarnings()
    for action, message, category, module, lineno in init['warning_filters']:
        warnings.filterwarnings(
            action,
            message=message or '',
            category=_class_from_name(category),
            module=module or '',
            lineno=lineno,
            append=True,
        )
    pv.OFF_SCREEN = True
    pv.BUILDING_GALLERY = init['building_gallery']
    pv.FIGURE_PATH = init['figure_path']
    pv.PLOT_DIRECTIVE_THEME = init['plot_directive_theme']
    pv.set_plot_theme(Theme.from_dict(init['theme']))
    if init['error_output_file'] is not None:
        set_error_output_file(init['error_output_file'])
    reply({'ready': True})
    for line in sys.stdin:
        message = json.loads(line)
        try:
            results, messages, ns, source = _execute_pieces(**message['job'])
            records = None
            if message['want_records'] and source:
                from sphinx_autocodelink import _records_for  # noqa: PLC0415
                from sphinx_autocodelink import _to_jsonable  # noqa: PLC0415

                records = [_to_jsonable(record) for record in _records_for(source, ns)]
            reply({'results': results, 'warnings': messages, 'records': records})
        except PlotError as error:
            reply({'error': str(error)})
        except Exception:  # noqa: BLE001
            reply({'error': traceback.format_exc()})
