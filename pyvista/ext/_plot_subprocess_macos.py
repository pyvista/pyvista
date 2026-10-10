"""Render ``pyvista-plot`` snippets in a separate process for forked Sphinx workers on macOS.

Sphinx runs parallel builds with forked worker processes. On macOS a forked process that
never calls ``exec`` cannot use Cocoa, Metal or libdispatch, and VTK needs all three to
render, so rendering inside such a worker aborts the build. Each worker therefore starts one
long-lived interpreter with :mod:`subprocess` and sends it every snippet it has to render as
a JSON line on stdin. The render process executes the snippet, writes the images, and replies
with the image names, the warnings to log and the sphinx-autocodelink records.

The render process is only used when :func:`renders_in_subprocess` is true. Every other
platform and every serial build renders in the Sphinx process itself.
"""

from __future__ import annotations

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

if TYPE_CHECKING:
    from typing import IO

    from docutils.parsers.rst.states import RSTState
    from sphinx.environment import BuildEnvironment


def renders_in_subprocess() -> bool:
    """Return whether this is a forked Sphinx worker on macOS, which cannot render itself."""
    return sys.platform == 'darwin' and multiprocessing.parent_process() is not None


def _class_name(cls: type) -> str:
    """Return ``module:qualname`` for ``cls``."""
    return f'{cls.__module__}:{cls.__qualname__}'


def _class_from_name(name: str) -> type:
    """Return the class ``module:qualname`` names."""
    module, _, qualname = name.partition(':')
    return getattr(importlib.import_module(module), qualname)


class RenderProcess:
    """A spawned interpreter that renders the snippets of one Sphinx worker."""

    def __init__(self) -> None:
        """Start the interpreter and send it this process's path, theme and warning filters."""
        self._proc = subprocess.Popen(
            [sys.executable, '-c', 'from pyvista.ext._plot_subprocess_macos import main; main()'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            env={**os.environ, 'PYTHONPATH': os.pathsep.join(p for p in sys.path if p)},
        )
        self._stdin = cast('IO[str]', self._proc.stdin)
        self._stdout = cast('IO[str]', self._proc.stdout)
        self._send(
            {
                'sys_path': sys.path,
                'warning_filters': [
                    [
                        action,
                        getattr(message, 'pattern', message),
                        [_class_name(c) for c in category]
                        if isinstance(category, tuple)
                        else _class_name(category),
                        getattr(module, 'pattern', module),
                        lineno,
                    ]
                    for action, message, category, module, lineno in warnings.filters
                ],
                'theme': pv.global_theme.name,
                'plot_directive_theme': pv.PLOT_DIRECTIVE_THEME,
                'building_gallery': pv.BUILDING_GALLERY,
                'figure_path': pv.FIGURE_PATH,
            }
        )
        self._recv()

    def _send(self, message: dict[str, Any]) -> None:
        """Write one JSON message to the render process."""
        self._stdin.write(json.dumps(message) + '\n')
        self._stdin.flush()

    def _recv(self) -> dict[str, Any]:
        """Read one JSON message from the render process."""
        line = self._stdout.readline()
        if not line:
            msg = f'The pyvista-plot render process exited with code {self._proc.wait()}'
            raise RuntimeError(msg)
        return json.loads(line)

    def run(
        self, job: dict[str, Any], *, want_records: bool
    ) -> tuple[list[tuple[str, list[str]]], list[str], list[dict[str, Any]] | None]:
        """Run ``job``, the keyword arguments of ``_execute_pieces``, in the render process."""
        self._send({'job': job, 'want_records': want_records})
        reply = self._recv()
        if 'error' in reply:
            raise RuntimeError(reply['error'])
        return reply['results'], reply['warnings'], reply['records']


_render_process: RenderProcess | None = None


def get_render_process() -> RenderProcess:
    """Return this process's render process, starting it on first use."""
    global _render_process  # noqa: PLW0603
    if _render_process is None:
        _render_process = RenderProcess()
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
    sys.path[:] = init['sys_path']
    warnings.resetwarnings()
    for action, message, category, module, lineno in init['warning_filters']:
        warnings.filterwarnings(
            action,
            message=message or '',
            category=tuple(_class_from_name(c) for c in category)  # type: ignore[arg-type]
            if isinstance(category, list)
            else _class_from_name(category),
            module=module or '',
            lineno=lineno,
            append=True,
        )
    pv.OFF_SCREEN = True
    pv.BUILDING_GALLERY = init['building_gallery']
    pv.FIGURE_PATH = init['figure_path']
    pv.PLOT_DIRECTIVE_THEME = init['plot_directive_theme']
    if init['theme'] is not None:
        pv.set_plot_theme(init['theme'])
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
