"""Embed Python file directive module."""

from __future__ import annotations

import json
import multiprocessing
from pathlib import Path
import subprocess
import sys
from typing import TYPE_CHECKING

from docutils import nodes
from docutils.parsers.rst import Directive
from sphinx.util import logging
from sphinx.util.nodes import set_source_info

from pyvista.examples.downloads import download_file

if TYPE_CHECKING:
    from typing import Any

    from sphinx.application import Sphinx

logger = logging.getLogger(__name__)


def _download(name: str) -> str | list[str]:
    """Download ``name``, in a subprocess when this is a forked Sphinx worker."""
    if multiprocessing.parent_process() is None:
        return download_file(name)
    code = (
        'import json, sys\n'
        'from pyvista.examples.downloads import download_file\n'
        'print(json.dumps(download_file(sys.argv[1])))'
    )
    result = subprocess.run(
        [sys.executable, '-c', code, name], check=True, capture_output=True, text=True
    )
    return json.loads(result.stdout.splitlines()[-1])


class EmbedPyFileDirective(Directive):
    """Embed a Python file from PyVista's example data source as a code block."""

    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = False

    def run(self) -> list[nodes.Node]:
        """Download the file and return it as a Python code block."""
        name = self.arguments[0]
        try:
            downloaded = _download(name)
            if not isinstance(downloaded, str):
                msg = f'{name} downloads to more than one file'
                raise TypeError(msg)
            path = Path(downloaded)
            text = path.read_text(encoding='utf-8')
        except Exception as e:  # noqa: BLE001
            logger.warning(f'Failed to embed {name}: {e}')
            return []

        self.state.document.settings.env.note_dependency(str(path))

        node = nodes.literal_block(text, text)
        node['language'] = 'python'
        node['classes'].append('no-search')
        set_source_info(self, node)
        return [node]


def setup(app: Sphinx) -> dict[str, Any]:  # numpydoc ignore=RT01
    """Register the ``embed-py-file`` directive."""
    app.add_directive('embed-py-file', EmbedPyFileDirective)
    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
