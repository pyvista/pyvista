"""Embed Python file directive module."""

from __future__ import annotations

from pathlib import Path
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


class EmbedPyFileDirective(Directive):
    """Embed a Python file from PyVista's example data source as a code block."""

    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = False

    def run(self):
        """Download the file and return it as a Python code block."""
        name = self.arguments[0]
        try:
            path = Path(download_file(name))
            text = path.read_text(encoding='utf-8')
        except Exception as e:  # noqa: BLE001
            logger.warning(f'Failed to embed {name}: {e}')
            return []

        self.state.document.settings.env.note_dependency(str(path))

        node = nodes.literal_block(text, text)
        node['language'] = 'python'
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
