"""Example file directive module."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from typing import ClassVar

from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.parsers.rst import directives
from sphinx.util import logging
from sphinx.util.nodes import set_source_info

from pyvista.examples.downloads import download_file

if TYPE_CHECKING:
    from sphinx.application import Sphinx

logger = logging.getLogger(__name__)

_LANGUAGES = {'.json': 'json', '.md': 'markdown', '.py': 'python', '.toml': 'toml'}


class ExampleFileDirective(Directive):
    """Include a file from PyVista's example data source as a code block."""

    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = False
    option_spec: ClassVar[dict[str, object]] = {'language': directives.unchanged}

    def run(self):  # pragma: no cover
        """Download the file and return it as a literal block."""
        name = self.arguments[0]
        try:
            path = Path(download_file(name))
            text = path.read_text(encoding='utf-8')
        except Exception as e:  # noqa: BLE001
            logger.warning(f'Failed to include example file {name}: {e}')
            return []

        self.state.document.settings.env.note_dependency(str(path))

        node = nodes.literal_block(text, text)
        node['language'] = self.options.get('language', _LANGUAGES.get(path.suffix, 'none'))
        set_source_info(self, node)
        return [node]


def setup(app: Sphinx):
    """Register the directive."""
    app.add_directive('example-file', ExampleFileDirective)

    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
