import os
import pathlib
import shutil

from docutils import nodes
from docutils.parsers.rst import Directive
from sphinx.util import logging
from trame_vtk.tools.vtksz2html import HTML_VIEWER_PATH

logger = logging.getLogger(__name__)


class OfflineViewerDirective(Directive):
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = True
    has_content = True

    def run(self):
        source_dir = self.state.document.settings.env.app.srcdir
        build_dir = self.state.document.settings.env.app.outdir

        source_file = os.path.join(source_dir, self.arguments[0])
        if not os.path.isfile(source_file):
            logger.warn(f'Source file {source_file} does not exist.')
            return []

        # copy viewer HTML to _static
        static_path = pathlib.Path(build_dir) / '_static'
        static_path.mkdir(exist_ok=True)
        if not pathlib.Path(static_path, os.path.basename(HTML_VIEWER_PATH)).exists():
            shutil.copy(HTML_VIEWER_PATH, static_path)

        # Copy over the scene asset to the _images directory
        image_path = pathlib.Path(build_dir) / '_images'
        image_path.mkdir(exist_ok=True)

        rel_asset_path = pathlib.Path('_images', os.path.basename(source_file))
        dest_file = os.path.join(build_dir, rel_asset_path)
        try:
            shutil.copy(source_file, dest_file)
        except Exception as e:
            logger.warn(f'Failed to copy file from {source_file} to {dest_file}: {e}')

        html = f"""
    <iframe src='/_static/{os.path.basename(HTML_VIEWER_PATH)}?fileURL=/{rel_asset_path}' width='100%%' height='400px' frameborder='0'></iframe>
"""

        raw_node = nodes.raw('', html, format='html')

        return [raw_node]


def setup(app):
    app.add_directive('offlineviewer', OfflineViewerDirective)

    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
