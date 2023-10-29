"""Viewer directive module."""
import os
import pathlib
import shutil

from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.utils import relative_path  # pragma: no cover
from sphinx.util import logging
from trame_vtk.tools.vtksz2html import HTML_VIEWER_PATH

logger = logging.getLogger(__name__)


class OfflineViewerDirective(Directive):
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = True
    has_content = True

    def run(self):  # pragma: no cover
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

        dest_file = pathlib.Path(build_dir) / '_images' / os.path.basename(source_file)
        try:
            shutil.copy(source_file, dest_file)
        except Exception as e:
            logger.warn(f'Failed to copy file from {source_file} to {dest_file}: {e}')

        # Compute the relative path of the current source to the source directory,
        # which is the same as the relative path of the '_static' directory to the
        # generated HTML file.
        relpath_to_source_root = relative_path(self.state.document.current_source, source_dir)
        rel_viewer_path = (
            pathlib.Path(".")
            / relpath_to_source_root
            / '_static'
            / os.path.basename(HTML_VIEWER_PATH)
        ).as_posix()
        rel_asset_path = pathlib.Path(os.path.relpath(dest_file, static_path)).as_posix()
        html = f"""
    <iframe src='{rel_viewer_path}?fileURL={rel_asset_path}' width='100%%' height='400px' frameborder='0'></iframe>
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
