"""Open Graph images chosen by position, independent of what rendered them.

This is deliberately not specific to :mod:`pyvista.ext.plot_directive`: it numbers
every image on a page -- in document order, regardless of whether a
``pyvista-plot`` directive, a plain ``.. image::``, or anything else produced it --
and lets a page point at the one it wants as its ``og:image``. The default is the
first image, which is what most pages want without any selection at all.

Sphinx-Gallery pages are handled separately, since they already have a thumbnail:
their ``og:image`` always matches the gallery's own selection, using the full
resolution version of it rather than the (too small to preview well) thumbnail file
Sphinx-Gallery renders.

The result is written to the page's ``og:image`` field before ``sphinxext-opengraph``
renders its tags, so its own default (``ogp_image``) is only ever used as a fallback
for pages with no image of their own.

"""

from __future__ import annotations

from pathlib import Path
import posixpath
from typing import TYPE_CHECKING
from typing import Any
import urllib.parse

from docutils import nodes
from docutils.parsers.rst import Directive
from sphinx.util import logging

from pyvista.ext import _opengraph

if TYPE_CHECKING:
    from collections.abc import Iterator

    from sphinx.application import Sphinx
    from sphinx.config import Config

logger = logging.getLogger(__name__)

#: Document attribute holding the ``pyvista-opengraph-thumbnail`` argument for the page
_THUMBNAIL_NUMBER = '_pyvista_opengraph_thumbnail_number'


class OpenGraphThumbnailDirective(Directive):
    """The ``.. pyvista-opengraph-thumbnail::`` directive.

    Selects which of the page's images is used as its Open Graph image. See
    :mod:`pyvista.ext._opengraph_image`.
    """

    has_content = False
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = False

    def run(self):
        """Record the page's thumbnail number and render nothing.

        Returns
        -------
        list[docutils.nodes.Node]
            Always empty; this directive has no visible output.

        """
        document = self.state_machine.document
        env = document.settings.env
        argument = self.arguments[0].strip()
        try:
            number = int(argument)
        except ValueError:
            msg = f"'pyvista-opengraph-thumbnail' expects an integer, got {argument!r}."
            raise self.error(msg)
        if number == 0:
            msg = (
                "'pyvista-opengraph-thumbnail' is one-based, so 0 is not a valid image "
                'number. Use 1 for the first image, or -1 for the last one.'
            )
            raise self.error(msg)
        if _is_sphinx_gallery_document(env.app, env.docname):
            raise self.error(_gallery_thumbnail_error(number))
        if _THUMBNAIL_NUMBER in document.attributes:
            # A warning rather than an error: Open Graph metadata is per-document, so a
            # page documenting several objects collides even when each of their own
            # sources is correct on its own generated page.
            logger.warning(
                'this page already selects image %d as its Open Graph image, and a page '
                'can only have one. Ignoring this selection of image %d.',
                document.attributes[_THUMBNAIL_NUMBER],
                number,
                location=(env.docname, self.lineno),
                type='pyvista',
                subtype='opengraph_thumbnail',
            )
            return []
        document.attributes[_THUMBNAIL_NUMBER] = number
        return []


def _gallery_thumbnail_error(number: int) -> str:
    """Return guidance for choosing a Sphinx-Gallery example's thumbnail."""
    return (
        "'pyvista-opengraph-thumbnail' cannot be used in a Sphinx-Gallery example, "
        'because its Open Graph image always follows the gallery thumbnail. '
        f"Use '# sphinx_gallery_thumbnail_number = {number}' instead."
    )


def setup(app: Sphinx) -> None:
    """Wire up Open Graph images.

    Called by :mod:`pyvista.ext.plot_directive`; this module is not a Sphinx
    extension of its own, though nothing about it is specific to the plot
    directive either.
    """
    app.add_directive('pyvista-opengraph-thumbnail', OpenGraphThumbnailDirective)
    # Must run before ``sphinxext.opengraph`` renders its tags at the default priority
    app.connect('html-page-context', _set_image, priority=400)


def _set_image(  # noqa: PLR0917
    app: Sphinx,
    pagename: str,
    templatename: str,  # noqa: ARG001
    context: dict[str, Any],
    doctree: nodes.document | None,
) -> None:
    """Point the page's ``og:image`` at the image it selects.

    This runs at write time rather than while reading, because that is the first
    point at which an image's final ``_images`` filename is known: hash-based
    naming, parallel reads and Sphinx's own de-duplication all mean a page cannot
    predict where its own output ends up while it is still being parsed.
    """
    if doctree is None or not _opengraph.is_enabled(app):
        return
    fields = _opengraph.page_fields(app, context)
    if fields is None or 'og:image' in fields:
        return

    if _is_sphinx_gallery_document(app, pagename):
        image = _gallery_image(app, pagename, doctree)
    else:
        image = _numbered_image(app, pagename, doctree)
    if image is not None:
        fields['og:image'] = image


def _numbered_image(app: Sphinx, docname: str, doctree: nodes.document) -> str | None:
    """Return the URL of the image a page selects by position."""
    images = list(_image_nodes(doctree))
    if not images:
        return None

    number = doctree.get(_THUMBNAIL_NUMBER, 1)
    index = number - 1 if number > 0 else number
    if not -len(images) <= index < len(images):
        # Not fatal: a build that deliberately skips rendering (e.g. PyVista's own
        # ``pyvista_plot_skip``) legitimately renders fewer images than the page
        # selects from, so fall back to the first image either way
        if not _skips_rendering(app.config):
            logger.warning(
                "'pyvista-opengraph-thumbnail' selects image %d, but this page only "
                'has %d image(s). Using the first one.',
                number,
                len(images),
                location=docname,
                type='pyvista',
                subtype='opengraph_thumbnail',
            )
        index = 0
    # Sphinx has already rewritten the URI to the image's path relative to this page
    return _absolute_url(app, docname, images[index]['uri'])


def _skips_rendering(config: Config) -> bool:
    """Return whether a build is known to deliberately render fewer images.

    Soft dependency on :mod:`pyvista.ext.plot_directive`'s own configuration: this
    module works without it, it just cannot explain a mismatch as well.
    """
    return bool(
        getattr(config, 'pyvista_plot_skip', False)
        or getattr(config, 'pyvista_plot_skip_optional', False)
    )


def _gallery_image(app: Sphinx, docname: str, doctree: nodes.document) -> str | None:
    """Return the URL of the image Sphinx-Gallery uses as an example's thumbnail.

    The full resolution image is preferred over the gallery's own thumbnail file,
    which is too small to make a good link preview, but it is always the same image
    the gallery shows.
    """
    source = Path(app.env.doc2path(docname))
    number, path = _gallery_thumbnail_selection(source.with_suffix('.py'))
    if path is None:
        prefix = f'sphx_glr_{source.stem}_'
        images = [
            image
            for image in _image_nodes(doctree)
            if posixpath.basename(image['uri']).startswith(prefix)
        ]
        index = number - 1 if number > 0 else number
        if -len(images) <= index < len(images):
            # Sphinx-Gallery copies its images into the output verbatim
            return _absolute_url(app, docname, _output_image_path(app, images[index]['uri']))

    # ``sphinx_gallery_thumbnail_path`` and failed examples both leave a thumbnail with
    # no full resolution counterpart on the page
    thumbnails = (source.parent / 'images' / 'thumb').glob(f'sphx_glr_{source.stem}_thumb.*')
    thumbnail = next(thumbnails, None)
    if thumbnail is None:
        return None
    return _absolute_url(app, docname, _output_image_path(app, thumbnail.name))


def _gallery_thumbnail_selection(source: Path) -> tuple[int, str | None]:
    """Return the ``sphinx_gallery_thumbnail_{number,path}`` chosen by an example."""
    try:
        from sphinx_gallery.py_source_parser import extract_file_config  # noqa: PLC0415
    except ImportError:  # pragma: no cover
        return 1, None
    try:
        file_conf = extract_file_config(source.read_text(encoding='utf-8'))
    except OSError:
        # Gallery index pages and ``sg_execution_times`` have no example source
        return 1, None
    # A number always wins over a path, matching ``sphinx_gallery.gen_rst.save_thumbnail``
    number = file_conf.get('thumbnail_number')
    if number is None:
        path = file_conf.get('thumbnail_path')
        return 1, None if path is None else str(path)
    return int(number), None


def _image_nodes(doctree: nodes.document) -> Iterator[nodes.Element]:
    """Yield every image-bearing node of a page, in document order.

    Sphinx-Gallery renders its images as ``imgsgnode`` rather than
    :class:`docutils.nodes.image`, so nodes are matched on carrying a ``uri``.
    """
    for node in doctree.findall(nodes.Element):
        if node.get('uri'):
            yield node


def _output_image_path(app: Sphinx, name: str) -> str:
    """Return the path of an output image, relative to the page being written."""
    return posixpath.join(app.builder.imgpath, posixpath.basename(name))


def _absolute_url(app: Sphinx, docname: str, path: str) -> str:
    """Return the public URL of *path*, which is relative to *docname*."""
    site_url = app.config.ogp_canonical_url or app.config.ogp_site_url
    page_url = urllib.parse.urljoin(site_url, app.builder.get_target_uri(docname))
    return urllib.parse.urljoin(page_url, path)


def _is_sphinx_gallery_document(app: Sphinx, docname: str) -> bool:
    """Return whether *docname* is a generated Sphinx-Gallery document."""
    gallery_conf = getattr(app.config, 'sphinx_gallery_conf', None)
    if not gallery_conf:
        return False

    gallery_dirs = gallery_conf.get('gallery_dirs', ())
    if isinstance(gallery_dirs, str):
        gallery_dirs = (gallery_dirs,)
    directories = [Path(directory).as_posix().strip('/') for directory in gallery_dirs]
    return any(
        docname == directory or docname.startswith(f'{directory}/') for directory in directories
    )
