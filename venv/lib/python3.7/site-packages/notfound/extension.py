import docutils
import os
import sphinx

from sphinx.errors import ExtensionError

from . import version
from .utils import replace_uris


class BaseURIError(ExtensionError):
    """Exception for malformed base URI."""
    pass


# https://www.sphinx-doc.org/en/stable/extdev/appapi.html#event-html-collect-pages
def html_collect_pages(app):
    """
    Create a ``404.html`` page.

    Uses ``notfound_template`` as a template to be rendered with
    ``notfound_context`` for its context. The resulting file generated is
    ``notfound_pagename``.html.

    If the user already defined a page with pagename title
    ``notfound_pagename``, we don't generate this page.

    :param app: Sphinx Application
    :type app: sphinx.application.Sphinx
    """
    if app.config.notfound_pagename in app.env.titles:
        # There is already a ``404.rst`` file rendered.
        # Skip generating our default one.
        return []

    return [(
        app.config.notfound_pagename,
        app.config.notfound_context,
        app.config.notfound_template,
    )]


# https://www.sphinx-doc.org/en/stable/extdev/appapi.html#event-html-page-context
def finalize_media(app, pagename, templatename, context, doctree):
    """
    Point media files at our media server.

    Generate absolute URLs for resources (js, images, css, etc) to point to the
    right. For example, if a URL in the page is ``_static/js/custom.js`` it will
    be replaced by ``/<notfound_default_language>/<notfound_default_version>/_static/js/custom.js``.

    On the other hand, if ``notfound_no_urls_prefix`` is set, it will be
    replaced by ``/_static/js/custom.js``.

    Also, all the links from the sidebar (toctree) are replaced with their
    absolute version. For example, ``../section/pagename.html`` will be replaced
    by ``/section/pagename.html``.

    :param app: Sphinx Application
    :type app: sphinx.application.Sphinx

    :param pagename: name of the page being rendered
    :type pagename: str

    :param templatename: template used to render the page
    :type templatename: str

    :param context: context used to render the page
    :type context: dict

    :param doctree: doctree of the page being rendered
    :type doctree: docutils.nodes.document
    """

    # https://github.com/sphinx-doc/sphinx/blob/7138d03ba033e384f1e7740f639849ba5f2cc71d/sphinx/builders/html.py#L1054-L1065
    def pathto(otheruri, resource=False, baseuri=None):
        """
        Hack pathto to display absolute URL's.

        Instead of calling ``relative_url`` function, we call
        ``app.builder.get_target_uri`` to get the absolut URL.

        .. note::

            If ``otheruri`` is a external ``resource`` it does not modify it.
        """
        if resource and '://' in otheruri:
            # allow non-local resources given by scheme
            return otheruri

        if not resource:
            otheruri = app.builder.get_target_uri(otheruri)

        if baseuri is None:
            if app.config.notfound_no_urls_prefix:
                baseuri = '/'
            else:
                baseuri = '/{language}/{version}/'.format(
                    language=app.config.notfound_default_language,
                    version=app.config.notfound_default_version,
                )

        if not baseuri.startswith('/'):
            raise BaseURIError('"baseuri" must be absolute')

        if otheruri and not otheruri.startswith('/'):
            otheruri = '/' + otheruri

        if otheruri:
            if baseuri.endswith('/'):
                baseuri = baseuri[:-1]
            otheruri = baseuri + otheruri

        uri = otheruri or '#'
        return uri

    # https://github.com/sphinx-doc/sphinx/blob/2adeb68af1763be46359d5e808dae59d708661b1/sphinx/builders/html.py#L1081
    def toctree(*args, **kwargs):
        try:
            # Sphinx >= 1.6
            from sphinx.environment.adapters.toctree import TocTree
            get_toctree_for = TocTree(app.env).get_toctree_for
        except ImportError:
            # Sphinx < 1.6
            get_toctree_for = app.env.get_toctree_for

        toc = get_toctree_for(
            app.config.notfound_pagename,
            app.builder,
            collapse=kwargs.pop('collapse', False),
            includehidden=kwargs.pop('includehidden', False),
            **kwargs  # not using trailing comma here makes this compatible with
                      # Python2 syntax
        )

        # If no TOC is found, just return ``None`` instead of failing here
        if not toc:
            return None

        replace_uris(app, toc, docutils.nodes.reference, 'refuri')
        return app.builder.render_partial(toc)['fragment']

    # Apply our custom manipulation to 404.html page only
    if pagename == app.config.notfound_pagename:
        # Override the ``pathto`` helper function from the context to use a custom ones
        # https://www.sphinx-doc.org/en/master/templating.html#pathto
        context['pathto'] = pathto

        # Override the ``toctree`` helper function from context to use a custom
        # one and generate valid links on not found page.
        # https://www.sphinx-doc.org/en/master/templating.html#toctree
        # NOTE: not used on ``singlehtml`` builder for RTD Sphinx theme
        context['toctree'] = toctree


# https://www.sphinx-doc.org/en/stable/extdev/appapi.html#event-doctree-resolved
def doctree_resolved(app, doctree, docname):
    """
    Generate and override URLs for ``.. image::`` Sphinx directive.

    When ``.. image::`` is used in the ``404.rst`` file, this function will
    override the URLs to point to the right place.

    :param app: Sphinx Application
    :type app: sphinx.application.Sphinx
    :param doctree: doctree representing the document
    :type doctree: docutils.nodes.document
    :param docname: name of the document
    :type docname: str
    """

    if docname == app.config.notfound_pagename:
        # Replace image ``uri`` to its absolute version
        replace_uris(app, doctree, docutils.nodes.image, 'uri')


def setup(app):
    default_context = {
        'title': 'Page not found',
        'body': '<h1>Page not found</h1>\n\nThanks for trying.',
    }

    # https://github.com/sphinx-doc/sphinx/blob/master/sphinx/themes/basic/page.html
    app.add_config_value('notfound_template', 'page.html', 'html')
    app.add_config_value('notfound_context', default_context, 'html')
    app.add_config_value('notfound_pagename', '404', 'html')

    # TODO: get these values from Project's settings
    default_version = os.environ.get('READTHEDOCS_VERSION', 'latest')

    app.add_config_value('notfound_default_language', 'en', 'html')
    app.add_config_value('notfound_default_version', default_version, 'html')
    app.add_config_value('notfound_no_urls_prefix', False, 'html')

    app.connect('html-collect-pages', html_collect_pages)
    app.connect('html-page-context', finalize_media)
    app.connect('doctree-resolved', doctree_resolved)

    # Sphinx injects some javascript files using ``add_js_file``. The path for
    # this file is rendered in the template using ``js_tag`` instead of
    # ``pathto``. The ``js_tag`` uses ``pathto`` internally to resolve these
    # paths, we call again the setup function for this tag *after* the context
    # was overriden by our extension with the patched ``pathto`` function.
    if sphinx.version_info >= (1, 8):
        from sphinx.builders.html import setup_js_tag_helper
        app.connect('html-page-context', setup_js_tag_helper)

    return {
        'version': version,
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
