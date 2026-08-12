"""Supporting functions for documentation build."""

from __future__ import annotations

import inspect
import os
import os.path as op
import sys


def linkcode_resolve(domain: str, info: dict[str, str], edit: bool = False) -> str | None:  # noqa: FBT001, FBT002
    """Determine the URL corresponding to a Python object.

    Parameters
    ----------
    domain : str
        Only useful when 'py'.

    info : dict
        With keys "module" and "fullname".

    edit : bool, default=False
        Link to the GitHub edit page instead of the blob view. The blob view
        gets the full line range highlighted; the edit page gets a short,
        two-line range starting at the same line -- a single-line anchor
        doesn't reliably scroll the edit view there on first load.

    Returns
    -------
    str
        The code URL. Empty string if there is no valid link.

    Notes
    -----
    This function is used by the `sphinx.ext.linkcode` extension to create the "[Source]"
    button whose link is edited in this function.

    This has been adapted to deal with our "verbose" decorator.

    Adapted from mne (mne/utils/docs.py), which was adapted from SciPy (doc/source/conf.py).

    """
    import pyvista as pv  # noqa: PLC0415

    if domain != 'py':
        return None

    modname = info['module']
    fullname = info['fullname']

    # Little clean up to avoid pyvista.pyvista
    if fullname.startswith(modname):
        fullname = fullname[len(modname) + 1 :]

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split('.'):
        try:
            obj = getattr(obj, part)
        except Exception:  # noqa: BLE001
            return None

    # deal with our decorators properly
    while hasattr(obj, 'fget'):
        obj = obj.fget

    # deal with wrapped object
    while hasattr(obj, '__wrapped__'):
        obj = obj.__wrapped__
    try:
        fn = inspect.getsourcefile(obj)
    except Exception:  # noqa: BLE001  # pragma: no cover
        fn = None

    if not fn:  # pragma: no cover
        try:
            fn = inspect.getsourcefile(sys.modules[obj.__module__])
        except Exception:  # noqa: BLE001
            return None
        return None

    fn = op.relpath(fn, start=op.dirname(pv.__file__))  # noqa: PTH120
    fn = '/'.join(op.normpath(fn).split(os.sep))  # in case on Windows # noqa: PTH206

    try:
        source, lineno = inspect.getsourcelines(obj)
    except Exception:  # noqa: BLE001 # pragma: no cover
        lineno = None

    if not lineno:
        linespec = ''
    elif edit:
        # A single-line #Lxx anchor doesn't reliably scroll GitHub's edit
        # view on first load (it opens at the top until the page is
        # refreshed); a short range seems to work, so use the smallest one.
        linespec = f'#L{lineno}-L{lineno + 1}'
    else:
        linespec = f'#L{lineno}-L{lineno + len(source) - 1}'

    if 'dev' in pv.__version__:
        kind = 'main'
    else:  # pragma: no cover
        kind = f'release/{".".join(pv.__version__.split(".")[:2])}'

    blob_or_edit = 'edit' if edit else 'blob'

    return f'https://github.com/pyvista/pyvista/{blob_or_edit}/{kind}/pyvista/{fn}{linespec}'


def fix_edit_link_button(pagename: str, link: str) -> str:
    """Rewrite an "edit on GitHub" link to point at the actual source file.

    The default "edit this page" link points at the rendered page source
    under ``doc/source`` (for example, the ``.rst`` generated for a gallery
    example), which 404s on GitHub because that file is not in the repo.
    Two cases need rewriting:

    - Gallery examples ``.rst`` to the source ``.py`` under ``examples/``.
    - Autosummary stubs to the object's definition, at the same line span as
      the page's ``[source]`` button, so the implementation doesn't have to
      be found by hand once the edit page opens.

    Parameters
    ----------
    pagename : str
        The Sphinx pagename for the page being rendered.

    link : str
        The default GitHub edit URL, used as a fallback.

    Returns
    -------
    str
        The corrected edit URL, or the original ``link`` for pages that do
        not need rewriting or whose source could not be resolved.

    """
    if pagename == 'contributing':
        return 'https://github.com/pyvista/pyvista/edit/main/CONTRIBUTING.rst'
    if pagename.startswith('examples') and 'index' not in pagename:
        # Gallery example. The ``examples`` segment in ``pagename`` matches
        # the ``examples`` directory in the repo, so we can use it directly.
        return f'https://github.com/pyvista/pyvista/edit/main/{pagename}.py'
    if '_autosummary' in pagename:
        # API summary stub: resolve the source via the Python object.
        fullname = pagename.split('_autosummary')[1].lstrip('/')
        resolved = linkcode_resolve('py', {'module': 'pyvista', 'fullname': fullname}, edit=True)
        # ``linkcode_resolve`` returns ``None`` for objects it cannot locate
        return resolved or link
    # Fall back to the default link for everything else.
    return link


def _fix_edit_button(pagename: str, context) -> None:
    """Point the "suggest edit" button at the file the page is generated from.

    ``sphinx-book-theme`` builds the pencil button in Python rather than in a
    template: ``header_buttons`` is populated by its ``add_source_buttons``
    handler (priority 501) from ``get_edit_provider_and_url()``. So the URL is
    patched here after the fact, which requires ``pv_html_page_context`` to be
    connected with a priority greater than 501.

    ``get_edit_provider_and_url`` is replaced as well so that any template
    calling it directly -- such as the ``edit-this-page.html`` component of
    ``pydata-sphinx-theme`` -- gets the corrected URL too. Templates render
    after every ``html-page-context`` handler has run, so the override is
    picked up regardless of priority.

    """
    get_edit_provider_and_url = context.get('get_edit_provider_and_url')
    if get_edit_provider_and_url is None:  # pragma: no cover
        # ``use_edit_page_button`` is disabled, or a theme without the
        # pydata "edit this page" machinery is in use.
        return

    def fixed_provider_and_url() -> tuple[str, str]:
        provider, link = get_edit_provider_and_url()
        return provider, fix_edit_link_button(pagename, link)

    context['get_edit_provider_and_url'] = fixed_provider_and_url

    for button in context.get('header_buttons', []):
        # A single enabled repo button is added flat, several are nested in a group
        buttons = button['buttons'] if button.get('type') == 'group' else [button]
        for repo_button in buttons:
            if repo_button.get('label') == 'source-edit-button':
                repo_button['url'] = fix_edit_link_button(pagename, repo_button['url'])


def _drop_download_button(context) -> None:
    """Remove the header "download this page" button.

    ``sphinx-book-theme`` adds it whenever a page has a source suffix, without
    checking whether that source is actually served: it links to a ``.rst``
    under ``_sources`` that 404s the same way the unfixed edit button does,
    since ``html_copy_source`` is disabled for these docs. There is nothing
    else worth downloading either -- "Print to PDF" is a browser feature, not
    a real download. Gallery examples keep their own ``.py``/``.ipynb``/``.zip``
    downloads from sphinx-gallery's links at the bottom of the page.

    """
    if 'header_buttons' not in context:
        return

    context['header_buttons'] = [
        button for button in context['header_buttons'] if button.get('label') != 'download-buttons'
    ]


def pv_html_page_context(  # noqa: PLR0917
    app,  # noqa: ARG001
    pagename: str,
    templatename: str,  # noqa: ARG001
    context,
    doctree,  # noqa: ARG001
) -> None:
    """Fix up the ``sphinx-book-theme`` header buttons for the page being rendered.

    Must be connected to ``html-page-context`` with a priority above the 501
    used by the theme's own handlers, which is where the buttons are built.

    """
    _fix_edit_button(pagename, context)
    _drop_download_button(context)
