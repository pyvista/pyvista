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
        Jump right to the edit page.

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

    linespec = f'#L{lineno}-L{lineno + len(source) - 1}' if lineno and not edit else ''

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
    - Autosummary stubs to the file defining the Python object.

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


def pv_html_page_context(  # noqa: PLR0917
    app,  # noqa: ARG001
    pagename: str,
    templatename: str,  # noqa: ARG001
    context,
    doctree,  # noqa: ARG001
) -> None:
    """Point the "suggest edit" button at the file the page is generated from.

    ``sphinx-book-theme`` builds the pencil button in Python rather than in a
    template: ``header_buttons`` is populated by its ``add_source_buttons``
    handler (priority 501) from ``get_edit_provider_and_url()``. So the URL is
    patched here after the fact, which requires this handler to be connected
    with a priority greater than 501.

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
