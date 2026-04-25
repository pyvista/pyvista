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

    return f'http://github.com/pyvista/pyvista/{blob_or_edit}/{kind}/pyvista/{fn}{linespec}'


def fix_edit_link_button(pagename: str, link: str) -> str | None:
    """Rewrite an "edit on GitHub" link to point at the actual source file.

    The default Sphinx "edit this page" link points at the rendered page
    (for example, the generated ``.rst`` for a gallery example), which
    404s on GitHub. Two cases need rewriting:

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
    str or None
        The corrected edit URL, or the original ``link`` for pages that
        do not need rewriting. ``None`` if no source location can be
        resolved for an autosummary stub.

    """
    if pagename.startswith('examples') and 'index' not in pagename:
        # Gallery example. The ``examples`` segment in ``pagename`` matches
        # the ``examples`` directory in the repo, so we can use it directly.
        return f'https://github.com/pyvista/pyvista/edit/main/{pagename}.py'
    if '_autosummary' in pagename:
        # API summary stub: resolve the source via the Python object.
        fullname = pagename.split('_autosummary')[1].lstrip('/')
        return linkcode_resolve('py', {'module': 'pyvista', 'fullname': fullname}, edit=True)
    # Fall back to the default link for everything else.
    return link


def pv_html_page_context(  # noqa: PLR0917
    app,  # noqa: ARG001
    pagename: str,
    templatename: str,  # noqa: ARG001
    context,
    doctree,  # noqa: ARG001
) -> None:  # pragma: no cover
    """Inject the ``fix_edit_link_button`` helper into the Jinja context.

    The Sphinx ``html-page-context`` event fires per page, so the helper
    is bound here to the current ``pagename`` and called from the
    ``edit-this-page.html`` template.
    """
    context['fix_edit_link_button'] = lambda link: fix_edit_link_button(pagename, link)
