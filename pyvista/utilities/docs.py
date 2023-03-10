"""Supporting functions for documentation build."""
import inspect
import os
import os.path as op
import sys


def linkcode_resolve(domain, info):
    """Determine the URL corresponding to a Python object.

    Parameters
    ----------
    domain : str
        Only useful when 'py'.

    info : dict
        With keys "module" and "fullname".

    Returns
    -------
    url : str
        The code URL.

    Notes
    -----
    This has been adapted to deal with our "verbose" decorator.

    Adapted from mne (mne/utils/docs.py), which was adapted from SciPy (doc/source/conf.py).
    """
    import pyvista

    if domain != 'py':
        return None

    modname = info['module']
    fullname = info['fullname']

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split('.'):
        try:
            obj = getattr(obj, part)
        except Exception:
            return None

    # deal with our decorators properly
    while hasattr(obj, 'fget'):
        obj = obj.fget

    try:
        fn = inspect.getsourcefile(obj)
    except Exception:  # pragma: no cover
        fn = None

    if not fn:  # pragma: no cover
        try:
            fn = inspect.getsourcefile(sys.modules[obj.__module__])
        except Exception:
            return None
        return None

    fn = op.relpath(fn, start=op.dirname(pyvista.__file__))
    fn = '/'.join(op.normpath(fn).split(os.sep))  # in case on Windows

    try:
        source, lineno = inspect.getsourcelines(obj)
    except Exception:  # pragma: no cover
        lineno = None

    if lineno:
        linespec = f"#L{lineno}-L{lineno + len(source) - 1}"
    else:
        linespec = ""

    if 'dev' in pyvista.__version__:
        kind = 'main'
    else:  # pragma: no cover
        kind = 'release/%s' % ('.'.join(pyvista.__version__.split('.')[:2]))

    return f"http://github.com/pyvista/pyvista/blob/{kind}/pyvista/{fn}{linespec}"
