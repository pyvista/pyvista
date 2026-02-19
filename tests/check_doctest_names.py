"""A helper script to check names in doctests.

This module is intended to be called from pyvista's root directory with

    python tests/check_doctest_names.py

The problem is that pytest doctests (following the standard-library
doctest module) see the module-global namespace. So when a doctest looks
like this:

Examples
--------
    >>> import numpy
    >>> from pyvista import CellType
    >>> offset = np.array([0, 9])
    >>> cell0_ids = [8, 0, 1, 2, 3, 4, 5, 6, 7]
    >>> cell1_ids = [8, 8, 9, 10, 11, 12, 13, 14, 15]
    >>> cells = np.hstack((cell0_ids, cell1_ids))
    >>> cell_type = np.array([CellType.HEXAHEDRON, CellType.HEXAHEDRON], np.int8)

there will be a ``NameError`` when the code block is copied into Python
because the ``np`` name is undefined. However, pytest and sphinx test
runs will not catch this, as the ``np`` name is typically also available
in the global namespace of the module where the doctest resides.

In order to fix this, we build a tree of pyvista's public and private
API, using the standard-library doctest module as a doctest parser. We
execute examples with a clean empty namespace to ensure that mistakes
such as the above can be caught.

Note that we don't try to verify that the actual results from each
example are correct; that's still pytest's responsibility. As long
as the examples run without error, this module will be happy.

The implementation is not very robust or smart, it just gets the job
done to find the rare name mistake in our examples.

If you need off-screen plotting, set the ``PYVISTA_OFF_SCREEN``
environmental variable to ``True`` before running the script.

"""

from __future__ import annotations

from argparse import ArgumentParser
from doctest import DocTestFinder
import re
import sys
from textwrap import indent
from types import ModuleType

import pyvista as pv


def discover_modules(entry=pv, recurse=True):
    """Discover the submodules present under an entry point.

    If ``recurse=True``, search goes all the way into descendants of the
    entry point. Only modules are gathered, because within a module
    ``doctest``'s discovery can work recursively.

    Should work for ``pyvista`` as entry, but no promises for its more
    general applicability.

    Parameters
    ----------
    entry : module, optional
        The entry point of the submodule search. Defaults to the main
        ``pyvista`` module.

    recurse : bool, optional
        Whether to recurse into submodules.

    Returns
    -------
    modules : dict of modules
        A (module name -> module) mapping of submodules under ``entry``.

    """
    entry_name = entry.__name__
    found_modules = {}
    next_entries = {entry}
    while next_entries:
        next_modules = {}
        for ent in next_entries:
            for attr_short_name in dir(ent):
                attr = getattr(ent, attr_short_name)
                if not isinstance(attr, ModuleType):
                    continue

                module_name = attr.__name__

                if module_name.startswith(entry_name):
                    next_modules[module_name] = attr

        if not recurse:
            return next_modules

        # find as-of-yet-undiscovered submodules
        next_entries = {
            module
            for module_name, module in next_modules.items()
            if module_name not in found_modules
        }
        found_modules.update(next_modules)

    return found_modules


def check_doctests(modules=None, respect_skips=True, verbose=True):
    """Check whether doctests can be run as-is without errors.

    Parameters
    ----------
    modules : dict, optional
        (module name -> module) mapping of submodules defined in a
        package as returned by ``discover_modules()``. If omitted,
        ``discover_modules()`` will be called for ``pyvista``.

    respect_skips : bool, optional
        Whether to ignore doctest examples that contain a DOCTEST:+SKIP
        directive.

    verbose : bool, optional
        Whether to print passes/failures as the testing progresses.
        Failures are printed at the end in every case.

    Returns
    -------
    failures : dict of (Exception, str)  tuples
        An (object name -> (exception raised, failing code)) mapping
        of failed doctests under the specified modules.

    """
    skip_pattern = re.compile(r'doctest: *\+SKIP')

    if modules is None:
        modules = discover_modules()

    # find and parse all docstrings; this will also remove any duplicates
    doctests = {
        dt.name: dt
        for module_name, module in modules.items()
        for dt in DocTestFinder(recurse=True).find(module, globs={})
    }

    # loop over doctests in alphabetical order for sanity
    sorted_names = sorted(doctests)
    failures = {}
    for dt_name in sorted_names:
        dt = doctests[dt_name]
        if not dt.examples:
            continue

        # mock print to suppress output from a few talkative tests
        globs = {'print': (lambda *args, **kwargs: ...)}  # noqa: ARG005
        for iline, example in enumerate(dt.examples, start=1):
            if not example.source.strip() or (
                respect_skips and skip_pattern.search(example.source)
            ):
                continue
            try:
                exec(example.source, globs)
            except Exception as exc:  # noqa: BLE001
                if verbose:
                    print(f'FAILED: {dt.name} -- {exc!r}')
                erroring_code = ''.join([example.source for example in dt.examples[:iline]])
                failures[dt_name] = exc, erroring_code
                break
        else:
            if verbose:
                print(f'PASSED: {dt.name}')

    total = len(doctests)
    fails = len(failures)
    passes = total - fails
    print(f'\n{passes} passes and {fails} failures out of {total} total doctests.\n')
    if not fails:
        return failures

    print('List of failures:')
    for name, (exc, erroring_code) in failures.items():
        print('-' * 60)
        print(f'{name}:')
        print(indent(erroring_code, '    '))
        print(repr(exc))
    print('-' * 60)

    return failures


if __name__ == '__main__':
    parser = ArgumentParser(description='Look for name errors in doctests.')
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        help='print passes and failures as tests progress',
    )
    parser.add_argument(
        '--no-respect-skips',
        action='store_false',
        dest='respect_skips',
        help='ignore doctest SKIP directives',
    )
    args = parser.parse_args()

    failures = check_doctests(verbose=args.verbose, respect_skips=args.respect_skips)

    if failures:
        # raise a red flag for CI
        sys.exit(1)
