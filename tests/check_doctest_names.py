"""Check that names used in doctests are defined by the doctests themselves.

Call this from pyvista's root directory with

    python tests/check_doctest_names.py

The examples of each docstring are concatenated and analysed with ``symtable``.
Any name referenced without ever being bound is reported.

"""

from __future__ import annotations

import argparse
import ast
import builtins
import doctest
import re
import symtable
import sys
import textwrap
from types import ModuleType

import pyvista as pv

MODULE_DUNDERS = frozenset(
    {'__name__', '__file__', '__doc__', '__package__', '__spec__', '__loader__', '__builtins__'}
)


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


def _iter_scopes(table):
    """Yield a symbol table and every scope nested inside it."""
    yield table
    for child in table.get_children():
        yield from _iter_scopes(child)


def _has_import_star(tree):
    """Return whether the parsed source contains a star import."""
    return any(
        isinstance(node, ast.ImportFrom) and any(alias.name == '*' for alias in node.names)
        for node in ast.walk(tree)
    )


def _bound_names(table):
    """Return the names bound at module scope of a symbol table."""
    bound = {
        sym.get_name() for sym in table.get_symbols() if sym.is_assigned() or sym.is_imported()
    }
    for scope in _iter_scopes(table):
        bound.update(
            sym.get_name()
            for sym in scope.get_symbols()
            if sym.is_declared_global() and sym.is_assigned()
        )
    return bound


def undefined_names(source):
    """Return the sorted names a source uses without ever binding them.

    Parameters
    ----------
    source : str
        Python source to analyse.

    Returns
    -------
    list of str
        Names referenced but never bound. Empty for sources with a star import.

    """
    tree = ast.parse(source)
    if _has_import_star(tree):
        return []

    table = symtable.symtable(source, '<doctest>', 'exec')
    bound = _bound_names(table)

    undefined = set()
    for scope in _iter_scopes(table):
        for sym in scope.get_symbols():
            name = sym.get_name()
            if not sym.is_referenced() or name in bound or name in MODULE_DUNDERS:
                continue
            if hasattr(builtins, name):
                continue
            if scope is table:
                if not (sym.is_assigned() or sym.is_imported()):
                    undefined.add(name)
            elif sym.is_global():
                undefined.add(name)
    return sorted(undefined)


def check_doctests(modules=None, respect_skips=True, verbose=True):
    """Check whether doctests define every name they use.

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
        Whether to print passes/failures as the checking progresses.
        Failures are printed at the end in every case.

    Returns
    -------
    failures : dict of (Exception, str) tuples
        An (object name -> (exception, offending code)) mapping of
        doctests that use names they never define.

    """
    skip_pattern = re.compile(r'doctest: *\+SKIP')

    if modules is None:
        modules = discover_modules()

    # find and parse all docstrings; this will also remove any duplicates
    doctests = {
        dt.name: dt
        for module in modules.values()
        for dt in doctest.DocTestFinder(recurse=True).find(module, globs={})
    }

    # loop over doctests in alphabetical order for sanity
    sorted_names = sorted(doctests)
    failures = {}
    for dt_name in sorted_names:
        dt = doctests[dt_name]
        if not dt.examples:
            continue

        sources = [
            example.source
            for example in dt.examples
            if example.source.strip()
            and not (respect_skips and skip_pattern.search(example.source))
        ]
        if not sources:
            continue
        source = ''.join(sources)

        try:
            missing = undefined_names(source)
        except SyntaxError as exc:
            failures[dt_name] = exc, source
            if verbose:
                print(f'FAILED: {dt.name} -- {exc!r}')
            continue

        if missing:
            listed = ', '.join(repr(name) for name in missing)
            exc = NameError(f'name {listed} is not defined')
            failures[dt_name] = exc, source
            if verbose:
                print(f'FAILED: {dt.name} -- {exc!r}')
        elif verbose:
            print(f'PASSED: {dt.name}')

    total = len(doctests)
    fails = len(failures)
    passes = total - fails
    print(f'\n{passes} passes and {fails} failures out of {total} total doctests.\n')
    if not fails:
        return failures

    print('List of failures:')
    for name, (exc, offending_code) in failures.items():
        print('-' * 60)
        print(f'{name}:')
        print(textwrap.indent(offending_code, '    '))
        print(repr(exc))
    print('-' * 60)

    return failures


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Look for name errors in doctests.')
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        help='print passes and failures as checks progress',
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
