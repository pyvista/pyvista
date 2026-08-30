"""Test the static name analysis in :mod:`tests.check_doctest_names` itself."""

from __future__ import annotations

import pytest

from tests.check_doctest_names import undefined_names


@pytest.mark.parametrize(
    ('source', 'expected'),
    [
        pytest.param('arr = np.array([1, 2])\n', ['np'], id='missing-import'),
        pytest.param('import numpy as np\narr = np.array([1, 2])\n', [], id='import-in-example'),
        pytest.param(
            'from numpy import array as arr_fn\nvalue = arr_fn([1])\n', [], id='import-from-as'
        ),
        pytest.param('text = str(len([1, 2]))\nprint(text)\n', [], id='builtins-only'),
        pytest.param('from numpy import *\narr = array([1])\n', [], id='star-import'),
        pytest.param('def fn():\n    return missing_global\n', ['missing_global'], id='nested'),
        pytest.param(
            'import numpy as np\n\n\ndef fn():\n    return np.pi\n', [], id='nested-bound'
        ),
        pytest.param(
            'mesh = pv.Sphere()\nmesh.plot(color=col)\n', ['col', 'pv'], id='several-names'
        ),
    ],
)
def test_undefined_names(source, expected):
    """Names are reported only when the source never binds them."""
    assert undefined_names(source) == expected


def test_names_bound_by_an_earlier_line_are_clean():
    """A name defined by an earlier example is available to later ones."""
    source = 'import pyvista as pv\nmesh = pv.Sphere()\nmesh.plot()\n'
    assert undefined_names(source) == []


def test_loop_and_context_targets_bind():
    """Loop, ``with`` and ``except`` targets count as bound names."""
    source = (
        'for item in [1, 2]:\n'
        '    print(item)\n'
        "with open('f') as fh:\n"
        '    print(fh)\n'
        'try:\n'
        '    pass\n'
        'except ValueError as exc:\n'
        '    print(exc)\n'
    )
    assert undefined_names(source) == []


def test_syntax_error_propagates():
    """Unparsable source raises so the caller can report it."""
    with pytest.raises(SyntaxError):
        undefined_names('def (:\n')
