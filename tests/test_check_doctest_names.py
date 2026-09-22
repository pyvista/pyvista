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
        pytest.param('count: int = 1\nprint(count)\n', [], id='annotated-assign'),
        pytest.param('class Foo:\n    bar = 1\n    baz = bar + 1\n', [], id='class-body'),
        pytest.param('class Foo(Base):\n    pass\n', ['Base'], id='class-base'),
        pytest.param('class Foo:\n    bar = missing\n', ['missing'], id='class-body-undefined'),
        pytest.param('[b + k for b in [1]]\n', ['k'], id='listcomp-free-name'),
        pytest.param('print(__name__)\n', [], id='module-dunder'),
    ],
)
def test_undefined_names(source, expected):
    """Names are reported only when the source never binds them."""
    assert undefined_names(source) == expected


@pytest.mark.parametrize(
    ('source', 'expected'),
    [
        pytest.param('arr = np.array([1])\nimport numpy as np\n', ['np'], id='import-after-use'),
        pytest.param('print(val)\nval = 1\n', ['val'], id='assign-after-use'),
        pytest.param('def fn():\n    return later\n\n\nlater = 1\n', [], id='deferred-body'),
        pytest.param(
            'items = [1]\nall(isinstance(b, int) for b in items)\n', [], id='genexp-target'
        ),
        pytest.param('items = [1]\n[b for b in items]\n', [], id='listcomp-target'),
        pytest.param(
            'items = [1]\n[y for x in items if (y := x)]\nprint(y)\n', [], id='walrus-in-comp'
        ),
        pytest.param(
            '@deco\ndef f():\n    pass\n\n\ndef deco(fn):\n    return fn\n',
            ['deco'],
            id='decorator-after-use',
        ),
        pytest.param(
            'def f(a=later):\n    return a\n\n\nlater = 1\n', ['later'], id='default-after-use'
        ),
    ],
)
def test_names_used_before_they_are_bound(source, expected):
    """A name is reported when it is used above the line that binds it."""
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
