from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest

from pyvista.typing.mypy_plugin import promote_type

TEST_DIR = str(Path(__file__).parent)
ROOT_DIR = str(Path(TEST_DIR).parent.parent.parent)
assert 'tests' in os.listdir(ROOT_DIR)  # noqa: PTH208
MYPY_PLUGIN_MODULE = str(Path(ROOT_DIR) / 'pyvista' / 'typing' / 'mypy_plugin.py')
MYPY_CONFIG_FILE_NO_PLUGIN = str(Path(TEST_DIR) / 'mypy_no_plugin.ini')
MYPY_CONFIG_FILE_USE_PLUGIN = str(Path(TEST_DIR) / 'mypy_use_plugin.ini')


@pytest.fixture
def decorated_single():
    @promote_type(float)
    class Foo: ...

    return Foo


@pytest.fixture
def decorated_double():
    @promote_type(float, str)
    class Foo: ...

    return Foo


def test_promote_type_runtime(decorated_single, decorated_double):
    klass = decorated_single()
    assert isinstance(klass, decorated_single)

    klass = decorated_double()
    assert isinstance(klass, decorated_double)


cases = [
    dict(  # Test that the duck type is not a float by default
        use_plugin=False,
        promote_type=float,
        arg_type=float,
        expected_output='py:9: error: Argument 1 to "foo" has incompatible type "DuckType"; expected "float"  [arg-type]',
    ),
    dict(  # Same as above, but use the plugin.
        use_plugin=True, promote_type=float, arg_type=float, expected_output=''
    ),
    dict(  # Test promotion as a subclass.
        use_plugin=True, promote_type=int, arg_type=float, expected_output=''
    ),
    dict(  # Test promotion is one-way.
        use_plugin=True,
        promote_type=float,
        arg_type=int,
        expected_output='py:9: error: Argument 1 to "foo" has incompatible type "DuckType"; expected "int"  [arg-type]',
    ),
]


@pytest.mark.parametrize(
    ('use_plugin', 'promote_type', 'arg_type', 'expected_output'), [case.values() for case in cases]
)
def test_promote_type_static(use_plugin, promote_type, arg_type, expected_output, tmp_path):
    code = f"""
from pyvista.typing.mypy_plugin import promote_type

@promote_type({promote_type.__name__})
class DuckType: ...
y: DuckType

def foo(x: {arg_type.__name__}) -> None: ...
foo(y)
"""
    mypy_output = _run_mypy_code(
        code, use_plugin=use_plugin, follow_imports=False, tmp_path=tmp_path
    )
    stdout = str(mypy_output.stdout.decode('utf-8'))
    code = mypy_output.returncode
    if expected_output == '':
        assert code == 0, f'Mypy did not return success status:\n{stdout}'
    else:
        assert code != 0, f'Mypy returned success status, but an error was expected:\n{stdout}'
        assert expected_output in stdout


def _run_mypy_code(code, use_plugin, follow_imports, tmp_path):
    """Call mypy from the project root dir with or without the plugin enabled."""
    # Save code to tmp file
    tmp_file = tmp_path / 'code.py'
    with open(tmp_file, 'w', encoding='utf-8') as file:  # noqa: PTH123
        file.write(code)

    cwd = Path.cwd()
    try:
        args = ['mypy', '--show-traceback']
        if not follow_imports:
            # Use '--follow-imports=skip' to only analyze the files passed to mypy
            # otherwise it will analyze the entire pyvista library
            args.append('--follow-imports=skip')

        # Set config file
        config = MYPY_CONFIG_FILE_USE_PLUGIN if use_plugin else MYPY_CONFIG_FILE_NO_PLUGIN
        args.extend(['--config-file', config])

        # Only run mypy on the code block and plugin module files
        args.extend([tmp_file, MYPY_PLUGIN_MODULE])
        return subprocess.run(args, capture_output=True)
    finally:
        os.chdir(cwd)


cases = [
    dict(  # Test that NDArray[float]
        use_plugin=False,
        generic_arg=float,
        expected_error=[
            ':3: error: Type argument "float" of "NDArray" must be a subtype of "generic[Any]"'
        ],
        expected_note=[
            ':4: note: Revealed type is "numpy.ndarray[builtins.tuple[builtins.int, ...], numpy.dtype[builtins.float]]"'
        ],
    ),
    dict(  # Test that NDArray[float]
        use_plugin=True,
        generic_arg=float,
        expected_error=None,
        expected_note=[
            ':4: note: Revealed type is "numpy.ndarray[builtins.tuple[builtins.int, ...], numpy.dtype[builtins.float]]"'
        ],
    ),
]


@pytest.mark.parametrize(
    ('use_plugin', 'generic_arg', 'expected_error', 'expected_note'),
    [case.values() for case in cases],
)
def test_generic_arraylike(use_plugin, generic_arg, expected_error, expected_note, tmp_path):
    code = f"""
from numpy.typing import NDArray
x: NDArray[{generic_arg.__name__}]
reveal_type(x)
"""
    mypy_output = _run_mypy_code(
        code, use_plugin=use_plugin, follow_imports=True, tmp_path=tmp_path
    )
    stdout = str(mypy_output.stdout.decode('utf-8'))
    code = mypy_output.returncode

    if expected_error:
        assert code != 0, f'Mypy returned success status, but an error was expected:\n{stdout}'
        for error in expected_error:
            assert error in stdout
    else:
        assert code == 0, f'Mypy did not return success status:\n{stdout}'

    if expected_note:
        for note in expected_note:
            assert note in stdout
