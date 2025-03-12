"""Test static and runtime type annotations revealed by Mypy.

This test will automatically analyze all files in `tests/typing/test_typing_cases`
directory. To add new test cases, simply add a new .py file with each test case
following the format:

    reveal_type(arg)  # EXPECTED_TYPE: "<T>"

where `arg` is any argument you want mypy to analyze, and <T> is the expected
revealed type returned by mypy. Note: the output types from mypy are truncated
based on the mappings in the `REPLACE_TYPES` dictionary.

"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import re
import sys
from typing import Any
from typing import NamedTuple
from typing import Union  # noqa: F401

from mypy import api as mypy_api
import pyanalyze
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
assert 'tests' in os.listdir(PROJECT_ROOT)
TYPING_CASES_REL_PATH = 'tests/typing/test_typing_cases'
TYPING_CASES_PACKAGE = TYPING_CASES_REL_PATH.replace('/', '.')
TYPING_CASES_ABS_PATH = PROJECT_ROOT / TYPING_CASES_REL_PATH
TEST_FILE_NAMES = [f for f in os.listdir(TYPING_CASES_ABS_PATH) if f.endswith('.py')]

# Define types to simplify in the "revealed type" output string from Mypy.
# The key will be replaced by the value.
REPLACE_TYPES = {
    'pyvista.core.partitioned.PartitionedDataSet': 'PartitionedDataSet',
    'pyvista.core.dataset.DataSet': 'DataSet',
    'pyvista.core.composite.MultiBlock': 'MultiBlock',
    'pyvista.core.pointset.ExplicitStructuredGrid': 'ExplicitStructuredGrid',
    'pyvista.core.pointset.StructuredGrid': 'StructuredGrid',
    'pyvista.core.pointset.PolyData': 'PolyData',
    'pyvista.core.pointset.UnstructuredGrid': 'UnstructuredGrid',
    'pyvista.core.pointset.PointSet': 'PointSet',
    'pyvista.core.grid.ImageData': 'ImageData',
    'pyvista.core.grid.RectilinearGrid': 'RectilinearGrid',
    'pyvista.core.objects.Table': 'Table',
    'pyvista.core.pyvista_ndarray.pyvista_ndarray': 'pyvista_ndarray',
    'typing.Iterator': 'Iterator',
}

# Import the REPLACE_TYPES values into the global namespace to make
# available for runtime tests
globals().update(
    {
        class_name: getattr(importlib.import_module(module_name), class_name)
        for full_path in REPLACE_TYPES
        for module_name, class_name in [full_path.rsplit('.', 1)]
    }
)
REPLACE_TYPES.update({'builtins.str': 'str'})
REPLACE_TYPES.update({'builtins.int': 'int'})
REPLACE_TYPES.update({'builtins.tuple': 'tuple'})


class _TestCaseTuple(NamedTuple):
    file: str
    line_num: int
    arg: str | None
    expected: str | None
    revealed: str | None


def _reveal_types():
    # Call mypy from the project root dir on the typing test case files
    # Calling from root ensures the config is loaded and imports are found
    # NOTE: running mypy can be slow, avoid making excessive calls
    if importlib.util.find_spec('mypy') is None:
        raise ModuleNotFoundError("Package 'mypy' is required for this test.")

    if importlib.util.find_spec('npt_promote') is None:
        raise ModuleNotFoundError("Package 'npt-promote' is required for this test.")

    cur = Path().cwd()
    try:
        os.chdir(PROJECT_ROOT)

        std_out, std_err, exit_status = mypy_api.run(
            ['--show-absolute-path', '--show-traceback', '--package', TYPING_CASES_PACKAGE]
        )

        if exit_status != 0:
            if std_out:
                raise RuntimeError(f'Error running mypy.\n{std_out}')
            else:
                raise RuntimeError(f'Error running mypy.\n{std_err}')
        assert 'Cannot find implementation' not in std_out

        # Group the revealed types by (filepath), (line num), and (type)
        pattern = r'^(.*?):(\d*?):\snote: Revealed type is "([^"]+)"'
        match = re.findall(pattern, std_out, re.MULTILINE)
        assert match is not None

        # Make revealed types less verbose
        for i, group in enumerate(match):
            filepath, line_num, revealed = group
            for key, value in REPLACE_TYPES.items():
                revealed = revealed.replace(key, value)
            match[i] = (filepath, line_num, revealed)
        return match

    finally:
        os.chdir(cur)


def _get_expected_types():
    """Parse all case files and extract expected types."""
    cases = []
    pattern = r'^\s*?reveal_type\((.*?)\)\s*?#\sEXPECTED_TYPE: "([^"]+)"'
    for file in TEST_FILE_NAMES:
        with (Path(TYPING_CASES_ABS_PATH) / file).open() as f:
            split_lines = f.read().splitlines()
        for line_num, line in enumerate(split_lines):
            if not line.strip().startswith('#'):
                match = re.search(pattern, line)
                if match is not None:
                    arg, expected = match.groups()
                    cases.append((file, line_num + 1, arg, expected))
    assert cases is not None
    return cases


def _generate_test_cases():
    """Generate a list of line-by-line test cases from the typing test directory.

    This function:
        (1) calls mypy to get the revealed types, and
        (2) parses the code files to get the `reveal_type(arg)` argument and the
            expected type.

    The two outputs are then merged to create individual test cases.
    """
    test_cases_dict = {}

    def add_to_dict(filepath, line_num: str, key: str, val: str):
        # Function for stuffing parsed data into a dict.
        # We use a dict to allow for any entry to be made based on line number alone.
        # This way, we can defer checking for any errors with the parsed data to test time.
        nonlocal test_cases_dict
        filename = Path(filepath).name
        line_num = int(line_num)
        try:
            test_cases_dict[filename]
        except KeyError:
            test_cases_dict[filename] = {}
        try:
            test_cases_dict[filename][line_num]
        except KeyError:
            test_cases_dict[filename][line_num] = {}
        test_cases_dict[filename][line_num][key] = val

    # run mypy
    revealed_types: list[tuple[str, str, str]] = _reveal_types()
    for filepath, line_num, revealed in revealed_types:
        add_to_dict(filepath, line_num, 'revealed', revealed)

    # parse code files
    expected_types: list[tuple[str, str, str, str]] = _get_expected_types()
    for filepath, line_num, arg, expected in expected_types:
        add_to_dict(filepath, line_num, 'arg', arg)
        add_to_dict(filepath, line_num, 'expected', expected)

    # flatten dict
    test_cases_list = []
    for file, lines in test_cases_dict.items():
        for line_num, content in sorted(lines.items()):
            arg = content.get('arg', None)
            rev = content.get('revealed', None)
            exp = content.get('expected', None)
            test_case = _TestCaseTuple(
                file=file, line_num=line_num, arg=arg, expected=exp, revealed=rev
            )
            test_cases_list.append(test_case)

    return test_cases_list


def _load_module_namespace(module_path: Path) -> dict[str, Any]:
    namespace = {}
    with module_path.open() as f:
        exec(f.read(), namespace)
    return namespace


def pytest_generate_tests(metafunc):
    """Generate parametrized tests."""

    # Skip tests on Linux and Python < 3.12
    if sys.platform == 'linux' and sys.version_info < (3, 12):
        pytest.skip('Issue with mypy plugin.')

    if 'test_case' in metafunc.fixturenames:
        # Generate separate tests for static and runtime checks
        test_cases = _generate_test_cases()
        test_cases_runtime = [(*case, 'runtime') for case in test_cases]
        test_cases_static = [(*case, 'static') for case in test_cases]

        # Interleave cases
        all_cases = [x for y in zip(test_cases_runtime, test_cases_static) for x in y]
        all_cases = all_cases[::-1]

        # Name test cases with file line number
        parent = Path(TYPING_CASES_REL_PATH).name
        ids = [
            f'{parent}/{file}, line {line}, {static_or_runtime}'
            for file, line, _, _, _, static_or_runtime in all_cases
        ]
        metafunc.parametrize('test_case', all_cases, ids=ids)


def test_typing(test_case):
    file, line_num, arg, expected, revealed, static_or_runtime = test_case
    # Test set-up
    assert file in TEST_FILE_NAMES
    assert isinstance(line_num, int)
    if arg is None or revealed is None or expected is None:
        pytest.fail(
            f'Test setup failed for test case in {file}:{line_num}. Got:\n'
            f'\targ: {arg}\n'
            f'\texpected: {expected}\n'
            f'\trevealed: {revealed}\n'
        )
    if static_or_runtime == 'static':
        # Test statically revealed type from mypy is correct
        assert f'{arg} -> {revealed}' == f'{arg} -> {expected}'
    else:
        # Test that the actual runtime type is compatible with the revealed type

        # Load the test case file's namespace into the local namespace
        # so we can evaluate code defined in the test case
        namespace = _load_module_namespace(Path(TYPING_CASES_ABS_PATH) / file)
        locals().update(namespace)

        try:
            revealed_type = eval(revealed)
        except Exception as e:
            pytest.fail(
                f'Test setup failed for runtime test case in {file}:{line_num}.\n'
                f'Could not evaluate revealed type:\n '
                f'\t{revealed}\n'
                f'An exception was raised:\n{e!r}'
            )
        try:
            runtime_val = eval(arg)
        except Exception as e:
            pytest.fail(
                f'Test setup failed for runtime test case in {file}:{line_num}.\n'
                f'Could not evaluate runtime argument:\n '
                f'\t{arg}\n'
                f'An exception was raised:\n{e!r}'
            )
        compat_error_msg = pyanalyze.runtime.get_assignability_error(runtime_val, revealed_type)
        if compat_error_msg:
            error_prefix = (
                f'\nRuntime value:\n'
                f'\t{arg} = {runtime_val}\n'
                f'is not compatible with statically revealed type:\n'
                f'\t{revealed_type}\n\n'
                f'Reason:\n'
            )

            pytest.fail(error_prefix + compat_error_msg)
