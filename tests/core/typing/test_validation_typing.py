"""Test static type annotations revealed by Mypy.

This test will automatically analyze all files in the test validation_cases directory.
To add new test validation_cases, simply add a new .py file with each test case following
the format:

    reveal_type(arg)  # EXPECTED_TYPE: "<T>"

where `arg` is any argument you want mypy to analyze, and <T> is the expected
revealed type returned by mypy. Note: the output types from mypy are truncated
with the module names removed, e.g. `typing.Sequence` -> `Sequence`,
`builtins.float` -> `float`, etc.

"""

from collections import namedtuple
import importlib
import os
import re
import sys
from typing import List, Tuple

from mypy import api as mypy_api
import pyanalyze
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
TYPING_CASES_REL_PATH = 'tests/core/typing/validation_cases'
TYPING_CASES_PACKAGE = TYPING_CASES_REL_PATH.replace('/', '.')
TYPING_CASES_ABS_PATH = os.path.join(PROJECT_ROOT, TYPING_CASES_REL_PATH)
TEST_FILE_NAMES = [f for f in os.listdir(TYPING_CASES_ABS_PATH) if f.endswith('.py')]

_TestCaseTuple = namedtuple('_TestCaseTuple', ['file', 'line_num', 'arg', 'expected', 'revealed'])


def _reveal_types():
    # Call mypy from the project root dir on the typing test case files
    # Calling from root ensures the config is loaded and imports are found
    # NOTE: running mypy can be slow, avoid making excessive calls
    cur = os.getcwd()
    if importlib.util.find_spec('npt_promote') is None:
        raise ModuleNotFoundError("Package 'npt-promote' is required for this test.")
    try:
        os.chdir(PROJECT_ROOT)

        result = mypy_api.run(['--show-absolute-path', '--package', TYPING_CASES_PACKAGE])
        assert 'usage: mypy' not in result[1]
        assert 'Cannot find implementation' not in result[0]

        # Clean up output
        stdout = str(result[0])

        # group revealed types by (filepath), (line num), and (type)
        pattern = r'^(.*?):(\d*?):\snote: Revealed type is "([^"]+)"'
        match = re.findall(pattern, stdout, re.MULTILINE)
        assert match is not None

        # Make revealed types less verbose
        for i, group in enumerate(match):
            filepath, line_num, revealed = group
            revealed = revealed.replace('Tuple', 'tuple')
            revealed = revealed.replace('builtins.', '')
            revealed = revealed.replace('numpy.', '')
            revealed = revealed.replace('typing.', '')
            match[i] = (filepath, line_num, revealed)
        return match

    finally:
        os.chdir(cur)


def _get_expected_types():
    """Parse all case files and extract expected types."""
    cases = []
    pattern = r'^\s.*?reveal_type\((.*?)\)\s*?#\sEXPECTED_TYPE: "([^"]+)"'
    for file in TEST_FILE_NAMES:
        with open(os.path.join(TYPING_CASES_ABS_PATH, file)) as f:
            split_lines = f.read().splitlines()
        for line_num, line in enumerate(split_lines):
            match = re.search(pattern, line)
            if match is not None:
                arg, expected = match.groups()
                cases.append((file, line_num + 1, arg, expected))
    assert cases is not None
    return cases


def _generate_test_cases():
    """Generate a list of line-by-line test validation_cases from the typing test directory.

    This function:
        (1) calls mypy to get the revealed types, and
        (2) parses the code files to get the `reveal_type(arg)` argument and the
            expected type.

    The two outputs are then merged to create individual test validation_cases.
    """
    test_cases_dict = {}

    def add_to_dict(filepath, line_num: str, key: str, val: str):
        # Function for stuffing parsed data into a dict.
        # We use a dict to allow for any entry to be made based on line number alone.
        # This way, we can defer checking for any errors with the parsed data to test time.
        nonlocal test_cases_dict
        filename = os.path.basename(filepath)
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
    revealed_types: List[Tuple[str, str, str]] = _reveal_types()
    for filepath, line_num, revealed in revealed_types:
        add_to_dict(filepath, line_num, 'revealed', revealed)

    # parse code files
    expected_types: List[Tuple[str, str, str, str]] = _get_expected_types()
    for filepath, line_num, arg, expected in expected_types:
        add_to_dict(filepath, line_num, 'arg', arg)
        add_to_dict(filepath, line_num, 'expected', expected)

    # flatten dict
    test_cases_list = []
    for file, lines in test_cases_dict.items():
        for line_num, content in sorted(lines.items()):
            arg = content['arg'] if 'arg' in content else None
            rev = content['revealed'] if 'revealed' in content else None
            exp = content['expected'] if 'expected' in content else None
            test_case = _TestCaseTuple(
                file=file, line_num=line_num, arg=arg, expected=exp, revealed=rev
            )
            test_cases_list.append(test_case)

    return test_cases_list


def pytest_generate_tests(metafunc):
    """Generate parametrized tests."""
    if 'test_case' in metafunc.fixturenames:
        test_cases = _generate_test_cases()
        test_cases_runtime = [(*case, 'runtime') for case in test_cases]
        test_cases_static = [(*case, 'static') for case in test_cases]

        # Interleave cases
        all_cases = [x for y in zip(test_cases_runtime, test_cases_static) for x in y]
        all_cases = all_cases[::-1]

        # Name test validation_cases with file line number
        ids = [f"{file.split('.py')[0]} : line {line} : {static_or_runtime}" for file, line, _, _, _, static_or_runtime in all_cases]
        metafunc.parametrize('test_case', all_cases, ids=ids)

@pytest.mark.skipif(sys.version_info < (3, 9), reason="Type subscription requires python >= 3.9")
def test_typing(test_case):
    file, line_num, arg, expected, revealed, static_or_runtime = test_case
    # Test set-up
    assert file in TEST_FILE_NAMES
    assert isinstance(line_num, int)
    if arg is None or revealed is None or expected is None:
        pytest.fail(
            f"Test setup failed for test case in {file}:{line_num}. Got:\n"
            f"\targ: {arg}\n"
            f"\texpected: {expected}\n"
            f"\trevealed: {revealed}\n"
        )
    if static_or_runtime == 'static':
        # Test statically revealed type from mypy is correct
        assert f"{arg} -> {revealed}" == f"{arg} -> {expected}"
    else:
        # Test that the actual runtime type is compatible with the revealed type

        # imports needed for runtime evaluation
        from typing import Any, Sequence, Union

        import numpy as np
        from numpy import dtype, ndarray

        from pyvista.core._validation._array_wrapper import _ArrayLikeWrapper
        from pyvista.core._validation.validate import validate_array
        try:
            revealed_type = eval(revealed)
            runtime_val = eval(arg)
        except Exception as e:
            pytest.fail(
                f"Test setup failed for runtime test case in {file}:{line_num}.\n"
                f"An exception was raised:\n{repr(e)}"

            )
        compat_error_msg = pyanalyze.runtime.get_compatibility_error(runtime_val, revealed_type)
        if compat_error_msg:
            error_prefix = (f"\nRuntime value:\n"
                         f"\t{arg} = {runtime_val}\n"
                         f"is not compatible with statically revealed type:\n"
                         f"\t{revealed_type}\n\n"
                            f"Reason:\n")

            pytest.fail(error_prefix + compat_error_msg)
