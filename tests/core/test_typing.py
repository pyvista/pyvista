import importlib
import os
from re import findall

from mypy import api as mypy_api
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
TYPING_CASES_DIR = PROJECT_ROOT + '/tests/core/typing/'


def _reveal_type_from_code(code_snippet: str):
    # Call `mypy -c CODE` from the project root dir
    # This ensures the config is loaded and imports are found
    # NOTE: running mypy can be slow, avoid making excessive calls
    cur = os.getcwd()
    if importlib.util.find_spec('npt_promote') is None:
        raise ModuleNotFoundError("Package 'npt-promote' is required for this test.")
    try:
        os.chdir(PROJECT_ROOT)

        result = mypy_api.run(['-c', code_snippet])
        assert 'usage: mypy' not in result[1]
        assert 'Cannot find implementation' not in result[0]

        # Clean up output
        stdout = str(result[0])
        stdout = stdout.replace('Tuple', 'tuple')
        stdout = stdout.replace('builtins.', '')
        stdout = stdout.replace('numpy.', '')
        stdout = stdout.replace('typing.', '')

        match = findall(r'note: Revealed type is "([^"]+)"', stdout)
        assert match is not None
        return match

    finally:
        os.chdir(cur)


def _generate_test_cases(test_case_filename: str):
    # Create code snippet for mypy to analyze
    with open(os.path.join(TYPING_CASES_DIR, test_case_filename)) as f:
        code = f.read()

    revealed_types = _reveal_type_from_code(code)
    expected_types = findall(r'# EXPECTED_TYPE: "([^"]+)"', code)
    reveal_type_args = findall(r'reveal_type\((.*?)\)(?=(\s*?#))', code)
    reveal_type_args = [x[0] for x in reveal_type_args]
    assert len(expected_types) == len(revealed_types)
    assert len(expected_types) == len(reveal_type_args)
    return zip(reveal_type_args, revealed_types, expected_types)


@pytest.mark.parametrize('case', _generate_test_cases('case_array_wrapper.py'))
def test_array_wrapper(case):
    arg, revealed, expected = case
    assert f"{arg} -> {revealed}" == f"{arg} -> {expected}"


@pytest.mark.parametrize('case', _generate_test_cases('case_validate_array_default.py'))
def test_validate_array_default(case):
    arg, revealed, expected = case
    assert f"{arg} -> {revealed}" == f"{arg} -> {expected}"


@pytest.mark.parametrize('case', _generate_test_cases('case_validate_array_dtype_out.py'))
def test_validate_array_dtype_out(case):
    arg, revealed, expected = case
    assert f"{arg} -> {revealed}" == f"{arg} -> {expected}"


@pytest.mark.parametrize('case', _generate_test_cases('case_validate_array_return_numpy.py'))
def test_validate_array_return_numpy(case):
    arg, revealed, expected = case
    assert f"{arg} -> {revealed}" == f"{arg} -> {expected}"
