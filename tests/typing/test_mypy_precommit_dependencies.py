from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

from packaging.requirements import Requirement
import pytest
import yaml

REPO = 'https://github.com/pyvista/pre-commit-mypy'
ROOT_PATH = Path(__file__).parent.parent.parent
REQUIREMENTS_PATH = ROOT_PATH / 'requirements_typing.txt'
PRE_COMMIT_CONFIG_PATH = ROOT_PATH / '.pre-commit-config.yaml'
SKIP_PACKAGES = ['vtk']


class _TestCaseTuple(NamedTuple):
    package_name: str
    project_specifier: str
    precommit_specifier: str


def _generate_test_cases():
    """Generate a list of requirement test cases.
    This function:
        (1) Generates a list of typing requirements defined in a requirements file
        (2) Generates a list of mypy pre-commit requirements from the pre-commit config
        (3) Merges the two lists together and returns separate test cases to
            comparing all typing requirements to all pre-commit requirements
    """
    test_cases_dict: dict = {}

    def add_to_dict(package_name: str, specifier: str | None, key: str):
        # Function for stuffing image paths into a dict.
        # We use a dict to allow for any entry to be made based on image path alone.
        # This way, we can defer checking for any mismatch between the cached and docs
        # images to test time.
        nonlocal test_cases_dict
        test_name = package_name
        try:
            test_cases_dict[test_name]
        except KeyError:
            test_cases_dict[test_name] = {}
        value = package_name if specifier is None else package_name + specifier
        test_cases_dict[test_name].setdefault(key, value)

    # process project requirements
    project_key = REQUIREMENTS_PATH.name
    typing_requirements = _parse_requirements_file(REQUIREMENTS_PATH)
    [
        add_to_dict(package, specifier, key=project_key)
        for package, specifier in typing_requirements.items()
    ]

    # process pre-commit requirements
    precommit_key = PRE_COMMIT_CONFIG_PATH.name
    precommit_requirements = _get_mypy_precommit_requirements(PRE_COMMIT_CONFIG_PATH)
    [
        add_to_dict(package, specifier, key=precommit_key)
        for package, specifier in precommit_requirements.items()
    ]

    # flatten dict
    test_cases_list = []
    for test_name, content in sorted(test_cases_dict.items()):
        project = content.get(project_key, None)
        precommit = content.get(precommit_key, None)
        test_case = _TestCaseTuple(
            package_name=test_name,
            project_specifier=project,
            precommit_specifier=precommit,
        )
        test_cases_list.append(test_case)

    return test_cases_list


def pytest_generate_tests(metafunc):
    """Generate parametrized tests."""
    if 'requirement_test_case' in metafunc.fixturenames:
        test_cases = _generate_test_cases()
        ids = [case.package_name for case in test_cases]
        metafunc.parametrize('requirement_test_case', test_cases, ids=ids)


def _parse_requirements_file(file_path):
    requirements = {}
    with open(file_path) as file:  # noqa: PTH123
        for line in file:
            line = line.strip()

            # Update dict from file
            r_arg = '-r'
            if line.startswith(r_arg):
                new_file_path = ROOT_PATH / line[len(r_arg) :].strip()
                more_requirements = _parse_requirements_file(new_file_path)
                requirements.update(more_requirements)
                continue
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue

            # Parse each requirement line
            req = Requirement(line)
            requirements[req.name] = str(req.specifier) if req.specifier else None
    return requirements


def _get_mypy_precommit_requirements(file_path):
    # Get pre-commit config
    with open(file_path) as file:  # noqa: PTH123
        config = yaml.safe_load(file)

    # Extract mypy hook dependencies
    for repo in config['repos']:
        if repo['repo'] == REPO:
            # Update repo config
            hook = repo['hooks'][0]
            assert hook['id'] == 'mypy'
            requirements_list = hook['additional_dependencies']
            requirements_dict = {}
            for item in requirements_list:
                req = Requirement(item)
                requirements_dict[req.name] = str(req.specifier) if req.specifier else None
            return requirements_dict

    raise RuntimeError(
        f'Pre-commit repo:\n\t{REPO}\n' f'does not exist in config:\n\t{PRE_COMMIT_CONFIG_PATH!s}\n'
    )


def test_requirement(requirement_test_case):
    package_name, project_specifier, precommit_specifier = requirement_test_case
    if package_name in SKIP_PACKAGES:
        pytest.skip('Known skips.')

    msg = _test_both_requirements_exist(*requirement_test_case)
    if msg:
        pytest.fail(msg)
    if project_specifier != precommit_specifier:
        pytest.fail(
            f"The `mypy` dependency '{package_name+precommit_specifier}' in '{PRE_COMMIT_CONFIG_PATH.name}'\n'"
            f"must match the requirement '{package_name+project_specifier}' in '{REQUIREMENTS_PATH.name}'."
        )


def _test_both_requirements_exist(package_name, project_requirement, precommit_requirement):
    if project_requirement is None or precommit_requirement is None:
        if project_requirement is None:
            assert precommit_requirement is not None
            missing = REQUIREMENTS_PATH.name
            exists = PRE_COMMIT_CONFIG_PATH.name
            action_precommit = 'removed from'
            action_project = 'added to'
        else:
            assert project_requirement is not None
            missing = PRE_COMMIT_CONFIG_PATH.name
            exists = REQUIREMENTS_PATH.name
            action_precommit = 'added to'
            action_project = 'removed from'
        return (
            f"Test setup failed for package '{package_name}'.\n"
            f"The requirement exists in '{exists}' but is missing from '{missing}'.\n"
            f'The package should be {action_precommit} the `pre-commit` `mypy` config '
            f'or {action_project} the project requirements.'
        )
    return None
