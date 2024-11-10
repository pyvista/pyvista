from __future__ import annotations

from pathlib import Path
import sys
from typing import NamedTuple

from packaging.requirements import Requirement
import pytest
import yaml

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

MYPY_PRECOMMIT_REPO = 'https://github.com/pyvista/pre-commit-mypy'
ROOT_PATH = Path(__file__).parent.parent.parent
PYPROJECT_TOML_PATH = ROOT_PATH / 'pyproject.toml'
OPTIONAL_DEPENDENCIES = 'typing'  # From`pip install pyvista[typing]`
PRE_COMMIT_CONFIG_PATH = ROOT_PATH / '.pre-commit-config.yaml'
SKIP_PACKAGES = [
    'mypy',  # Mypy is part of typing env but not core dependencies
]


class _TestCaseTuple(NamedTuple):
    package_name: str
    project_specifier: str
    precommit_specifier: str


def _generate_test_cases():
    """Generate a list of dependency test cases.
    This function:
        (1) Generates a list of typing dependencies defined in `pyproject.toml`
        (2) Generates a list of mypy pre-commit dependencies from the pre-commit config
        (3) Merges the two lists together and returns separate test cases for
            each dependency.

    """
    test_cases_dict: dict = {}

    def add_to_dict(package_name: str, specifier: set[str], key: str):
        # Function for stuffing dependencies into a dict.
        # We use a dict to allow for any entry to be made based on package name alone.
        # This way, we can defer checking for any mismatch between the typing
        # dependencies and the pre-commit dependencies to test time.
        nonlocal test_cases_dict
        test_name = package_name
        test_cases_dict.setdefault(test_name, {})
        test_cases_dict[test_name].setdefault(key, specifier)

    # process project dependencies
    project_key = PYPROJECT_TOML_PATH.name
    typing_dependencies = _get_project_dependencies(
        PYPROJECT_TOML_PATH, extra=OPTIONAL_DEPENDENCIES
    )
    [
        add_to_dict(package, specifier, key=project_key)
        for package, specifier in typing_dependencies.items()
    ]

    # process pre-commit dependencies
    precommit_key = PRE_COMMIT_CONFIG_PATH.name
    precommit_dependencies = _get_mypy_precommit_dependencies(PRE_COMMIT_CONFIG_PATH)
    [
        add_to_dict(package, specifier, key=precommit_key)
        for package, specifier in precommit_dependencies.items()
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
    if 'dependency_test_case' in metafunc.fixturenames:
        test_cases = _generate_test_cases()
        ids = [case.package_name for case in test_cases]
        metafunc.parametrize('dependency_test_case', test_cases, ids=ids)


def _get_project_dependencies(file_path, extra: str | None = None) -> dict[str, set[str]]:
    """Parse dependencies in `pyproject.toml` file.

    A dict is returned with all package names as keys with a set of corresponding
    specifiers as its values.
    """

    with open(file_path, 'rb') as file:  # noqa: PTH123
        data = tomllib.load(file)
    project_dependencies = data['project']['dependencies']
    optional_dependencies = data['project']['optional-dependencies']

    deps_to_parse = project_dependencies
    if extra:
        deps_to_parse.extend(optional_dependencies[extra])
    dependencies: dict[str, set[str]] = {}
    for dependency in deps_to_parse:
        if 'extra ==' not in dependency:
            dep = dependency
        elif f'extra == "{extra}"' in dependency:
            dep = dependency.split(';')[0]
        else:
            continue

        if 'pyvista[' in dep:
            new_extra = dependency.split('[')[1].split(']')[0]
            dependencies.update(_get_project_dependencies(file_path, extra=new_extra))
            continue

        _add_dependency_to_dependencies_dict(dependencies, dep)
    return dependencies


def _add_dependency_to_dependencies_dict(dependencies: dict[str, set[str]], dep: str):
    dep = Requirement(dep)
    dependencies.setdefault(dep.name, set())
    specifier = str(dep.specifier)
    if specifier != '':
        if ',' in specifier:
            dependencies[dep.name].update(specifier.split(','))
        else:
            dependencies[dep.name].add(specifier)


def _get_mypy_precommit_dependencies(file_path):
    # Get pre-commit config
    with open(file_path) as file:  # noqa: PTH123
        config = yaml.safe_load(file)

    # Extract mypy hook dependencies
    for repo in config['repos']:
        if repo['repo'] == MYPY_PRECOMMIT_REPO:
            hook = repo['hooks'][0]
            assert hook['id'] == 'mypy'
            dependencies_list = hook['additional_dependencies']
            dependencies_dict = {}
            for item in dependencies_list:
                _add_dependency_to_dependencies_dict(dependencies_dict, item)
            return dependencies_dict

    raise RuntimeError(
        f'Pre-commit repo:\n\t{MYPY_PRECOMMIT_REPO}\n'
        f'does not exist in config:\n\t{PRE_COMMIT_CONFIG_PATH!s}\n'
    )


def test_dependency(dependency_test_case):
    package_name, project_specifier, precommit_specifier = dependency_test_case
    if package_name in SKIP_PACKAGES:
        pytest.skip('Known skips.')

    msg = _test_both_dependencies_exist(*dependency_test_case)
    if msg:
        pytest.fail(msg)
    if project_specifier != precommit_specifier:
        project_specifier_str = ','.join(str(s) for s in project_specifier)
        precommit_specifier_str = ','.join(str(s) for s in precommit_specifier)
        pytest.fail(
            f"The `mypy` pre-commit dependency '{package_name+precommit_specifier_str}' in '{PRE_COMMIT_CONFIG_PATH.name}'\n'"
            f"must match the `pyvista[typing]` dependency '{package_name+project_specifier_str}' in '{PYPROJECT_TOML_PATH.name}'."
        )


def _test_both_dependencies_exist(package_name, project_dependency, precommit_dependency):
    if project_dependency is None or precommit_dependency is None:
        if project_dependency is None:
            assert precommit_dependency is not None
            missing = f"`typing` dependencies in '{PYPROJECT_TOML_PATH.name}'"
            exists = f"'{PRE_COMMIT_CONFIG_PATH.name}'"
            action_precommit = 'removed from'
            action_project = 'added to'
        else:
            assert project_dependency is not None
            missing = f"'{PRE_COMMIT_CONFIG_PATH.name}'"
            exists = f"`typing` dependencies in '{PYPROJECT_TOML_PATH.name}'"
            action_precommit = 'added to'
            action_project = 'removed from'
        return (
            f"Test setup failed for package '{package_name}'.\n"
            f'The dependency exists in {exists} but is missing from {missing}.\n'
            f'The package should be {action_precommit} the `pre-commit` `mypy` config '
            f'or {action_project} the project dependencies.'
        )
    return None
