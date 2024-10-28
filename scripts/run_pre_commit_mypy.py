# from __future__ import annotations
#
# from pathlib import Path
# import subprocess
#
# from packaging.requirements import Requirement
# import yaml
#
# MYPY = 'mypy'
# MYPY_TEMPLATE = MYPY + '_template'
# TARGET_REPO = 'https://github.com/pyvista/pre-commit-mypy'
# MYPY_ARGS = ['pyvista']  # List of pre-commit `args` passed when executing `mypy`
#
#
# def main():
#     # def _get_compatible_version(package_name, requirement):
#     #     # Run pip to get available versions of `package_name`
#     #     result = subprocess.run(
#     #         ["pip", "index", "versions", package_name],
#     #         capture_output=True,
#     #         text=True
#     #     )
#     #
#     #     if result.returncode != 0:
#     #         print("Error fetching package versions.")
#     #         return None
#     #
#     #     # Parse the versions output from pip
#     #     lines = result.stdout.splitlines()
#     #     versions = []
#     #     for line in lines:
#     #         if line.startswith("Available versions:"):
#     #             versions = line.split("Available versions: ")[1].split(", ")
#     #             break
#     #
#     #     if not versions:
#     #         print("No versions found for the package.")
#     #         return None
#     #
#     #     # Check each version to see if it meets the requirement
#     #     specifier = SpecifierSet(requirement)
#     #     for version_str in versions:
#     #         version = Version(version_str)
#     #         if version in specifier:
#     #             return str(version)
#     #
#     #     print("No compatible version found.")
#     #     return None
#
#     def _parse_requirements_file(file_path):
#         requirements = {}
#         with open(file_path) as file:
#             for line in file:
#                 line = line.strip()
#
#                 # Update dict from file
#                 r_arg = '-r'
#                 if line.startswith(r_arg):
#                     new_file_path = line[len(r_arg) :].strip()
#                     more_requirements = _parse_requirements_file(new_file_path)
#                     requirements.update(more_requirements)
#                     continue
#                 # Skip comments and empty lines
#                 if not line or line.startswith('#'):
#                     continue
#
#                 # Parse each requirement line
#                 req = Requirement(line)
#                 requirements[req.name] = str(req.specifier) if req.specifier else None
#         return requirements
#
#     # Get requirements
#     requirements = _parse_requirements_file('requirements_test.txt')
#
#     # Get pre-commit config
#     project_root = Path(__file__).parent.parent
#     pre_commit_config_filepath = project_root / '.pre-commit-config.yaml'
#     with open(pre_commit_config_filepath) as file:
#         config = yaml.safe_load(file)
#
#     # Extract the mypy hook config and update
#     new_config = None
#     for repo in config['repos']:
#         if repo['repo'] == TARGET_REPO:
#             # Update repo config
#             hook = repo['hooks'][0]
#             assert hook['id'] == MYPY_TEMPLATE
#             requirements_list = [key + value for key, value in requirements.items()]
#             requirements_list.append('vtk')
#             hook['additional_dependencies'] = requirements_list
#             hook['args'] = ['pyvista']
#
#             new_config = {'repos': [repo]}
#             break
#     if new_config is None:
#         raise RuntimeError(f'Pre-commit repo:\n\t{TARGET_REPO}\n'
#                            f'does not exist in config:\n\t{str(pre_commit_config_filepath)}\n')
#
#
#     # Writing YAML data to a file
#     new_config_filepath = project_root / '.pre-commit-config-mypy.yaml'
#     with open(new_config_filepath, 'w') as file:
#         yaml.dump(new_config, file)
#
#     subprocess.run(
#         ['pre-commit', 'run', '--hook-stage', 'manual', '--config', new_config_filepath, MYPY_TEMPLATE]
#     )
#
#
# if __name__ == '__main__':
#     main()
#
#
#
#
#
#
#
#
#
#
#
"""Run `mypy` executable using a templated pre-commit hook.

This script reads the `.pre-commit-config.yaml` file and populates the templated
values for the `mypy_template` hook and then runs it.

`requirements_test.txt` is parsed to fill the 'additional_dependencies' field
`args` is also populated.
"""

from __future__ import annotations

from pathlib import Path
import subprocess

import yaml

# Need `packaging` to parse project requirements
subprocess.run(
    ['pip', 'install', '--upgrade', 'packaging'],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.STDOUT,
)
from packaging.requirements import Requirement

MYPY = 'mypy'
REPO = 'https://github.com/pyvista/pre-commit-mypy'
REQUIREMENTS_FILE = 'requirements_typing.txt'

import sys


def print_error(*args, **kwargs):
    """Print to stderr."""
    print(*args, file=sys.stderr, **kwargs)


def main():
    """Read `mypy` config, populate dependencies, and run hook."""

    def _parse_requirements_file(file_path):
        requirements = {}
        with open(file_path) as file:  # noqa: PTH123
            for line in file:
                line = line.strip()

                # Update dict from file
                r_arg = '-r'
                if line.startswith(r_arg):
                    new_file_path = line[len(r_arg) :].strip()
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

    # Get requirements
    requirements = _parse_requirements_file(REQUIREMENTS_FILE)

    # Get pre-commit config
    project_root = Path(__file__).parent.parent
    pre_commit_config_filepath = project_root / '.pre-commit-config.yaml'
    with open(pre_commit_config_filepath) as file:  # noqa: PTH123
        config = yaml.safe_load(file)

    # Extract the mypy hook config and update it
    new_config = None
    for repo in config['repos']:
        if repo['repo'] == REPO:
            # Update repo config
            hook = repo['hooks'][0]
            assert hook['id'] == MYPY
            requirements_list = [key + value for key, value in requirements.items()]
            requirements_list.append('vtk')
            hook['additional_dependencies'] = requirements_list

            new_config = {'repos': [repo]}
            break
    if new_config is None:
        raise RuntimeError(
            f'Pre-commit repo:\n\t{REPO}\n'
            f'does not exist in config:\n\t{pre_commit_config_filepath!s}\n'
        )

    # Writing YAML data to a file
    new_config_filepath = project_root / '.pre-commit-config-mypy.yaml'
    with open(new_config_filepath, 'w') as file:  # noqa: PTH123
        yaml.dump(new_config, file)

    subprocess.run(
        [
            'pre-commit',
            'run',
            '--verbose',
            '--hook-stage',
            'manual',
            '--config',
            new_config_filepath,
            MYPY,
        ],
        stderr=subprocess.STDOUT,
    )


if __name__ == '__main__':
    main()
