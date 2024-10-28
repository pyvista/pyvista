"""Run `mypy` executable using a templated pre-commit hook.

This script reads the `.pre-commit-config.yaml` file and populates the templated
values for the `mypy_template` hook and then runs it.

`requirements_test.txt` is parsed to fill the 'additional_dependencies' field
`args` is also populated.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

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
ROOT_PATH = Path(__file__).parent.parent
PRE_COMMIT_CONFIG_PATH = ROOT_PATH / '.pre-commit-config.yaml'
PRE_COMMIT_MYPY_CONFIG_PATH = ROOT_PATH / '.pre-commit-config-mypy.yaml'


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
    with open(PRE_COMMIT_CONFIG_PATH) as file:  # noqa: PTH123
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
            f'does not exist in config:\n\t{PRE_COMMIT_CONFIG_PATH!s}\n'
        )

    # Save new mypy config file
    with open(PRE_COMMIT_MYPY_CONFIG_PATH, 'w') as file:  # noqa: PTH123
        yaml.dump(new_config, file)

    subprocess.check_call(
        [
            'pre-commit',
            'run',
            '--verbose',
            '--hook-stage',
            'manual',
            '--config',
            PRE_COMMIT_MYPY_CONFIG_PATH,
            MYPY,
        ],
        stdout=sys.stdout,
        stderr=subprocess.STDOUT,
    )


if __name__ == '__main__':
    main()
