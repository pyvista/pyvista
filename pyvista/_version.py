"""Version info for pyvista.

On the ``main`` branch, use 'dev0' to denote a development version.
For example:

``version_info`` = 0, 27, 'dev0'

When generating pre-release wheels, use '0rcN', for example:

``version_info`` = 0, 28, '0rc1'

Denotes the first release candidate.

"""

# major, minor, patch
from __future__ import annotations

version_info = 0, 50, 'dev0'

# Nice string for the version
__version__ = '.'.join(map(str, version_info))


def _is_deprecation_due(version: tuple[int, int]) -> bool:
    """Check a release deadline, excluding its development versions."""
    return version_info[:2] > version or (
        version_info[:2] == version and not str(version_info[2]).startswith('dev')
    )
