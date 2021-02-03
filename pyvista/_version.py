"""Version info for pyvista.

On the ``master`` branch, use 'dev0' to denote a development version.
For example:

version_info = 0, 27, 'dev0'

When generating pre-release wheels, use '0rcN', for example:

version_info = 0, 28, '0rc1'

"""
# major, minor, patch
version_info = 0, 28, 1

# Nice string for the version
__version__ = '.'.join(map(str, version_info))
