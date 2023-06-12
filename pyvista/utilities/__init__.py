"""Deprecated utilities subpackage."""
from ._getattr import _getattr_factory

__getattr__ = _getattr_factory(globals())
