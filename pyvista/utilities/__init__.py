"""Deprecated utilities subpackage."""
from ._getattr import _GetAttr

__getattr__ = _GetAttr(globals())
