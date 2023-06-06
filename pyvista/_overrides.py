"""Functionality to override PyVista object creation with other subclasses."""
from functools import wraps
import inspect

from pyvista.core import DataSet


def set_up_overrides(root=DataSet):
    """Set up type overriding for all subclasses of a PyVista type.

    Setup entails recursively finding all subclasses of the root
    type and defining an optional ``type_override`` class
    attribute on each. __new__ is decorated for each type to
    return the overriding type if provided.

    Parameters
    ----------
    root : type, default: pyvista.DataSet
        Parent type whose (recursive) subclasses will be set up.
        Root itself is included.
    """
    for cls in OVERRIDABLES:
        cls.type_override = None
        old_new = cls.__new__

        @wraps(old_new)
        def new_new(cls, *args, __old_new=old_new, **kwargs):
            other_cls = cls.type_override
            if other_cls is None or other_cls is cls:
                return __old_new(cls, *args, **kwargs)
            # non-trivial override is set: return other type
            instance = other_cls.__new__(other_cls, *args, **kwargs)
            return instance

        new_new.__signature__ = inspect.signature(cls.__init__)

        cls.__new__ = new_new


def _find_subclasses(root):
    """Recursively build a set of subclasses for a root type.

    Parameters
    ----------
    root : type
        Root type to find subclasses for.

    Returns
    -------
    subclasses : set
        Set of types containing (recursive) subclasses. Includes
        the root type.
    """
    return {root}.union(*(_find_subclasses(child) for child in root.__subclasses__()))


# mainly exposed for ease of resetting:
OVERRIDABLES = frozenset(_find_subclasses(DataSet))
