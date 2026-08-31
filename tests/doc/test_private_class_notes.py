"""Private classes documented for their inherited public members carry a note.

Since private classes are documented, a reader can land on ``_BoundsSizeMixin``
with nothing saying why a private name has a page. Any private class whose own
public members are inherited by a public class has to say so.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil

import pytest

import pyvista as pv

NOTE = 'private internal implementation detail'


def _all_classes():
    """Return every class defined anywhere in ``pyvista``."""
    modules = [pv]
    for info in pkgutil.walk_packages(pv.__path__, f'{pv.__name__}.'):
        try:
            modules.append(importlib.import_module(info.name))
        except Exception:  # noqa: BLE001, S112  # pragma: no cover
            continue  # a module whose optional dependency is missing
    classes = {}
    for module in modules:
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if getattr(obj, '__module__', '').startswith(pv.__name__):
                classes[obj] = obj
    return list(classes)


def _own_public_members(cls) -> list[str]:
    """Return the public members ``cls`` itself defines."""
    return sorted(
        name
        for name, value in vars(cls).items()
        if not name.startswith('_')
        and (
            inspect.isfunction(value)
            or isinstance(value, (property, staticmethod, classmethod))
            or type(value).__name__ in ('_classproperty', 'cached_property')
        )
    )


def _classes_needing_the_note():
    """Return ``{private class: public subclasses}`` for classes the note applies to."""
    classes = _all_classes()
    needed = {}
    for cls in classes:
        if not cls.__name__.startswith('_') or not _own_public_members(cls):
            continue
        subclasses = sorted(
            {
                other.__name__
                for other in classes
                if not other.__name__.startswith('_') and other is not cls and cls in other.__mro__
            }
        )
        if subclasses:
            needed[cls] = subclasses
    return needed


@pytest.fixture(scope='module')
def needs_note():
    return _classes_needing_the_note()


def test_some_classes_need_the_note(needs_note):
    """Guard the test above: the set it checks must not silently become empty."""
    assert len(needs_note) > 10


def test_private_classes_exposing_public_members_carry_the_note(needs_note):
    missing = sorted(
        f'{cls.__module__}.{cls.__name__} (inherited by {", ".join(subs[:3])})'
        for cls, subs in needs_note.items()
        if NOTE not in (cls.__doc__ or '')
    )
    assert not missing, (
        'These private classes expose public members through public subclasses, '
        'so their docstring must contain a note saying so:\n  ' + '\n  '.join(missing)
    )
