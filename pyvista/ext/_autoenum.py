"""Sphinx autodoc documenter for ``Enum`` subclasses, replacing ``enum_tools.autoenum``.

Modeled on
https://www.sphinx-doc.org/en/master/development/tutorials/autodoc_ext.html
"""

from __future__ import annotations

from enum import Enum
from enum import EnumMeta
from enum import Flag
import inspect
from typing import TYPE_CHECKING
from typing import Any

from sphinx.ext.autodoc import ClassDocumenter
from sphinx.ext.autodoc import ClassLevelDocumenter
from sphinx.ext.autodoc import ObjectMember
from sphinx.ext.autodoc import PropertyDocumenter
from sphinx.util import logging

if TYPE_CHECKING:
    from sphinx.application import Sphinx
    from sphinx.ext.autodoc import ObjectMembers

logger = logging.getLogger(__name__)

_SKIP_METACLASSES = (type, EnumMeta)


def _is_enum(obj: Any) -> bool:
    """Return whether ``obj`` is an ``Enum`` subclass."""
    return isinstance(obj, type) and issubclass(obj, Enum)


def _metaclass_properties(cls: type) -> dict[str, property]:
    """Return ``{name: property}`` for every ``property`` defined on ``cls``'s metaclass.

    ``getattr(cls, name)`` evaluates a metaclass property eagerly, losing the descriptor
    (and its docstring) that produced the value -- so this has to look at the metaclass
    directly instead.
    """
    properties: dict[str, property] = {}
    for meta in type(cls).__mro__:
        if meta in _SKIP_METACLASSES:
            continue
        for name, value in vars(meta).items():
            if isinstance(value, property) and name not in properties:
                properties[name] = value
    return properties


def metaclass_property_names(module: str, objname: str) -> list[str]:  # numpydoc ignore=RT01
    """Return the sorted metaclass property names of ``module.objname``.

    Takes strings, not the live object, since that's all ``enum.rst`` (via
    ``autosummary_context``) has to give it.
    """
    import importlib  # noqa: PLC0415

    cls = importlib.import_module(module)
    for part in objname.split('.'):
        cls = getattr(cls, part)
    return sorted(_metaclass_properties(cls))


def _is_bitmask_like(cls: type[Enum]) -> bool:
    """Return whether every member of ``cls`` looks like a bit flag (0 or a power of two)."""
    if issubclass(cls, Flag):
        return True
    values = [int(member.value) for member in cls]
    return bool(values) and all(v == 0 or (v & (v - 1)) == 0 for v in values)


def _format_value(value: int, *, as_hex: bool) -> str:
    """Format an enum member's value the way it should appear after ``:value:``."""
    return hex(value) if as_hex else str(value)


class MetaclassPropertyDocumenter(PropertyDocumenter):
    """Documents a ``property`` defined on a class's metaclass.

    Constructed directly by :class:`EnumDocumenter`, not resolved via member dispatch.
    Only ``import_object`` needs to change from stock ``PropertyDocumenter``: fetch the
    property itself rather than its (eagerly evaluated) value.
    """

    objtype = 'metaclassproperty'
    directivetype = 'property'

    def import_object(self, raiseerror: bool = False) -> bool:
        ClassLevelDocumenter.import_object(self, raiseerror=False)
        if self.parent is None:
            return False
        name = self.objpath[-1]
        prop = _metaclass_properties(self.parent).get(name)
        if prop is None:
            logger.warning('%s is not a metaclass property of %r.', name, self.parent)
            return False
        self.object = prop
        self.isclassmethod = True
        return True


class EnumDocumenter(ClassDocumenter):
    """Documents an ``Enum`` subclass, as its own ``enum`` objtype/``autoenum`` directive.

    ``directivetype`` stays ``'class'`` so xrefs and domain indexing match a normal class.
    """

    objtype = 'enum'
    directivetype = 'class'
    priority = ClassDocumenter.priority + 5

    @classmethod
    def can_document_member(
        cls: type[ClassDocumenter], member: Any, membername: str, isattr: bool, parent: Any
    ) -> bool:
        return _is_enum(member)

    def filter_members(
        self, members: ObjectMembers, want_all: bool
    ) -> list[tuple[str, Any, bool]]:
        if not _is_enum(self.object):
            return super().filter_members(members, want_all)

        # Enum members and metaclass properties are documented by hand in add_content.
        skip_names = set(self.object.__members__) | set(_metaclass_properties(self.object))
        kept = [
            member
            for member in members
            if (member.__name__ if isinstance(member, ObjectMember) else member[0])
            not in skip_names
        ]
        return super().filter_members(kept, want_all)

    def add_content(self, more_content: Any) -> None:
        super().add_content(more_content)
        if not _is_enum(self.object):
            return
        sourcename = self.get_sourcename()
        self._document_members(sourcename)
        self._document_metaclass_properties(sourcename)

    def _document_members(self, sourcename: str) -> None:
        cls = self.object
        as_hex = _is_bitmask_like(cls)

        self.add_line('', sourcename)
        self.add_line('**Valid values are as follows:**', sourcename)
        self.add_line('', sourcename)

        self.indent += self.content_indent
        try:
            for member in cls:
                self.add_line(f'.. py:attribute:: {cls.__name__}.{member.name}', sourcename)
                self.add_line(
                    f'   :value: {_format_value(int(member.value), as_hex=as_hex)}',
                    sourcename,
                )
                self.add_line('', sourcename)

                doc = member.__doc__
                if doc and doc != type(member).__doc__:
                    for line in inspect.cleandoc(doc).splitlines():
                        self.add_line(f'   {line}'.rstrip(), sourcename)
                    self.add_line('', sourcename)
        finally:
            self.indent = self.indent[: -len(self.content_indent)]

    def _document_metaclass_properties(self, sourcename: str) -> None:
        names = sorted(_metaclass_properties(self.object))
        if not names:
            return

        self.add_line('', sourcename)
        self.add_line('**Class properties:**', sourcename)
        self.add_line('', sourcename)

        real_modname = getattr(self, 'real_modname', None) or self.modname
        for name in names:
            full_mname = f'{self.modname}::{".".join([*self.objpath, name])}'
            documenter = MetaclassPropertyDocumenter(self.directive, full_mname, self.indent)
            documenter.generate(all_members=True, real_modname=real_modname, check_module=False)


def setup(app: Sphinx) -> dict[str, Any]:  # numpydoc ignore=RT01
    """Register the ``autoenum`` directive."""
    app.add_autodocumenter(EnumDocumenter)
    app.add_autodocumenter(MetaclassPropertyDocumenter)
    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
