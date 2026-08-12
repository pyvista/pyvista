"""Sphinx autodoc documenter for :class:`enum.Enum` subclasses.

Replaces the third-party ``enum_tools.autoenum`` extension, which has three
problems this module is built to avoid:

* It renders any non-instance class attribute (e.g. a ``@property`` defined on
  a class's *metaclass*, such as :attr:`~pyvista.CellType.dimension_map`) by
  dumping its ``repr()``, since by the time ``getattr(cls, name)`` returns, the
  fact that the value came from a property with its own docstring is gone --
  see :func:`_metaclass_properties` below.
* It reprs the *member itself* for each value's ``:value:`` field
  (``EMPTY_CELL = <CellType.EMPTY_CELL: 0>``) instead of the plain value.
* It registers a new ``enum`` autodoc objtype, which ``sphinx.ext.autosummary``
  has no logic to populate the ``methods``/``attributes`` template variables
  for (see ``sphinx/ext/autosummary/generate.py``, the
  ``elif doc.objtype == 'class':`` branch, keyed on that exact string) --
  so PyVista needed its own ``enum.rst`` autosummary template, which in turn
  needed a hardcoded, per-class list of extra attributes to document.

This extension still registers a dedicated ``enum`` objtype/``autoenum``
directive -- mirroring ``sphinx.ext.autodoc``'s own built-in
``ExceptionDocumenter``, which does the same thing for ``exception`` (also
without a matching ``exception.rst``, falling back to the generic
``base.rst``). There's no way to make an Enum subclass reuse the stock
``class`` objtype's autosummary handling without overriding Sphinx's built-in
class documenter for *every* class in the docs (risky and invasive) or
monkeypatching ``sphinx.ext.autosummary.generate`` (fragile). Both are worse
than owning a small, generic ``enum.rst`` -- see
``doc/source/_templates/autosummary/enum.rst``, which (unlike the template it
replaces) no longer hardcodes a per-class attribute list: it calls
:func:`metaclass_property_names` through ``autosummary_context`` instead.

Modeled on the Sphinx custom-documenter tutorial:
https://www.sphinx-doc.org/en/master/development/tutorials/autodoc_ext.html
"""

from __future__ import annotations

import enum
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

#: Metaclasses that are never the source of a "class property" -- ``type`` never
#: defines one, and ``EnumMeta``/``EnumType`` is the (non-custom) metaclass every
#: :class:`~enum.Enum` already has, so it would otherwise be walked for every enum.
_SKIP_METACLASSES = (type, enum.EnumMeta)


def _is_enum(obj: Any) -> bool:
    """Return whether ``obj`` is an :class:`~enum.Enum` subclass."""
    return isinstance(obj, type) and issubclass(obj, enum.Enum)


def _metaclass_properties(cls: type) -> dict[str, property]:
    """Return ``{name: property}`` for every ``property`` defined on ``cls``'s metaclass.

    A ``@property`` defined on a class's *metaclass* -- rather than on the class itself --
    evaluates eagerly when accessed as ``cls.name`` instead of returning a descriptor. This
    is a common trick to get a real class-level computed attribute (plain ``@property``
    only computes on *instance* access), e.g. :attr:`pyvista.CellType.dimension_map`. But it
    means that by the time ``getattr(cls, name)`` returns, the fact that the value came from
    a property -- with its own docstring, separate from the evaluated value -- is gone:
    autodoc just sees a plain (possibly huge) value sitting in the class's namespace, and
    reprs it verbatim.
    """
    properties: dict[str, property] = {}
    for meta in type(cls).__mro__:
        if meta in _SKIP_METACLASSES:
            continue
        for name, value in vars(meta).items():
            if isinstance(value, property) and name not in properties:
                properties[name] = value
    return properties


def metaclass_property_names(module: str, objname: str) -> list[str]:
    """Return the sorted names of ``objname``'s metaclass properties.

    Called from ``enum.rst`` (via ``autosummary_context``) so the template does not need to
    hardcode a per-class attribute list the way the old template did. Takes the same
    ``module``/``objname`` strings the template already has -- autosummary template context
    is string-only, it never hands templates the live object -- and imports the class itself,
    the same way ``.. autoenum:: {{ objname }}`` already does.
    """
    import importlib  # noqa: PLC0415

    cls = importlib.import_module(module)
    for part in objname.split('.'):
        cls = getattr(cls, part)
    return sorted(_metaclass_properties(cls))


def _is_bitmask_like(cls: type[enum.Enum]) -> bool:
    """Return whether every member of ``cls`` looks like a bit flag (0 or a power of two)."""
    if issubclass(cls, enum.Flag):
        return True
    values = [int(member.value) for member in cls]
    return bool(values) and all(v == 0 or (v & (v - 1)) == 0 for v in values)


def _format_value(value: int, *, as_hex: bool) -> str:
    """Format an enum member's value the way it should appear after ``:value:``."""
    return hex(value) if as_hex else str(value)


class MetaclassPropertyDocumenter(PropertyDocumenter):
    """Documents a single ``property`` defined on a class's metaclass.

    Never resolved through the usual priority-based member dispatch (nothing here calls
    ``can_document_member`` for it) -- :class:`EnumDocumenter` constructs it directly for
    every name :func:`_metaclass_properties` finds, once it already has the class and the
    property object in hand. The only piece that needs to change from stock
    :class:`~sphinx.ext.autodoc.PropertyDocumenter` is ``import_object``: the standard
    implementation resolves ``self.object`` with a plain attribute walk, which -- same as
    everywhere else in this module -- would silently re-evaluate the metaclass property and
    hand us the computed value instead of the descriptor. Everything else (docstring
    extraction/processing, ``:type:`` from the getter's return annotation, ...) is inherited
    unchanged, since once ``self.object`` is the real ``property``, it *is* a normal one.
    """

    objtype = 'metaclassproperty'
    directivetype = 'property'

    def import_object(self, raiseerror: bool = False) -> bool:
        # Walk the module/class path as normal so self.module, self.parent, and the
        # analyzer are set up correctly. This *does* resolve self.object to the evaluated
        # value (not the property) -- that part is simply discarded below.
        ClassLevelDocumenter.import_object(self, raiseerror=False)
        if self.parent is None:
            return False
        name = self.objpath[-1]
        prop = _metaclass_properties(self.parent).get(name)
        if prop is None:
            logger.warning(
                '%s is not a metaclass property of %r; not documenting it.',
                name,
                self.parent,
                type='autoenum',
            )
            return False
        self.object = prop
        self.isclassmethod = True
        return True


class EnumDocumenter(ClassDocumenter):
    """Documents an :class:`~enum.Enum` subclass.

    Registered as its own ``enum`` objtype/``autoenum`` directive (see the module
    docstring for why), but with ``directivetype = 'class'`` so it still emits a plain
    ``.. py:class::`` -- keeping ``:class:`~pyvista.CellType``` and friends, inheritance
    diagrams, and domain indexing identical to an ordinary class. Only the *autodoc*
    dispatch (this class) and the *autosummary* template selection (``enum.rst``) are enum
    specific.
    """

    objtype = 'enum'
    directivetype = 'class'
    # Must outrank ClassDocumenter (15) so real Enum subclasses are routed here instead.
    priority = ClassDocumenter.priority + 5

    @classmethod
    def can_document_member(
        cls: type[ClassDocumenter], member: Any, membername: str, isattr: bool, parent: Any
    ) -> bool:
        return _is_enum(member)

    def filter_members(
        self, members: ObjectMembers, want_all: bool
    ) -> list[tuple[str, Any, bool]]:
        # Enum members and metaclass properties are documented by hand in add_content --
        # keep them out of the normal member pipeline so they aren't *also* auto-documented
        # (with the wrong formatting) as generic class attributes.
        if not _is_enum(self.object):
            return super().filter_members(members, want_all)

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

        # self.real_modname is only set partway through Documenter._generate(); add_content()
        # (and so this method) always runs after that point in practice, but fall back to
        # self.modname rather than assume it, since nothing guarantees that ordering across
        # Sphinx versions.
        real_modname = getattr(self, 'real_modname', None) or self.modname

        for name in names:
            full_mname = f'{self.modname}::{".".join([*self.objpath, name])}'
            documenter = MetaclassPropertyDocumenter(self.directive, full_mname, self.indent)
            documenter.generate(all_members=True, real_modname=real_modname, check_module=False)


def setup(app: Sphinx) -> dict[str, Any]:
    """Register the ``autoenum`` directive."""
    app.add_autodocumenter(EnumDocumenter)
    app.add_autodocumenter(MetaclassPropertyDocumenter)
    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
