"""Sphinx extension documenting ``_classproperty`` members as properties, not their values."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING
from typing import Any

from sphinx.ext.autosummary import generate
from sphinx.util.inspect import safe_getattr

from pyvista.core.utilities.misc import _classproperty

if TYPE_CHECKING:
    from sphinx.application import Sphinx


def _get_classproperty(cls: type, name: str) -> _classproperty | None:
    """Return the ``_classproperty`` that ``cls.name`` resolves to, or ``None``."""
    value = inspect.getattr_static(cls, name, None)
    return value if isinstance(value, _classproperty) else None


def classproperty_attrgetter(obj: type, name: str, *defargs: Any) -> Any:
    """Return ``obj.name``, or the ``_classproperty`` itself rather than its value."""
    prop = _get_classproperty(obj, name)
    return prop if prop is not None else safe_getattr(obj, name, *defargs)


def _patch_autosummary_content() -> None:
    """Make autosummary write ``autoproperty`` stubs for ``_classproperty`` members."""
    original = generate.generate_autosummary_content
    if getattr(original, '_classproperty_patch', False):
        return

    def generate_autosummary_content(
        name: str, obj: Any, parent: Any, *args: Any, **kwargs: Any
    ) -> str:
        """Pass the ``_classproperty`` instead of its value when ``name`` is one."""
        if isinstance(parent, type):
            prop = _get_classproperty(parent, name.rsplit('.', 1)[-1])
            if prop is not None:
                obj = prop
        return original(name, obj, parent, *args, **kwargs)

    generate_autosummary_content._classproperty_patch = True  # type: ignore[attr-defined]
    generate.generate_autosummary_content = generate_autosummary_content


def setup(app: Sphinx) -> dict[str, Any]:  # numpydoc ignore=RT01
    """Register the attribute getter and the autosummary stub patch."""
    app.add_autodoc_attrgetter(type, classproperty_attrgetter)
    _patch_autosummary_content()
    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
