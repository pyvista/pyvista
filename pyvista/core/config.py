"""Global configuration for PyVista core (non-plotting) behavior.

This module exposes :data:`pyvista.global_config`, a singleton :class:`Config`
instance that holds process-wide settings for the ``pyvista.core`` layer. It is
a sibling to :data:`pyvista.global_theme` (defined in
:mod:`pyvista.plotting.themes`) and shares the same machinery: both inherit
from :class:`_ConfigBase` and behave the same way for attribute access,
dict-style item access, ``to_dict`` / ``from_dict`` serialization, and equality
comparison.

The base class lives here in :mod:`pyvista.core` rather than in
:mod:`pyvista.plotting` so that the core layer does not depend on plotting.
:mod:`pyvista.plotting.themes` imports :class:`_ConfigBase` from here to build
the plotting theme hierarchy.

To add a new core setting:

1. Add an underscore-prefixed slot to ``Config.__slots__``.
2. Initialize it in :meth:`Config.__init__`.
3. Expose it via a public ``@property`` getter / setter pair that reads and
   writes the underscore slot. The setter should validate its input.

That is the same pattern used by every theme subclass, so the two hierarchies
stay symmetrical and ``to_dict`` / ``from_dict`` round-tripping works without
any extra code.

Examples
--------
Disable the default array-length check that :func:`pyvista.wrap` performs on
every VTK object it wraps:

>>> import pyvista as pv
>>> pv.global_config.validate_on_wrap = False
>>> pv.global_config.validate_on_wrap = True  # restore default

Access a setting via dict-style lookup:

>>> pv.global_config['validate_on_wrap']
True

Show the VTK-inherited API in :func:`dir` (and IDE tab-completion):

>>> pv.global_config.show_vtk_api = True
>>> pv.global_config.show_vtk_api = False  # restore default

Dump the current config to a plain dict (useful for logging or round-tripping):

>>> pv.global_config.to_dict()
{'show_vtk_api': False, 'validate_on_wrap': True}

"""

from __future__ import annotations

from itertools import chain
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar

if TYPE_CHECKING:
    from typing_extensions import Self


# Mostly from https://stackoverflow.com/questions/56579348/how-can-i-force-subclasses-to-have-slots
class _ForceSlots(type):
    """Metaclass to force classes and subclasses to have ``__slots__``."""

    @classmethod
    def __prepare__(  # type: ignore[override]
        cls,
        name: str,
        bases: tuple[type, ...],
        **kwargs: Any,
    ) -> dict[str, Any]:
        super_prepared = super().__prepare__(cls, name, bases, **kwargs)  # type: ignore[arg-type, call-arg, misc]
        super_prepared['__slots__'] = ()
        return super_prepared  # type: ignore[return-value]


class _ConfigBase(metaclass=_ForceSlots):
    """Shared base class for PyVista config objects.

    Provides dict-style item access, ``from_dict`` / ``to_dict`` serialization,
    and equality comparison. Used as the base for both the core
    :class:`Config` (this module) and every node of the plotting
    :class:`pyvista.plotting.themes.Theme` hierarchy.

    Subclasses must list every attribute as an underscore-prefixed entry in
    their ``__slots__`` and expose each one via a public ``@property`` getter
    / setter pair that reads and writes the underscore slot.

    """

    __slots__: ClassVar[list[str]] = []

    # Slot names (without the leading underscore) that should be omitted from
    # ``to_dict`` output even though they live in ``__slots__``. Subclasses may
    # override this with their own ``frozenset``.
    _TO_DICT_SKIP: ClassVar[frozenset[str]] = frozenset()

    @classmethod
    def from_dict(cls, dict_: dict[str, Any]) -> Self:
        """Create an instance from a dictionary of attribute values.

        Parameters
        ----------
        dict_ : dict
            Mapping of public attribute name to value, as produced by
            :meth:`to_dict`. Nested config objects are recursively
            reconstructed via their own ``from_dict``.

        Returns
        -------
        Self
            New instance of ``cls`` populated from ``dict_``.

        """
        inst = cls()
        for key, value in dict_.items():
            attr = getattr(inst, key)
            if hasattr(attr, 'from_dict'):
                setattr(inst, key, attr.from_dict(value))
            else:
                setattr(inst, key, value)
        return inst

    def to_dict(self) -> dict[str, Any]:
        """Return config parameters as a dictionary.

        Returns
        -------
        dict
            Mapping of public attribute name to its current value. Nested
            config objects are recursively serialized via their own
            ``to_dict``. Names listed in :attr:`_TO_DICT_SKIP` are omitted.

        """
        skip = type(self)._TO_DICT_SKIP
        dict_: dict[str, Any] = {}
        for key in self._all__slots__():
            name = key[1:]  # strip the leading underscore
            if name in skip:
                continue
            value = getattr(self, key)
            if hasattr(value, 'to_dict'):
                dict_[name] = value.to_dict()
            else:
                dict_[name] = value
        return dict_

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _ConfigBase):
            return False
        for attr_name in other._all__slots__():
            attr = getattr(self, attr_name)
            other_attr = getattr(other, attr_name)
            if (
                isinstance(attr, (tuple, list)) and tuple(attr) != tuple(other_attr)
            ) or not attr == other_attr:
                return False
        return True

    __hash__ = None  # type: ignore[assignment]  # https://github.com/pyvista/pyvista/pull/7671

    def __getitem__(self, key: str) -> Any:
        """Get a value via a key (backwards-compatible dict access)."""
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Set a value via a key (backwards-compatible dict access)."""
        setattr(self, key, value)

    @classmethod
    def _all__slots__(cls) -> tuple[str, ...]:
        """Return all slot names including parent classes."""
        mro = cls.mro()
        return tuple(chain.from_iterable(c.__slots__ for c in mro if c is not object))  # type: ignore[attr-defined]


class Config(_ConfigBase):
    """PyVista core configuration.

    Holds process-wide settings that affect ``pyvista.core`` behavior. The
    singleton instance is exposed as ``pyvista.global_config``. This is the
    sibling of ``pyvista.global_theme`` for plotting (rendering) settings.

    Examples
    --------
    Disable the default array-length check performed by :func:`pyvista.wrap`:

    >>> import pyvista as pv
    >>> pv.global_config.validate_on_wrap = False
    >>> pv.global_config.validate_on_wrap = True  # restore default

    """

    __slots__ = ['_show_vtk_api', '_validate_on_wrap']

    def __init__(self) -> None:
        self._validate_on_wrap: bool = True
        self._show_vtk_api: bool = False

    def __repr__(self) -> str:
        header = 'PyVista Config'
        lines = [header, '-' * len(header)]
        lines.extend(f'{key[1:]:<25}: {getattr(self, key)}' for key in self._all__slots__())
        return '\n'.join(lines)

    @property
    def validate_on_wrap(self) -> bool:  # numpydoc ignore=RT01
        """Return or set whether :func:`pyvista.wrap` validates data arrays.

        When ``True`` (the default), :func:`pyvista.wrap` performs a cheap
        array-length sanity check on every VTK object it wraps and emits a
        :class:`~pyvista.InvalidMeshWarning` if any point or cell data array
        has a tuple count that does not match the dataset's point or cell
        count. Set to ``False`` to skip this check globally when the cost
        matters in tight loops and the caller trusts their inputs.

        Notes
        -----
        Per-call control is also available via the ``validate`` keyword on
        :func:`pyvista.wrap`, :func:`pyvista.read`, and
        :meth:`pyvista.BaseReader.read`. The per-call keyword takes
        precedence; this global setting is consulted only when the per-call
        keyword is left at its default ``None``.

        .. versionadded:: 0.48

        Examples
        --------
        >>> import pyvista as pv
        >>> pv.global_config.validate_on_wrap
        True
        >>> pv.global_config.validate_on_wrap = False
        >>> pv.global_config.validate_on_wrap = True  # restore default

        """
        return self._validate_on_wrap

    @validate_on_wrap.setter
    def validate_on_wrap(self, value: bool) -> None:
        # Defensive runtime check for dynamic call sites (e.g. JSON-driven
        # configuration). Static callers are already constrained by the
        # ``bool`` annotation above, so mypy treats the branch below as
        # unreachable.
        if not isinstance(value, bool):
            msg = f'`validate_on_wrap` must be a bool, got {type(value).__name__}.'  # type: ignore[unreachable]
            raise TypeError(msg)
        self._validate_on_wrap = value

    @property
    def show_vtk_api(self) -> bool:  # numpydoc ignore=RT01
        """Return or set whether VTK-inherited attributes appear in :func:`dir`.

        When ``False`` (the default), attributes inherited from VTK base
        classes are hidden from :func:`dir` and tab-completion on PyVista
        objects that wrap VTK types (data objects, :class:`~pyvista.Renderer`,
        :class:`~pyvista.Actor`, :class:`~pyvista.Property`, etc.). This keeps
        the public surface curated for data-science IDEs such as Positron's
        Variables pane and VS Code's Jupyter extension, and for IPython /
        Jupyter tab-completion. VTK methods remain fully callable regardless
        of this setting.

        When ``True``, the full VTK API is enumerated alongside the PyVista
        API, which is useful for VTK developers who want to discover the raw
        VTK method surface via introspection.

        Notes
        -----
        The snake_case VTK aliases (``get_bounds``, ``deep_copy``, ...) are
        controlled separately by :func:`pyvista.vtk_snake_case`. When
        snake_case is not ``'allow'`` (the default), those names are hidden
        from :func:`dir` regardless of this setting, because accessing them
        would already raise ``PyVistaAttributeError``.
        Enabling snake_case surfaces the snake_case names in :func:`dir`;
        ``show_vtk_api`` only controls the CamelCase VTK API.

        .. versionadded:: 0.48

        Examples
        --------
        >>> import pyvista as pv
        >>> pv.global_config.show_vtk_api
        False
        >>> pv.global_config.show_vtk_api = True
        >>> pv.global_config.show_vtk_api = False  # restore default

        """
        return self._show_vtk_api

    @show_vtk_api.setter
    def show_vtk_api(self, value: bool) -> None:
        if not isinstance(value, bool):
            msg = f'`show_vtk_api` must be a bool, got {type(value).__name__}.'  # type: ignore[unreachable]
            raise TypeError(msg)
        self._show_vtk_api = value


global_config = Config()
