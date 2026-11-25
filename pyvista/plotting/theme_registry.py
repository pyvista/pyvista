"""Registry for named pyvista themes."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING
from typing import Literal
from typing import NamedTuple
from typing import TypedDict

from pyvista._warn_external import warn_external
from pyvista.core.utilities._registry import _HandlerRegistry
from pyvista.core.utilities._registry import _HandlerRegistryState
from pyvista.core.utilities._registry import handler_source

if TYPE_CHECKING:
    from .themes import Theme


THEME_ENTRY_POINT_GROUP = 'pyvista.themes'


class ThemeRegistration(NamedTuple):
    """Describe how a theme name became registered.

    Returned by :func:`~pyvista.registered_themes`. ``kind`` distinguishes
    the registration path: a :class:`~pyvista.plotting.themes.Theme`
    subclass with a class-level ``_default_name`` (``'subclass'``), a
    ``pyvista.themes`` entry point from an installed plugin
    (``'entry_point'``), or a built-in legacy alias such as ``'default'``
    or ``'vtk'`` (``'alias'``).

    Attributes
    ----------
    name : str
        The registered (normalized) theme name.
    kind : {'subclass', 'entry_point', 'alias'}
        How the name was registered.
    source : str
        Human-readable origin (for example, ``'my_package.theme.MyTheme'``).

    Examples
    --------
    Inspect how a name became registered.

    >>> import pyvista as pv
    >>> next(r for r in pv.registered_themes() if r.name == 'dark')
    ThemeRegistration(name='dark', kind='subclass', source='pyvista.plotting.themes.DarkTheme')

    """

    name: str
    kind: Literal['subclass', 'entry_point', 'alias']
    source: str


class _ThemeRegistryState(TypedDict):
    """Stored registry state used by tests."""

    registry: _HandlerRegistryState
    aliases: set[str]


def _normalize_theme_name(name: str) -> str:
    """Normalize a theme name."""
    return name.strip().lower()


def _accept_theme(_name: str, obj: object) -> str | None:
    """Return why ``obj`` cannot serve as a theme, or ``None`` when it can."""
    from .themes import Theme  # local import breaks circular dependency  # noqa: PLC0415

    if isinstance(obj, Theme) or (isinstance(obj, type) and issubclass(obj, Theme)):
        return None
    return 'value is not a pyvista Theme subclass or instance.'


_registry: _HandlerRegistry[type[Theme] | Theme] = _HandlerRegistry(
    THEME_ENTRY_POINT_GROUP,
    label='theme',
    normalize=_normalize_theme_name,
    accept=_accept_theme,
    separate_discovered=True,
    unpack_mappings=True,
)
_registered_theme_classes = _registry.explicit
_discovered_entry_point_themes = _registry.discovered
_plugins = _registry.plugins
_registered_theme_aliases: set[str] = set()


def _save_registry_state() -> _ThemeRegistryState:
    """Snapshot the current registry state for later restoration."""
    return {'registry': _registry.snapshot(), 'aliases': _registered_theme_aliases.copy()}


def _restore_registry_state(state: _ThemeRegistryState) -> None:
    """Restore registry state from a snapshot."""
    _registry.restore(state['registry'])
    _registered_theme_aliases.clear()
    _registered_theme_aliases.update(state['aliases'])


def _register_theme_class(name: str, cls: type[Theme], *, source: str) -> None:
    """Register a ``Theme`` subclass. Called from ``Theme.__init_subclass__``.

    Collisions warn (do not raise) so that a subclass definition at
    import time cannot crash the interpreter. Last wins, mirroring the
    behavior of every other PyVista plugin registry.
    """
    if not _normalize_theme_name(name):
        warn_external(
            f'Theme class {source} declared an empty ``_default_name``; skipped.',
        )
        return
    _registry.register(name, cls, source)


def _register_alias(name: str, cls: type[Theme]) -> None:
    """Register a built-in name alias (for example, ``'default' → DocumentTheme``).

    Bypasses collision detection so built-in aliases can coexist with
    auto-registered subclasses without warning.
    """
    _registered_theme_aliases.add(
        _registry.register(name, cls, handler_source(cls), override=True)
    )


def _instantiate(theme: type[Theme] | Theme | None) -> Theme | None:
    """Return a fresh instance of a registered class, so callers never share one."""
    return theme() if isinstance(theme, type) else theme


def _resolve_theme(name: str) -> Theme | None:
    """Look up a theme by name and return a usable ``Theme`` instance.

    Explicit subclass registrations win over entry-point discoveries.
    Entry-point plugins are imported lazily—only the plugin claiming
    ``name`` loads, sibling plugins stay pending.
    """
    resolved = _instantiate(_registry.get(name))
    if resolved is not None:
        return resolved

    # A Mapping-style entry point may expose names that differ from its
    # ``ep.name``, so load the remaining pending plugins one by one.
    for pending_name in _plugins.pending_keys():
        _plugins.resolve(pending_name)
        resolved = _instantiate(_registry.get(name))
        if resolved is not None:
            return resolved
    return None


def _available_theme_names() -> tuple[str, ...]:
    """Return all currently registered theme names.

    Pending entry-point names appear in the result without triggering
    plugin imports—only metadata is consulted.
    """
    return tuple(sorted(_registry.keys()))


def registered_themes() -> tuple[ThemeRegistration, ...]:
    """Return all registered themes.

    Use this to discover which names can be passed to
    :func:`~pyvista.set_plot_theme` or the ``PYVISTA_PLOT_THEME``
    environment variable. Entry-point plugins are loaded on the first
    call so they appear in the result.

    .. versionchanged:: 0.50.0
        A plugin whose import failed stays pending and is retried on
        every call, so a theme whose dependency was installed after the
        failure becomes available again.

    Returns
    -------
    tuple[pyvista.ThemeRegistration, ...]
        One record per registered theme, sorted by name. Each record
        exposes ``name``, ``kind``, and ``source``.

    Examples
    --------
    List available theme names.

    >>> import pyvista as pv
    >>> for record in pv.registered_themes():
    ...     print(record.name)
    dark
    default
    document
    document_build
    document_pro
    paraview
    testing
    vtk

    Inspect how a name became registered.

    >>> next(r for r in pv.registered_themes() if r.name == 'dark').kind
    'subclass'

    """
    records = []
    for name, _theme, source in _registry.registered():
        kind: Literal['subclass', 'entry_point', 'alias'] = (
            'alias'
            if name in _registered_theme_aliases
            else 'subclass'
            if name in _registered_theme_classes
            else 'entry_point'
        )
        records.append(ThemeRegistration(name=name, kind=kind, source=source))
    return tuple(sorted(records, key=lambda r: r.name))


def _resolve_theme_like(theme: str | Theme) -> Theme:
    """Coerce a theme name, dotted path, or ``Theme`` instance to a ``Theme``.

    Shared by ``set_plot_theme`` and ``Plotter`` so name resolution and
    error messages stay consistent. Raises ``ValueError`` for an
    unresolvable string, ``TypeError`` for any other non-``Theme`` input.
    """
    from .themes import Theme  # local import breaks circular dependency  # noqa: PLC0415

    if isinstance(theme, str):
        if ':' in theme:
            cls = _resolve_dotted_path(theme)
            return cls()
        resolved = _resolve_theme(theme)
        if resolved is None:
            allowed = ', '.join(_available_theme_names())
            msg = (
                f'Theme "{theme}" not found. Available themes: {allowed}. '
                'To load from an arbitrary module use "package.module:ClassName".'
            )
            raise ValueError(msg)
        return resolved
    if isinstance(theme, Theme):
        return theme
    msg = (  # type: ignore[unreachable]
        f'Expected a ``pyvista.plotting.themes.Theme`` or ``str``, not {type(theme).__name__}'
    )
    raise TypeError(msg)


def _resolve_dotted_path(spec: str) -> type[Theme]:
    """Resolve a ``'package.module:ClassName'`` spec to a ``Theme`` subclass."""
    from .themes import Theme  # local import breaks circular dependency  # noqa: PLC0415

    if ':' not in spec:
        msg = f'Theme spec "{spec}" must use "package.module:ClassName" form.'
        raise ValueError(msg)
    module_path, _, class_name = spec.partition(':')
    if not module_path or not class_name:
        msg = f'Invalid theme spec "{spec}".'
        raise ValueError(msg)
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        msg = f'Cannot import "{module_path}" from theme spec "{spec}": {exc}'
        raise ValueError(msg) from exc
    cls = getattr(module, class_name, None)
    if cls is None:
        msg = f'"{class_name}" not found in module "{module_path}".'
        raise ValueError(msg)
    if not (isinstance(cls, type) and issubclass(cls, Theme)):
        # ValueError (not TypeError) keeps parity with the other dotted-path
        # failure modes and lets _set_plot_theme_from_env catch it uniformly.
        msg = f'"{spec}" does not resolve to a pyvista Theme subclass.'
        raise ValueError(msg)  # noqa: TRY004
    return cls
